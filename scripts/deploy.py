#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""One-pass rebuild + deploy for the Hackathon 2 service.

    uv run scripts/deploy.py                   local: test, rebuild image, recreate container, verify
    uv run scripts/deploy.py azure             Azure: Bicep infra, build, push to ACR, roll out, verify
    uv run scripts/deploy.py status [--azure]  what is running, and which image it is
    uv run scripts/deploy.py down [--volumes]  stop the local stack
    uv run scripts/deploy.py teardown          delete this project's Azure resources (asks first)

Flags: --tag TAG  --skip-tests  --no-cache  --dry-run  --yes  --timeout SECONDS

"One pass" means every run goes all the way from source to a verified, running
container: the image is always rebuilt, the app container is always recreated,
and the run only succeeds once /health answers with the image tag THIS run
built. A stale container can never pass for a fresh deploy.

Stdlib only, so `uv run` starts it instantly without syncing the project, and
the same code runs on Windows, macOS, Linux and in GitHub Actions (cd.yml calls
`deploy.py azure`) -- local and CI deploys cannot drift apart. Every run is
logged to logs/deploy-NNNN-<timestamp>-<command>.log.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"
BICEP_FILE = ROOT / "deployment" / "main.bicep"
LOG_DIR = ROOT / "logs"

IMAGE_NAME = "hackathon2-app"
APP_SERVICE = "app"
DB_SERVICE = "db"
COMPOSE = ["docker", "compose", "--project-directory", str(ROOT), "-f", str(ROOT / "docker-compose.yml")]

# The only values with no shared default -- everything else ships in .env.example.
REQUIRED_LLM_KEYS = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
)
SECRET_KEYS = {"AZURE_OPENAI_API_KEY", "POSTGRES_PASSWORD"}
AZURE_DEFAULTS = {
    "AZURE_RESOURCE_GROUP": "rg-hackathon2",
    "AZURE_LOCATION": "germanywestcentral",
    "AZURE_CONTAINERAPP_NAME": "hackathon2-app",
}
# Real environment variables win over .env (same rule as Compose and pydantic-settings).
ENV_OVERRIDE_PREFIXES = ("AZURE_", "OPENAI_", "APP_", "POSTGRES_")
# main.bicep tags everything it creates with project=<PROJECT_TAG>; teardown deletes
# exactly those, dependents first (the app before its environment, etc.).
PROJECT_TAG = "hackathon2"
TEARDOWN_ORDER = (
    "microsoft.app/containerapps",
    "microsoft.app/managedenvironments",
    "microsoft.insights/components",
    "microsoft.operationalinsights/workspaces",
    "microsoft.containerregistry/registries",
)
DEFAULT_APP_PORT = "8020"
DEFAULT_TIMEOUT = {"local": 180, "azure": 420}
DOCKER_START_TIMEOUT = 300  # a cold Docker Desktop start boots a WSL VM first


def local_now() -> datetime:
    """Timezone-aware local time (log names, tags and deployment names are local-clock)."""
    return datetime.now().astimezone()


class DeployError(Exception):
    """A step failed. The message says what, and what to do about it."""


# ---------------------------------------------------------------------------
# Pure helpers -- unit-tested in tests/test_deploy_script.py
# ---------------------------------------------------------------------------


def parse_env(text: str) -> dict[str, str]:
    """Parse .env text: KEY=VALUE lines, optional `export`, quotes stripped, comments skipped."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def set_env_value(text: str, key: str, value: str) -> str:
    """Return .env text with KEY set to value: replace its line if present, else append."""
    quote = "'" if '"' in value else '"'
    new_line = f"{key}={quote}{value}{quote}"
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        existing = raw.strip().removeprefix("export ").split("=", 1)[0].strip()
        if "=" in raw and not raw.lstrip().startswith("#") and existing == key:
            lines[i] = new_line
            return "\n".join(lines) + "\n"
    return "\n".join([*lines, new_line]) + "\n"


def make_image_tag(sha: str | None, dirty: bool, now: datetime) -> str:
    """Commit SHA for a clean tree. A dirty tree or no git gets a timestamp, so every
    rebuild of uncommitted code is a distinct tag -- Azure only rolls out a new
    revision when the tag changes, and /health verification relies on it too."""
    stamp = now.strftime("%Y%m%d%H%M%S")
    if sha and not dirty:
        return sha
    if sha:
        return f"{sha}-dirty-{stamp}"
    return f"dev-{stamp}"


def mask(text: str, secrets: Iterable[str]) -> str:
    """Replace every secret value in text with ***."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


def redact(args: Iterable[str], secrets: Iterable[str]) -> str:
    """Render a command for display with every secret value masked."""
    secrets = list(secrets)
    shown = [mask(arg, secrets) for arg in args]
    return " ".join(f'"{arg}"' if not arg or any(c.isspace() for c in arg) else arg for arg in shown)


def bicep_parameters(cfg: dict[str, str], *, deploy_app: bool, image_tag: str) -> dict:
    """ARM parameters file for deployment/main.bicep. Written to a temp file so the
    API key never appears on a command line or in the log."""
    params: dict[str, object] = {
        "location": cfg["AZURE_LOCATION"],
        "appName": cfg["AZURE_CONTAINERAPP_NAME"],
        "imageName": IMAGE_NAME,
        "imageTag": image_tag,
        "deployApp": deploy_app,
        "azureOpenAiEndpoint": cfg.get("AZURE_OPENAI_ENDPOINT", ""),
        "azureOpenAiApiVersion": cfg.get("OPENAI_API_VERSION", ""),
        "azureOpenAiDeployment": cfg.get("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
        "azureOpenAiEmbeddingDeployment": cfg.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", ""),
        "azureOpenAiApiKey": cfg.get("AZURE_OPENAI_API_KEY", ""),
    }
    if cfg.get("AZURE_ACR_NAME"):
        params["acrName"] = cfg["AZURE_ACR_NAME"]
    return {
        "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
        "contentVersion": "1.0.0.0",
        "parameters": {name: {"value": value} for name, value in params.items()},
    }


def read_outputs(outputs: dict | None) -> dict[str, str]:
    """Flatten ARM deployment outputs to {lowercased name: value}."""
    return {name.lower(): str(entry.get("value", "")) for name, entry in (outputs or {}).items()}


DOCKER_CRASH_MARKER = "backend crashed, dumping error to file and reporting to user: "


def crash_reason_from_log(lines: list[str], since: datetime) -> str | None:
    """Latest Docker Desktop backend crash reason logged at or after `since` (UTC).
    Log lines look like: [2026-09-22T22:22:38.895115900Z][com.docker.backend.exe] backend crashed, ..."""
    for line in reversed(lines):
        if DOCKER_CRASH_MARKER not in line or not line.startswith("["):
            continue
        try:
            stamp = datetime.fromisoformat(line[1:20] + "+00:00")  # to the second is enough
        except ValueError:
            return None
        return line.split(DOCKER_CRASH_MARKER, 1)[1].strip() if stamp >= since.replace(microsecond=0) else None
    return None


def teardown_plan(resource_ids: Iterable[str], keep_registry: str | None = None) -> list[str]:
    """Order this project's resources for deletion (dependents first), dropping a
    registry that was named explicitly -- it may be shared with other work."""

    def resource_type(resource_id: str) -> str:
        provider_path = resource_id.split("/providers/", 1)[-1].split("/")
        return "/".join(provider_path[:2]).lower()

    def resource_name(resource_id: str) -> str:
        return resource_id.rstrip("/").rsplit("/", 1)[-1].lower()

    ids = [
        rid for rid in resource_ids
        if not (keep_registry and resource_type(rid) == "microsoft.containerregistry/registries"
                and resource_name(rid) == keep_registry.lower())
    ]
    rank = {t: i for i, t in enumerate(TEARDOWN_ORDER)}
    return sorted(ids, key=lambda rid: rank.get(resource_type(rid), len(rank)))


# ---------------------------------------------------------------------------
# Console + process runner
# ---------------------------------------------------------------------------


class Console:
    """Prints to the terminal and mirrors everything into the run's log file."""

    def __init__(self, log_path: Path | None) -> None:
        self.log_path = log_path
        self._log = log_path.open("a", encoding="utf-8") if log_path else None

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        if self._log:
            self._log.write(text)
            self._log.flush()

    def say(self, message: str = "") -> None:
        self.write(message + "\n")

    def step(self, title: str) -> None:
        self.say(f"\n==> {title}")

    def close(self) -> None:
        if self._log:
            self._log.close()


@dataclass
class Result:
    returncode: int
    output: str = ""
    error: str = ""


class Runner:
    def __init__(self, console: Console, dry_run: bool = False) -> None:
        self.console = console
        self.dry_run = dry_run
        self.secrets: set[str] = set()

    def run(
        self,
        args: list[str],
        *,
        mutating: bool = True,
        capture: bool = False,
        check: bool = True,
        env: dict[str, str] | None = None,
        quiet: bool = False,
        timeout: float | None = None,
    ) -> Result:
        """Run a command. Streams output (terminal + log) unless capture=True.
        In --dry-run, mutating commands are printed but not executed.
        `timeout` (capture only) turns a hung command into returncode 124."""
        exe = shutil.which(args[0])
        if exe is None:
            raise DeployError(f"'{args[0]}' is not installed or not on PATH.")
        if not quiet:
            self.console.say("$ " + redact(args, self.secrets))
        if self.dry_run and mutating:
            self.console.say("  (dry run: not executed)")
            return Result(0, "")

        full_env = {**os.environ, **(env or {})}
        # `uv run scripts/deploy.py` points VIRTUAL_ENV at this script's throwaway
        # env; drop it so nested `uv run pytest` uses the project's .venv quietly.
        full_env.pop("VIRTUAL_ENV", None)
        cmd = [exe, *args[1:]]

        if capture:
            try:
                proc = subprocess.run(
                    cmd, cwd=ROOT, env=full_env, capture_output=True, check=False,
                    text=True, encoding="utf-8", errors="replace", timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                if check:
                    raise DeployError(f"`{redact(args[:3], self.secrets)} ...` timed out after {timeout:.0f}s") from None
                return Result(124, "", f"timed out after {timeout:.0f}s")
            if check and proc.returncode != 0:
                detail = mask((proc.stderr or proc.stdout).strip(), self.secrets)
                raise DeployError(f"`{redact(args[:3], self.secrets)} ...` failed (exit {proc.returncode}):\n{detail}")
            return Result(proc.returncode, proc.stdout, mask(proc.stderr, self.secrets))

        proc = subprocess.Popen(cmd, cwd=ROOT, env=full_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            self.console.write(raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n"))
        returncode = proc.wait()
        if check and returncode != 0:
            raise DeployError(f"command failed (exit {returncode}): {redact(args, self.secrets)}")
        return Result(returncode)


# ---------------------------------------------------------------------------
# Preflight building blocks
# ---------------------------------------------------------------------------


def next_log_path(command: str) -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    counter = LOG_DIR / ".run_counter"
    try:
        number = int(counter.read_text().strip()) + 1
    except (FileNotFoundError, ValueError):
        number = 1
    counter.write_text(str(number))
    return LOG_DIR / f"deploy-{number:04d}-{local_now():%Y-%m-%dT%H-%M-%S}-{command}.log"


def load_config() -> dict[str, str]:
    """Defaults < .env < real environment variables."""
    file_values = parse_env(ENV_FILE.read_text(encoding="utf-8")) if ENV_FILE.exists() else {}
    config = {**AZURE_DEFAULTS, **{k: v for k, v in file_values.items() if v}}
    config.update({k: v for k, v in os.environ.items() if k.startswith(ENV_OVERRIDE_PREFIXES) and v})
    return config


def ensure_env_file(console: Console) -> None:
    if ENV_FILE.exists():
        return
    if not ENV_EXAMPLE.exists():
        raise DeployError(f"neither .env nor .env.example exists in {ROOT}")
    shutil.copyfile(ENV_EXAMPLE, ENV_FILE)
    console.say("created .env from .env.example")


def prompt_for_missing(console: Console, missing: list[str]) -> None:
    """Ask once for each missing value and save it to .env; fail loudly without a terminal."""
    if not missing:
        return
    if not sys.stdin.isatty():
        raise DeployError(
            "missing required settings: " + ", ".join(missing)
            + "\n  Put them in .env (or export them), or run from a terminal to be prompted."
        )
    ensure_env_file(console)
    console.say(f"\n  {len(missing)} value(s) needed -- asked once, then saved to .env (gitignored).")
    text = ENV_FILE.read_text(encoding="utf-8")
    for key in missing:
        value = ""
        while not value:
            value = (getpass.getpass(f"  {key} (hidden): ") if key in SECRET_KEYS else input(f"  {key}: ")).strip()
        text = set_env_value(text, key, value)
    ENV_FILE.write_text(text, encoding="utf-8")
    console.say("  saved to .env")


def docker_running(runner: Runner) -> bool:
    # A half-started daemon accepts the connection and never answers -- without
    # a timeout this probe hangs forever and the startup deadline never fires.
    probe = runner.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                       mutating=False, capture=True, check=False, quiet=True, timeout=15)
    return probe.returncode == 0


def docker_desktop_crash(since: datetime) -> str | None:
    """Windows: the reason Docker Desktop's backend crashed after `since`, if it did.

    On a startup crash the GUI stays open on an error dialog, so the process looks
    alive; the backend log is the reliable signal. Best effort -- None if unknown."""
    if platform.system() != "Windows":
        return None
    log = Path(os.environ.get("LOCALAPPDATA", "")) / "Docker" / "log" / "host" / "com.docker.backend.exe.log"
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()[-400:]
    except OSError:
        return None
    return crash_reason_from_log(lines, since)


def start_docker_desktop() -> bool:
    system = platform.system()
    if system == "Windows":
        exe = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Docker" / "Docker" / "Docker Desktop.exe"
        if not exe.exists():
            return False
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        subprocess.Popen([str(exe)], creationflags=flags, close_fds=True)
        return True
    if system == "Darwin":
        return subprocess.run(["open", "-a", "Docker"], check=False).returncode == 0
    return False


def docker_desktop_alive() -> bool | None:
    """Windows: is the Docker Desktop process still running? None where we cannot tell."""
    if platform.system() != "Windows":
        return None
    tasks = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Docker Desktop.exe", "/NH"],
                           capture_output=True, text=True, errors="replace", check=False)
    return "Docker Desktop.exe" in tasks.stdout


def ensure_docker(runner: Runner, console: Console) -> None:
    if shutil.which("docker") is None:
        raise DeployError("'docker' is not on PATH. Install Docker Desktop (or the Docker Engine) first.")
    if docker_running(runner):
        console.say("docker daemon: running")
        return
    if runner.dry_run:
        console.say("docker daemon: NOT running (dry run: not starting it)")
        return
    if not start_docker_desktop():
        raise DeployError("the Docker daemon is not running. Start it (e.g. `sudo systemctl start docker`) and retry.")
    console.say(f"docker daemon: not running -- starting Docker Desktop (waiting up to {DOCKER_START_TIMEOUT}s)")
    launched_at = datetime.now(tz=UTC)
    started = time.monotonic()
    while time.monotonic() - started < DOCKER_START_TIMEOUT:
        time.sleep(3)
        if docker_running(runner):
            console.say(f"docker daemon: running (after {time.monotonic() - started:.0f}s)")
            return
        # Stop as soon as the answer is known: a Docker Desktop that crashed on
        # startup does not recover by waiting out the full timeout.
        reason = docker_desktop_crash(launched_at)
        if reason:
            raise DeployError(
                f"Docker Desktop crashed during startup:\n  {reason}\n"
                "  A 'listening on unix://...sock: rename ...' error means socket files left by an unclean\n"
                "  shutdown: quit Docker Desktop, move the folder named in the error aside, and retry."
            )
        if time.monotonic() - started > 20 and docker_desktop_alive() is False:
            raise DeployError(
                "Docker Desktop started and then quit -- it failed during startup.\n"
                r"  The reason is in its error dialog and in %LOCALAPPDATA%\Docker\log\host\com.docker.backend.exe.log"
            )
    raise DeployError(f"Docker did not come up within {DOCKER_START_TIMEOUT}s. Open Docker Desktop, check for errors, retry.")


def git_image_tag(runner: Runner) -> str:
    sha, dirty = None, False
    if shutil.which("git"):
        head = runner.run(["git", "rev-parse", "--short=12", "HEAD"], mutating=False, capture=True, check=False, quiet=True)
        if head.returncode == 0:
            sha = head.output.strip() or None
        if sha:
            status = runner.run(["git", "status", "--porcelain"], mutating=False, capture=True, check=False, quiet=True)
            dirty = bool(status.output.strip())
    return make_image_tag(sha, dirty, local_now())


def lock_and_test(args: argparse.Namespace, console: Console, runner: Runner) -> None:
    uv = os.environ.get("UV") or "uv"  # `uv run` exports UV = path of the running uv binary
    lock = runner.run([uv, "lock", "--check"], mutating=False, capture=True, check=False)
    if lock.returncode != 0:
        console.say("uv.lock is out of date with pyproject.toml -- relocking (the image builds with --locked)")
        runner.run([uv, "lock"])
    else:
        console.say("uv.lock: in sync with pyproject.toml")
    if args.skip_tests:
        console.say("tests: skipped (--skip-tests)")
        return
    runner.run([uv, "run", "--locked", "pytest", "-q"])


def confirm(args: argparse.Namespace, question: str) -> None:
    if args.yes or args.dry_run:
        return
    if not sys.stdin.isatty():
        raise DeployError(f"{question}\n  No terminal to confirm on -- pass --yes to proceed non-interactively.")
    if input(f"{question} [y/N] ").strip().lower() not in ("y", "yes"):
        raise DeployError("aborted -- nothing was changed")


def fetch_health(url: str, timeout: float = 5) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def wait_for_health(console: Console, url: str, expected_tag: str, timeout: int) -> dict:
    """Poll until /health answers with THIS run's image tag -- proof the new image is serving."""
    console.say(f"polling {url} for image_tag={expected_tag} (up to {timeout}s)")
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        body = fetch_health(url)
        if body is not None:
            last = body
            if body.get("image_tag") == expected_tag:
                return body
        time.sleep(3)
    if last is not None:
        raise DeployError(
            f"{url} answers, but with image_tag={last.get('image_tag')!r} instead of {expected_tag!r} "
            "-- the new image is not the one serving."
        )
    raise DeployError(f"{url} did not answer within {timeout}s.")


def describe_checks(body: dict) -> str:
    return "  ".join(f"{k}={v}" for k, v in body.get("checks", {}).items())


def azure_account(runner: Runner) -> dict:
    try:
        return json.loads(runner.run(["az", "account", "show", "-o", "json"], mutating=False, capture=True).output)
    except DeployError as exc:
        raise DeployError(f"not logged in to Azure -- run `az login` first.\n{exc}") from None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_local(args: argparse.Namespace, console: Console, runner: Runner) -> None:
    timeout = args.timeout or DEFAULT_TIMEOUT["local"]

    console.step("1/6 Preflight")
    ensure_docker(runner, console)
    ensure_env_file(console)
    env_values = parse_env(ENV_FILE.read_text(encoding="utf-8"))
    prompt_for_missing(console, [k for k in REQUIRED_LLM_KEYS if not env_values.get(k)])
    env_values = parse_env(ENV_FILE.read_text(encoding="utf-8"))
    runner.secrets.update(v for k, v in env_values.items() if k in SECRET_KEYS and v)
    config = load_config()
    port = config.get("APP_PORT", DEFAULT_APP_PORT)
    tag = args.tag or git_image_tag(runner)
    console.say(f"image: {IMAGE_NAME}:{tag}")

    console.step("2/6 Lockfile and tests")
    lock_and_test(args, console, runner)

    compose_env = {"IMAGE_TAG": tag}
    console.step("3/6 Build image")
    build = [*COMPOSE, "build", APP_SERVICE]
    if args.no_cache:
        build += ["--no-cache", "--pull"]
    runner.run(build, env=compose_env)

    console.step("4/6 Database (pgvector)")
    runner.run([*COMPOSE, "up", "-d", "--wait", "--wait-timeout", str(timeout), DB_SERVICE], env=compose_env)

    try:
        console.step("5/6 Recreate app container")
        runner.run(
            [*COMPOSE, "up", "-d", "--no-deps", "--force-recreate", "--remove-orphans",
             "--wait", "--wait-timeout", str(timeout), APP_SERVICE],
            env=compose_env,
        )
        console.step("6/6 Verify")
        if runner.dry_run:
            console.say("(dry run: skipping health verification)")
            return
        body = wait_for_health(console, f"http://127.0.0.1:{port}/health", tag, timeout)
    except DeployError:
        console.say("\n--- diagnostics ---")
        runner.run([*COMPOSE, "ps", "-a"], mutating=False, check=False, env=compose_env)
        runner.run([*COMPOSE, "logs", "--tail", "60", APP_SERVICE], mutating=False, check=False, env=compose_env)
        raise

    console.say(f"healthy: {describe_checks(body)}")
    console.say("")
    console.say(f"  App       http://127.0.0.1:{port}        (API docs: /docs, health: /health)")
    console.say(f"  Image     {IMAGE_NAME}:{tag}")
    console.say(f"  Database  127.0.0.1:{config.get('POSTGRES_PORT', '5446')}  db={config.get('POSTGRES_DB', 'hackathon2')}")
    console.say("  Logs      docker compose logs -f app")


def cmd_azure(args: argparse.Namespace, console: Console, runner: Runner) -> None:
    timeout = args.timeout or DEFAULT_TIMEOUT["azure"]

    console.step("1/6 Preflight")
    if shutil.which("az") is None:
        raise DeployError("Azure CLI ('az') is not on PATH: https://learn.microsoft.com/cli/azure/install-azure-cli")
    ensure_docker(runner, console)
    config = load_config()
    missing = [k for k in REQUIRED_LLM_KEYS if not config.get(k)]
    if missing:
        prompt_for_missing(console, missing)
        config = load_config()
    runner.secrets.update(v for k, v in config.items() if k in SECRET_KEYS and v)
    account = azure_account(runner)
    group, location, app = config["AZURE_RESOURCE_GROUP"], config["AZURE_LOCATION"], config["AZURE_CONTAINERAPP_NAME"]
    tag = args.tag or git_image_tag(runner)
    console.say(f"subscription    {account.get('name')} ({account.get('id')})")
    console.say(f"resource group  {group}  ({location})")
    console.say(f"container app   {app}")
    console.say(f"image tag       {tag}")
    confirm(args, f"Deploy {IMAGE_NAME}:{tag} to '{group}' in subscription '{account.get('name')}'?")

    console.step("2/6 Lockfile and tests")
    lock_and_test(args, console, runner)

    console.step("3/6 Infrastructure (Bicep: registry, monitoring, environment)")
    ensure_resource_group(runner, console, group, location)
    outputs = bicep_deploy(runner, console, config, deploy_app=False, image_tag=tag)
    acr_name = outputs.get("acrname", "<acr-name>")
    image = f"{outputs.get('acrloginserver', '<acr-name>.azurecr.io')}/{IMAGE_NAME}:{tag}"

    console.step("4/6 Build and push image")
    # Container Apps runs linux/amd64; pinning it keeps Apple-silicon builds deployable.
    build = ["docker", "build", "--platform", "linux/amd64", "--build-arg", f"IMAGE_TAG={tag}", "-t", image]
    if args.no_cache:
        build += ["--no-cache", "--pull"]
    runner.run([*build, str(ROOT)])
    runner.run(["az", "acr", "login", "-n", acr_name])
    runner.run(["docker", "push", image])

    console.step("5/6 Roll out the new revision")
    outputs = bicep_deploy(runner, console, config, deploy_app=True, image_tag=tag)
    fqdn = outputs.get("appfqdn", "")

    console.step("6/6 Verify")
    if runner.dry_run:
        console.say("(dry run: skipping health verification)")
        return
    if not fqdn:
        raise DeployError("the deployment returned no app FQDN -- check the Container App in the portal.")
    body = wait_for_health(console, f"https://{fqdn}/health", tag, timeout)
    console.say(f"healthy: {describe_checks(body)}")
    console.say("")
    console.say(f"  App        https://{fqdn}   (API docs: /docs)")
    console.say(f"  Image      {image}")
    console.say(f"  Telemetry  Application Insights '{outputs.get('appinsightsname', '')}' in {group}")
    console.say(f"  Logs       az containerapp logs show -n {app} -g {group} --follow")


def ensure_resource_group(runner: Runner, console: Console, group: str, location: str) -> None:
    """Use the group if we can read it; create it only if it truly does not exist.

    `az group show` needs access to this one group only. `az group exists` needs
    subscription-wide read, which course/sandbox accounts (access to a single
    assigned group) do not have -- it fails with a bare 'Forbidden'."""
    show = runner.run(["az", "group", "show", "-n", group, "-o", "none"], mutating=False, capture=True, check=False)
    if show.returncode == 0:
        console.say(f"resource group '{group}': exists")
        return
    if "ResourceGroupNotFound" in show.error:
        runner.run(["az", "group", "create", "-n", group, "-l", location, "-o", "none"])
        return
    visible = runner.run(["az", "group", "list", "--query", "[].name", "-o", "tsv"],
                         mutating=False, capture=True, check=False, quiet=True).output.split()
    hint = f"Groups you can use: {', '.join(visible)}." if visible else "You cannot see any resource group."
    raise DeployError(
        f"no access to resource group '{group}' (and no permission to create it).\n"
        f"  {hint} Set AZURE_RESOURCE_GROUP in .env (or as a CI variable)."
    )


def bicep_deploy(runner: Runner, console: Console, config: dict[str, str], *, deploy_app: bool, image_tag: str) -> dict[str, str]:
    params = bicep_parameters(config, deploy_app=deploy_app, image_tag=image_tag)
    fd, params_path = tempfile.mkstemp(prefix="hackathon2-params-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(params, fh)
        name = f"hackathon2-{'app' if deploy_app else 'infra'}-{local_now():%Y%m%d%H%M%S}"
        console.say("(ARM deployment -- usually 1-3 minutes, no output until it finishes)")
        result = runner.run(
            ["az", "deployment", "group", "create", "-g", config["AZURE_RESOURCE_GROUP"], "-n", name,
             "-f", str(BICEP_FILE), "-p", f"@{params_path}", "--query", "properties.outputs", "-o", "json"],
            capture=True,
        )
    finally:
        os.remove(params_path)
    outputs = read_outputs(json.loads(result.output)) if result.output.strip() else {}
    for key, value in outputs.items():
        if value:
            console.say(f"  {key} = {value}")
    return outputs


def cmd_status(args: argparse.Namespace, console: Console, runner: Runner) -> None:
    config = load_config()
    console.step("Local")
    if shutil.which("docker") and docker_running(runner):
        runner.run([*COMPOSE, "ps", "-a"], mutating=False, check=False)
    else:
        console.say("docker daemon: not running")
    url = f"http://127.0.0.1:{config.get('APP_PORT', DEFAULT_APP_PORT)}/health"
    body = fetch_health(url)
    console.say(json.dumps(body, indent=2) if body else f"{url}: not answering")

    if not args.azure:
        return
    console.step("Azure")
    fqdn = runner.run(
        ["az", "containerapp", "show", "-n", config["AZURE_CONTAINERAPP_NAME"], "-g", config["AZURE_RESOURCE_GROUP"],
         "--query", "properties.configuration.ingress.fqdn", "-o", "tsv"],
        mutating=False, capture=True, check=False,
    ).output.strip()
    if not fqdn:
        console.say(f"no container app '{config['AZURE_CONTAINERAPP_NAME']}' in '{config['AZURE_RESOURCE_GROUP']}'")
        return
    body = fetch_health(f"https://{fqdn}/health", timeout=15)
    console.say(json.dumps(body, indent=2) if body else f"https://{fqdn}/health: not answering")


def cmd_down(args: argparse.Namespace, console: Console, runner: Runner) -> None:
    command = [*COMPOSE, "down", "--remove-orphans"]
    if args.volumes:
        confirm(args, "Also delete the local database volume (all indexed data and checkpoints)?")
        command.append("--volumes")
    runner.run(command)


def cmd_teardown(args: argparse.Namespace, console: Console, runner: Runner) -> None:
    """Default: delete only the resources main.bicep tagged project=hackathon2, so a
    shared group (course subscription: one assigned group you cannot recreate) survives.
    --whole-group deletes the resource group itself."""
    config = load_config()
    group = config["AZURE_RESOURCE_GROUP"]
    account = azure_account(runner)

    if args.whole_group:
        console.say(
            f"This DELETES resource group '{group}' and EVERYTHING in it -- including resources that are "
            f"not part of this project -- in subscription '{account.get('name')}'."
        )
        if not (args.yes or args.dry_run):
            if not sys.stdin.isatty():
                raise DeployError("pass --yes to delete non-interactively")
            if input("Type the resource group name to confirm: ").strip() != group:
                raise DeployError("name did not match -- nothing was deleted")
        runner.run(["az", "group", "delete", "-n", group, "--yes", "--no-wait"])
        console.say("deletion started in the background (takes 2-5 minutes); billing stops when it completes")
        return

    listing = runner.run(
        ["az", "resource", "list", "-g", group, "--tag", f"project={PROJECT_TAG}", "--query", "[].id", "-o", "tsv"],
        mutating=False, capture=True,
    )
    plan = teardown_plan(listing.output.split(), keep_registry=config.get("AZURE_ACR_NAME"))
    if config.get("AZURE_ACR_NAME"):
        console.say(f"keeping registry '{config['AZURE_ACR_NAME']}' -- AZURE_ACR_NAME is set, so it may be shared")
    if not plan:
        console.say(f"nothing tagged project={PROJECT_TAG} in '{group}' -- nothing to delete")
        return
    console.say(f"resources of this project in '{group}' (deleted in this order):")
    for resource_id in plan:
        console.say(f"  - {resource_id.split('/providers/', 1)[-1]}")
    confirm(args, f"Delete these {len(plan)} resources? Everything else in '{group}' is left alone.")
    for resource_id in plan:
        runner.run(["az", "resource", "delete", "--ids", resource_id])
    console.say("done -- note: Log Analytics workspaces stay soft-deleted (recoverable) for 14 days")


COMMANDS = {
    "local": cmd_local,
    "azure": cmd_azure,
    "status": cmd_status,
    "down": cmd_down,
    "teardown": cmd_teardown,
}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="deploy.py",
        description="Rebuild and deploy the Hackathon 2 service in one pass.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 2)[1] if __doc__ else None,
    )
    parser.add_argument("command", nargs="?", default="local", choices=COMMANDS, help="default: local")
    parser.add_argument("--tag", help="image tag to build/deploy (default: git SHA, +timestamp if dirty)")
    parser.add_argument("--skip-tests", action="store_true", help="do not run pytest before building")
    parser.add_argument("--no-cache", action="store_true", help="rebuild every layer and re-pull base images")
    parser.add_argument("--dry-run", action="store_true", help="print mutating commands instead of running them")
    parser.add_argument("-y", "--yes", action="store_true", help="skip confirmation prompts (CI)")
    parser.add_argument("--timeout", type=int, help="seconds to wait for health (default: 180 local, 420 azure)")
    parser.add_argument("--volumes", action="store_true", help="down: also delete the database volume")
    parser.add_argument("--azure", action="store_true", help="status: also check the Azure deployment")
    parser.add_argument("--whole-group", action="store_true",
                        help="teardown: delete the entire resource group, not just this project's resources")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")  # never die on a non-UTF-8 Windows console
    args = parse_args(argv)
    log_path = next_log_path(args.command)
    console = Console(log_path)
    runner = Runner(console, dry_run=args.dry_run)
    started = time.monotonic()
    console.say(f"== hackathon2 deploy: {args.command}{' (dry run)' if args.dry_run else ''} "
                f"-- {local_now():%Y-%m-%d %H:%M:%S} ==")
    console.say(f"   log: {log_path.relative_to(ROOT)}")
    try:
        COMMANDS[args.command](args, console, runner)
    except DeployError as exc:
        console.say(f"\n[FAILED] {exc}")
        console.say(f"         full log: {log_path}")
        console.close()
        return 1
    except KeyboardInterrupt:
        console.say("\n[ABORTED] interrupted")
        console.close()
        return 130
    console.say(f"\n[OK] {args.command} finished in {time.monotonic() - started:.0f}s")
    console.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
