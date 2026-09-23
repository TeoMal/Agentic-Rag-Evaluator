"""The deploy script is pipeline code -- it gets tests like everything else."""

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy.py"
_spec = importlib.util.spec_from_file_location("deploy_script", SCRIPT)
deploy = importlib.util.module_from_spec(_spec)
sys.modules["deploy_script"] = deploy  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(deploy)

NOW = datetime(2026, 9, 23, 14, 5, 9, tzinfo=UTC)


def test_parse_env_handles_quotes_comments_and_export():
    text = (
        "# comment\n"
        'AZURE_OPENAI_API_KEY="abc=123"\n'
        "export OPENAI_API_VERSION='2024-12-01-preview'\n"
        "APP_PORT=8020 # inline comment\n"
        "EMPTY=\n"
        "not a setting\n"
    )
    assert deploy.parse_env(text) == {
        "AZURE_OPENAI_API_KEY": "abc=123",
        "OPENAI_API_VERSION": "2024-12-01-preview",
        "APP_PORT": "8020",
        "EMPTY": "",
    }


def test_set_env_value_replaces_the_key_only():
    text = "# AZURE_OPENAI_API_KEY= keep this comment\nAZURE_OPENAI_API_KEY=\nAZURE_OPENAI_API_KEY_OLD=x\n"
    updated = deploy.set_env_value(text, "AZURE_OPENAI_API_KEY", "s3cret")
    assert updated.splitlines() == [
        "# AZURE_OPENAI_API_KEY= keep this comment",
        'AZURE_OPENAI_API_KEY="s3cret"',
        "AZURE_OPENAI_API_KEY_OLD=x",
    ]
    assert deploy.parse_env(deploy.set_env_value("", "NEW", "v")) == {"NEW": "v"}


def test_image_tag_is_the_commit_for_a_clean_tree():
    assert deploy.make_image_tag("0123456789ab", dirty=False, now=NOW) == "0123456789ab"


def test_image_tag_is_unique_per_rebuild_of_uncommitted_code():
    assert deploy.make_image_tag("0123456789ab", dirty=True, now=NOW) == "0123456789ab-dirty-20260923140509"
    assert deploy.make_image_tag(None, dirty=False, now=NOW) == "dev-20260923140509"


def test_redact_masks_secrets_in_displayed_commands():
    shown = deploy.redact(["az", "login", "--password", "hunter2", "C:\\Program Files\\x"], {"hunter2"})
    assert "hunter2" not in shown
    assert shown == 'az login --password *** "C:\\Program Files\\x"'


def test_bicep_parameters_carry_the_key_and_optional_acr_name():
    cfg = {
        **deploy.AZURE_DEFAULTS,
        "AZURE_OPENAI_API_KEY": "k",
        "AZURE_OPENAI_ENDPOINT": "https://e/",
        "OPENAI_API_VERSION": "v",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "d",
    }
    params = deploy.bicep_parameters(cfg, deploy_app=True, image_tag="t1")["parameters"]
    assert params["azureOpenAiApiKey"] == {"value": "k"}
    assert params["deployApp"] == {"value": True}
    assert params["imageTag"] == {"value": "t1"}
    assert "acrName" not in params  # left to the template's uniqueString() default
    with_acr = deploy.bicep_parameters({**cfg, "AZURE_ACR_NAME": "acrteam1"}, deploy_app=False, image_tag="t1")
    assert with_acr["parameters"]["acrName"] == {"value": "acrteam1"}


def test_read_outputs_flattens_arm_outputs():
    arm = {"acrLoginServer": {"type": "String", "value": "acr1.azurecr.io"}, "appFqdn": {"type": "String", "value": ""}}
    assert deploy.read_outputs(arm) == {"acrloginserver": "acr1.azurecr.io", "appfqdn": ""}
    assert deploy.read_outputs(None) == {}


def test_teardown_deletes_dependents_first_and_spares_a_named_registry():
    rg = "/subscriptions/s/resourceGroups/rg-gtgh-14/providers/"
    ids = [
        rg + "Microsoft.ContainerRegistry/registries/acrshared",
        rg + "Microsoft.OperationalInsights/workspaces/law-hackathon2-app",
        rg + "Microsoft.App/managedEnvironments/cae-hackathon2-app",
        rg + "Microsoft.Insights/components/appi-hackathon2-app",
        rg + "Microsoft.App/containerApps/hackathon2-app",
    ]
    plan = deploy.teardown_plan(ids)
    assert [p.rsplit("/", 1)[-1] for p in plan] == [
        "hackathon2-app", "cae-hackathon2-app", "appi-hackathon2-app", "law-hackathon2-app", "acrshared",
    ]
    assert not any("registries" in p for p in deploy.teardown_plan(ids, keep_registry="ACRSHARED"))


def test_docker_desktop_crash_is_read_from_its_backend_log():
    crashed = "[com.docker.backend.exe] backend crashed, dumping error to file and reporting to user: "
    lines = [
        "[2026-09-22T22:17:37.270377500Z]" + crashed + "starting services: initializing Ingest server: old",
        "[2026-09-22T22:22:38.895115900Z][com.docker.backend.exe.engines] all local engines stopped",
        "[2026-09-22T22:22:38.895115900Z]" + crashed + "starting services: initializing Secrets Engine: rename engine.sock",
    ]
    launched = datetime(2026, 9, 22, 22, 22, 34, 500000, tzinfo=UTC)
    assert deploy.crash_reason_from_log(lines, launched) == (
        "starting services: initializing Secrets Engine: rename engine.sock"
    )
    # A crash from an earlier launch must not be blamed on this one.
    assert deploy.crash_reason_from_log(lines[:2], launched) is None
    assert deploy.crash_reason_from_log([], launched) is None
