# Deployment — Azure Container Apps

`main.bicep` is the course template (units 35–38) extended for this project. One deploy creates or
updates, in a single resource group:

| Resource | Name | Why |
|---|---|---|
| Container Registry (Basic) | `acr<uniqueString(rg)>` or `AZURE_ACR_NAME` | holds the images |
| Log Analytics workspace | `law-<app>` | container logs + App Insights store |
| Application Insights | `appi-<app>` | traces, requests, exceptions, (agent spans) — handout §11 |
| Container Apps environment | `cae-<app>` | runtime, logs to Log Analytics |
| Container App | `<app>` (default `hackathon2-app`) | the service, external HTTPS ingress → port 8000 |

The app gets the Azure OpenAI settings as env vars (the key as a Container Apps **secret**) and
`APPLICATIONINSIGHTS_CONNECTION_STRING`, which switches on Azure Monitor OpenTelemetry in
`src/hackathon2/telemetry.py`. It runs exactly **one replica** (`minReplicas = maxReplicas = 1`)
because, without Postgres on Azure, HITL checkpoints and the vector index live in memory.

## Deploy from your machine

```bash
az login
uv run scripts/deploy.py azure --dry-run    # see every command first; nothing is created
uv run scripts/deploy.py azure              # asks for confirmation, then deploys
```

Settings come from `.env` (or real environment variables, which win):

| Variable | Default | Note |
|---|---|---|
| `AZURE_RESOURCE_GROUP` | `rg-hackathon2` | **Course subscription: use your assigned group** (e.g. `rg-gtgh-14`) — accounts there cannot create groups; the script lists the ones you can use if it is wrong |
| `AZURE_LOCATION` | `germanywestcentral` | only used if the group must be created |
| `AZURE_CONTAINERAPP_NAME` | `hackathon2-app` | ≤ 32 chars, lowercase |
| `AZURE_ACR_NAME` | derived | globally unique; set it to reuse an existing registry |

The deploy runs the template twice — infra first (so the registry exists), then again with the new
image tag — and succeeds only when `https://<fqdn>/health` reports that tag.

## Deploy from GitHub Actions (cd.yml)

Every push to `main` runs CI, then the same `deploy.py azure`. One-time setup:

1. **Service principal** scoped to the resource group (Contributor is enough — the template needs
   no role assignments):
   ```bash
   az ad sp create-for-rbac --name "gh-hackathon2" --role contributor \
     --scopes /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP> --sdk-auth
   ```
   On the course subscription you may lack rights to create one — ask the instructor for the
   credentials JSON, or deploy from your machine instead.
2. **Repository secrets** (Settings → Secrets and variables → Actions):
   `AZURE_CREDENTIALS` (the JSON above), `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`.
3. **Repository variables** (optional; defaults in `cd.yml`): `AZURE_RESOURCE_GROUP` (set this on
   the course subscription), `AZURE_LOCATION`, `AZURE_CONTAINERAPP_NAME`, `AZURE_ACR_NAME`,
   `OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT_NAME`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`.
4. Optional: add **required reviewers** to the `production` environment for a manual approval gate.

## Observe

- Portal → Application Insights `appi-<app>` → **Live metrics**, **Transaction search**, **Failures**.
- Container logs: `az containerapp logs show -n <app> -g <group> --follow`
- `uv run scripts/deploy.py status --azure`

## Cost and cleanup

ACR Basic and Log Analytics ingestion are the steady costs; the single always-on 0.5 vCPU / 1 GiB
replica is small. When the hackathon is over:

```bash
uv run scripts/deploy.py teardown                 # this project's resources only (asks first)
uv run scripts/deploy.py teardown --whole-group   # the entire resource group (type its name to confirm)
```

The default is deliberately narrow: every resource the template creates is tagged
`project=hackathon2`, and teardown deletes exactly those, dependents first (app → environment →
App Insights → Log Analytics → registry). That matters on the course subscription, where your one
assigned group also holds other coursework (e.g. `acrlanggraphdemo`) **and you cannot recreate the
group once it is gone** — never use `--whole-group` there. A registry named explicitly through
`AZURE_ACR_NAME` is always kept, since it may be shared.
