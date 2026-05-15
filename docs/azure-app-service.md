# Deploy Transformer Risk Agent to Azure App Service (Linux)

This guide wires your Azure subscription and deploys the Streamlit app from your local repository root.

## 1) Connect Azure subscription

```bash
# Install Azure CLI first if needed: https://learn.microsoft.com/cli/azure/install-azure-cli
az login
az account list -o table
az account set --subscription "<SUBSCRIPTION_ID_OR_NAME>"
az account show -o table
```

## 2) Set deployment variables

```bash
RG="rg-transformer-risk-agent"
LOCATION="eastus"
PLAN="asp-transformer-risk-agent"
APP_NAME="transformer-risk-agent-<unique-suffix>"   # must be globally unique
RUNTIME="PYTHON:3.11"
```

## 3) Create resource group, plan, and web app

```bash
az group create --name "$RG" --location "$LOCATION"

az appservice plan create \
  --name "$PLAN" \
  --resource-group "$RG" \
  --sku B1 \
  --is-linux

az webapp create \
  --name "$APP_NAME" \
  --resource-group "$RG" \
  --plan "$PLAN" \
  --runtime "$RUNTIME"
```

## 4) Configure Streamlit startup and app settings

Set startup command so Streamlit binds to host `0.0.0.0` and platform port (`$PORT` fallback `8000`):

```bash
az webapp config set \
  --name "$APP_NAME" \
  --resource-group "$RG" \
  --startup-file "python -m streamlit run src/ui/app.py --server.address=0.0.0.0 --server.port \${PORT:-8000}"
```

Set required app settings:

```bash
az webapp config appsettings set \
  --name "$APP_NAME" \
  --resource-group "$RG" \
  --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true WEBSITES_PORT=8000
```

Optional: enable real LLM calls (otherwise app can run with mock fallback):

```bash
az webapp config appsettings set \
  --name "$APP_NAME" \
  --resource-group "$RG" \
  --settings OPENAI_API_KEY="<YOUR_KEY>" OPENAI_MODEL="gpt-4o-mini" LLM_PROVIDER="openai"
```

## 5) Deploy code from repo root

Run this from your repository root (the directory containing `README.md` and `requirements.txt`).

```bash
az webapp up \
  --name "$APP_NAME" \
  --resource-group "$RG" \
  --location "$LOCATION" \
  --runtime "$RUNTIME" \
  --sku B1
```

## 6) Validate deployment

```bash
az webapp show --name "$APP_NAME" --resource-group "$RG" --query defaultHostName -o tsv
az webapp log config --name "$APP_NAME" --resource-group "$RG" --application-logging filesystem --level information
az webapp log tail --name "$APP_NAME" --resource-group "$RG"
```

Then open:

`https://<defaultHostName>`

## 7) Hardening recommendations

- Enable Application Insights and Log Analytics.
- Restrict inbound access (IP restrictions or private endpoint).
- Use Key Vault references for secrets.
- Add GitHub Actions + OIDC for CI/CD.
