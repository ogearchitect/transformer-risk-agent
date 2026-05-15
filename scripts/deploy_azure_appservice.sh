#!/usr/bin/env bash
set -euo pipefail

if ! command -v az >/dev/null 2>&1; then
  echo "Azure CLI (az) is required. Install it first."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ ! -d "$REPO_ROOT" ]]; then
  echo "Repo directory not found: $REPO_ROOT"
  exit 1
fi

RG="${RG:-rg-transformer-risk-agent}"
LOCATION="${LOCATION:-eastus}"
PLAN="${PLAN:-asp-transformer-risk-agent}"
APP_NAME="${APP_NAME:-}"
RUNTIME="${RUNTIME:-PYTHON:3.11}"
SKU="${SKU:-B1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
LLM_PROVIDER="${LLM_PROVIDER:-openai}"

if [[ -z "$APP_NAME" ]]; then
  echo "APP_NAME is required and must be globally unique."
  echo "Example: APP_NAME=transformer-risk-agent-1234 $0"
  exit 1
fi

echo "Using subscription:"
az account show -o table

echo "Creating resource group and plan..."
az group create --name "$RG" --location "$LOCATION"
az appservice plan create --name "$PLAN" --resource-group "$RG" --sku "$SKU" --is-linux

echo "Creating web app..."
az webapp create --name "$APP_NAME" --resource-group "$RG" --plan "$PLAN" --runtime "$RUNTIME"

echo "Configuring startup command and app settings..."
az webapp config set \
  --name "$APP_NAME" \
  --resource-group "$RG" \
  --startup-file "python -m streamlit run src/ui/app.py --server.address=0.0.0.0 --server.port \${PORT:-8000}"

az webapp config appsettings set \
  --name "$APP_NAME" \
  --resource-group "$RG" \
  --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true WEBSITES_PORT=8000

if [[ -n "$OPENAI_API_KEY" ]]; then
  az webapp config appsettings set \
    --name "$APP_NAME" \
    --resource-group "$RG" \
    --settings OPENAI_API_KEY="$OPENAI_API_KEY" OPENAI_MODEL="$OPENAI_MODEL" LLM_PROVIDER="$LLM_PROVIDER"
fi

echo "Deploying from: $REPO_ROOT"
cd "$REPO_ROOT"
az webapp up --name "$APP_NAME" --resource-group "$RG" --location "$LOCATION" --runtime "$RUNTIME" --sku "$SKU"

HOSTNAME="$(az webapp show --name "$APP_NAME" --resource-group "$RG" --query defaultHostName -o tsv)"
echo "Deployment complete: https://$HOSTNAME"
echo "To stream logs: az webapp log tail --name $APP_NAME --resource-group $RG"
