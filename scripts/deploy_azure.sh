#!/usr/bin/env bash
# Deploy the Transformer Risk Agent (Streamlit app) to Azure Container Apps.
#
# Prereqs:
#   - Azure CLI logged in (`az login`)
#   - Subscription selected (`az account set -s <SUB_ID>`)
#   - `az extension add -n containerapp` (the script auto-installs if missing)
#   - `./scripts/provision_foundry.sh` already run for the target resource group
#
# Usage:
#   ./scripts/deploy_azure.sh                      # uses defaults below
#   LOCATION=westus3 APP_NAME=my-app ./scripts/deploy_azure.sh
#
# What it does:
#   1. Registers required resource providers (idempotent).
#   2. Creates the resource group if missing.
#   3. Runs `az containerapp up` which:
#        - creates an ACR, builds the image from this repo via ACR cloud build,
#        - creates a Container Apps environment + Log Analytics workspace,
#        - deploys/updates the container app with HTTPS ingress on port 8501.
#   4. Prints the public FQDN.
#
# Re-running this script is safe; it updates the existing app in place.

set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-transformer-risk-agent}"
LOCATION="${LOCATION:-swedencentral}"
APP_NAME="${APP_NAME:-transformer-risk-agent}"
ENV_NAME="${ENV_NAME:-cae-transformer-risk-agent}"
TARGET_PORT="${TARGET_PORT:-8501}"
FOUNDRY_ACCOUNT="${FOUNDRY_ACCOUNT:-cog-transformer-risk-agent}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Subscription:    $(az account show --query name -o tsv)"
echo "==> Resource group:  $RESOURCE_GROUP ($LOCATION)"
echo "==> Container app:   $APP_NAME"
echo "==> Environment:     $ENV_NAME"
echo "==> Target port:     $TARGET_PORT"
echo

if ! az extension show -n containerapp >/dev/null 2>&1; then
  echo "==> Installing containerapp CLI extension..."
  az extension add -n containerapp --upgrade -y >/dev/null
fi

echo "==> Registering required providers (idempotent)..."
az provider register -n Microsoft.App --wait >/dev/null
az provider register -n Microsoft.OperationalInsights --wait >/dev/null
az provider register -n Microsoft.ContainerRegistry --wait >/dev/null

echo "==> Ensuring resource group exists..."
az group create -n "$RESOURCE_GROUP" -l "$LOCATION" -o none

echo "==> Loading Azure AI Foundry settings..."
if ! az cognitiveservices account show -n "$FOUNDRY_ACCOUNT" -g "$RESOURCE_GROUP" >/dev/null 2>&1; then
  echo "❌ Foundry account '$FOUNDRY_ACCOUNT' not found in '$RESOURCE_GROUP'."
  echo "   Run ./scripts/provision_foundry.sh first."
  exit 1
fi
AZURE_OPENAI_ENDPOINT=$(az cognitiveservices account show -n "$FOUNDRY_ACCOUNT" -g "$RESOURCE_GROUP" --query properties.endpoint -o tsv)
FOUNDRY_SCOPE=$(az cognitiveservices account show -n "$FOUNDRY_ACCOUNT" -g "$RESOURCE_GROUP" --query id -o tsv)
AZURE_OPENAI_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-gpt-4.1-mini}"
AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2025-01-01-preview}"

echo "==> Building image and deploying container app (this can take ~5 min)..."
az containerapp up \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --environment "$ENV_NAME" \
  --source . \
  --target-port "$TARGET_PORT" \
  --ingress external \
  --env-vars \
    "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT" \
    "AZURE_OPENAI_DEPLOYMENT=$AZURE_OPENAI_DEPLOYMENT" \
    "AZURE_OPENAI_API_VERSION=$AZURE_OPENAI_API_VERSION"

echo "==> Enabling system-assigned managed identity on the container app..."
az containerapp identity assign \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --system-assigned \
  -o none

APP_PRINCIPAL_ID=$(az containerapp identity show \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query principalId -o tsv)

echo "==> Granting 'Cognitive Services User' role on Foundry account to the app..."
az role assignment create \
  --assignee-object-id "$APP_PRINCIPAL_ID" \
  --assignee-principal-type ServicePrincipal \
  --role "Cognitive Services User" \
  --scope "$FOUNDRY_SCOPE" \
  -o none 2>&1 | grep -v "already exists" || true

echo "==> Restarting the container app revision to pick up the identity..."
az containerapp revision restart \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --revision "$(az containerapp revision list --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --query "[?properties.active].name | [0]" -o tsv)" \
  -o none 2>&1 || true

FQDN=$(az containerapp show \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn -o tsv)

echo
echo "✅ Deployed. App URL: https://$FQDN"
