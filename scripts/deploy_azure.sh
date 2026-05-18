#!/usr/bin/env bash
# Deploy the Transformer Risk Agent (Streamlit app) to Azure Container Apps.
#
# Prereqs:
#   - Azure CLI logged in (`az login`)
#   - Subscription selected (`az account set -s <SUB_ID>`)
#   - `./scripts/provision_foundry.sh` already run for the target resource group
#
# Usage:
#   ./scripts/deploy_azure.sh                      # uses defaults below
#   LOCATION=westus3 APP_NAME=my-app ./scripts/deploy_azure.sh
#
# What it does:
#   1. Registers required resource providers (idempotent).
#   2. Creates the resource group, Container Apps environment, and ACR if missing.
#   3. Builds + pushes the image via `az acr build` (cloud build, no local Docker).
#   4. Creates or updates the container app with HTTPS ingress on port 8501.
#   5. Enables system-assigned managed identity on the app and grants it
#      'Cognitive Services User' on the Foundry account (Entra ID auth — the
#      AIServices kind hard-locks disableLocalAuth=true so API keys cannot be
#      used).
#   6. Restarts the active revision so the identity is picked up.
#   7. Prints the public FQDN.
#
# Re-running this script is safe; it updates the existing app in place.
#
# NOTE: We intentionally do NOT use `az containerapp up` (or the `containerapp`
# CLI extension) because the current beta has a bug that breaks ACR cloud
# builds on darwin (`'NoneType' object has no attribute 'linux'`). The built-in
# `az containerapp create/update` commands work fine on their own.

set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-transformer-risk-agent}"
LOCATION="${LOCATION:-swedencentral}"
APP_NAME="${APP_NAME:-transformer-risk-agent}"
ENV_NAME="${ENV_NAME:-cae-transformer-risk-agent}"
TARGET_PORT="${TARGET_PORT:-8501}"
FOUNDRY_ACCOUNT="${FOUNDRY_ACCOUNT:-cog-transformer-risk-agent}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d%H%M%S)}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Subscription:    $(az account show --query name -o tsv)"
echo "==> Resource group:  $RESOURCE_GROUP ($LOCATION)"
echo "==> Container app:   $APP_NAME"
echo "==> Environment:     $ENV_NAME"
echo "==> Target port:     $TARGET_PORT"
echo "==> Image tag:       $IMAGE_TAG"
echo

# Remove the buggy preview extension if present so the built-in commands win.
if az extension show -n containerapp >/dev/null 2>&1; then
  echo "==> Removing buggy containerapp preview extension..."
  az extension remove -n containerapp >/dev/null 2>&1 || true
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

echo "==> Ensuring Azure Container Registry..."
ACR_NAME=$(az acr list -g "$RESOURCE_GROUP" --query "[0].name" -o tsv)
if [ -z "$ACR_NAME" ]; then
  ACR_NAME="acrtra$(date +%s | tail -c 8)"
  echo "    creating new registry $ACR_NAME..."
  az acr create -g "$RESOURCE_GROUP" -n "$ACR_NAME" --sku Basic --admin-enabled true -o none
else
  echo "    using existing registry $ACR_NAME"
  az acr update -n "$ACR_NAME" --admin-enabled true -o none
fi
ACR_LOGIN_SERVER=$(az acr show -n "$ACR_NAME" --query loginServer -o tsv)
IMAGE="$ACR_LOGIN_SERVER/$APP_NAME:$IMAGE_TAG"

echo "==> Building + pushing image $IMAGE via ACR cloud build..."
az acr build --registry "$ACR_NAME" --image "$APP_NAME:$IMAGE_TAG" --file Dockerfile . -o none

echo "==> Ensuring Container Apps environment..."
if ! az containerapp env show -n "$ENV_NAME" -g "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az containerapp env create -n "$ENV_NAME" -g "$RESOURCE_GROUP" -l "$LOCATION" -o none
fi

ACR_USERNAME=$(az acr credential show -n "$ACR_NAME" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show -n "$ACR_NAME" --query "passwords[0].value" -o tsv)

if az containerapp show -n "$APP_NAME" -g "$RESOURCE_GROUP" >/dev/null 2>&1; then
  echo "==> Updating existing container app..."
  az containerapp registry set \
    --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
    --server "$ACR_LOGIN_SERVER" --username "$ACR_USERNAME" --password "$ACR_PASSWORD" \
    -o none
  az containerapp update \
    --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
    --image "$IMAGE" \
    --set-env-vars \
      "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT" \
      "AZURE_OPENAI_DEPLOYMENT=$AZURE_OPENAI_DEPLOYMENT" \
      "AZURE_OPENAI_API_VERSION=$AZURE_OPENAI_API_VERSION" \
    -o none
else
  echo "==> Creating container app..."
  az containerapp create \
    --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
    --environment "$ENV_NAME" \
    --image "$IMAGE" \
    --registry-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_USERNAME" \
    --registry-password "$ACR_PASSWORD" \
    --target-port "$TARGET_PORT" \
    --ingress external \
    --min-replicas 0 --max-replicas 2 \
    --cpu 1.0 --memory 2.0Gi \
    --env-vars \
      "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT" \
      "AZURE_OPENAI_DEPLOYMENT=$AZURE_OPENAI_DEPLOYMENT" \
      "AZURE_OPENAI_API_VERSION=$AZURE_OPENAI_API_VERSION" \
    -o none
fi

echo "==> Enabling system-assigned managed identity on the container app..."
az containerapp identity assign \
  --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --system-assigned -o none

APP_PRINCIPAL_ID=$(az containerapp identity show \
  --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --query principalId -o tsv)

echo "==> Granting 'Cognitive Services User' role on Foundry account to the app..."
az role assignment create \
  --assignee-object-id "$APP_PRINCIPAL_ID" \
  --assignee-principal-type ServicePrincipal \
  --role "Cognitive Services User" \
  --scope "$FOUNDRY_SCOPE" \
  -o none 2>&1 | grep -v "already exists" || true

echo "==> Restarting active revision to pick up the identity..."
ACTIVE_REV=$(az containerapp revision list --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --query "[?properties.active].name | [0]" -o tsv)
if [ -n "$ACTIVE_REV" ]; then
  az containerapp revision restart \
    --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
    --revision "$ACTIVE_REV" -o none 2>&1 || true
fi

FQDN=$(az containerapp show \
  --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn -o tsv)

echo
echo "✅ Deployed. App URL: https://$FQDN"
