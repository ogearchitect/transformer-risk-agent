#!/usr/bin/env bash
# Provision Azure AI Foundry (Cognitive Services kind=AIServices) + model deployments.
#
# Prereqs:
#   - az logged in to the correct sub (`az account show`)
#
# Usage:
#   ./scripts/provision_foundry.sh
#   LOCATION=swedencentral ACCOUNT_NAME=cog-transformer-risk-agent ./scripts/provision_foundry.sh
#
# Idempotent: re-running is safe and will only create missing resources.

set -euo pipefail

RG="${RESOURCE_GROUP:-rg-transformer-risk-agent}"
LOCATION="${LOCATION:-swedencentral}"
ACCOUNT="${ACCOUNT_NAME:-cog-transformer-risk-agent}"
CHAT_DEPLOYMENT="${CHAT_DEPLOYMENT:-gpt-4.1-mini}"
CHAT_MODEL_NAME="${CHAT_MODEL_NAME:-gpt-4.1-mini}"
CHAT_MODEL_VERSION="${CHAT_MODEL_VERSION:-2025-04-14}"
CHAT_SKU="${CHAT_SKU:-GlobalStandard}"
EMBED_DEPLOYMENT="${EMBED_DEPLOYMENT:-text-embedding-3-small}"
EMBED_MODEL_NAME="${EMBED_MODEL_NAME:-text-embedding-3-small}"
EMBED_MODEL_VERSION="${EMBED_MODEL_VERSION:-1}"
EMBED_SKU="${EMBED_SKU:-GlobalStandard}"
CHAT_CAPACITY="${CHAT_CAPACITY:-20}"
EMBED_CAPACITY="${EMBED_CAPACITY:-10}"

echo "==> Subscription: $(az account show --query name -o tsv)"
echo "==> Resource group: $RG ($LOCATION)"
echo "==> Foundry account: $ACCOUNT"
echo

az provider register -n Microsoft.CognitiveServices --wait >/dev/null

echo "==> Ensuring resource group..."
az group create -n "$RG" -l "$LOCATION" -o none

echo "==> Ensuring Cognitive Services account (kind=AIServices)..."
if ! az cognitiveservices account show -n "$ACCOUNT" -g "$RG" >/dev/null 2>&1; then
  az cognitiveservices account create \
    -n "$ACCOUNT" -g "$RG" -l "$LOCATION" \
    --kind AIServices --sku S0 \
    --custom-domain "$ACCOUNT" \
    --yes -o none
else
  echo "    (already exists)"
fi

echo "==> Ensuring local (API-key) auth is enabled on the account..."
az resource update \
  --resource-group "$RG" \
  --name "$ACCOUNT" \
  --resource-type "Microsoft.CognitiveServices/accounts" \
  --set properties.disableLocalAuth=false \
  -o none

echo "==> Deploying chat model: $CHAT_DEPLOYMENT ($CHAT_MODEL_NAME $CHAT_MODEL_VERSION, sku=$CHAT_SKU)..."
az cognitiveservices account deployment create \
  -g "$RG" -n "$ACCOUNT" \
  --deployment-name "$CHAT_DEPLOYMENT" \
  --model-name "$CHAT_MODEL_NAME" \
  --model-version "$CHAT_MODEL_VERSION" \
  --model-format OpenAI \
  --sku-name "$CHAT_SKU" --sku-capacity "$CHAT_CAPACITY" \
  -o none 2>&1 || echo "    (deployment may already exist — continuing)"

echo "==> Deploying embedding model: $EMBED_DEPLOYMENT ($EMBED_MODEL_NAME $EMBED_MODEL_VERSION, sku=$EMBED_SKU)..."
az cognitiveservices account deployment create \
  -g "$RG" -n "$ACCOUNT" \
  --deployment-name "$EMBED_DEPLOYMENT" \
  --model-name "$EMBED_MODEL_NAME" \
  --model-version "$EMBED_MODEL_VERSION" \
  --model-format OpenAI \
  --sku-name "$EMBED_SKU" --sku-capacity "$EMBED_CAPACITY" \
  -o none 2>&1 || echo "    (deployment may already exist — continuing)"

echo "==> Endpoint:"
ENDPOINT=$(az cognitiveservices account show -n "$ACCOUNT" -g "$RG" --query properties.endpoint -o tsv)
echo "$ENDPOINT"
echo
echo "ℹ️  This Foundry account (kind=AIServices) is Entra-ID-only."
echo "    Grant 'Cognitive Services User' role to callers; do NOT use API keys."
echo
echo "==> Granting current user 'Cognitive Services User' role for local dev..."
CURRENT_USER=$(az ad signed-in-user show --query id -o tsv 2>/dev/null || true)
if [ -n "$CURRENT_USER" ]; then
  SCOPE=$(az cognitiveservices account show -n "$ACCOUNT" -g "$RG" --query id -o tsv)
  az role assignment create \
    --assignee-object-id "$CURRENT_USER" \
    --assignee-principal-type User \
    --role "Cognitive Services User" \
    --scope "$SCOPE" \
    -o none 2>&1 | grep -v "already exists" || true
  echo "    ✓ role granted (or already present)"
fi

# Write a local .env.foundry for local runs (gitignored).
# NOTE: no API key — local dev relies on DefaultAzureCredential / `az login`.
cat > .env.foundry <<EOF
AZURE_OPENAI_ENDPOINT=$ENDPOINT
AZURE_OPENAI_DEPLOYMENT=$CHAT_DEPLOYMENT
AZURE_OPENAI_EMBED_DEPLOYMENT=$EMBED_DEPLOYMENT
AZURE_OPENAI_API_VERSION=2025-01-01-preview
EOF
echo "==> Wrote .env.foundry (gitignored)"
