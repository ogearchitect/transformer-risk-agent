#!/usr/bin/env bash
# Provision an Azure Digital Twins instance, upload the Transformer DTDL model,
# and grant the container app's managed identity 'Azure Digital Twins Data
# Owner' on the instance. Idempotent.
#
# Usage:
#   ./scripts/provision_adt.sh                       # uses env defaults below
#   ADT_NAME=adt-tra LOCATION=swedencentral ./scripts/provision_adt.sh
#
# Outputs:
#   - Echoes the ADT hostname (suitable for ADT_ENDPOINT env var)

set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-transformer-risk-agent}"
LOCATION="${LOCATION:-swedencentral}"
ADT_NAME="${ADT_NAME:-adt-transformer-risk-agent}"
APP_NAME="${APP_NAME:-transformer-risk-agent}"
MODEL_FILE="${MODEL_FILE:-dtdl/Transformer.v1.json}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Registering Microsoft.DigitalTwins provider (idempotent)..."
az provider register -n Microsoft.DigitalTwins --wait >/dev/null

echo "==> Ensuring resource group exists..."
az group create -n "$RESOURCE_GROUP" -l "$LOCATION" -o none

echo "==> Ensuring Azure Digital Twins extension..."
if ! az extension show -n azure-iot >/dev/null 2>&1; then
  az extension add -n azure-iot -y -o none
fi
if ! az dt --help >/dev/null 2>&1; then
  az extension add -n dt -y -o none
fi

echo "==> Ensuring ADT instance $ADT_NAME..."
if ! az dt show -n "$ADT_NAME" -g "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az dt create -n "$ADT_NAME" -g "$RESOURCE_GROUP" -l "$LOCATION" -o none
fi

ADT_HOSTNAME=$(az dt show -n "$ADT_NAME" -g "$RESOURCE_GROUP" --query hostName -o tsv)
ADT_ID=$(az dt show -n "$ADT_NAME" -g "$RESOURCE_GROUP" --query id -o tsv)

CALLER_OBJECT_ID=$(az ad signed-in-user show --query id -o tsv 2>/dev/null || true)
if [ -n "$CALLER_OBJECT_ID" ]; then
  echo "==> Granting current user 'Azure Digital Twins Data Owner' (for model upload)..."
  az role assignment create \
    --assignee-object-id "$CALLER_OBJECT_ID" \
    --assignee-principal-type User \
    --role "Azure Digital Twins Data Owner" \
    --scope "$ADT_ID" \
    -o none 2>&1 | grep -v "already exists" || true
  echo "    waiting 30s for RBAC to propagate..."
  sleep 30
fi

echo "==> Uploading DTDL model from $MODEL_FILE..."
MODEL_ID=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["@id"])' "$MODEL_FILE")
if az dt model show -n "$ADT_NAME" --dtmi "$MODEL_ID" >/dev/null 2>&1; then
  echo "    model $MODEL_ID already present"
else
  az dt model create -n "$ADT_NAME" --models "$MODEL_FILE" -o none
  echo "    uploaded $MODEL_ID"
fi

if az containerapp show -n "$APP_NAME" -g "$RESOURCE_GROUP" >/dev/null 2>&1; then
  APP_PRINCIPAL_ID=$(az containerapp identity show -n "$APP_NAME" -g "$RESOURCE_GROUP" --query principalId -o tsv 2>/dev/null || true)
  if [ -n "$APP_PRINCIPAL_ID" ] && [ "$APP_PRINCIPAL_ID" != "null" ]; then
    echo "==> Granting container app MSI 'Azure Digital Twins Data Owner'..."
    az role assignment create \
      --assignee-object-id "$APP_PRINCIPAL_ID" \
      --assignee-principal-type ServicePrincipal \
      --role "Azure Digital Twins Data Owner" \
      --scope "$ADT_ID" \
      -o none 2>&1 | grep -v "already exists" || true
  fi
fi

echo
echo "✅ ADT ready. Hostname: $ADT_HOSTNAME"
echo "   export ADT_ENDPOINT=$ADT_HOSTNAME"
echo "   Model:    $MODEL_ID"
