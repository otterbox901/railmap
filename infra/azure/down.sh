#!/usr/bin/env bash
# Delete everything the campaign created.
# shellcheck source-path=SCRIPTDIR
set -o pipefail -o errexit -o nounset
source "$(dirname "$0")/config.env"

if az group exists -n "$RG" | grep -q true; then
  az group delete -n "$RG" --yes --no-wait
  echo "deleting resource group $RG (async); check with: az group exists -n $RG"
else
  echo "resource group $RG does not exist"
fi
rm -rf "$STATE_DIR"
