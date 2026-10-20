#!/usr/bin/env bash
# Create the ND H100 v5 scale set for the railmap campaign.
# shellcheck source-path=SCRIPTDIR
set -o pipefail -o errexit -o nounset
source "$(dirname "$0")/config.env"

az account show --query '{subscription:name, id:id}' -o table
az vm image show --urn "$IMAGE" --query name -o tsv >/dev/null || { echo "image $IMAGE not found" >&2; exit 1; }

echo "H100 quota in $LOCATION:"
az vm list-usage --location "$LOCATION" --query "[?contains(name.value, 'H100')].{family:name.value, used:currentValue, limit:limit}" -o table
echo "need $((COUNT * 96)) vCPUs in the ND H100 v5 family"

cost=$((COUNT * PRICE_PER_VM_HOUR * HOURS))
echo "About to create $COUNT x $SKU ($PRIORITY) in $LOCATION: ~\$$cost for $HOURS h at \$$PRICE_PER_VM_HOUR/VM-hour."
read -r -p "Type 'yes' to continue: " answer
[[ "$answer" == "yes" ]] || { echo "aborted"; exit 1; }

expires=$(date -u -d "+$((HOURS + 1)) hours" +%Y-%m-%dT%H:%MZ)
az group create -n "$RG" -l "$LOCATION" --tags purpose=railmap expires="$expires" -o none

spot_args=()
if [[ "$PRIORITY" == "Spot" ]]; then
  spot_args=(--eviction-policy Delete --max-price -1)
fi
# InfiniBand only connects VMs in one scale set with a single placement group.
az vmss create -g "$RG" -n "$VMSS" \
  --image "$IMAGE" --vm-sku "$SKU" --instance-count "$COUNT" \
  --orchestration-mode Uniform --single-placement-group true --platform-fault-domain-count 1 \
  --upgrade-policy-mode Manual \
  --admin-username "$ADMIN" --ssh-key-values "$SSH_PUB" \
  --public-ip-per-vm --lb "" --accelerated-networking true \
  --priority "$PRIORITY" "${spot_args[@]}" \
  --tags purpose=railmap expires="$expires" -o none

mkdir -p "$STATE_DIR"
az vmss list-instance-public-ips -g "$RG" -n "$VMSS" --query "[].ipAddress" -o tsv >"$STATE_DIR/public-ips"
az vmss nic list -g "$RG" --vmss-name "$VMSS" --query "[].ipConfigurations[0].privateIPAddress" -o tsv \
  >"$STATE_DIR/private-ips"
echo "public:  $(paste -sd' ' "$STATE_DIR/public-ips")"
echo "private: $(paste -sd' ' "$STATE_DIR/private-ips")"
echo "next: infra/azure/bootstrap.sh, then infra/azure/campaign.sh"
