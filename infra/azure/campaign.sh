#!/usr/bin/env bash
# Run the measurement campaign on node0, fetch results, and delete the VMs on exit (unless KEEP=1).
# shellcheck source-path=SCRIPTDIR
set -o pipefail -o errexit -o nounset
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/config.env"
mapfile -t pub <"$STATE_DIR/public-ips"
mapfile -t priv <"$STATE_DIR/private-ips"
stamp=$(date -u +%Y%m%d-%H%M)
remote_out="railmap-results/azure-ndv5-$stamp"
local_out="$here/../../results/azure-ndv5-$stamp"
ssh_opts=(-o StrictHostKeyChecking=accept-new -o BatchMode=yes)

finish() {
  mkdir -p "$local_out"
  rsync -az -e "ssh ${ssh_opts[*]}" "$ADMIN@${pub[0]}:$remote_out/" "$local_out/" || echo "result fetch failed" >&2
  if [[ "$KEEP" != 1 ]]; then
    "$here/down.sh"
  fi
}
trap finish EXIT

mkdir -p "$local_out"
# shellcheck disable=SC2029  # the paths and IPs expand here on purpose
ssh "${ssh_opts[@]}" "$ADMIN@${pub[0]}" "bash railmap/infra/azure/remote-campaign.sh $remote_out ${priv[0]} ${priv[1]}" \
  2>&1 | tee "$local_out/campaign.log"
rsync -az -e "ssh ${ssh_opts[*]}" "$ADMIN@${pub[0]}:$remote_out/" "$local_out/"
uv run --project "$here/../.." python "$here/summarize.py" "$local_out" | tee "$local_out/summary.md"
