#!/usr/bin/env bash
# Copy railmap to every VM, run node-setup.sh, and give node0 passwordless ssh to the others for mpirun/perftest.
# shellcheck source-path=SCRIPTDIR
set -o pipefail -o errexit -o nounset
source "$(dirname "$0")/config.env"
repo="$(cd "$(dirname "$0")/../.." && pwd)"
mapfile -t pub <"$STATE_DIR/public-ips"
mapfile -t priv <"$STATE_DIR/private-ips"
ssh_opts=(-o StrictHostKeyChecking=accept-new -o BatchMode=yes)

pids=()
for ip in "${pub[@]}"; do
  rsync -az --delete -e "ssh ${ssh_opts[*]}" --exclude .venv --exclude .git --exclude results --exclude .state \
    "$repo/" "$ADMIN@$ip:railmap/"
  ssh "${ssh_opts[@]}" "$ADMIN@$ip" 'bash railmap/infra/azure/node-setup.sh' &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

# node0 drives the campaign: give it a key every node (itself included, for mpirun -H) trusts.
ssh "${ssh_opts[@]}" "$ADMIN@${pub[0]}" \
  '[[ -f ~/.ssh/id_ed25519 ]] || ssh-keygen -q -t ed25519 -N "" -f ~/.ssh/id_ed25519; cat ~/.ssh/id_ed25519.pub' \
  >"$STATE_DIR/node0.pub"
for ip in "${pub[@]}"; do
  ssh "${ssh_opts[@]}" "$ADMIN@$ip" 'cat >>~/.ssh/authorized_keys' <"$STATE_DIR/node0.pub"
done
for p in "${priv[@]}"; do
  ssh "${ssh_opts[@]}" "$ADMIN@${pub[0]}" "ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes $p hostname"
done
echo "bootstrap done"
