#!/usr/bin/env bash
# Smoke-test an installed railmap on a machine without GPUs: version, a plan + budget from the ND H100 v5 fixture,
# doctor (must report, not crash) and, with SMOKE_UI=1, the web page.
#
#   scripts/smoke.sh                                    # the `railmap` on PATH
#   SMOKE_UI=1 scripts/smoke.sh venv/bin/railmap
#   FIXTURES=/f scripts/smoke.sh docker run --rm -v "$PWD/tests/fixtures:/f:ro" railmap:ci
set -o pipefail -o errexit -o nounset
repo="$(cd "$(dirname "$0")/.." && pwd)"
fixtures=${FIXTURES:-$repo/tests/fixtures}
if [[ $# -gt 0 ]]; then rm=("$@"); else rm=(railmap); fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

"${rm[@]}" --version | tee "$tmp/version"
grep -q '^railmap [0-9]' "$tmp/version"

"${rm[@]}" plan --topo-file "$fixtures/azure-ndv5-topo.xml" --nodes 16 --json >"$tmp/plan.json"
python3 - "$tmp/plan.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
ranks = d["plan"]["ranks"]
assert len(ranks) == 8, ranks
assert all(r["path"] == "PIX" for r in ranks), [r["path"] for r in ranks]
assert d["budget"]["qps_per_nic"] > 0, d["budget"]
print(f"plan: 8 ranks on PIX NICs, {d['budget']['qps_per_nic']} QPs/NIC at 16 nodes")
EOF

set +o errexit
"${rm[@]}" doctor >"$tmp/doctor" 2>&1
code=$?
set -o errexit
cat "$tmp/doctor"
if [[ $code -gt 1 ]] || grep -q Traceback "$tmp/doctor"; then
  echo "doctor crashed (exit $code)" >&2
  exit 1
fi

if [[ "${SMOKE_UI:-0}" == 1 ]]; then
  "${rm[@]}" ui --no-open --port 0 --topo-file "$fixtures/azure-ndv5-topo.xml" >"$tmp/ui" 2>&1 &
  pid=$!
  url=""
  for _ in $(seq 50); do
    url=$(grep -o 'http://[^ ]*' "$tmp/ui" | head -1 || true)
    [[ -n "$url" ]] && break
    sleep 0.2
  done
  status=$(curl -s -o /dev/null -w '%{http_code}' "$url" || true)
  kill "$pid"
  [[ "$status" == 200 ]] || { echo "ui did not answer at '$url' ($status)" >&2; cat "$tmp/ui" >&2; exit 1; }
  echo "ui: $url answered 200"
fi
echo "smoke ok"
