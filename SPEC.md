# SPEC: deployable railmap, then a measured, public release

Interviewed 2026-10-23. Work happens in two phases: **build** (this spec, in full) and then **measure** (only
constraints are fixed so far; see Open questions).

## Goal

Anyone can install railmap on a cloud GPU node and have it work: in a plain venv, in a container, or under Slurm or
Kubernetes. It should be packaged, licensed and tested well enough to make public on GitHub and PyPI. Once the build
is done, an Azure campaign within the budget replaces the docs' "findings pending" with measured numbers.

## Non-goals

- Clouds other than Azure for the campaign. AWS EFA uses SRD rather than RC QPs, so the QP model doesn't apply.
- arm64 / Grace (GH200, GB200) support claims. The wheel installs there, but Grace exposes GPU memory as NUMA nodes and
  the placement logic hasn't been checked against that.
- Contributor scaffolding: CONTRIBUTING, issue templates, a profile-submission flow, CHANGELOG. GitHub release notes are
  enough for now.
- Shipping nccl-tests or an MPI/NCCL stack in the container image. Users bring their own training stack.
- Running anything on cloud hardware during the build phase.

## Requirements

### Packaging and venv installs

1. `requires-python = ">=3.10"`. The full test suite passes on 3.10, 3.11, 3.12, 3.13 and 3.14. Ruff `target-version` is `py310`.
2. Every module that is imported is declared as a dependency (`numpy` was missing). With `uv pip install
   --resolution lowest-direct` on 3.10, the suite still passes, which proves the declared floors are real. This check
   found that typer 0.12.x can't parse `Path | None` options, so the floor is now `typer>=0.13`.
3. `python3 -m venv v && v/bin/pip install <wheel>` with no uv, no root and no compiler gives a working `railmap`.
   `railmap --version`, `railmap plan --topo-file <ndv5 xml>` and `railmap ui --no-open` (port 0, which starts and
   answers `/`) all run.
4. The wheel contains `profiles/*.json`, `ui/index.html` and any shipped topology files. `uv build` produces an sdist
   and a wheel, and `twine check` passes on both.
5. `railmap --version` prints the version from package metadata, so there is one source of truth: `pyproject.toml`.
6. The pyproject metadata includes license (Apache-2.0, SPDX expression), authors, project URLs, classifiers and
   keywords.

### Container- and cgroup-aware runtime

7. Live discovery restricts each NUMA node's CPUs to the CPUs the process's cgroup allows. That is cgroup v2
   `cpuset.cpus.effective`, or the cgroup v1 `cpuset.effective_cpus`, with `sched_getaffinity` as the fallback. The
   plan then splits only CPUs that the ranks can actually use. A note is added when the set is restricted: "planning
   on N of M CPUs allowed by the cgroup".
8. The restriction applies only when the topology describes this host: live sysfs, or a topology file that matched
   the live GPUs. `--snapshot` and topology files that don't match plan on the full CPU set. `show`/`plan`/`ui`/`exec`
   take `--all-cpus` to turn the restriction off.
9. All ranks of a node compute identical plans, even when the launcher has already pinned each process. To get this,
   rule 7 prefers the cgroup cpuset over per-process affinity.
10. A NUMA node with no allowed CPUs gets the same warning as today's "ranks share cores" case. Its ranks fall back to
    all allowed CPUs, and the plan never contains a CPU outside the allowed set.
11. `exec` only adds `numactl --membind=<n>` when node `n` is in the process's `Mems_allowed_list`. Otherwise it warns
    and runs without memory binding.
12a. Live discovery keeps only what this process was given, because `/sys` shows the whole host inside a container
     or job step:
     - GPUs that NVML can open, renumbered in PCI order as CUDA numbers them;
     - NICs that have a `/dev/infiniband/uverbs*` node.
     Each filter applies only when some, but not all, devices are usable. When none are usable (no NVML, no device
     nodes), everything is kept so `show`/`plan` still work for inspection, and `doctor` reports the problem.
12. When the PCI tree is flattened and no topology file is found or given, railmap matches the live GPU addresses
    against Azure topology files shipped inside the package (ND A100 v4 and ND H100 v5). The match works the same way
    as the existing `/opt/microsoft` lookup, so a container on Azure works without mounting `/opt/microsoft`. Files on
    the host still take precedence. The shipped files carry their upstream license and attribution.

### `railmap doctor`

13. `railmap doctor` checks the environment and reports each check as ok, warn or fail, with a one-line fix. It covers:
    - railmap and Python versions;
    - GPUs from sysfs and whether NVML loads;
    - RDMA devices and active ports, and whether `/dev/infiniband/uverbs*` is accessible;
    - the topology source, which fails when the tree is flattened and no topology file is available;
    - whether railmap runs in a container, and which one;
    - CPUs allowed versus online, and NUMA memory nodes allowed;
    - the locked-memory limit (`RLIMIT_MEMLOCK`; RDMA needs it unlimited or large);
    - whether `numactl`, `rdma`, `ib_write_bw` (and its CUDA support) and `nvidia-smi` are present;
    - which launcher local-rank variable is set, if any.
14. `doctor --json` prints machine-readable results. The exit code is 1 if any check fails and 0 otherwise. Doctor
    never throws a traceback on a host with no GPUs, no NICs or no tools.

### Container image

15. A `Dockerfile` builds `ghcr.io/otterbox901/railmap:<version>` (plus `:latest`) for linux/amd64. It is multi-stage:
    perftest with CUDA support is built in a CUDA devel stage and copied into a CUDA base runtime stage. The runtime
    stage holds railmap (installed in a venv), rdma-core and its providers, ibverbs-utils, iproute2 (`rdma`),
    numactl and perftest. The entrypoint is `railmap`. perftest is pinned to the commit `node-setup.sh` uses.
16. `docker run --rm ghcr.io/otterbox901/railmap --version` and `... plan --topo-file /fixture.xml` work on a machine
    with no GPU. On a GPU+RDMA host, `docker run --gpus all --device /dev/infiniband --ulimit memlock=-1 --network host
    ghcr.io/otterbox901/railmap doctor` is the documented smoke test.

### Scheduler and container recipes (`examples/`)

17. **Slurm:** an sbatch script that runs `srun --cpu-bind=none railmap exec -- <cmd>`, a variant that uses
    `railmap env --format slurm` for the mask, and a Pyxis/Enroot variant (`--container-image`).
18. **Kubernetes:** a single-node Pod manifest that runs `railmap doctor` / `plan`, and a multi-node `torchrun` job
    that wraps the trainer with `railmap exec`. The multi-node job is a core Indexed Job plus a headless Service rather
    than a PyTorchJob: it needs no operator, and kubeconform can validate it, while the public CRD catalog has no
    PyTorchJob schema. PyTorchJob and JobSet users wrap the same command. Both include GPU and RDMA resource requests (NVIDIA GPU and Network
    operators), `IPC_LOCK`, Guaranteed QoS, and a note about the static CPU manager.
19. **Own image:** a Dockerfile snippet that adds railmap to a user's training image with `pip install railmap`.
20. CI validates the recipes statically: shellcheck for the scripts and kubeconform for the manifests. Testing them on
    hardware belongs to the measure phase.

### CI, release and public readiness

21. GitHub Actions CI runs on every push and PR:
    - ruff check and format check;
    - pytest on 3.10–3.14;
    - the lowest-direct job (req 2);
    - a build that installs the wheel into a clean venv and runs a smoke test (reqs 3–4);
    - building the container image and smoke-testing it, including `doctor` running without a traceback inside a
      `--cpuset-cpus`-limited container;
    - shellcheck and kubeconform.
22. A release workflow runs on `v*` tags. It builds, publishes to PyPI with trusted publishing (no stored token),
    pushes the image to GHCR and creates a GitHub release. Its only manual prerequisite is a PyPI pending publisher,
    which is documented.
23. `LICENSE` (Apache-2.0) is at the repo root, and the README has a license line.
24. Before anything goes public, the git history and tree are scanned for secrets and personal data, and the findings
    are reported.
25. Outward-facing steps happen only after explicit confirmation from the user at that moment: making the repo public,
    pushing tags, publishing to PyPI or GHCR, and changing GitHub settings.

### Docs

26. `docs/install.md` covers pip/venv, `uv tool`, pipx, the container, Azure notes (the topology file, Python on
    22.04 HPC images), Slurm, Kubernetes, and a troubleshooting table: memlock, `/dev/infiniband`, cgroup CPUs, missing
    numactl, flattened PCI tree. The README quick start leads with `pip install railmap` and links to install.md.
    `usage.md` documents `doctor` and `--all-cpus`.

## Design / approach

- **CPU restriction lives in `topology/discover.py`.** `discover()` gets a `restrict_cpus: bool = True` parameter and
  applies the restriction after choosing a source. It reads the cgroup through `root`, so fake sysfs trees can test it.
  The model and placement stay pure.
- **Cgroup cpuset is preferred over affinity** because launchers such as `srun` with default binding and
  `mpirun --bind-to core` pin each rank differently. A plan built from per-process affinity would differ from rank to
  rank and overlap. The cgroup is the same for the whole pod or step.
- **Shipped topology files go in `railmap/topology/vendor/`.** They are searched after the host's `/opt/microsoft`
  files. Upstream source: Azure/azhpc-images. Before copying, check its license; if redistribution isn't allowed, drop
  req 12 and make mounting `/opt/microsoft` the documented path.
- **`doctor` lives in `railmap/doctor.py`.** It holds pure check functions that take `root`/`env`, plus a report in
  `emit/report.py` that matches the existing table style.
- **Python 3.10** because Ubuntu 22.04 HPC images still ship it. The code uses no 3.11+ features. CI covers up to 3.14.
- **License: Apache-2.0**, for its patent grant and because it is the norm in GPU/HPC tooling.
- **One smoke script.** `scripts/smoke.sh` runs `--version`, `plan` with a budget, `doctor` (must not crash) and
  optionally `ui`. CI and local checks use it for both the wheel and the image.
- **The runtime image is `ubuntu:24.04`, not a CUDA base.** perftest at the pinned commit loads `libcuda` at run time
  (`src/cuda_loader.c`), so the CUDA devel image is needed only to compile it. The image doesn't set
  `NVIDIA_VISIBLE_DEVICES`, so it can't expose every GPU to a pod that requested none.
- **Version 0.1.0 stays** for the first public release. Bump it if the measure phase changes behaviour before the tag.

## Edge cases and error handling

- **cgroup v1, v2, hybrid or none.** If the cpuset file isn't found, fall back to `sched_getaffinity`. Never fail
  because of this.
- **The allowed CPU set covers only part of a NUMA node, or excludes a node entirely.** Handled by reqs 7 and 10.
- **SMT siblings split by the cpuset.** Already handled: `_core_groups` keeps only allowed siblings.
- **A container sees the host's `/sys` but not its `/opt`.** Handled by req 12.
- **No GPUs.** `plan` keeps its current error; `doctor` reports fail instead of throwing.
- **RDMA in exclusive netns mode** shows only the devices assigned to the container. `doctor` reports what it sees.
- **`ulimit -l` is small in Docker by default.** `doctor` warns and recommends `--ulimit memlock=-1`.
- **numactl is missing from the user's image.** Already a warning; `doctor` reports it too.
- **Rootless use.** `observe` may see only the user's own QPs without `CAP_NET_ADMIN`. Documented, not changed.

## Testing plan

- **Unit tests with fake sysfs/cgroup trees** for:
  - cgroup v2, cgroup v1 and the affinity fallback;
  - a partial NUMA cpuset, an empty NUMA cpuset and SMT splits;
  - snapshots and topology-file mismatches staying unrestricted;
  - `--all-cpus`;
  - the `Mems_allowed` gate in `exec`;
  - each `doctor` check, including no-GPU and no-tools hosts;
  - matching the shipped vendor file on a flattened tree without `/opt/microsoft`.
- **Packaging:** the clean-venv wheel smoke test and the lowest-direct job, run locally with uv first and then in CI.
- **Container:** build locally with podman and run the smoke tests from req 16 without GPUs. CI repeats this with
  docker.
- **Recipes:** shellcheck and kubeconform.
- **Hardware:** none in this phase. Covered by the measure phase below.

## Open questions

- **Measure phase plan, under a $250 Azure hard cap.** Unknowns:
  - the subscription's offer type and GPU/HB quota (free-trial and VS credits can't run GPU VMs);
  - which SKUs fit the budget. The staged HBv4 → 1× ND H100 v5 → 2× ND H100 v5 plan doesn't fit: ND H100 v5 is about
    $98 per VM-hour. Candidates are an HBv4 pair (CX-7 knee) plus 2× ND A100 v4, or a single ND H100 v5 node;
  - a spend guard that deletes the resource group at the cap;
  - the `az` CLI, which isn't installed locally.
- **When to go public:** right after the build phase, or only once measured results are in the docs (the public bar
  includes "measured results in docs").
