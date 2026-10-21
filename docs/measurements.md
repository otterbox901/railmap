# Measurements

The Azure ND H100 v5 campaign (`infra/azure/`) will measure four things:

- the GPUDirect RDMA bandwidth for every GPU×NIC pair, grouped by path class;
- the message-rate knee against QP count on ConnectX-7 NDR;
- how closely the QP-count model matches QPs counted on the hardware;
- nccl-tests bus bandwidth under no binding, railmap's placement and a deliberately bad placement.

Until then, the only connection-budget profile is a one-point prior built from the 59% / 3,000 figure. Below 3,000
QPs, the expected loss is reported as unknown.

## Probes

These need RDMA hardware, or Soft-RoCE for a code-path check (`sudo scripts/softroce-up.sh`):

```bash
railmap observe -- <command>      # peak QPs per NIC owned by the command's process tree
railmap observe --duration 60     # whole host minus baseline (for nodes where ranks were launched remotely)
railmap calibrate --hca mlx5_ib0 --server node1 -o profile.json   # perftest -q sweep, knee fit
railmap gdr-matrix --remote node1 # GPUDirect RDMA Gb/s for every GPU x NIC pair
railmap validate --launcher "mpirun -np 16 -H a:8,b:8 --bind-to none"   # nccl-tests: none/best/worst
```

## Azure campaign

```bash
infra/azure/up.sh          # 2 x Standard_ND96isr_H100_v5 in one VMSS; prints cost, asks to confirm
infra/azure/bootstrap.sh   # railmap, CUDA perftest, nccl-tests; node0 -> node1 ssh
infra/azure/campaign.sh    # runs everything, fetches results/, then deletes the resource group
```

Requirements:

- An ND H100 v5 quota of at least 192 vCPUs in the region (`LOCATION` in `infra/azure/config.env`).
- Check the price per VM-hour before running; `PRICE_PER_VM_HOUR` in the same file feeds the confirmation prompt.

`campaign.sh` deletes the resource group when it exits. Set `KEEP=1` to keep it, then run `infra/azure/down.sh`
yourself.

The campaign runs on InfiniBand, not Ethernet. Profiles are keyed by NIC `board_id` and link layer, so a
RoCE profile measured elsewhere slots in without code changes.

The results show up in the [UI](ui.md)'s Campaign results view.
