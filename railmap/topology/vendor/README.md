# Vendor topology files

NCCL topology files that Azure HPC images install under `/opt/microsoft/`, copied from
[Azure/azhpc-images](https://github.com/Azure/azhpc-images/tree/master/topology) at commit
`4a4565a0c0aa9d6944c53420155936061b9c3a98` under the MIT license in [LICENSE](LICENSE).

| file | VM size |
|---|---|
| `ndv4-topo.xml` | ND A100 v4 (`Standard_ND96asr_v4`) |
| `ndv5-topo.xml` | ND H100 v5 (`Standard_ND96isr_H100_v5`) |

railmap uses one only on a virtualized PCI tree whose GPU addresses match the file, and only when the host has no
matching file of its own (for example, inside a container that does not mount `/opt/microsoft`).
