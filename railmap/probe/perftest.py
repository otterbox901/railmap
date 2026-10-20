"""Run perftest (ib_write_bw and friends) as a server/client pair and parse its result line."""

from __future__ import annotations

import re
import shlex
import subprocess
import time
from dataclasses import dataclass

# "#bytes #iterations BW peak[Gb/sec] BW average[Gb/sec] MsgRate[Mpps]"
RESULT_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", re.MULTILINE)


@dataclass
class Result:
    size: int
    bw_avg_gbps: float
    msg_rate_mpps: float


def parse(output: str) -> Result:
    rows = RESULT_RE.findall(output)
    if not rows:
        raise ValueError("no perftest result line found in output:\n" + output[-2000:])
    size, _, _, bw_avg, rate = rows[-1]
    return Result(int(size), float(bw_avg), float(rate))


@dataclass
class Endpoint:
    hca: str
    port: int = 1
    host: str = ""  # empty: this machine; otherwise run over ssh
    gid_index: int | None = None  # RoCE needs the RoCE v2 GID index
    use_cuda: int | None = None  # GPU index for GPUDirect RDMA buffers

    def args(self) -> list[str]:
        a = ["-d", self.hca, "-i", str(self.port)]
        if self.gid_index is not None:
            a += ["-x", str(self.gid_index)]
        if self.use_cuda is not None:
            a += [f"--use_cuda={self.use_cuda}"]
        return a

    def wrap(self, argv: list[str]) -> list[str]:
        return ["ssh", "-o", "BatchMode=yes", self.host, shlex.join(argv)] if self.host else argv


def run_pair(
    tool: str,
    server: Endpoint,
    client: Endpoint,
    connect_to: str,
    common: list[str],
    tcp_port: int,
    retries: int = 10,
    timeout: float = 600,
) -> Result:
    """Start the server, connect the client to `connect_to`, and return the client's result."""
    base = [tool, *common, "-p", str(tcp_port), "-F", "--report_gbits"]
    # A remote server outlives a killed ssh client, so bound its lifetime on the far side.
    srv_argv = (["timeout", str(int(timeout))] if server.host else []) + base + server.args()
    srv = subprocess.Popen(server.wrap(srv_argv), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out = ""
        for _ in range(retries):
            time.sleep(0.5)
            proc = subprocess.run(
                client.wrap(base + client.args() + [connect_to]),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = proc.stdout + proc.stderr
            if proc.returncode == 0:
                return parse(out)
            if srv.poll() is not None:
                break
        srv_out = srv.stdout.read() if srv.stdout and srv.poll() is not None else ""
        raise RuntimeError(f"{tool} client failed:\n{out[-2000:]}\nserver:\n{srv_out[-2000:]}")
    finally:
        if srv.poll() is None:
            srv.terminate()
            try:
                srv.wait(timeout=5)
            except subprocess.TimeoutExpired:
                srv.kill()


def supports_cuda(tool: str) -> bool:
    try:
        out = subprocess.run([tool, "--help"], check=False, capture_output=True, text=True).stdout
    except FileNotFoundError:
        return False
    return "use_cuda" in out
