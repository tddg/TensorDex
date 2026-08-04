#!/usr/bin/env python3
"""Full HF CLI campaign: end-to-end `hf download` (xet verify + write to
disk) per model, with concurrent route tracing that does NOT touch the
download path: sample the CLI process's own established TCP peers
(ss -tnp filtered by pid) and classify them against resolved HF/CDN
hostnames, plus NIC rx-byte accounting per run.

Local HF cache is wiped before every model, so every run pays the full
download + verify + write cost. One record per model in --out (JSONL).
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import threading
import time

CANDIDATES = {
    "us.aws.cdn.hf.co": "hf-own-cdn",
    "cas-server.xethub.hf.co": "cas-server",
    "transfer.xethub.hf.co": "cloudfront-xet",
    "cas-bridge.xethub.hf.co": "cloudfront-xet",
    "huggingface.co": "hub-cloudfront",
    "cdn-lfs.huggingface.co": "cdn-lfs",
    "cdn-lfs-us-1.hf.co": "cdn-lfs",
}


def resolve_all():
    m = {}
    for host, cls in CANDIDATES.items():
        try:
            for info in socket.getaddrinfo(host, 443,
                                           proto=socket.IPPROTO_TCP):
                m[info[4][0]] = (cls, host)
        except OSError:
            pass
    return m


def nic_rx(iface):
    with open(f"/sys/class/net/{iface}/statistics/rx_bytes") as f:
        return int(f.read())


class PeerSampler(threading.Thread):
    """Poll `ss -tnp` and count samples per remote IP for one pid."""

    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid_tag = f"pid={pid}"
        self.stop_ev = threading.Event()
        self.peers = {}
        self.ipmap = resolve_all()

    def run(self):
        last_resolve = time.monotonic()
        while not self.stop_ev.is_set():
            out = subprocess.run(["ss", "-tnp", "state", "established"],
                                 capture_output=True, text=True).stdout
            for line in out.splitlines():
                if self.pid_tag not in line:
                    continue
                # with a state filter ss omits the State column:
                # Recv-Q Send-Q Local:Port Peer:Port Process
                parts = line.split()
                if len(parts) >= 4:
                    peer = parts[3].rsplit(":", 1)[0]
                    if peer.startswith(("10.", "172.", "127.", "[")):
                        continue
                    self.peers[peer] = self.peers.get(peer, 0) + 1
            if time.monotonic() - last_resolve > 10:
                self.ipmap.update(resolve_all())
                last_resolve = time.monotonic()
            time.sleep(0.25)

    def summary(self):
        by_cls, by_ip = {}, {}
        for ip, n in sorted(self.peers.items(), key=lambda kv: -kv[1]):
            cls, host = self.ipmap.get(ip, ("other", None))
            d = by_cls.setdefault(cls, {"ips": 0, "samples": 0})
            d["ips"] += 1
            d["samples"] += n
            by_ip[ip] = {"samples": n, "cls": cls, "host": host}
        return by_cls, by_ip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-bin", default="hf")
    ap.add_argument("--iface", default="ens5")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    with open(args.out, "a") as out_f:
        for repo in args.repos:
            hf_home = os.path.join(args.workdir, "hf_home")
            shutil.rmtree(hf_home, ignore_errors=True)
            env = dict(os.environ, HF_HOME=hf_home,
                       HF_HUB_DISABLE_TELEMETRY="1")
            log_path = os.path.join(
                args.workdir, repo.replace("/", "__") + ".log")
            rx0 = nic_rx(args.iface)
            t0 = time.monotonic()
            with open(log_path, "w") as log_f:
                proc = subprocess.Popen(
                    [args.hf_bin, "download", repo],
                    env=env, stdout=log_f, stderr=subprocess.STDOUT)
                sampler = PeerSampler(proc.pid)
                sampler.start()
                rc = proc.wait()
            wall = time.monotonic() - t0
            sampler.stop_ev.set()
            sampler.join(timeout=5)
            rx1 = nic_rx(args.iface)
            by_cls, by_ip = sampler.summary()
            du = 0
            if os.path.isdir(hf_home):
                r = subprocess.run(["du", "-sb", hf_home],
                                   capture_output=True, text=True)
                if r.stdout:
                    du = int(r.stdout.split()[0])
            rec = {"repo": repo, "rc": rc, "e2e_s": round(wall, 2),
                   "nic_rx_bytes": rx1 - rx0, "disk_bytes": du,
                   "disk_GiB": round(du / 2**30, 2),
                   "wire_MiBps": round((rx1 - rx0) / 2**20 / wall, 1),
                   "disk_MiBps": round(du / 2**20 / wall, 1),
                   "route_by_class": by_cls, "route_by_ip": by_ip}
            if rc != 0:
                tail = open(log_path).read()[-500:]
                rec["error_tail"] = tail
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            print(f"{repo}: rc={rc} {wall:.1f}s "
                  f"{du/2**30:.2f} GiB disk, "
                  f"{(rx1-rx0)/2**30:.2f} GiB wire, "
                  f"routes={ {k: v['samples'] for k, v in by_cls.items()} }",
                  flush=True)
            shutil.rmtree(hf_home, ignore_errors=True)


if __name__ == "__main__":
    main()
