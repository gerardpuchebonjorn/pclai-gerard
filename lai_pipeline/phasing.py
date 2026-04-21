"""
phasing.py

Detection of phasing status in a VCF file.
"""

from __future__ import annotations

import re, subprocess
from pathlib import Path

from lai_pipeline.utils import run, LOG
from lai_pipeline.models import ToolConfig 


def is_vcf_phased(cfg, vcf_gz: Path, contig: str, max_lines: int = 2000) -> bool:
    """
    Detect whether a VCF is phased by inspecting genotype separators.
    Phased genotypes use '|' (e.g. 0|1), unphased use '/' (e.g. 0/1).
    Checks up to max_lines records for efficiency.
    """
    hdr = run([cfg.bcftools, "view", "-h", str(vcf_gz)], capture_stdout=True).stdout or ""
    header_says_phased = False
    for line in hdr.splitlines():
        if line.startswith("##FORMAT=<ID=GT"):
            if re.search(r"phased", line, flags=re.IGNORECASE):
                header_says_phased = True
            break

    cmd = [cfg.bcftools, "query", "-r", contig, "-f", "[%GT\t]\n", str(vcf_gz)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    pipes = 0
    slashes = 0
    n_lines = 0
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            n_lines += 1
            for gt in line.rstrip("\n").split("\t"):
                if not gt or gt == ".":
                    continue
                if "|" in gt:
                    pipes += 1
                if "/" in gt:
                    slashes += 1
            if n_lines >= max_lines:
                break
            if slashes > 0 and pipes == 0:
                break
            if pipes > 0 and slashes > 0:
                break
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    rc = proc.returncode
    if rc not in (0, -13, 141, -15):
        stderr = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(f"Phasing check failed rc={rc}\nSTDERR:\n{stderr}")

    if slashes > 0 and pipes == 0:
        return False
    if pipes > 0 and slashes == 0:
        return True
    if pipes > 0 and slashes > 0:
        return False
    return header_says_phased