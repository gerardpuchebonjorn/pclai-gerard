"""
io.py

Reading, writing, and indexing VCF files via bcftools.
"""

from __future__ import annotations
from lai_pipeline.utils import run, popen_lines, count_stream_lines, LOG

import logging, re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from lai_pipeline.utils import LOG
from lai_pipeline.models import ToolConfig


def ensure_index(cfg, vcf_gz: Path, *, prefer: str = "tbi", force: bool = False) -> None:
    """Create or rebuild a .tbi or .csi index for a bgzipped VCF."""
    v_mtime = vcf_gz.stat().st_mtime if vcf_gz.exists() else 0.0
    tbi = Path(str(vcf_gz) + ".tbi")
    csi = Path(str(vcf_gz) + ".csi")

    idx_exists = tbi.exists() or csi.exists()
    idx_stale = False
    if tbi.exists() and tbi.stat().st_mtime < v_mtime:
        idx_stale = True
    if csi.exists() and csi.stat().st_mtime < v_mtime:
        idx_stale = True

    if not force and idx_exists and not idx_stale:
        return

    if idx_stale and not force:
        LOG.warning("Index older than VCF for %s -> rebuilding", vcf_gz)
        force = True

    cmd = [cfg.bcftools, "index"]
    if force:
        cmd.append("-f")
    if prefer.lower() == "tbi":
        cmd.append("-t")
    else:
        cmd.append("--csi")
    cmd.append(str(vcf_gz))
    run(cmd)


def bcftools_count_records(cfg, vcf_gz: Path) -> int:
    """Count the number of variant records in a VCF using bcftools."""
    LOG.info("Counting records (streaming): %s", vcf_gz)
    proc = popen_lines([cfg.bcftools, "view", "-H", str(vcf_gz)])
    n = count_stream_lines(proc, label=f"count({vcf_gz.name})")
    stderr = proc.stderr.read() if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"bcftools count failed rc={rc}\nSTDERR:\n{stderr}")
    return n


def read_samples_from_vcf_header(cfg, vcf: Path) -> List[str]:
    """Return the list of sample names from the VCF header."""
    hdr = run([cfg.bcftools, "view", "-h", str(vcf)], capture_stdout=True).stdout or ""
    for line in reversed(hdr.splitlines()):
        if line.startswith("#CHROM"):
            parts = line.rstrip("\n").split("\t")
            return parts[9:] if len(parts) > 9 else []
    return []


def get_vcf_contigs(cfg, vcf_gz: Path) -> List[str]:
    """Return all contig IDs declared in the VCF header."""
    p = run([cfg.bcftools, "view", "-h", str(vcf_gz)], capture_stdout=True)
    hdr = p.stdout or ""
    contigs: List[str] = []
    for line in hdr.splitlines():
        if line.startswith("##contig=<"):
            m = re.search(r"ID=([^,>]+)", line)
            if m:
                contigs.append(m.group(1))
    return contigs


def extract_chrom_variant_vcf(cfg, input_vcf: Path, contig: str, out_vcf: Path) -> None:
    """Extract biallelic SNPs for a single contig into a new VCF."""
    cmd = [
        cfg.bcftools, "view",
        "-r", contig,
        "-v", "snps",
        "-m2", "-M2",
        "-e", 'ALT="."',
        "-Oz", "-o", str(out_vcf),
        str(input_vcf),
    ]
    run(cmd)
    ensure_index(cfg, out_vcf, prefer="tbi", force=True)


def _iter_pos_ref_alt(cfg: ToolConfig, vcf: Path) -> Iterable[Tuple[int, str, str]]:
    proc = popen_lines([cfg.bcftools, "query", "-f", "%POS\t%REF\t%ALT\n", str(vcf)])
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        pos_s, ref, alt = line.split("\t")
        yield int(pos_s), ref, alt
    stderr = proc.stderr.read() if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"bcftools query failed on {vcf} rc={rc}\nSTDERR:\n{stderr}")


def _iter_vcf_data_lines(cfg, vcf: Path) -> Iterable[str]:
    """Stream raw VCF data lines (no header) via bcftools view -H."""
    proc = popen_lines([cfg.bcftools, "view", "-H", str(vcf)])
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line:
            yield line
    stderr = proc.stderr.read() if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"bcftools view -H failed on {vcf} rc={rc}\nSTDERR:\n{stderr}")

def load_bundle_manifest(bundle_dir: Path) -> dict:
    """Load and return the bundle manifest.json."""
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")
    import json
    with manifest_path.open("r") as f:
        manifest = json.load(f)
    LOG.info("Loaded bundle manifest: %s", manifest_path)
    LOG.info("Bundle format=%s models=%d", manifest.get("format"), len(manifest.get("models", [])))
    return manifest


def bundle_entries_for_chrom(bundle_dir: Path, chrom: str) -> list:
    """Return sorted bundle entries for a given chromosome."""
    manifest = load_bundle_manifest(bundle_dir)
    entries = [
        m for m in manifest["models"]
        if str(m["chrom"]) == str(chrom)
    ]
    entries = sorted(entries, key=lambda x: (999 if x["subset_idx"] is None else x["subset_idx"]))
    if not entries:
        raise RuntimeError(f"No bundle entries found for chr{chrom} in {bundle_dir}")
    return entries


def available_bundle_chroms(bundle_dir: Path) -> list:
    """Return sorted list of chromosomes available in the bundle."""
    manifest = load_bundle_manifest(bundle_dir)
    chroms = sorted({str(m["chrom"]) for m in manifest["models"]}, key=lambda x: int(x))
    return chroms


def combined_snp_manifest_for_chrom(bundle_dir: Path, chrom: str, target_contig: Optional[str] = None):
    """
    Concatenate all per-subset SNP manifests for a chromosome in bundle order.
    Returns a pandas DataFrame with columns: chrom, pos, rsid, ref, alt.
    If target_contig is provided, overwrites the chrom column with it.
    """
    import pandas as pd
    entries = bundle_entries_for_chrom(bundle_dir, chrom)
    dfs = []
    for e in entries:
        p = bundle_dir / e["snp_manifest_file"]
        if not p.exists():
            raise FileNotFoundError(f"Missing SNP manifest file: {p}")
        df = pd.read_csv(p, sep="\t", dtype={"chrom": str, "pos": int, "rsid": str, "ref": str, "alt": str})
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    if target_contig is not None:
        out = out.copy()
        out["chrom"] = str(target_contig)

    LOG.info(
        "Combined SNP manifest chr%s: %d rows from %d subset file(s)",
        chrom, len(out), len(entries)
    )
    return out


def build_key_to_tail_list(cfg, vcf: Path) -> Dict[Tuple[int, str, str], List[List[str]]]:
    """
    Build a mapping of (POS, REF, ALT) -> list of tails (columns[5:]).

    Duplicates with identical key are preserved, used to fill genotypes
    in the final VCF assembly.

    """
    m: Dict[Tuple[int, str, str], List[List[str]]] = defaultdict(list)
    for line in _iter_vcf_data_lines(cfg, vcf):
        cols = line.split("\t")
        if len(cols) < 8:
            continue
        pos = int(cols[1])
        ref = cols[3]
        alt = cols[4]

        tail = cols[5:]  # QUAL, FILTER, INFO, FORMAT, SAMPLES...
        m[(pos, ref, alt)].append(tail)
    return m