"""
harmonize.py

Contig name detection and renaming (e.g. chr1 vs 1).
"""

from __future__ import annotations

from lai_pipeline.utils import run, LOG
from lai_pipeline.io import get_vcf_contigs, ensure_index

import re
from pathlib import Path
from typing import Dict, List, Optional

from lai_pipeline.utils import LOG


def detect_canonical_chrom_mapping(contigs: List[str]) -> Dict[str, str]:
    """
    Given a list of contig names from a VCF, return a mapping
    of canonical chrom (e.g. '1', 'X') to the actual contig name used
    in that VCF (e.g. 'chr1', 'chrX').
    """
    contig_set = set(contigs)

    def choose(canonical: str) -> Optional[str]:
        if canonical == "MT":
            candidates = ["chrM", "MT", "chrMT", "M"]
        else:
            candidates = [f"chr{canonical}", canonical]
        for c in candidates:
            if c in contig_set:
                return c
        return None

    mapping: Dict[str, str] = {}
    for c in [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]:
        picked = choose(c)
        if picked:
            mapping[c] = picked
    return mapping


def contig_for_canonical_chrom(cfg, vcf_gz: Path, chrom: str) -> str:
    """
    Return the contig name used in vcf_gz for a given canonical chrom.
    E.g. canonical '1' -> 'chr1' if that VCF uses chr-prefixed names.
    """
    contigs = get_vcf_contigs(cfg, vcf_gz)
    mapping = detect_canonical_chrom_mapping(contigs)
    if chrom in mapping:
        return mapping[chrom]
    if len(contigs) == 1:
        return contigs[0]
    raise RuntimeError(f"Could not find contig for canonical chr{chrom} in {vcf_gz}")


def rename_chrom_if_needed(cfg, in_vcf: Path, old_contig: str, new_contig: str, out_vcf: Path) -> Path:
    """
    Rename a contig in a VCF using bcftools annotate --rename-chrs.
    If old_contig == new_contig, returns in_vcf unchanged.
    """
    if old_contig == new_contig:
        return in_vcf

    out_vcf.parent.mkdir(parents=True, exist_ok=True)
    map_txt = out_vcf.parent / f"rename_{old_contig}_to_{new_contig}.txt"
    map_txt.write_text(f"{old_contig}\t{new_contig}\n")

    run([cfg.bcftools, "annotate", "--rename-chrs", str(map_txt), "-Oz", "-o", str(out_vcf), str(in_vcf)])
    ensure_index(cfg, out_vcf, prefer="tbi", force=True)
    return out_vcf


def clean_snps_biallelic(cfg: ToolConfig, in_vcf: Path, out_vcf: Path) -> Path:
    out_vcf.parent.mkdir(parents=True, exist_ok=True)
    run([
        cfg.bcftools, "view",
        "-v", "snps",
        "-m2", "-M2",
        "-e", 'ALT="."',
        "-Oz", "-o", str(out_vcf),
        str(in_vcf),
    ])
    ensure_index(cfg, out_vcf, prefer="tbi", force=True)
    return out_vcf