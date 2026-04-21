"""
impute.py

Phasing and imputation using Beagle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from lai_pipeline.utils import run, LOG
from lai_pipeline.io import ensure_index
from lai_pipeline.models import ToolConfig


def run_beagle_phasing(cfg, in_vcf: Path, out_prefix: Path, genetic_map: Optional[Path]) -> Path:
    """
    Phase an unphased VCF using Beagle.
    Returns the path to the phased output VCF.
    """
    if cfg.beagle_jar is None:
        raise ValueError("beagle_jar must be provided to phase unphased targets.")

    cmd = [
        cfg.java, "-jar", "-Xmx200g", str(cfg.beagle_jar),
        f"gt={in_vcf}",
        f"out={out_prefix}",
        f"nthreads={cfg.threads}",
    ]
    if genetic_map:
        cmd.append(f"map={genetic_map}")

    run(cmd)

    phased_vcf = Path(str(out_prefix) + ".vcf.gz")
    if not phased_vcf.exists():
        raise RuntimeError(f"Beagle phasing output not found: {phased_vcf}")
    ensure_index(cfg, phased_vcf, prefer="tbi", force=True)
    return phased_vcf


def run_beagle_imputation(cfg, gt_vcf: Path, ref_vcf: Path, out_prefix: Path, genetic_map: Optional[Path]) -> Path:
    """
    Impute missing variants using Beagle with a reference panel.
    Returns the path to the imputed output VCF.
    """
    if cfg.beagle_jar is None:
        raise ValueError("beagle_jar must be provided for Beagle imputation.")

    cmd = [
        cfg.java, "-jar", "-Xmx200g", str(cfg.beagle_jar),
        f"gt={gt_vcf}",
        f"ref={ref_vcf}",
        f"out={out_prefix}",
        "impute=true",
        f"nthreads={cfg.threads}",
    ]
    if genetic_map:
        cmd.append(f"map={genetic_map}")

    run(cmd)

    out_vcf = Path(str(out_prefix) + ".vcf.gz")
    if not out_vcf.exists():
        raise RuntimeError(f"Beagle imputation output not found: {out_vcf}")
    ensure_index(cfg, out_vcf, prefer="tbi", force=True)
    return out_vcf