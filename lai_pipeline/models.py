"""
models.py

Shared dataclasses used across the pipeline modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ToolConfig:
    """External tool paths and runtime settings."""
    bcftools: str
    java: str
    beagle_jar: Optional[Path]
    minimac4: str
    threads: int


@dataclass
class Templates:
    """Path templates for per-chromosome reference files."""
    reference_split_template: Optional[str]
    genetic_map_template: Optional[str]


@dataclass
class ChromStats:
    """Per-chromosome processing results."""
    chrom: str
    input_contig: str
    total_manifest_records: int
    qc_passed: bool
    allele_exact_match_pct: float
    allele_inverted: int
    allele_other_mismatch: int
    target_is_phased: bool
    phased_or_imputed_vcf: Optional[str]
    final_model_order_vcf: str


@dataclass
class AlleleConcordanceStats:
    """Results of allele concordance QC check."""
    shared_pos: int
    exact_match: int
    inverted_ref_alt: int
    other_mismatch: int
    missing_in_model: int
    exact_match_pct: float
    inverted_pct: float
    other_mismatch_pct: float
    examples: List[str]