"""
cli.py

Command-line interface for the LAI harmonization pipeline.

Usage:
  python cli.py \
    --input-vcf path/to/input.vcf.gz \
    --workdir path/to/workdir \
    --bundle-dir path/to/pclai_bundle \
    --reference-vcf-template "path/to/chr{chrom}.snps.vcf.gz" \
    --beagle-jar path/to/beagle.jar \
    --threads 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lai_pipeline.utils import LOG
from lai_pipeline.pipeline import LAIPipeline
from lai_pipeline.models import ToolConfig, Templates


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "LAI harmonization pipeline: harmonizes any input VCF to match "
            "the format expected by a Local Ancestry Inference model."
        )
    )

    # --- Required inputs ---
    ap.add_argument("--input-vcf", required=True, type=Path,
                    help="Input VCF file to harmonize (.vcf.gz).")
    ap.add_argument("--workdir", required=True, type=Path,
                    help="Directory where intermediate and output files will be saved.")
    ap.add_argument("--bundle-dir", required=True, type=Path,
                    help="Path to the PCLAI bundle directory (contains manifest.json and snp_manifests/).")

    # --- Optional inputs ---
    ap.add_argument("--reference-vcf-template", default=None,
                    help="Path template for per-chromosome reference panel VCFs (required for imputation). "
                         "E.g. '/data/ref/chr{chrom}.snps.vcf.gz'")
    ap.add_argument("--reference-fasta", type=Path, default=None,
                    help="Reference genome FASTA file. Used to normalize the input VCF if QC fails.")
    ap.add_argument("--genetic-map-template", default=None,
                    help="Path template for per-chromosome genetic maps used by Beagle. "
                         "E.g. '/data/maps/chr{chrom}.map'")

    # --- Imputation ---
    # minimac4 is not yet implemented but reserved for future use
    ap.add_argument("--impute-engine", choices=["beagle", "none"], default="beagle",
                    help="Imputation engine to use (default: beagle). Note: minimac4 is planned but not yet implemented.")
    ap.add_argument("--beagle-jar", type=Path, default=None,
                    help="Path to Beagle JAR file (required if --impute-engine is beagle).")

    # --- Tools ---
    ap.add_argument("--bcftools", default="bcftools",
                    help="Path to bcftools executable (default: bcftools).")
    ap.add_argument("--java", default="java",
                    help="Path to java executable (default: java).")
    ap.add_argument("--threads", type=int, default=8,
                    help="Number of threads for Beagle (default: 8).")

    # --- QC ---
    ap.add_argument("--qc-strict", action="store_true",
                    help="If set, QC failures stop the pipeline with an error.")
    ap.add_argument("--min-exact-match-pct", type=float, default=99.999,
                    help="Minimum allele exact match %% required to pass QC (default: 99.999).")
    ap.add_argument("--allow-inversions", action="store_true",
                    help="Allow inverted REF/ALT alleles without failing QC.")
    ap.add_argument("--allow-other-mismatch", action="store_true",
                    help="Allow other allele mismatches without failing QC.")
    ap.add_argument("--auto-normalize-on-qc-fail", action="store_true",
                    help="If QC fails, automatically normalize the input VCF and retry.")

    # --- Logging ---
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging verbosity (default: INFO).")

    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()

    from lai_pipeline.utils import setup_logging
    setup_logging(args.log_level)

    LOG.info("=== LAI HARMONIZATION PIPELINE ===")
    LOG.info("  input-vcf:              %s", args.input_vcf)
    LOG.info("  workdir:                %s", args.workdir)
    LOG.info("  bundle-dir:             %s", args.bundle_dir)
    LOG.info("  reference-vcf-template: %s", args.reference_vcf_template or "(none)")
    LOG.info("  reference-fasta:        %s", args.reference_fasta or "(none)")
    LOG.info("  impute-engine:          %s", args.impute_engine)
    LOG.info("  beagle-jar:             %s", args.beagle_jar or "(none)")
    LOG.info("  threads:                %s", args.threads)
    LOG.info("  qc-strict:              %s", args.qc_strict)
    LOG.info("  log-level:              %s", args.log_level)
    LOG.info("==================================")

    # --- Input validation ---
    if not args.input_vcf.exists():
        print(f"Error: input VCF not found: {args.input_vcf}")
        return 1

    if not args.bundle_dir.exists():
        print(f"Error: bundle directory not found: {args.bundle_dir}")
        return 1

    if not args.workdir.exists():
        args.workdir.mkdir(parents=True, exist_ok=True)

    if args.impute_engine == "beagle" and args.beagle_jar is None:
        print("Error: --beagle-jar is required when --impute-engine is beagle.")
        return 1

    if args.beagle_jar is not None and not args.beagle_jar.exists():
        print(f"Error: beagle JAR not found: {args.beagle_jar}")
        return 1

    cfg = ToolConfig(
        bcftools=args.bcftools,
        java=args.java,
        beagle_jar=args.beagle_jar,
        minimac4="minimac4",
        threads=args.threads,
    )

    templates = Templates(
        reference_split_template=args.reference_vcf_template,
        genetic_map_template=args.genetic_map_template,
    )

    pipe = LAIPipeline(
        cfg,
        templates,
        args.bundle_dir,
        args.workdir,
        impute_engine=args.impute_engine,
        qc_strict=args.qc_strict,
        min_exact_match_pct=args.min_exact_match_pct,
        require_zero_inversions=not args.allow_inversions,
        require_zero_other_mismatch=not args.allow_other_mismatch,
        reference_fasta=args.reference_fasta,
        auto_normalize_on_qc_fail=args.auto_normalize_on_qc_fail,
    )

    stats = pipe.run(args.input_vcf)

    print("\n=== SUMMARY ===")
    for s in stats:
        print(
            f"  chr{s.chrom} | "
            f"qc={'PASS' if s.qc_passed else 'FAIL'} | "
            f"exact={s.allele_exact_match_pct:.3f}% | "
            f"inv={s.allele_inverted} | "
            f"other={s.allele_other_mismatch} | "
            f"records={s.total_manifest_records} | "
            f"output={s.final_model_order_vcf}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())