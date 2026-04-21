"""
assembly.py

Final VCF reconstruction in exact SNP manifest order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lai_pipeline.utils import run, LOG
from lai_pipeline.io import ensure_index, read_samples_from_vcf_header, build_key_to_tail_list, _iter_vcf_data_lines
from lai_pipeline.models import ToolConfig


def write_final_vcf_in_manifest_order(
    cfg: ToolConfig,
    *,
    chrom: str,
    snp_df,
    header_source_vcf: Path,
    beagle_fill_vcf: Optional[Path],
    target_fill_vcf: Path,
    out_vcf_gz: Path,
) -> None:
    """
    Emit final VCF in exact SNP-manifest row order.
    For each manifest record (POS, REF, ALT), fill FORMAT + samples from:
      - beagle_fill_vcf (preferred) if available and contains exact allele
      - else target_fill_vcf if contains exact allele
      - else missing (./.)

    Output is bgzipped via bcftools view -Oz and indexed.
    """
    samples = read_samples_from_vcf_header(cfg, header_source_vcf)
    n_samples = len(samples)

    LOG.info("FINAL assembly chr%s:", chrom)
    LOG.info("  manifest_rows     = %d", len(snp_df))
    LOG.info("  header_source     = %s", header_source_vcf)
    LOG.info("  beagle_fill_vcf   = %s", str(beagle_fill_vcf) if beagle_fill_vcf else "(none)")
    LOG.info("  target_fill_vcf   = %s", target_fill_vcf)
    LOG.info("  n_samples         = %d", n_samples)

    beagle_map = build_key_to_tail_list(cfg, beagle_fill_vcf) if beagle_fill_vcf else {}
    target_map = build_key_to_tail_list(cfg, target_fill_vcf)

    # Write uncompressed .vcf then compress with bcftools (bgzip) for indexing.
    tmp_vcf = out_vcf_gz.with_suffix("")  # drops .gz
    if tmp_vcf.suffix != ".vcf":
        tmp_vcf = tmp_vcf.with_suffix(".vcf")

    hdr = run([cfg.bcftools, "view", "-h", str(header_source_vcf)], capture_stdout=True).stdout or ""
    with tmp_vcf.open("w") as fout:
        fout.write(hdr)

        missing_count = 0
        beagle_used = 0
        target_used = 0
        total_manifest = 0

        for row in snp_df.itertuples(index=False):
            total_manifest += 1
            chrom_s = str(row.chrom)
            pos = int(row.pos)
            vid = str(row.rsid)
            ref = str(row.ref)
            alt = str(row.alt)

            key = (pos, ref, alt)

            tail: Optional[List[str]] = None
            if key in beagle_map and beagle_map[key]:
                tail = beagle_map[key].pop(0)
                beagle_used += 1
            elif key in target_map and target_map[key]:
                tail = target_map[key].pop(0)
                target_used += 1
            else:
                missing_count += 1
                # Minimal tail with GT only (keeps VCFReaderPolars happy)
                # QUAL FILTER INFO FORMAT [samples...]
                sample_cols = ["./."] * n_samples
                tail = [".", "PASS", ".", "GT"] + sample_cols

            out_cols = [chrom_s, str(pos), vid, ref, alt] + tail
            fout.write("\t".join(out_cols) + "\n")

    LOG.info("FINAL assembly chr%s done:", chrom)
    LOG.info("  total_manifest_records = %d", total_manifest)
    LOG.info("  filled_from_beagle     = %d", beagle_used)
    LOG.info("  filled_from_target     = %d", target_used)
    LOG.info("  missing_unfilled       = %d", missing_count)

    # Compress + index
    out_vcf_gz.parent.mkdir(parents=True, exist_ok=True)
    run([cfg.bcftools, "view", "-Oz", "-o", str(out_vcf_gz), str(tmp_vcf)])
    ensure_index(cfg, out_vcf_gz, prefer="tbi", force=True)

    # (Optional) keep tmp for debugging; comment out next line if you want to inspect
    try:
        tmp_vcf.unlink()
    except Exception:
        pass