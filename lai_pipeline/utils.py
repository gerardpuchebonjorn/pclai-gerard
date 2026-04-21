"""
utils.py

Subprocess helpers and logging setup.
"""

from __future__ import annotations

import logging, shlex, subprocess
from pathlib import Path
from typing import List, Optional

LOG = logging.getLogger("lai-pipeline")


def setup_logging(level: str) -> None:
    """Configure root logger format and level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    LOG.info("Logging initialized at level=%s", level.upper())


def shjoin(cmd: List[str]) -> str:
    """Return a shell-quoted string representation of a command list."""
    return " ".join(shlex.quote(c) for c in cmd)


def run(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    check: bool = True,
    capture_stdout: bool = False,
    capture_stderr: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    """Run a command, log it, and return the CompletedProcess."""
    LOG.info("RUN: %s", shjoin(cmd))
    if cwd:
        LOG.info("  cwd: %s", cwd)

    stdout = subprocess.PIPE if capture_stdout else None
    stderr = subprocess.PIPE if capture_stderr else None

    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=stdout,
        stderr=stderr,
        text=text,
    )
    LOG.info("  exit_code=%s", p.returncode)

    if p.stdout:
        LOG.debug("  STDOUT (first 2000 chars):\n%s", p.stdout[:2000])
    if p.stderr:
        LOG.debug("  STDERR (first 2000 chars):\n%s", p.stderr[:2000])

    if check and p.returncode != 0:
        msg = f"Command failed ({p.returncode}): {shjoin(cmd)}"
        if p.stderr:
            msg += f"\nSTDERR:\n{p.stderr}"
        raise RuntimeError(msg)
    return p



def popen_lines(cmd: List[str], *, cwd: Optional[Path] = None) -> subprocess.Popen:
    """Open a subprocess and return the Popen object for line-by-line streaming."""
    LOG.info("POPEN: %s", shjoin(cmd))
    if cwd:
        LOG.info("  cwd: %s", cwd)

    return subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

def count_stream_lines(proc: subprocess.Popen, *, label: str) -> int:
    """Count lines from a Popen stdout, logging progress every 1M lines."""
    assert proc.stdout is not None
    n = 0
    for _ in proc.stdout:
        n += 1
        if n % 1_000_000 == 0:
            LOG.info("%s: counted %d lines so far...", label, n)
    return n