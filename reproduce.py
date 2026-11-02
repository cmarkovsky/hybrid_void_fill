#!/usr/bin/env python3
"""Reproduce manuscript results from the data archive, excluding RGI60-14.17287 by default."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

REPO = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    output = args.out.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty --out directory")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".cache/matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output / ".cache"))
    data = args.data_root.expanduser().resolve()
    scripts = REPO / "analysis/manuscript"

    def run(script, *options):
        command = [sys.executable, "-B", str(scripts / script), *map(str, options)]
        print(script, flush=True)
        subprocess.run(command, check=True, cwd=REPO)

    run("make_membership.py", "--data-root", data, "--out", output / "metadata")
    run("recalculate_core.py", "--data-root", data, "--out", output / "evidence")
    run("recalculate_transfer_coordinates.py", "--data-root", data, "--out", output / "evidence")
    run("build_tables.py", "--evidence", output / "evidence", "--out", output)
    run("run_production_stages.py", "--data-root", data, "--out", output / "pipeline_stages")
    if not args.skip_figures:
        run("make_figures.py", "--data-root", data, "--recalc", output / "evidence",
            "--subset-tree", output / "pipeline_stages", "--out", output / "figures")
    options = ["--figures", output / "figures"] if not args.skip_figures else []
    run("verify_results.py", "--data-root", data, "--stage-tree", output / "pipeline_stages",
        "--out", output / "verification.json", *options)
    print(f"Reproduction complete: {output}")


if __name__ == "__main__":
    main()
