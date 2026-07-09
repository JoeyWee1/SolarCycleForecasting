#!/usr/bin/env python3
"""
analyse_star.py
Batch runner: apply the GPR pipeline to every dataset in Data/mwd/ that
passes quality checks. One pickle file per star is written under --output_dir.

Usage
-----
python analyse_star.py \
    --data_dir  /path/to/Data/mwd \
    --output_dir /path/to/results \
    [--star_type G] \
    [--n_windows 5] \
    [--skip_existing]
"""

import argparse
import logging
import pickle
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from helpers.pipeline import star_window_analysis


def run_one_star(datapath, output_dir, star_type, n_windows, log):
    star_name = Path(datapath).stem.replace("_caii", "").upper()
    out_path  = Path(output_dir) / f"{star_name}_results.pkl"

    log.info("── %s ──", star_name)

    result = star_window_analysis(
        datapath=str(datapath),
        star_name=star_name,
        star_type=star_type,
        n_target_windows=n_windows,
    )

    if result.get("skipped"):
        log.warning("  SKIP: %s", result.get("skip_reason", "unknown"))
        return False

    log.info("  %d obs, %.1f yr, %d split(s)",
             result["n_obs"], result["span_years"], len(result["splits"]))

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    log.info("  saved → %s", out_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch GPR star analyser")
    parser.add_argument("--data_dir",      required=True,
                        help="Directory containing *_caii.txt files")
    parser.add_argument("--output_dir",    required=True,
                        help="Directory to write per-star pickle files")
    parser.add_argument("--star_type",     default="G",
                        help="Spectral type passed to get_priors (default: G)")
    parser.add_argument("--n_windows",     type=int, default=5,
                        help="Target number of train/valid windows per star (default: 5)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip stars whose output pickle already exists")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "analyse_star.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("analyse_star")
    log.info("data_dir   : %s", args.data_dir)
    log.info("output_dir : %s", output_dir)
    log.info("star_type  : %s", args.star_type)
    log.info("n_windows  : %d", args.n_windows)

    data_files = sorted(Path(args.data_dir).glob("*_caii.txt"))
    log.info("Found %d dataset files", len(data_files))

    n_ok = n_skip = n_fail = 0

    for i, datapath in enumerate(data_files):
        star_name = datapath.stem.replace("_caii", "").upper()
        out_path  = output_dir / f"{star_name}_results.pkl"

        if args.skip_existing and out_path.exists():
            log.info("[%d/%d] SKIP (exists): %s", i + 1, len(data_files), star_name)
            n_skip += 1
            continue

        log.info("[%d/%d] %s", i + 1, len(data_files), star_name)
        try:
            ok = run_one_star(datapath, output_dir, args.star_type, args.n_windows, log)
            if ok:
                n_ok += 1
            else:
                n_skip += 1
        except Exception:
            log.error("FAILED: %s\n%s", star_name, traceback.format_exc())
            n_fail += 1

    log.info("Done.  processed=%d  skipped=%d  failed=%d", n_ok, n_skip, n_fail)


if __name__ == "__main__":
    main()
