#!/usr/bin/env python3
"""
cadence_analysis.py
This is a batch runner to apply the cadence analysis in helpers.pipeline ot the simulated stars
which live in the Data/simulated/<rate>/ folders

The <rate> folders correspond to the cadences to which the simulated stars were sampled.

The analysis is run on each star and is saved as a pickle file for each star.
This makes the run resumable in the case of any crash or bug.
Each pickle saves the dict returned by star_window_analysis():
- star, n_obs, span_years, skipped, skip_reason
- splits: list of per-train and per-valid-split dicts, each with
  - best_combo, valid_metric_value, rho_mid_years, Q_mid, n_cycles_obs, is_const, aleatoric_frac_at_t0
  - windows: list of per-lookahead dicts, each with
    - lookahead_years, t0_year
    - gpr_med, gpr_lb68, gpr_ub68, gpr_lb95, gpr_ub95 (GPR predicted minimum + CIs)
    - truth_med, truth_lb, truth_ub (Fourier ground-truth minimum + 68% CI)
    - error, abs_error, gpr_width_68, truth_width, in_68, in_95

After each folder is complete a summary CSV for that cadence is saved.
A combined CSV is writted from all of these once the end of the run is complete.
These CSVs contain one row per (star, split, lookahead window), with all the
split_idx and sampling_rate_days.


Usage
-----
# All folders, all 100 stars, 5 windows:
python cadence_analysis.py --sim_root Data/simulated --out_dir Results/simulated

# One folder only (useful for parallelising over folders on HPC):
python cadence_analysis.py --sim_root Data/simulated --out_dir Results/simulated \
    --rate 1.0d --n_windows 10

# Quick test: 3 G-type stars, 5 windows:
python cadence_analysis.py --sim_root Data/simulated --out_dir Results/simulated \
    --n_stars 3 --star_type_filter G
"""

import argparse
import logging
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from helpers.pipeline import star_window_analysis, results_to_df

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Batch cadence analysis runner on simulated data')
    parser.add_argument('--sim_root',         default='Data/simulated',
                        help='Root of simulated data (contains <rate>d/ subfolders)')
    parser.add_argument('--out_dir',          default='Results/simulated',
                        help='Output directory; per-star pickles and per-rate CSVs written here')
    parser.add_argument('--n_windows',        type=int, default=5,
                        help='Target number of train/valid windows per star (default: 5)')
    parser.add_argument('--rate',             default=None,
                        help='Run only this folder, e.g. "1.0d" (default: all folders)')
    parser.add_argument('--n_stars',          type=int, default=None,
                        help='Limit number of stars per folder (default: all)')
    parser.add_argument('--star_type_filter', default=None,
                        help='Only run stars of this type, e.g. "G" (default: all)')
    parser.add_argument('--skip_existing',    action='store_true',
                        help='Skip stars whose output pickle already exists')
    parser.add_argument('--verbose',          action='store_true')
    args = parser.parse_args()

    sim_root = Path(args.sim_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / 'cadence_analysis.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-7s  %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger('cadence_analysis')
    log.info('sim_root  : %s', sim_root)
    log.info('out_dir   : %s', out_dir)
    log.info('n_windows : %d', args.n_windows)

    params = np.load(sim_root / 'star_params.npz', allow_pickle=True)
    star_labels = params['labels']          # (N,) string array, e.g. 'G', 'ME'

    def prior_type(stype): return stype  # pass full type; get_priors has ME and MM keys

    # Select folders
    all_folders = sorted(
        [f for f in sim_root.iterdir() if f.is_dir()],
        key=lambda f: float(f.name.rstrip('d'))
    )
    folders = [f for f in all_folders if args.rate is None or f.name == args.rate]
    if not folders:
        log.error('No folders matched --rate %s', args.rate)
        sys.exit(1)

    # Select star indices
    indices = np.arange(len(star_labels))
    if args.star_type_filter:
        indices = indices[star_labels[indices] == args.star_type_filter]
    if args.n_stars is not None:
        indices = indices[:args.n_stars]

    log.info('Folders: %s', [f.name for f in folders])
    log.info('Stars  : %d', len(indices))

    all_dfs = []

    for folder in folders:
        rate_str = folder.name
        rate_days = float(rate_str.rstrip('d'))
        pkl_dir = out_dir / rate_str
        pkl_dir.mkdir(exist_ok=True)
        log.info('=== %s ===', rate_str)

        n_ok = n_skip = n_fail = 0

        for i in indices:
            stype = star_labels[i]
            star_name = f'sim{i:04d}_{stype}'
            pkl_path = pkl_dir / f'{star_name}.pkl'
            fpath = folder / f'{star_name}_caii.txt'

            if args.skip_existing and pkl_path.exists():
                log.info('  SKIP (exists): %s', star_name)
                n_skip += 1
                continue

            if not fpath.exists():
                log.warning('  MISSING: %s', fpath)
                n_fail += 1
                continue

            log.info('  %s  (P_cyc=%.1f yr, P_rot=%.1f d)',
                     star_name, params['P_cycs'][i], params['P_rots'][i])
            try:
                result = star_window_analysis(
                    datapath=str(fpath),
                    star_name=star_name,
                    star_type=prior_type(stype),
                    add_prefix=False,
                    n_target_windows=args.n_windows,
                    verbose=args.verbose,
                )
                with open(pkl_path, 'wb') as f:
                    pickle.dump(result, f)
                n_ok += 1
            except Exception:
                log.error('  FAILED: %s\n%s', star_name, traceback.format_exc())
                n_fail += 1

        log.info('%s done — ok=%d  skipped=%d  failed=%d', rate_str, n_ok, n_skip, n_fail)

        # Combine all pickles for this folder into a CSV
        results = []
        for pkl_path in sorted(pkl_dir.glob('*.pkl')):
            with open(pkl_path, 'rb') as f:
                results.append(pickle.load(f))

        if results:
            df = results_to_df(results)
            df['sampling_rate_days'] = rate_days
            csv_path = out_dir / f'{rate_str}.csv'
            df.to_csv(csv_path, index=False)
            log.info('  CSV → %s  (%d rows)', csv_path, len(df))
            all_dfs.append(df)

    # Combined CSV across all rates
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = out_dir / 'all_rates.csv'
        combined.to_csv(combined_path, index=False)
        log.info('Combined CSV → %s  (%d rows)', combined_path, len(combined))

    log.info('All done.')


if __name__ == '__main__':
    main()
