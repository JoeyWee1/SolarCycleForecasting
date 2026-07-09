# Solar and Stellar Activity Cycle Forecasting with Gaussian Process Regression
---
Author: Joey Wee
Dissertation project submitted for the MPhil Data Intensive Science Programme.

## Project Overview
---
Exoplanets are detected by the radial velocity or transit methods, which rely on spectroscopic or photometric measurements respectively. However, non-exoplanetary noise in these measurements can be introduced by stellar activity. The high-precision instruments being used to take these measurements are also heavily oversubscribed. The magnitude of this noise is proportional to the stellar activity and has been shown to significantly affect exoplanet detectability. As such, by predicting the periods of lower stellar activity using data that can be taken using more modest instruments, the high-precision instrument observation schedules can be optimised such that they observe at the stellar activity minima to maximise detectability.

This project has produced a scalable pipeline which creates these forecasts.

## Repository Structure
---

```
Thesis/
├── Analysis/                                # Batch run scripts and shell wrappers
│   ├── analyse_star.py                 # Batch GPR window analysis pipeline runner (per-star pickle output)
│   ├── analyse_star.sh                 # Shell wrapper for analyse_star.py
│   ├── cadence_analysis.py          # Observing cadence analysis script
│   └── cadence_analysis.sh         # Shell wrapper for cadence_analysis.py
│
├── Data/                                     # Input datasets
│   ├── benchmark/                      # Benchmark star observations
│   ├── misc/                                 # Miscellaneous / auxiliary data
│   ├── mwd/                                 # Mount Wilson Observatory S-index data files
│   ├── simulated/                        # Simulated activity cycle datasets
│   └── table_A2_full.csv            # Full stellar catalogue table used to generate simulated stars
│
├── helpers/                                 # Shared Python library
│   ├── df_ops.py                          # Data loading, splitting, and cleaning
│   ├── eval.py                               # Forecast evaluation metrics 
│   ├── gpr.py                                # GPR kernel setup and NLL optimisation
│   ├── LSP_peaks.py                     # Lomb-Scargle periodogram peak detection
│   ├── MCMC.py                            # emcee log-posterior for GPR parameters
│   ├── pipeline.py                         # High-level pipeline functions (star_window_analysis, run_star)
│   ├── priors.py                             # Prior generation and signal classification
│   └── __init__.py
│
├── Notebooks/                              # Exploratory and development notebooks
│   ├── ARIMA/                              # ARIMA baseline experiments
│   ├── GPR/                                    # GPR model development iterations (GPR1–GPR25)
│   ├── Priors/                                 # Prior design and automation notebooks
│   └── data-preproc.ipynb              #  Data preprocessing and exploration
│
├── Report/                                     # Written report and figures
│   ├── Figures/                                # All figures included in the report
│   ├── report.pdf                              # Delicious dissertation report
│   └── summary.pdf                         # Executive summary
│
├── Results/                                      # Pipeline output
│   ├── benchmark/                          # Per-star result pickles for benchmark stars
│   └── simulated/                              # Results for simulated cadence experiments
│
├── tests/                                           # Unit and integration tests
│   ├── conftest.py
│   ├── test_gpr.py
│   └── test_priors.py
│
├── .gitignore
├── Instructions.md
├── LICENSE
├── notes.md
├── readme.md
└── requirements.txt
```

## Setup
---
```

```


## How to Run
---



## Test Suite
---



## Autogeneration Tools
---



## Appendix: Detailed Notebook Breakdown
---

