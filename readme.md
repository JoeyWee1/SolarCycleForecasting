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
# Clone this majestic repository
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/assessments/projects/zyw26.git

# Create a virtual environment
python3 -m venv VenvStellarForecasting

# Activate the virtual environment
source VenvStellarForecasting/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## How to Run
---
There are a few key tasks the code can accomplish; we will go through them here one by one.

### Predict a star
Our predictions work on S-Index data (refer to the report for details on the S-Index).
With a dataset of S-Index measurements and the Julian date (JD) times at which they were taken, the code can create the predictions.
``` 
from helpers.pipeline import run_star
result = run_star(
    datapath='../../Data/benchmark/HD201091_Mt_wilson_data.txt',
    star_name='HD81809', star_type='G', add_prefix=False,
    lookahead_years=[1, 2, 3, 5],
    verbose=False, plot=True, # This will plot the predictions
)
```
The result object contains the predictions and best times to observe in the lookahead windows.
See documentation for details.

### Evaluate pipeline performance
In the report, we evaluated the performance of the predictions on the benchmark stars with 25 splits per dataset.
Due to HPC failure this was not completed on the full mwd dataset. 
A similar analysis can be performed on other data easily using star_window_analysis.py.
``` 
# In command line
python analyse_star.py \ # From the Analysis folder
    --data_dir  /path/to/Data/mwd \
    --output_dir /path/to/results \ # Outputs one pkl per star
    [--star_type G] \  # Star type data; here we use G because it is valid with the SM2016 relation
    [--n_windows 5] \
    [--skip_existing]
```
There is also a bash wrapper for HPC use. Edit the username for it to work.

### Cadence analysis
In the report, how the cadence affects the accuracy of the predictions is reported. 
Due to the HPC failure this was only run with 15/100 simulated stars.
To perform the full analysis, use cadence_analysis.py.
```
# All folders, all 100 stars, 5 windows:
python cadence_analysis.py --sim_root Data/simulated --out_dir Results/simulated
```

## Test Suite
---



## Notes
---
- "RuntimeWarning: invalid value encountered in scalar subtract lnpdiff = f + nlp - state.log_prob[j]" is a common warning but does not affect the performance of the modelling. If it bothers you, you can silence it with
```
import warnings
warnings.filterwarnings('ignore', message='invalid value encountered in scalar subtract')
```


## Autogeneration Tools
---



## Appendix: Detailed Notebook Breakdown
---

