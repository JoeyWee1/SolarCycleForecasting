# Solar and Stellar Activity Cycle Forecasting with Gaussian Process Regression
---
Author: Joey Wee
Dissertation project submitted for the MPhil Data Intensive Science Programme.

## Project Overview
---
Exoplanets are detected by the radial velocity or transit methods, which rely on spectroscopic or photometric measurements respectively. However, non-exoplanetary noise in these measurements can be introduced by stellar activity. The high-precision instruments being used to take these measurements are also heavily oversubscribed. The magnitude of this noise is proportional to the stellar activity and has been shown to significantly affect exoplanet detectability. As such, by predicting the periods of lower stellar activity using data that can be taken using more modest instruments, the high-precision instrument observation schedules can be optimised such that they observe at the stellar activity minima to maximise detectability.

This project has produced a scalable pipeline which creates these forecasts. The code for this pipeline lives in the helpers module.

The executive summary and full report describing this work is found in the Results folder.

## Repository Structure
---

```
Thesis/
├── Analysis/                                # Batch run scripts and shell wrappers
│   ├── star_window_analysis.py         # Batch GPR window analysis pipeline runner (per-star pickle output)
│   ├── star_window_analysis.sh         # Shell wrapper for star_window_analysis.py
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
├── helpers/                                 # Python library: main product of this work.
│   ├── df_ops.py                          # Data loading, splitting, and cleaning
│   ├── eval.py                               # Forecast evaluation metrics 
│   ├── gpr.py                                # GPR kernel setup and NLL optimisation
│   ├── LSP_peaks.py                     # Lomb-Scargle periodogram peak detection
│   ├── MCMC.py                            # emcee log-posterior for GPR parameters
│   ├── pipeline.py                         # High-level pipeline functions (star_window_analysis, run_star)
│   ├── priors.py                             # Prior generation and signal classification
│   └── __init__.py
│
├── Notebooks/                              # Code development notebooks
│   ├── ARIMA/                              # ARIMA baseline experiments
│   │   ├── ARIMA1_solar.ipynb          # ARIMA/Fourier+ARIMA baseline on solar S-index; motivates GPR
│   │   └── ARIMA2_baseline_stars.ipynb # Same on benchmark stars; highlights ARIMA inadequacy
│   ├── GPR/                                    # GPR model development iterations
│   │   ├── GPR1.ipynb                      # First GPR attempt; priors identified as necessary
│   │   ├── GPR2.ipynb                      # Manual kernel tuning on solar data
│   │   ├── GPR3.ipynb                      # Moving-window SHO+ROT kernel on solar data
│   │   ├── GPR4_Spectrum_Kernel.ipynb  # Spectrum kernel introduced
│   │   ├── GPR5_Moving_Window_Spectrum.ipynb  # Moving-window eval of spectrum kernel
│   │   ├── GPR6_Refined_Priors_GPR.ipynb      # LSP-derived priors applied to benchmark stars
│   │   ├── GPR7_Removing_Outliers.ipynb        # MAD outlier filter; motivates model selection
│   │   ├── GPR8_Model_Selection.ipynb          # NLPD-based selection across kernel configurations
│   │   ├── GPR9_Pipeline.ipynb                 # Consolidated train_gpr() pipeline
│   │   ├── GPR10_Moving_Window_Pipeline.ipynb  # Moving-window sweep of pipeline; motivates MCMC
│   │   ├── GPR11_Further_Baseline_Manual.ipynb # LSP detection over full MWD catalogue stats
│   │   ├── GPR12a_Examining_Each_Detected_Combo_3sml_HD18256.ipynb  # 3sml case (all detected)
│   │   ├── GPR12b_Examining_Each_Detected_Combo_2sm_HD10072.ipynb   # 2sm case (no long)
│   │   ├── GPR12c_Examining_Each_Detected_Combo_2ml_HD16673.ipynb   # 2ml case (no short)
│   │   ├── GPR12d_Examining_Each_Detected_Combo_1m_HD11131.ipynb    # 1m case (cycle only)
│   │   ├── GPR12d_Examining_Each_Detected_Como_0_HD10700.ipynb      # 0 case (flat star)
│   │   ├── GPR13_MCMC.ipynb                    # emcee MCMC for kernel parameter uncertainty
│   │   ├── GPR14_BMA.ipynb                     # Bayesian Model Averaging across kernel candidates
│   │   ├── GPR14b_BMA_window.ipynb             # BMA in moving-window analysis; motivates CRPS
│   │   ├── GPR15_CRPS_window.ipynb             # CRPS-weighted BMA; single-model MCMC instead chosen
│   │   ├── GPR16_Minima_MCMC.ipynb             # Lookahead windows 
│   │   ├── GPR17_Results.ipynb                 # Full CRPS + MCMC + lookahead pipeline on HD201091
│   │   ├── GPR18_Improved_Results.ipynb        # Fixes boundary ringing and data leakage
│   │   ├── GPR19_Results_Window.ipynb          # Multi-window sweep with lookahead vs ground truth
│   │   ├── GPR20_Min_Gap_Window.ipynb          # Min-gap downsampling to fix boundary bias
│   │   ├── GPR21_New_Opt.ipynb                 # Forward-NLPD optimisation strategy
│   │   ├── GPR22_Init_Cond_Var.ipynb           # Multi-start optimisation
│   │   ├── GPR23a_Revised_Ground_Truth.ipynb   # Fourier MCMC probabilistic ground truth
│   │   ├── GPR23b_Revised_Ground_Truth_Pipeline.ipynb  # Fourier truth integrated into windowed pipeline
│   │   ├── GPR23c_Functionalised.ipynb         # star_window_analysis() and results_to_df() finalised
│   │   ├── GPR24a_Simulated_Data.ipynb         # Synthetic stellar population generator
│   │   ├── GPR24b_Testing_Simulated_Data.ipynb # Pipeline evaluation on simulated population
│   │   └── GPR25_Run_Star.ipynb                # Production run_star() for prospective predictions
│   ├── Priors/                                 # Prior design and automation notebooks
│   │   ├── priors1_rationalising_priors.ipynb  # LSP peak detection; motivates SM2016 fallback
│   │   ├── priors2_manual.ipynb                # Manual iterative prior extraction on HD81809
│   │   ├── priors3_automate_manual.ipynb       # fit_peaks() function; verified on all benchmark stars
│   │   ├── priors4_clean_automated.ipynb       # Production-ready fit_peaks()
│   │   └── priors5_prior_generating_func.ipynb # get_priors() with SM2016 and type-mean fallbacks
│   └── data-preproc.ipynb                      # Data preprocessing, visualisation, and split setup
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
│   ├── test_df_ops.py
│   ├── test_eval.py
│   ├── test_gpr.py
│   ├── test_mcmc.py
│   └── test_priors.py
│
├── docs/                                       # Sphinx API documentation
│   ├── _build/                             # Generated HTML output (not tracked)
│   ├── analysis.rst                        # Autodoc page for Analysis scripts
│   ├── conf.py                             # Sphinx configuration
│   ├── helpers.rst                         # Autodoc page for helpers package
│   ├── index.rst                           # Documentation landing page
│   └── make.bat                            # Windows build script
│
├── .gitignore
├── Instructions.md
├── LICENSE
├── notes.md
├── readme.md
├── pytest.ini                                 # Initialisation for the test suite
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

### Key result: predict stellar cycle
Our predictions work on S-Index data (refer to the report for details on the S-Index).
With a dataset of S-Index measurements and the Julian date (JD) times at which they were taken, the code can create the predictions.
``` python
from helpers.pipeline import run_star
result = run_star(
    datapath='Data/benchmark/HD81809_Mt_wilson_data.txt',
    star_name='HD81809', star_type='G', add_prefix=False,
    lookahead_years=[1, 2, 3, 5],
    verbose=False, plot=True, # This will plot the predictions
)
```
The result object contains the predictions and best times to observe in the lookahead windows.

### Evaluate pipeline performance
In the report, we evaluated the performance of the predictions on the benchmark stars with 25 splits per dataset.
Due to HPC failure this was not completed on the full mwd dataset. 
A similar analysis can be performed on other data easily using star_window_analysis.py.
``` 
# In command line (from project root)
python Analysis/star_window_analysis.py \
    --data_dir  /path/to/Data/mwd \
    --output_dir /path/to/results \ # Outputs one pkl per star
    [--star_type G] \  # Star type data; here we use G because it is valid with the SM2016 relation
    [--n_windows 5] \
    [--skip_existing]
```
There is also a bash wrapper for HPC use. Edit the username for it to work.

To analyse the results, load the pickles and flatten them with `results_to_df`, then plot:
```python
import pickle, glob
from helpers.pipeline import results_to_df
from helpers.eval import plot_return_errors

results = [pickle.load(open(f, 'rb')) for f in glob.glob('/path/to/results/*.pkl')]
df = results_to_df(results)
plot_return_errors(df, max_lookahead=5)
```

### Cadence analysis
In the report, how the cadence affects the accuracy of the predictions is reported. 
Due to the HPC failure this was only run with 15/100 simulated stars.
To perform the full analysis, use cadence_analysis.py.
```
# All folders, all 100 stars, 5 windows:
python Analysis/cadence_analysis.py --sim_root Data/simulated --out_dir Results/simulated
```
If it is desired to generate more than the current 100 stars,  run all cells in `Notebooks/GPR/GPR24a_Simulated_Data.ipynb`.
To plot the results,
```python
import pandas as pd
from helpers.eval import plot_cadence_analysis

df = pd.read_csv('Results/simulated/all_rates.csv')
plot_cadence_analysis(df, max_lookahead=5, savefig='Report/Figures/CadenceAnalysis.png')
```

## Documentation
---
API documentation is generated from docstrings using Sphinx. To build:
```powershell
cd docs
.\make.bat html
```
Then open `docs/_build/html/index.html` in a browser.

## Test Suite
---
A test suite was written to ensure things were always working during continuous integration.
Some of the tests take a long time to run. As such, they have been divided into fast unit tests and slow tests marked with `@pytest.mark.slow`.

| File | What it tests |
|------|--------------|
| `test_df_ops.py` | `prepare_df`, `split_df`, `clean_df`, `downsample_min_gap` |
| `test_eval.py` | `check_constant`, `best_in_x`, `truth_in_x` |
| `test_gpr.py` | `train_gpr` on the three benchmark stars (slow) |
| `test_mcmc.py` | `lnPost_gp` — bounds, prior penalty, multi-term kernels |
| `test_priors.py` | SM2016 relations, `get_priors` resolver, benchmark star priors (slow) |

From the project root:
```bash
# Fast unit tests only (seconds)
.\VenvName\Scripts\python.exe -m pytest -m "not slow"

# Slow integration tests only (runs full GPR/LSP on benchmark stars, 25 min)
.\VenvName\Scripts\python.exe -m pytest -m slow

# Full suite
.\VenvName\Scripts\python.exe -m pytest
```

## Notes
---
- "RuntimeWarning: invalid value encountered in scalar subtract lnpdiff = f + nlp - state.log_prob[j]" is a common warning but does not affect the performance of the modelling. If it bothers you, you can silence it with
```
import warnings
warnings.filterwarnings('ignore', message='invalid value encountered in scalar subtract')
```

## Autogeneration Tools
---
LLMs, specifically Claude Sonnet 5, were used to support this work. 
It was used to:

    - Help debug code when human attempts failed.
    - Improve low efficiency code: "how could this code be made more efficient?"
    - Help with some of the more esoteric Matplotlib functions: "how do I make this plot look like xyz?"
    - Help rephrase unwieldy sentences: "how can I make this sentence flow better?"
    - Help with generating the repository structure section in the readme.md. 
    - Help wrapping code for use in HPC: "how could I write this for HPC use?"
    - Help identify test cases for continuous integration: "what should I test in xyz function and how could I do that?"
    - Standardising docstring formatting.

I would like to emphasise that the analysis, algorithm design, and thought processes are  *my own*.  Model outputs were always reviewed and checked against literature. No generated response was incorporated into the work without thorough verification.



