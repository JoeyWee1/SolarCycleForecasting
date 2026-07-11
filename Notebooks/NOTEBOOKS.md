# Notebook Reference

Organised by directory, then by development order within each section.

---

## Data Preprocessing

### `data-preproc.ipynb`
Explores the raw S-index time series for the benchmark stars (HD81809, HD160346, HD201091) and the Sun, assessing sampling quality and structure. Loads data, converts Julian Date timestamps, and produces preliminary scatter plots contrasting well-sampled and poorly-sampled datasets. Establishes the 70/15/15 chronological train/validation/test split convention used throughout subsequent analysis.

---

## ARIMA (Baseline)

### `ARIMA/ARIMA1_solar.ipynb`
Investigates whether an ARIMA or Fourier+ARIMA model can forecast the solar S-index (~11-year magnetic activity cycle). Applies log transforms for stationarity, examines ACF/PACF plots, and uses the Hyndman-Khandakar algorithm (pmdarima) for automated order selection; also fits a Fourier series (k=3 harmonics) and applies ARIMA to the residuals. Both approaches perform poorly because fewer than two full cycles are available for training, motivating the switch to GPR.

### `ARIMA/ARIMA2_baseline_stars.ipynb`
Applies the same ARIMA pipeline to the three benchmark stellar activity datasets (HD81809, HD160346, HD201091). BoxCox variance stabilisation is used, with annual/semi-annual/monthly downsampling and automated seasonal ARIMA order selection. ARIMA achieves acceptable MAPEs at annual cadence but diverges at monthly resolution; the overall conclusion is that the method cannot capture the irregularity of stellar magnetic cycles, further motivating GPR.

---

## Priors

### `Priors/priors1_rationalising_priors.ipynb`
Investigates whether a Lomb-Scargle periodogram (LSP) can reliably identify the short (rotation), medium (activity cycle), and long (Gleissberg) period components for use as GPR kernel hyperparameter priors. Develops a synthetic signal generator to stress-test peak detection and implements `identify_peaks` to classify LSP peaks by frequency band. Finds that the mid-cycle period is reliably recovered while short-period detection is limited by cadence, motivating use of the Saar-Mackay 2016 (SM2016) empirical relation to derive rotation period from the detected cycle period.

### `Priors/priors2_manual.ipynb`
Develops an iterative prior extraction procedure for HD81809 using an MPFIT-style approach: applies LSP on training data, identifies the dominant above-FAP peak, checks for aliases, fits a Gaussian to measure SNR, accepts if SNR > 4, subtracts a simultaneous sinusoidal fit, and repeats on residuals. Correctly recovers HD81809's known rotation period (~40 days) and activity cycle (~8.2 years) from the data alone.

### `Priors/priors3_automate_manual.ipynb`
Refactors the manual iterative peak-finding from `priors2` into a reusable `fit_peaks` function and verifies it on all three benchmark stars. Adds an upper period cutoff at 3× the data timespan to suppress spurious long-period detections. All three stars' recovered periods match their literature values.

### `Priors/priors4_clean_automated.ipynb`
Repeats the priors3 verification with a production-ready version of `fit_peaks` (now imported from helpers), extended to include the Sun, and using logarithmic frequency spacing for better resolution across the wide period range. Introduces `find_classify_signals`, which bins detected peaks into short/mid/long categories. All four datasets produce literature-consistent periods.

### `Priors/priors5_prior_generating_func.ipynb`
Completes the prior-generation pipeline with a `get_priors` function that fills missing period components using the SM2016 empirical relation and population-mean fallbacks. Takes the output of `find_classify_signals`, resolves whichever of the short/mid/long periods were not directly detected, and returns a complete set of three priors ready for GPR kernel hyperparameters. Recovered priors match literature values closely across all three benchmark stars.

---

## GPR Development

### `GPR/GPR1.ipynb`
First attempt at GPR on the solar S-index using celerite2's RotationTerm (a sum of damped SHO terms) plus an optional long-term SHOTerm, optimised with L-BFGS-B. Q quality factors collapse to their bounds and the optimiser fails to find a coherent periodic solution, indicating that the model or kernel is misspecified and that physically motivated priors are required.

### `GPR/GPR2.ipynb`
Extends the GPR approach to the full solar dataset and compares three kernel configurations: SHOTerm only, RotationTerm only, and SHO+ROT combined. An LSP is used to identify the dominant ~10.57-year period as an initial prior. Multiple random restarts all converge to the same poor solution, pointing to kernel misspecification rather than optimisation failure, and motivating a spectral mixture (sum of SHOs) approach.

### `GPR/GPR3.ipynb`
Wraps the GPR2 SHO+ROT kernel in a 10-window moving-window evaluation to assess how forecast performance changes with more training data. Produces a multi-panel figure saved as `SolarGPRSHORot.png`. The model roughly tracks the trend but remains overconfident and fails to capture cycle-to-cycle variation.

### `GPR/GPR4_Spectrum_Kernel.ipynb`
Introduces a spectrum kernel — a sum of K underdamped SHOTerms — to capture multi-timescale variability. Parameterises by K triplets of (sigma, rho, Q) with log-spaced initial periods spanning rotation to centennial timescales, optimised via L-BFGS-B. Produces forward predictions on a held-out validation set as a flexible baseline; moving-window evaluation follows in GPR5.

### `GPR/GPR5_Moving_Window_Spectrum.ipynb`
Wraps the spectrum kernel GPR from GPR4 into a moving-window analysis across 10 sliding windows, with K=3 SHO terms. Produces a unified multi-panel figure demonstrating how the model extrapolates across different solar cycle regimes. Concludes that refined priors are the next necessary step.

### `GPR/GPR6_Refined_Priors_GPR.ipynb`
Applies the spectrum kernel GPR to the three benchmark stars using LSP-derived priors to initialise and bound SHO period parameters. Evaluates against a held-out validation set. Achieves modest fit quality but with underestimated uncertainties; identifies outliers in the training data as the likely cause and motivates GPR7.

### `GPR/GPR7_Removing_Outliers.ipynb`
Investigates whether applying a 4×MAD outlier filter to training data improves GPR forecast performance. Applies the filter before fitting the K=3 spectrum GPR with LSP priors on all three benchmark stars. Outlier removal alone is insufficient to resolve poor predictive performance, so model selection across kernel configurations is identified as the next step.

### `GPR/GPR8_Model_Selection.ipynb`
Systematically compares seven prior-combination configurations (k=1,2,3 with different combinations of short/mid/long period terms) on cleaned data for HD201091, selecting the best model via validation-set NLPD. The 3-term model combining all three timescales ("3sml") achieves the best NLPD (−1.1462), confirming that including rotation, activity-cycle, and secular components jointly gives the best-calibrated predictions.

### `GPR/GPR9_Pipeline.ipynb`
Consolidates data loading, outlier cleaning, LSP prior extraction, multi-configuration model selection by NLPD, and result plotting into a single `train_gpr()` function. Tested end-to-end on HD201091. The output is a reusable modular pipeline that reports the best kernel configuration and its NLPD automatically.

### `GPR/GPR10_Moving_Window_Pipeline.ipynb`
Wraps `train_gpr` into a moving-window sweep (training fractions 10%–90%) on HD201091 to assess how the best-fit model and predictions evolve as more historical data is included. Produces a multi-panel stacked forecast figure. Identifies the need for MCMC to quantify kernel-parameter uncertainty.

### `GPR/GPR11_Further_Baseline_Manual.ipynb`
Loops over the full stellar Ca II dataset, classifies each star by which periodic components (short/mid/long) the LSP detects, and saves a taxonomy CSV. Produces a breakdown showing 57 stars distributed across detection categories (3sml, 2sm, 2ml, 1m, 0), confirming that SM2016 fallbacks are needed for the majority of stars.

### `GPR/GPR12a_Examining_Each_Detected_Combo_3sml_HD18256.ipynb`
Tests the full pipeline on HD18256, a representative star with all three cycle types detected (the "best-case" 3sml scenario). Runs the model-selection loop and moving-window sweep as a sanity check. Confirms the pipeline works correctly and motivates adding MCMC for kernel-parameter uncertainty.

### `GPR/GPR12b_Examining_Each_Detected_Combo_2sm_HD10072.ipynb`
Investigates pipeline behaviour on HD10072, which has only short and mid cycles detected (2sm case), requiring a fallback prior for the long-term term. Refactors `train_gpr` to return per-cycle prior bounds with different tolerances depending on detection source (direct, SM2016-derived, or type mean). Overconfident fits on data-sparse stars motivate MCMC.

### `GPR/GPR12c_Examining_Each_Detected_Combo_2ml_HD16673.ipynb`
Applies the updated pipeline to HD16673 (mid and long cycles detected, 2ml case), testing the SM2016 fallback for the missing rotation component. Runs the moving-window analysis with the refactored `train_gpr`. Results are qualitatively similar to the 2sm case; MCMC treatment is again identified as necessary.

### `GPR/GPR12d_Examining_Each_Detected_Combo_1m_HD11131.ipynb`
Examines the pipeline on HD11131 (only mid-term cycle detected, 1m case), so both short and long priors use SM2016 or type-mean fallbacks. Runs the moving-window sweep. Reinforces the case for MCMC in low-data regimes.

### `GPR/GPR12d_Examining_Each_Detected_Como_0_HD10700.ipynb`
Applies the pipeline to HD10700 (tau Ceti), for which the LSP detects no periodic signal at all (class 0). The pipeline produces essentially flat predictions, consistent with the literature result that tau Ceti is chromospherically flat. Validates the pipeline's graceful handling of activity-flat stars.

### `GPR/GPR13_MCMC.ipynb`
Introduces MCMC (emcee affine-invariant sampler, 32 walkers, 10,000 steps) to quantify epistemic uncertainty in the fitted GP kernel parameters for HD201091. Draws 200 posterior samples to generate an ensemble of GP realisations, showing that total uncertainty bands are meaningfully wider than MAP-only estimates. Concludes that Bayesian Model Averaging may be needed to handle model-selection uncertainty.

### `GPR/GPR14_BMA.ipynb`
Addresses model-selection uncertainty by running MCMC independently on each kernel candidate, computing BMA weights via softmax on negative NLPDs, and drawing proportionally weighted posterior samples. Demonstrates BMA on HD201091 but finds that competing models (2sm and 3sml) are often out of phase, producing an excessively broad predictive band.

### `GPR/GPR14b_BMA_window.ipynb`
Extends the BMA pipeline into a 10-window moving-window analysis on HD201091. Wraps the full BMA workflow (MAP + per-model MCMC + weighted sampling) into `train_gpr`. The out-of-phase predictions between dominant models produce unsharp intervals, motivating a switch from NLPD to CRPS as the weighting score.

### `GPR/GPR15_CRPS_window.ipynb`
Re-implements the BMA pipeline with Continuous Ranked Probability Score (CRPS) based softmax weights and runs the moving-window analysis. CRPS-weighted BMA still struggles with out-of-phase models, leading to the conclusion that single-model MCMC is preferable to BMA and that the focus should shift to reliably predicting activity minima via lookahead windows.

### `GPR/GPR16_Minima_MCMC.ipynb`
Returns to single-best-model MCMC with CRPS for model selection, and attempts to extract practical observing recommendations from the posterior ensemble: (1) "best observing time in the next X years" via argmin of each posterior sample within a window, and (2) next activity minimum/maximum via `scipy.signal.find_peaks`. Peak-finding proves unreliable (100% of samples fail), so lookahead windows are adopted as the primary output.

### `GPR/GPR17_Results.ipynb`
Applies the full CRPS model-selection + MCMC + lookahead pipeline to HD201091 with the aim of comparing predictions against ground truth. Implements `check_constant`, `best_in_x`, and `next_maxmin` helpers. Produces a two-panel forecast plot plus numerical lookahead estimates with 68% credible intervals. Confirms that lookahead-window predictions are more robust than explicit minima/maxima fitting.

### `GPR/GPR18_Improved_Results.ipynb`
Addresses three shortcomings identified in GPR17: (1) transition ringing at the train/prediction boundary is suppressed by extending predictions back into the training period; (2) data leakage is eliminated by selecting the model on the validation set then retraining on combined train+validation data before predicting on the test set; (3) only lookahead window outputs are reported. Produces cleaner forecast plots with properly separated training, retraining, and test phases.

### `GPR/GPR19_Results_Window.ipynb`
Applies the improved GPR18 pipeline repeatedly across multiple train/validation/test splits of HD201091 to evaluate whether predictions are consistent across windows. Runs model selection, retraining, MCMC, and forecasting for each split, producing per-window forecast plots with lookahead panels comparing predicted minima to ground-truth estimates. Saves per-star results to `split_results.pkl`.

### `GPR/GPR20_Min_Gap_Window.ipynb`
Addresses a bias observed in GPR19 where densely clustered observations near a split boundary inflate the GP's confidence that the signal is constant. Introduces minimum-gap downsampling (enforcing at least 2.5–3 days between retained observations) before the windowed pipeline, tested on hd219834A. Downsampling regularises cadence and prevents overfitting to locally dense data near the prediction start.

### `GPR/GPR21_New_Opt.ipynb`
Explores replacing single-point maximum-likelihood fitting with a forward-NLPD strategy: the GP is trained on successive inner fractions (60%–90%) of the training set and scored on the held-out tail, with the best-performing fraction's parameters retained. Also expands the kernel search space to include Matérn-3/2 terms alongside SHOs. Aims to reduce kernel hyperparameter overfitting to the full training set.

### `GPR/GPR22_Init_Cond_Var.ipynb`
Addresses local-minima sensitivity in L-BFGS-B by running 25 random perturbations (±10% Gaussian noise) of the initial parameter guess for each kernel combination, keeping the result with the lowest NLL. Applies this multi-start strategy within the full windowed pipeline on HD81809.

### `GPR/GPR23a_Revised_Ground_Truth.ipynb`
Rethinks ground-truth definition for prediction assessment: instead of a Gaussian smooth of the raw data, fits a 3-harmonic Fourier model (with MCMC uncertainty) to the full stellar activity time series using the cycle period prior as the fundamental period. Produces a probabilistic ground-truth estimate of activity minima with proper uncertainty intervals, demonstrated on all three benchmark stars.

### `GPR/GPR23b_Revised_Ground_Truth_Pipeline.ipynb`
Integrates the Fourier MCMC ground truth from GPR23a into the full windowed prediction pipeline. For each training window, runs GPR model selection with multi-start optimisation, retrains with MCMC, forecasts forward, and compares predicted minimum times against the Fourier-derived ground truth. Outputs a compact per-star figure with lookahead panels and a table of predicted vs. true minimum years with 68% credible intervals.

### `GPR/GPR23c_Functionalised.ipynb`
Refactors the entire GPR23b pipeline into clean reusable functions: `star_window_analysis()` runs the full GPR + Fourier ground-truth pipeline on one star and returns a structured results dict; `results_to_df()` flattens results into a tidy per-(star, split, window) DataFrame. Runs on all four benchmark datasets (HD81809, HD160346, HD201091, Sun) and produces the error KDE figure showing prediction error distribution as a function of lookahead window length.

### `GPR/GPR24a_Simulated_Data.ipynb`
Generates a synthetic stellar population to supplement the limited real benchmark data. Draws 100 stars from realistic FGKM-type distributions, samples rotation and cycle periods from SM2016 statistics, and constructs time-series signals as a multiplicative hierarchy of rotation, activity-cycle, and long-period modulation with log-space KDE-sampled amplitudes. Saves simulated light curves across 10 observation cadences (1–90 days) for systematic cadence-effect tests.

### `GPR/GPR24b_Testing_Simulated_Data.ipynb`
Evaluates the GPR pipeline on the simulated stellar population from GPR24a, running `run_star()` on a subset of G-type stars across all 10 cadence folders. Compares predicted activity minimum times against ground-truth minima as a function of lookahead and cadence. Results (saved to `Results/simulated/all_rates.csv`) show that prediction accuracy degrades with sparser cadence and longer lookahead.

### `GPR/GPR25_Run_Star.ipynb`
Implements the production-ready `run_star()` function for genuine prospective predictions on a single star with no test set. Takes a data file, star name, spectral type, and a list of lookahead windows; returns posterior sample arrays and predicted best-in-x statistics with 68% intervals, with an optional diagnostic plot. Demonstrated on HD201091 with lookaheads of 1, 2, 3, and 5 years.
