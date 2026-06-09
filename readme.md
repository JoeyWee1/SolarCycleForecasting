- data-preproc is just a preliminary examination of the data
- ARIMA performs an ARIMA analysis of the data
- This ARIMA analysis has not done frightfully well as we do not have very many trcaining cycles in this data.
- Next step is to apply it to more data or with GPR
- GPR was performed in the ./GPR/ folder of ipynb
    - Could not get the Rot + SHO kernel working
    - Next used spectrum kernel in GPR4 spectrum kernel
    - Plotted the spectrum kernel moving window analysis in GPR5 Moving window spectrum
- Tried to automate prior selection using computational tools.
----
- Meeting 2:
    - Perform manual analysis on star data and compare with literature to verify pipeline: ./manual_pipeline/
    - Use the three stars indicated in the meeting.
        - HD201091
        - HD81809
        - HD160346
    - MPFIT them
        - Compute LS periodogram on current residuals (initially raw data).  Fix frequency grid from original data (don't recompute each iteration).
        - Find dominant peak:
            - Compare with SNR and FAP
            - Significant if peak_power > FAP threshold (and/or SNR > 4)
            - Set a maximum period cutoff at 3x the measurement timescale like in the SM2016 paper
            - else stop
        - Add best_freq to accepted frequency list
        - Simultaneous re-fit of ALL accepted frequencies
        - Calculate new residuals
        - Repeat with new residuals.
    - Flag and ignore window functions and aliases.
    - Write updated peak finder function.
    - Finish the prior generating function:
        - Rewrite the amplitudes as functions of the rotation frequency as this is the most commonly seen.
    - Perform GPR on the stars: GPR 6-9
    - Pipeline that takes in signal, finds the priors, fits the GPR, and plots it.
----
- Meeting 3:
    - Perform window analysis on baselines.
    - Double check NaN values are imputed in the df_ops.
    - Perform ARIMA on the baselines
    - Write introduction rough 
    - Test the pipeline on additional stars: GPR11 further baseline manual; GPR12 looks at one example of each baseline combination seen in GPR11
        - GPR12a updated the gpr code to include cleaning the validation set by the same limit.
        - GPR12b has very sparse data leading to very poor predictions. I cannot pick out the actual cycle. How can we identify non-cyclical stars? Include non-cyclical mode in GPR search? Could use a linear or RBF kernel? The kernel that ended up being used should be returned and if the non-cyclical kernel is used it should be flagged so that the prediction that anytime is ok for observations should be returned.
            - also updated the prior bounds: priors defined using mean and SM2016 get bigger leeway
            -Have a problem with total lack of data for HD10072. How can we quantify this?
            - Improved plotting code.
        -GPR12c motivates MCMC. Hard to quantify the quality of the fit.
    - Write MCMC in GPR13:
        - It is too expensive to run MCMC on all so use minimise to compare models first then MCMC to yield uncertainty.
        - Use BMA to not have FPTP model selection.
    - GPR14_BMA:
        - Include model choice uncertainty
        -  Weight BMA by NLPD. Strictly this is a bit odd given that NLPD is on the validation set but this is why we test on the test set. We're strictly meant to test on the marginal likelihood for weights which is hard to calc.
    - GPR14b: Did a window analysis with the scaling.
        - We seem to be allowing a lot of functions where the phase does not line up properly.
        - Need to consider a different loss function. ie CRPS {CITE: CRPS https://www.nature.com/articles/s44387-026-00073-7}
    -GPR15:
        - Try CRPS loss




    - Try MSE because realistically that is the most important.    

    - GPR_Minima:
        -  softmax pseudo-BMA is a computationally tractable approximation that weights models by held-out predictive performance. Acknowledge it is not theoretically equivalent to RJMCMC

    - Write Data rough

    -Write ARIMA rough

    - Examine making q_long less constrained (doesn't have to be cyclical it could be instrument drift).

    - Quantify minima prediction performance

    - Detection limit analysis on the amplitudes: how loud do the different peaks need to be for our prior detector to find them. 
    - Cadence limit analysis: what is the required cadence to detect different frequencies of signal well based on our analysis.

    - Plot PSD vs LSP

    - Look at using joint datasets.

    - Plot the prediction residuals.

    - Implement temperature weighting for the softmax in BMA weight






----------
- Extensions:
    - Finish off the SM2016 paper to improve amplitude estimates from magnitude ratios.
    - RJMCMC: justify not using it initially because it is hard to definet eh transitions between each. When do we add or remove a k mode. When do we stay at the same k but change the cycles involved.
    