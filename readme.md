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
        - This still doesn't capture the loss properly.

    - I think at this point the model averaging is a failed exercise. It is not really physical to define the NLPD ratio as the model average.
    - Instead, we can choose the best model from the validation and then do the MCMC within that model. 
    - That intra-model MCMC gives us the uncertainty in the calibration parameters for that model.
    - Then we can retrain the whole thing on the training and validation sets to test on the test set.
    
    - We can upweight the maxima and minima in the validation set.

    - Use GPR13 MCMC as the baseline.

    - GPR16: Try training the model not to predict every datapoint but to focus on the minima and maxima.
        - Perhaps we could take an RBF rolling average of the data; this accounts for the discontinuous measurements. 
        - Compare the minima
        - I am now fitting the maxima and minima to the whole dataset. It does not really matter if we catch all of them. But it is a problem if we catch ones that do not exist.
    
    - Labelling/:
        - I am now thinking it would probably be easier to manually label all the peaks and troughs. There aren't that many datasets and my dear old egg Claude would probably oblige me in writing some tool to help me do it interactively.
        - Most of the star have bad data. Maybe this search could be narrowed to stars we know we want to observe.
        - Now we can fit minima to the stars that are labelled.
        - For these we can report the accuracy in mnima prediction.
        - For the rest we can only report the MSE.

    - GPR17: Reporting results
        - In the final pipeline, we will report best in x = (0.25, 0.5, 1, 2, 3, 4, 5) years unless variation < tol then return that it is constant.
        - Return expected next minimum and uncertainty thereof.

        - Use the function in GPR16
        - In GPR17 though,
            - Bad data:
                - Split by % of data with fewer ie 5 splits.
                    - Report MSE and best in x years. Report on how correct those best in x years were based on a simple interpolation of the validation set.
            
            - Const data marker;
                - Split by % of data with more ie 10 splits.
                    - Test that it reports const ie variability of predictions < tol. Tol is defined by noise.
                    - Report best in x years and the error on those based on interpolation.

            - Else: labelled as osc
                - Split by date:
                    - Just after labelled max/min and bound by start and finish. If there is a thing near too near the finish, disregard the finish and use the max/min.
                    - Split at n_intervals points in between
                - Plot the predicted next maxima/minima with axvlines
                - Plot the labelled next maxima and minima with axvlines
                - Report best in x years and the error on those based on interpolation.

        - Challenge: how to choose only the next maximum/minimum
        
        -  Having now compelted GPR17, the conclusions are that fitting minima and maxima is too fragile.
        - The 1 year suffers from a transition ringing --> start predictions earlier but disregard them
        - We are LEAKING DATA!
            - We ought to perform the model selection on the vaid and do this testing on the test set.
        - Comparing with empirical can just use the time-based lookahead windows.

    - GPR18: Doing the rewrites to solve this.
        - Use lower cadence to prevent ringing. 
        - 2 phases? low cadence for short range forecasts?
        - Use the moving average to calculate the test set ground truths.
        
    - GPR19: Window analysis    
        - Window analysis on one dataset.
        - When verified that this works, check that this fails gracefully for bad datasets.
        - Run this on the full data folder.
        - The splits always have worse data when the GPR predictions start in the middle of a block of measurments. This is because each block is relatively stationary. Makes the model assume the data is stationary. Perhaps weight the datapoints down if there are too many in a short period? 
            - This is posterior anchoring.
            - These datapoints are very redundant.
            - Perhaps can enforce a minimum gap between datapoints

    - GPR20: Min gap window analysis
        - DOWNSAMPLING REALLY IMPROVES THINGS
        - The model is not so great at predicting when the cutoff is at a peak.
        - Testing on a few more, I think the takeaway is that the model only works well for clearly cyclical stars.





    - Write Data rough

    -Write ARIMA rough

    - Examine making q_long less constrained (doesn't have to be cyclical it could be instrument drift).

    - Detection limit analysis on the amplitudes: how loud do the different peaks need to be for our prior detector to find them. 
    - Cadence limit analysis: what is the required cadence to detect different frequencies of signal well based on our analysis.

    - Plot PSD vs LSP

    - Look at using joint datasets.

    - Plot the prediction residuals.

    - Compare the sunspot data with the sind data






----------
- Extensions:
    - Finish off the SM2016 paper to improve amplitude estimates from magnitude ratios.
    - RJMCMC: justify not using it initially because it is hard to definet eh transitions between each. When do we add or remove a k mode. When do we stay at the same k but change the cycles involved.
    