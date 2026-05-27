- data-preproc is just a preliminary examination of the data
- ARIMA performs an ARIMA analysis of the data
- This ARIMA analysis has not done frightfully well as we do not have very many trcaining cycles in this data.
- Next step is to apply it to more data or with GPR
- GPR was performed in the ./GPR/ folder of ipynb
    - Could not get the Rot + SHO kernel working
    - Next used spectrum kernel in GPR4 spectrum kernel
    - Plotted the spectrum kernel moving window analysis in GPR5 Moving window spectrum
- Tried to automate prior selection using computational tools.
- Meeting 2:
    - Perform manual analysis on star data and compare with literature to verify pipeline: ./manual_pipeline/
    - Use the three stars indicated in the meeting.
        - HD201091
        - HD81809
        - HD160346
    - Perform LSP on them.
    - Remove the first significant peak.
    - Look at the remaining peaks.
    - Flag and ignore window functions and aliases.
    - Write updated peak finder function.
    - Perform ARIMA analysis on the same stars to verify their failure.
    - Finish the prior generating function:
        - Rewrite the amplitudes as functions of the rotation frequency as this is the most commonly seen.
    - Detection limit analysis on the amplitudes: how loud do the different peaks need to be for our prior detector to find them. 
    - Cadence limit analysis: what is the required cadence to detect different frequencies of signal well based on our analysis.
    - Write pipeline that takes in signal, finds the priors, fits the GPR, and plots it.




----------
- Extensions:
    - MPFIT to improve on prior selection performance.
    - Finish off the SM2016 paper to improve amplitude estimates from magnitude ratios.
    