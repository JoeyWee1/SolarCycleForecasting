Solar and Stellar Activity Cycle Forecasting
=============================================

*Joey Wee — MPhil Data Intensive Science, University of Cambridge*

Exoplanets are detected via radial velocity or transit methods, which rely on
spectroscopic or photometric measurements. Non-exoplanetary noise introduced by
stellar activity can significantly affect exoplanet detectability, and the
high-precision instruments used for these measurements are heavily oversubscribed.
Because noise magnitude is proportional to stellar activity, predicting periods of
lower activity — using data from more modest instruments — allows observation
schedules to be optimised so that high-precision instruments are used at stellar
activity minima, maximising detectability.

This project produces a scalable GPR pipeline for those forecasts.

**Quick start:**

.. code-block:: python

   from helpers.pipeline import run_star
   result = run_star(
       datapath='Data/benchmark/HD81809_Mt_wilson_data.txt',
       star_name='HD81809', star_type='G',
       lookahead_years=[1, 2, 3, 5],
       verbose=False, plot=True,
   )

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   helpers
   analysis
