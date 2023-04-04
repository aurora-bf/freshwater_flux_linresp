Code and tools used to estimate freshwater fluxes from data using linear response theory. 


The tools folder contains functions needed for applying the method including functions for fitting the Gaussian mixture model, regridding data, and applying linear response theory. The folder model_tests applies the method to data from the CESM large ensemble. Here we focus on historical time period (1975-2019), but the folder also includes tests for future times (2011-2050). The folder app_to_obs applies the method to IAP data (see Cheng et al. 2020). Note some functions will need additional information to run - for example, the ocean only FAFMIP functions are needed to be able to execute this repo. These are publicly available but will need to be placed in a directory and then pointed to in tools/fafmip_gridding_tools.py

The conda environment used when creating this repo is in environment.yml and the environment can be recreated using conda env create -f environment.yml

The figures from our paper are generated in the following.

/app_to_obs/application_to_obs.ipynb: Fig 3c/d, Fig 1a, Fig 10, Fig 11, Fig 12

/app_to_obs/plot_compare_results.ipynb: Fig 13

/model_tests/linear_response_CESM_ensemblemean.ipynb: Fig 3 a/b, Fig 1b, Fig 4, Fig 5, Fig 6

/model_tests/true_freshwater_fluxes.ipynb: Fig 2, Fig 8




Still to account for:
Fig 7,Fig 9

