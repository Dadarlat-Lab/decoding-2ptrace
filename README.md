# Decoding multi-limb movements from two-photon calcium imaging of neuronal activity using deep learning

Codes for\
Park, S., Lipton, M., & Dadarlat, M. (2024). Decoding multi-limb movements from two-photon calcium imaging of neuronal activity using deep learning. *Journal of Neural Engineering*.

## Environment
- Python: 3.10.4
- PyTorch: 1.12.1
- scikit-learn: 1.1.2
- Optuna: 3.0.3

## Pipeline for training 
### 1. Prepare data
- Function: prepare_data() in prepare_data.py
- Save data in npy files to use in the pipeline.
- Saved files: The descriptions are in order of saving - see a code for a reference. A lot of extra files are being saved because they were necessary for checking or research. (*) is added for important files.
    - ops.npy: 'Lx' and 'Ly' (size of X/Y dimension of tiffs/h5) from the ops file from suite2p (See here for ops file information: https://suite2p.readthedocs.io/en/latest/outputs.html#ops-npy-fields).
    - stat.npy: stat from suite2p, but only for cells (See here for stat file information: https://suite2p.readthedocs.io/en/latest/outputs.html#stat-npy-fields).
    - stat_noncell.npy: stat from suite2p, but only for non-cells.
    - idx_coord_vid.npy: indices of video frames that match to limb coordinate frames.
    - behav_coord_ori.npy: limb coordinates, original.
    - coord_likelihood: likelihood of coordinate frames from Deeplabcut.
    - (*) behav_coord_likeli_ori.npy: limb coordinates, original, likelihood processed.
    - ffneu_ori_sel.npy: fluorescence(f)-0.7*neuropil(fneu) (**This is to remove neuropil signal from raw fluorescence**), original, selected (Select neural_data sequence period considering idx_coord_neural).
    - ffneu_z_sel.npy: f-0.7*fneu, z-scored, selected.
    - fneuf_ori_sel.npy: fneu-0.7*f (This is to remove soma signal from neuropil signal. "Neuropil, corrected" in Figure 8), original, selected.
    - fneuf_z_sel.npy: fneu-0.7*f, z-scored, selected.
    - spks_ori_sel.npy: spks (**Deconvolved fluorescence from Suite2p.** See here for spks.npy: https://suite2p.readthedocs.io/en/latest/outputs.html. See here for what deconvlution is: https://suite2p.readthedocs.io/en/latest/FAQ.html#deconvolution-means-what), original, selected.
    - (*) spks_z_sel.npy: spks, z-scored, selected.
    - neu_ori_sel.npy: Raw neuropil, original, selected.
    - neu_z_sel.npy: fneu, z-scored, selected.
    - fori_sel.npy: f, original, selected.
    - fz_sel.npy: f, z-scored, selected.
    - ffneu_noncell_ori_sel.npy: f-0.7*fneu of non-cells, original, selected.
    - ffneu_noncell_z_sel.npy: f-0.7*fneu of non-cells, z-scored, selected.
    - neu_noncell_ori_sel.npy: fneu of non-cells, original, selected.
    - neu_noncell_z_sel.npy: fneu of non-cells, z-scored, selected.
    - spks_noncell_ori_sel.npy: spks of non-cells, original, selected.
    - spks_noncell_z_sel.npy: spks of non-cells, z-scored, selected.
    - neudeconv_ori_sel.npy: deconvolved fneu, original, selected.
    - neudeconv_z_sel.npy: deconvolved fneu, z-scored, selected.
    - fneufdeconv_ori_sel.npy: deconvolved fneu-0.7*f, original, selected.
    - fneufdeconv_z_sel.npy: deconvolved fneu-0.7*f, z-scored, selected.
    - xoff_sel.npy: ops['xoff'], selected. 'xoff': x-shifts of recording at each timepoint (https://suite2p.readthedocs.io/en/latest/outputs.html#ops-npy-fields).
    - yoff_sel.npy: ops['yoff'], selected. 'yoff': y-shifts of recording at each timepoint (https://suite2p.readthedocs.io/en/latest/outputs.html#ops-npy-fields).
    - corrxy_sel.npy: ops['corrXY'], selected. 'corrXY': peak of phase correlation between frame and reference image at each timepoint (https://suite2p.readthedocs.io/en/latest/outputs.html#ops-npy-fields).
    - idx_coord_neural.npy: indices of neural frames that match limb coordinate frames.
    - (*) run_windows_likeli.npy: a list of [start, end(include)] coordinate frames of running periods, likelihood considered. 
    - stand_windows_likeli.npy: a list of [start, end(include)] coordinate frames of non-running periods, likelihood considered. 
    - ffneuzsel_match.npy: f-0.7*fneu, z-scored, selected, matched.
    - (*) spkszsel_match.npy: spks, z-scored, selected, matched.
    - idx_neural_match.npy: indices of neural frames for matched data.
    - idx_coord_match.npy: indices of coordinate frames for matched data.
    - (*) behavior_coord_likeli_match_ori.npy: limb coordinates, likelihood processed, matched, original.
    - (*) run_windows_likeli_match.npy: a list of [start, end(include)] coordinate frames of running periods, likelihood considered, matched.
    - stand_windows_likeli_match.npy: a list of [start, end(include)] coordinate frames of non-running periods, likelihood considered, matched.
    - behav_coord_likeli_demean.npy: limb coordinates, likelihood processed, demeaned.
    - behav_coord_likeli_demean_onlyrun.npy: limb coordinates, likelihood processed, demeaned, only running periods.
    - behav_coord_likeli_demean_match.npy: limb coordinates, likelihood processed, demeaned, matched.
    - behav_coord_likeli_demean_match_onlyrun.npy: limb coordinates, likelihood processed, demeaned, matched, only running periods.
    - ffneuzsel_dup.npy: f-0.7*fneu, z-scored, selected, duplicated.
    - (*) spkszsel_dup.npy: spks, z-scored, selected, duplicated.
    - ffneuzsel_intp.npy: f-0.7*fneu, z-scored, selected, interpolated.
    - (*) spkszsel_intp.npy: spks, z-scored, selected, interpolated.
    - (*) behav_coord_likeli_norm.npy: limb coordinates, likelihood processed, normalized.
    - (*) behav_coord_likeli_match_norm.npy: limb coordinates, likelihood processed, matched, normalized.
    - (*) test_idx_cv.npy: test index of neural frames for cross-validation.
    - (*) test_idx_cv_coord.npy: test index of coord frames for cross-validation.

### 2. Get optimal hyperparameters using Optuna
-  Functions:
    - run_optuna_multiple() in optuna_p.py: Run hyperparameter tuning.
    - save_result_excel() in optuna_p.py: Save the optuna result into an excel file.
    - get_best_result() in optuna_p.py: Print out the best case.
- Get optimal hyperparameters of multi-limb decoding with original data + LSTM-encdec (Figure 4), single-limb decoding with original data + LSTM-encdec (Figure 5), duplicated data + lstm / LSTM-encdec (Figure 4), interpolated data + lstm / LSTM-encdec (Figure 4), matching data + lstm / LSTM-encdec (Figure 4). See the case dictionary (commented) in the code to see what parameters to use.

### 3. Run training with the optimal parameters obtained from step 2
- Function: experiment_multiple() in train_experiment.py
- Input parameters (reference the code) to run_dict in experiment_multiple().

### 4. Convert normalized output
- Function: convert_norm_coord_dir() in analysis.py
- The saved outputs (in output/val; epoch_1_gt.npy, epoch_1_pred.npy, ...) are normalized. So converting normalization back using the original coordinates is needed.
- The final outputs you have to use for further analysis are 'gt_norm_converted_epoch{best_epoch_number}.npy' and 'pred_norm_converted_epoch{best_epoch_number}.npy'.

### Linear regression
- File: linear_regression.py
- Algorithms: 'linearlass', 'kalman', kalmanlass', 'kalmanlassl_lag_1'

## For further analysis besides basic training
### Neural importance
#### 1. Save neural importance
- Function: save_neuron_importance() in analysis.py
- Save neural importance.
#### 2. Save important neurons as data
- Function: save_important_neurons() in prepare_data.py
- Save important neurons as a dataset that can be used for training.
- Saved in '{dir_data}/save_neuron_importance'.
- 'impt-{10/25/50/75/90}.npy': the top 10%/25%/50%/75%/90% important neurons.
- 'impt-{10/25/50/75/90}_least.npy': the least 10%/25%/50%/75%/90% important neurons.

### Save random neurons
- function: save_random_neurons() in prepare_data.py

### Save top and low skew neurons
- function: save_skew_neurons() in prepare_data.py
- Save only neurons with 10% top and low skewness into a dataset for training.


