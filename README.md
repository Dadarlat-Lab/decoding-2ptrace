# Decoding multi-limb movements from two-photon calcium imaging of neuronal activity using deep learning

Codes for\
Park, S., Lipton, M., & Dadarlat, M. (2024). Decoding multi-limb movements from two-photon calcium imaging of neuronal activity using deep learning. *Journal of Neural Engineering*.

## Pipeline for training
### 1. Prepare data
- Function: prepare_data() in prepare_data.py
- Save data in npy files to use in the pipeline
- Saved files
    - ops.npy: 'Lx' and 'Ly' (size of X/Y dimension of tiffs/h5) from the ops file from suite2p (See here for ops file information: https://suite2p.readthedocs.io/en/latest/outputs.html#ops-npy-fields).
    - stat.npy: stat from suite2p, but only for cells (See here for stat file information: https://suite2p.readthedocs.io/en/latest/outputs.html#stat-npy-fields).
    - stat_noncell.npy: stat from suite2p, but only for non-cells.
    - idx_coord_vid.npy: indices of video frames that match to limb coordinate frames.
    - behav_coord_ori.npy: limb coordinates, original.
    - coord_likelihood: likelihood of coordinate frames from Deeplabcut.
    - behav_coord_likeli_ori.npy: limb coordinates, original, likelihood processed.
    - ffneu_ori_sel.npy: fluorescence(f)-0.7*neuropil(fneu) (This is to remove neuropil signal from raw fluorescence), original, selected (Select neural_data sequence period considering idx_coord_neural).
    - ffneu_z_sel.npy: f-0.7*fneu, z-scored, selected.
    - fneuf_ori_sel.npy: fneu-0.7*f (This is to remove soma signal from neuropil signal. "Neuropil, corrected" in Figure 8), original, selected.
    - fneuf_z_sel.npy: fneu-0.7*f, z-scored, selected.
    - spks_ori_sel.npy: spks (Deconvolved fluorescence from Suite2p. See here for spks.npy: https://suite2p.readthedocs.io/en/latest/outputs.html. See here for what deconvlution is: https://suite2p.readthedocs.io/en/latest/FAQ.html#deconvolution-means-what), original, selected.
    - 

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
