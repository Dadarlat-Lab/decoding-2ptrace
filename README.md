# Decoding multi-limb movements from two-photon calcium imaging of neuronal activity using deep learning

Codes for\
Park, S., Lipton, M., & Dadarlat, M. (2024). Decoding multi-limb movements from two-photon calcium imaging of neuronal activity using deep learning. *Journal of Neural Engineering*.

## Pipeline for training
### 1. Prepare data
- Function: prepare_data() in prepare_data.py
- Save data in npy files to use in the pipeline

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
