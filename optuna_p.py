import sys
import optuna
import torch
import random
import os
import logging
import numpy as np

from trainer import train_one_process
from LoggingPrinter import LoggingPrinter
from utils import get_name_with_time
import networks

'''
References
https://broutonlab.com/blog/efficient-hyperparameter-optimization-with-optuna-framework
https://pub.towardsai.net/tuning-pytorch-hyperparameters-with-optuna-470edcfd4dc
https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html#sphx-glr-tutorial-10-key-features-005-visualization-py
'''

class Objective:
	def __init__(self, args):
		self.args = args
	def __call__(self, trial):
		kwargs_one_process = {}
		
		kwargs_one_process['lr'] = trial.suggest_float('lr', 1e-5, 1e-1)
		kwargs_one_process['batch_size'] = trial.suggest_int("batch_size", 4, 256, step=4)
		kwargs_one_process['n_unit'] = trial.suggest_int("n_unit", 5, 1024)
		kwargs_one_process['n_layer'] = trial.suggest_int("num_layers", 1, 3, step=1)

		# Data
		kwargs_one_process['dir_data'] = self.args['dir_data']
		kwargs_one_process['name_neural'] = self.args['name_neural']
		kwargs_one_process['name_coord'] = self.args['name_coord']
		kwargs_one_process['input_size'] = self.args['input_size'] #3327 #Animal1=3327, Animal4=2978, Animal5=3909, Animal9=3314

		# Train
		kwargs_one_process['random_seed'] = 0
		kwargs_one_process['seq_len'] = 5
		kwargs_one_process['output_idx'] = self.args['output_idx'] #[None], [0,1], ...
		if kwargs_one_process['output_idx'] == [None]:
			kwargs_one_process['num_output'] = 8
		else:
			kwargs_one_process['num_output'] = len(kwargs_one_process['output_idx'])
		kwargs_one_process['dataset_name'] = self.args['dataset_name']

		kwargs_one_process['device'] = torch.device("cuda:0")
		kwargs_one_process['optimizer_name'] = "Adam"
		kwargs_one_process['num_epoch'] = 30
		kwargs_one_process['loss_func'] = self.args['loss_func'] #torch.nn.MSELoss() #networks.loss_seq2seq_mse()
		kwargs_one_process['flag_train_val'] = False
		kwargs_one_process['flag_save_model'] = False
		kwargs_one_process['flag_save_ni'] = False
		kwargs_one_process['flag_save_output'] = False
		kwargs_one_process['flag_print'] = False
		kwargs_one_process['flag_cv'] = False
		kwargs_one_process['early_stop_epoch'] = 10
		kwargs_one_process['early_stop_tol'] = 0.01
		kwargs_one_process['best_score_name'] = "r2" #'mse','r2'
		kwargs_one_process['best_score_direction'] = "max" #'min','max'

		# Network
		kwargs_one_process['model_name'] = self.args['model_name']   #simpleLSTM_many2many, lstm_encdec

		# Save logs
		if trial.number == 0:
			logger = LoggingPrinter(self.args['path_log'])
			logger.file_only("##### kwargs_one_process to train_one_process()\n")
			for k, v in kwargs_one_process.items():
				logger.file_only('   '+str(k) + ': '+ str(v)+"\n")
			logger.close()

		######################################################################################################
		r2_best = train_one_process(kwargs_one_process, optuna_trial=trial)
		return r2_best
		

def run_optuna_multiple():
	'''
	- I wrote codes to run multiple cases, but there is a bug. It seems something is run for multiple times in the second case.
	Therefore, only run one case.
	- Random seed for optuna has been set to be random, so optuna running cannot be reproducible. 
	But, a case should be able to be reproducible using the same hyperparameters and the random seed of 0.
	'''
	dir_result_optuna = r'E:\tmp\Optuna' # Directory to save the outputs. Change it.
	cases_load = { # Run only one case!!!
	
	### Multi-limb decoding, "Original, LSTM-encdec" (Figure 4)
	"Animal1_multilimb": {
		"dir_data": r"E:\tmp\prepare_data\Animal1-G8_53950_1L_Redo\Animal1_likeli01",
		"name_neural": "spks_z_sel",
		"name_coord": "behav_coord_likeli_norm" ,
		"input_size": 3327,
		"output_idx": [None],
		"dataset_name": ["seq2seq_run_overlapO", "seq2seq_run_overlapX"],
		"model_name": "seq2seq",
		'loss_func': networks.loss_seq2seq_mse()
	},

	# "Animal4_multilimb": {
	# 	"dir_data": r"E:\tmp\prepare_data\Animal4-G10_55902_1R\Animal4_likeli01",
	# 	"name_neural": "spks_z_sel",
	# 	"name_coord": "behav_coord_likeli_norm" ,
	# 	"input_size": 2978,
	# 	"output_idx": [None],
	# 	"dataset_name": ["seq2seq_run_overlapO", "seq2seq_run_overlapX"],
	# 	"model_name": "seq2seq",
	# 	'loss_func': networks.loss_seq2seq_mse()
	# },

	# "Animal9_multilimb": {
	# 	"dir_data": r"E:\tmp\prepare_data\Animal9-G12_55954_1L\Animal9_likeli095",
	# 	"name_neural": "spks_z_sel",
	# 	"name_coord": "behav_coord_likeli_norm" ,
	# 	"input_size": 3314,
	# 	"output_idx": [None],
	# 	"dataset_name": ["seq2seq_run_overlapO", "seq2seq_run_overlapX"],
	# 	"model_name": "seq2seq",
	# 	'loss_func': networks.loss_seq2seq_mse()
	# },

	### Single-limb decoding, "Original, LSTM-encdec", Animal1 (Figure 5)
	# "Animal1_singlelimb_01": {
	# 	"dir_data": r"E:\tmp\prepare_data\Animal1-G8_53950_1L_Redo\Animal1_likeli01",
	# 	"name_neural": "spks_z_sel",
	# 	"name_coord": "behav_coord_likeli_norm" ,
	# 	"input_size": 3327,
	# 	"output_idx": [0,1],
	# 	"dataset_name": ["seq2seq_run_overlapO", "seq2seq_run_overlapX"],
	# 	"model_name": "seq2seq",
	# 	'loss_func': networks.loss_seq2seq_mse()
	# },

	# "Animal1_singlelimb_23": {
	# 	"dir_data": r"E:\tmp\prepare_data\Animal1-G8_53950_1L_Redo\Animal1_likeli01",
	# 	"name_neural": "spks_z_sel",
	# 	"name_coord": "behav_coord_likeli_norm" ,
	# 	"input_size": 3327,
	# 	"output_idx": [2,3],
	# 	"dataset_name": ["seq2seq_run_overlapO", "seq2seq_run_overlapX"],
	# 	"model_name": "seq2seq",
	# 	'loss_func': networks.loss_seq2seq_mse()
	# },

	# "Animal1_singlelimb_45": {
	# 	"dir_data": r"E:\tmp\prepare_data\Animal1-G8_53950_1L_Redo\Animal1_likeli01",
	# 	"name_neural": "spks_z_sel",
	# 	"name_coord": "behav_coord_likeli_norm" ,
	# 	"input_size": 3327,
	# 	"output_idx": [4,5],
	# 	"dataset_name": ["seq2seq_run_overlapO", "seq2seq_run_overlapX"],
	# 	"model_name": "seq2seq",
	# 	'loss_func': networks.loss_seq2seq_mse()
	# },

	# "Animal1_singlelimb_67": {
	# 	"dir_data": r"E:\tmp\prepare_data\Animal1-G8_53950_1L_Redo\Animal1_likeli01",
	# 	"name_neural": "spks_z_sel",
	# 	"name_coord": "behav_coord_likeli_norm" ,
	# 	"input_size": 3327,
	# 	"output_idx": [6,7],
	# 	"dataset_name": ["seq2seq_run_overlapO", "seq2seq_run_overlapX"],
	# 	"model_name": "seq2seq",
	# 	'loss_func': networks.loss_seq2seq_mse()
	# },

	### Duplicated, Animal1 (Figure 4)
	# "Animal1_dup_lstm": {
	# 	"dir_data": r"E:\tmp\prepare_data\Animal1-G8_53950_1L_Redo\Animal1_likeli01",
	# 	"name_neural": "spkszsel_dup",
	# 	"name_coord": "behav_coord_likeli_norm",
	# 	"input_size": 3327,
	# 	"output_idx": [None],
	# 	"dataset_name": ['dupintp_run_overlapO', 'dupintp_run_overlapX'],
	# 	"model_name": "simpleLSTM_many2many",
	# 	'loss_func': torch.nn.MSELoss() #networks.loss_seq2seq_mse(), torch.nn.MSELoss()
	# },

	# "Animal1_dup_lstmencdec": {
	# 	"dir_data": r"E:\tmp\prepare_data\Animal1-G8_53950_1L_Redo\Animal1_likeli01",
	# 	"name_neural": "spkszsel_dup",
	# 	"name_coord": "behav_coord_likeli_norm",
	# 	"input_size": 3327,
	# 	"output_idx": [None],
	# 	"dataset_name": ['dupintp_run_overlapO', 'dupintp_run_overlapX'],
	# 	"model_name": "lstm_encdec",
	# 	'loss_func': torch.nn.MSELoss()
	# },

	### Interpolated, Animal1 (Figure 4)
	# "Animal1_intp_lstm": {
	# 	"dir_data": r"E:\tmp\prepare_data\Animal1-G8_53950_1L_Redo\Animal1_likeli01",
	# 	"name_neural": "spkszsel_intp",
	# 	"name_coord": "behav_coord_likeli_norm",
	# 	"input_size": 3327,
	# 	"output_idx": [None],
	# 	"dataset_name": ['dupintp_run_overlapO', 'dupintp_run_overlapX'],
	# 	"model_name": "simpleLSTM_many2many",
	# 	'loss_func': torch.nn.MSELoss() #networks.loss_seq2seq_mse(), torch.nn.MSELoss()
	# },

	# "Animal1_intp_lstmencdec": {
	# 	"dir_data": r"E:\tmp\prepare_data\Animal1-G8_53950_1L_Redo\Animal1_likeli01",
	# 	"name_neural": "spkszsel_intp",
	# 	"name_coord": "behav_coord_likeli_norm",
	# 	"input_size": 3327,
	# 	"output_idx": [None],
	# 	"dataset_name": ['dupintp_run_overlapO', 'dupintp_run_overlapX'],
	# 	"model_name": "lstm_encdec",
	# 	'loss_func': torch.nn.MSELoss()
	# },

	# ### Matching, Animal1 (Figure 4)
	# "Animal1_match_lstm": {
	# 	"dir_data": r"E:\tmp\prepare_data\Animal1-G8_53950_1L_Redo\Animal1_likeli01",
	# 	"name_neural": "spkszsel_match",
	# 	"name_coord": "behav_coord_likeli_match_norm",
	# 	"input_size": 3327,
	# 	"output_idx": [None],
	# 	"dataset_name": ['match_overlapO_run', 'match_overlapX_run'],
	# 	"model_name": "simpleLSTM_many2many",
	# 	'loss_func': torch.nn.MSELoss() #networks.loss_seq2seq_mse(), torch.nn.MSELoss()
	# },

	# "Animal1_match_lstmencdec": {
	# 	"dir_data": r"E:\tmp\prepare_data\Animal1-G8_53950_1L_Redo\Animal1_likeli01",
	# 	"name_neural": "spkszsel_match",
	# 	"name_coord": "behav_coord_likeli_match_norm",
	# 	"input_size": 3327,
	# 	"output_idx": [None],
	# 	"dataset_name": ['match_overlapO_run', 'match_overlapX_run'],
	# 	"model_name": "lstm_encdec",
	# 	'loss_func': torch.nn.MSELoss()
	# },

	}

	######################################################
	key = 'new' # new, load, best

	for study_name in list(cases_load.keys()):
		print("#########################################", study_name)

		seed = random.randint(0,5000)
		direction = "maximize" #minimize, maximize

		dir_result_study = os.path.join(dir_result_optuna, get_name_with_time(study_name))
		if os.path.isdir(dir_result_study) is False: os.mkdir(dir_result_study)

		args = cases_load[study_name]
		
		# log
		path_log = os.path.join(dir_result_study, 'log.txt')
		with LoggingPrinter(path_log):
			print('dir_result_optuna: ', dir_result_optuna)
			print('study_name: ', study_name)
			print('key: ', key)
			print('seed: ', seed)
			print('args: ')
			for k, v in args.items():
				print('   '+str(k) + ': '+ str(v))
		args['path_log'] = path_log
		
		# random seed
		random.seed(seed)
		os.environ['PYTHONHASHSEED'] = str(seed)
		np.random.seed(seed)

		storage_name = "sqlite:///{}.db".format(os.path.join(dir_result_study, study_name))

		# # # Deleted existing study
		# optuna.delete_study(study_name=study_name, storage=storage_name) 
		# exit()

		# Add stream handler of stdout to show the messages
		optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

		# Make the study
		if key == 'new': 
			study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed), study_name=study_name, storage=storage_name)
			study.optimize(Objective(args), n_trials=200) #, gc_after_trial=True)
		# elif key == 'load':
		#     study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed), study_name=study_name, storage=storage_name, load_if_exists=True)
		#     #study.optimize(objective, n_trials=5000)
		# elif key == 'best':
		#     study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed), study_name=study_name, storage=storage_name, load_if_exists=True)
		#     print('values: ', study.best_trial.values)
		#     print('params: ', study.best_trial.params)

		# Save into excel
		path_excel = os.path.join(dir_result_study, study_name+'.xlsx')
		df = study.trials_dataframe()
		df.to_excel(path_excel)

		#optuna.delete_study(study_name=study_name, storage=storage_name)


def save_result_excel():
	'''
	Save optuna result into an excel file
	'''
	dir_result = r"E:\tmp\Optuna\20241104_154547_Animal4_multilimb"
	study_name = "Animal4_multilimb"

	#######################################################################################################
	storage_name = "sqlite:///{}.db".format(os.path.join(dir_result, study_name))
	study = optuna.create_study(load_if_exists=True, direction="maximize", sampler=optuna.samplers.TPESampler(), study_name=study_name, storage=storage_name)

	# Save into excel
	path_excel = os.path.join(dir_result, study_name+'.xlsx')
	df = study.trials_dataframe()
	df.to_excel(path_excel)

	# best result
	best_trial = study.best_trial
	print("Best result")
	print('trial_id:', best_trial._trial_id)
	print('values: ', best_trial.values)
	print('params: ', best_trial.params)

	# best result, <200
	df_200 = df[df["number"]<200]
	df_200 = df_200.sort_values(by=["value"], ascending=False)
	best_result_200 = df_200.iloc[0]
	print("Best result, <200")
	print('number: ', best_result_200['number'])
	print('values: ', best_result_200['value'])
	print('params: ', study.trials[best_result_200['number']].params)


def get_best_result():
	'''
	Print out the best case
	'''
	dir_result = r"E:\tmp\Optuna\20241104_154547_Animal4_multilimb"
	study_name = "Animal4_multilimb"

	#######################################################################################################
	storage_name = "sqlite:///{}.db".format(os.path.join(dir_result, study_name))
	study = optuna.create_study(load_if_exists=True, direction="maximize", sampler=optuna.samplers.TPESampler(), study_name=study_name, storage=storage_name)

	best_trial = study.best_trial
	print('trial_id:', best_trial._trial_id)
	print('values: ', best_trial.values)
	print('params: ', best_trial.params)


if __name__ == '__main__':
	### Run optuna
	run_optuna_multiple()

	### Save optuna result into an excel file
	#save_result_excel()

	### Print out the best case
	#get_best_result()


	
	
	

	



