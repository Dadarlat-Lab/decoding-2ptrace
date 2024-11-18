import os
import torch

import networks
import utils
from trainer import train_one_process


def train_experiment(dir_result_parent, name_result, kwargs_train_process):
	# Create result directory
	dir_result = os.path.join(dir_result_parent, utils.get_name_with_time(name_result))
	if os.path.isdir(dir_result) is False: os.mkdir(dir_result)

	# train
	train_one_process(kwargs_train_process, dir_result=dir_result)

def experiment_multiple():
	'''
	Change dir_result_parent and run_dict
	dir_result_parent: Direcotory to save the outcomes
	run_dict: Cases to run training
	'''
	dir_result_parent = r"E:\tmp\NN"

	run_dict = {
		"Animal1_multilimb": {
		"dir_data": r"E:\tmp\prepare_data\Animal1-G8_53950_1L_Redo\Animal1_likeli01",
		"name_neural": "spks_z_sel",
		"name_coord": "behav_coord_likeli_norm",
		"input_size": 3327,
		"output_idx": [None],
		"dataset_name": ["seq2seq_run_overlapO", "seq2seq_run_overlapX"],
		"model_name": "seq2seq",
		'loss_func': networks.loss_seq2seq_mse(),
		'batch_size': 12, 'lr': 0.0019302726584225075, 'n_unit': 795, 'num_layers': 2
	},
 	}
	#################################################################################################################

	for case_name in run_dict: 
		case = run_dict[case_name]

		kwargs_train_process = {
		# Data
		'dir_data': case['dir_data'],
		'name_neural': case['name_neural'],
		'name_coord': case['name_coord'], 
		'input_size': case['input_size'], #Animal1=3327, Animal4=2978, Animal9=3314

		# Train
		'random_seed': 0, 
		'lr': case['lr'],
		'batch_size': case['batch_size'],
		'num_epoch':30,
		'optimizer_name': "Adam",
		'loss_func': case['loss_func'], #networks.loss_seq2seq_mse(),
		'seq_len': 5,
		'output_idx': case['output_idx'],
		'num_output': 8,
		'dataset_name': case["dataset_name"], 
		'device': torch.device("cuda:0"), 
		'flag_train_val': False,
		'flag_save_model': True,
		'flag_save_output':True,
		'flag_save_ni':False,#True,
		'flag_save_latent':True,
		'flag_print':True,
		'flag_cv':False,
		'early_stop_epoch':10,
		'early_stop_tol':0.01,
		'best_score_name': 'r2',
		'best_score_direction':'max',
		# Network
		'n_layer': case['num_layers'],
		'n_unit': case['n_unit'],
		'model_name':case['model_name']
		}
		train_experiment(dir_result_parent, case_name, kwargs_train_process)



if __name__ == '__main__':
	experiment_multiple()
