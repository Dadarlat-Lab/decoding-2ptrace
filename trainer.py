import numpy as np
import os
import time
import sys
import openpyxl
from copy import deepcopy

from torch.utils.data import DataLoader
import torch
import optuna

import networks
from scores import score_r2, score_rmse, score_mse
from dataset import NNDataset, collate_fn_custom
from saver import ValueSaver
from Logger import Logger

def get_model(kwargs):
	model_name = kwargs['model_name']
	num_layers = kwargs['n_layer']
	n_unit = kwargs['n_unit']
	input_size = kwargs['input_size']
	num_output = kwargs['num_output']
	device = kwargs['device']

	if model_name == 'seq2seq':
		model = networks.seq2seq(input_size, n_unit, num_layers, num_output, device)
	elif model_name == 'simpleLSTM_many2many':
		model = networks.simpleLSTM_many2many(input_size, n_unit, num_layers, num_output)
	elif model_name == 'lstm_encdec':
		model = networks.rnn_encdec(input_size, n_unit, num_layers, num_output, device, layer_type='lstm')

	model.to(device)
	# print('MODEL DEVICE, in function: ', next(model.parameters()).device)

	return model

def train_one_epoch(random_seed, model, model_name, dataloader_train, device, optimizer, loss_func, value_saver=None, epoch_cur=None, idx_step=None, flag_print=True, flag_optuna=False):
	if flag_print: 
		print("##### Train start, epoch=" + str(epoch_cur) + " #####")
	
	# Set random seed
	networks.set_random_seed(random_seed)

	# Init
	time_epoch_train_start = time.time()
	model.train()
	train_loss_sum = 0.0
	num_sample_loss = 0

	# Run batch
	for idx_batch_train, data_train in enumerate(dataloader_train):
		# Load data
		#inputs, labels, _, _ = data_train
		inputs, labels, idx_neural_list, idx_coord_list = data_train
		inputs = inputs.to(device)
		labels = labels.to(device)
		# print('inputs: ', inputs.shape)
		# print('labels: ', labels.shape)
		# print('idx_neural_list: ', idx_neural_list[0], idx_neural_list[1], idx_neural_list[-1])
		# print('idx_coord_list: ', idx_coord_list[0], idx_coord_list[1], idx_coord_list[-1])

		# Get prediction
		optimizer.zero_grad()
		if 'seq2seq' in model_name:
			output = model(inputs, labels)
		else:
			output = model(inputs)
		#print('output: ', output.shape)

		# When outputs are None, exit
		if output is None:
			print("Output from network is None.")
			if flag_optuna:
				raise optuna.TrialPruned()
			else:
				exit()

		if labels.shape != output.shape:
			print('labels and output have different shapes.')
			print('labels: ', labels.shape)
			print('output: ', output.shape)
			exit()
		
		# Train
		loss = loss_func(output.float(), labels.float())
		loss.backward()
		optimizer.step()

		# Save
		train_loss_sum += loss.item()*inputs.shape[0]
		num_sample_loss += inputs.shape[0]
		if value_saver is not None: 
			value_saver.update_result('train', 'loss_step', loss.item(), idx_step, True)
			idx_step += 1
		
		if flag_print:
			if idx_batch_train % 500 == 0: 
				print('idx_batch_train: ', idx_batch_train, ', loss: ', loss.item(), ', inputs: ', inputs.shape, ', labels: ', labels.shape, ', output: ', output.shape)
		
	train_loss_avg = train_loss_sum/float(num_sample_loss)
	if value_saver is not None: value_saver.update_result('train', 'loss_epoch', train_loss_avg)

	# time
	time_epoch_train_end = time.time()
	time_epoch_train = time_epoch_train_end - time_epoch_train_start
	if value_saver is not None: value_saver.update_result('train', 'time', time_epoch_train)
	
	if flag_print:
		print("Train is done. loss=", train_loss_avg)

	return train_loss_avg, idx_step

def val_one_epoch(random_seed, model, model_name, dataloader_test, device, loss_func, value_saver=None, kind=None, idx_step=None, flag_print=True, flag_optuna=False, flag_save_output=False, dir_result=None, epoch_cur=None):
	if flag_print:
		if kind is not None: print("##### Val start, kind=" + kind + " #####")
		else: print("##### Val start #####")
	
	if flag_save_output and dir_result is None:
		print("flag_save_output is true, but dir_result is None.")
		exit()
	if flag_save_output and kind is None:
		print("flag_save_output is true, but kind is None.")
		exit()
	
	# Set random seed
	networks.set_random_seed(random_seed)

	time_start = time.time()
	gt_stack, pred_stack = None, None
	idx_neural_batch, idx_coord_batch, idx_coord_stack = [], [], []
	val_loss_sum = 0.0
	num_sample_loss = 0
	model.eval()
	with torch.no_grad():
		for idx_batch_val, data_val in enumerate(dataloader_test):
			# Load data
			#inputs, labels, _, _ = data_val
			inputs, labels, idx_neural_list, idx_coord_list = data_val
			inputs = inputs.to(device)
			labels = labels.to(device)
			# print('inputs: ', inputs.shape)
			# print('labels: ', labels.shape)
			# print('idx_neural_list: ', idx_neural_list[0], idx_neural_list[1], idx_neural_list[-1])
			# print('idx_coord_list: ', idx_coord_list[0], idx_coord_list[1], idx_coord_list[-1])
			# #print ((labels == 0).nonzero(as_tuple=False)[0])

			# Get prediction
			if 'seq2seq' in model_name:
				pred = model(inputs, labels)
			else:
				pred = model(inputs)
			#print('pred: ', pred.shape) #torch.Size([16, 5, 8])

			if labels.shape != pred.shape:
				print('labels and output have different shapes.')
				print('labels: ', labels.shape)
				print('pred: ', pred.shape)
				exit()

			# loss
			loss = loss_func(pred.float(), labels.float())
			val_loss_sum += loss.item()*inputs.shape[0]
			num_sample_loss += inputs.shape[0]
			if value_saver is not None: 
				value_saver.update_result(kind, 'loss_step', loss.item(), idx_step, True)
				idx_step += 1
				
			# Stack gt
			labels_reshape = None
			if labels.dim() == 3: labels_reshape = np.reshape(labels.data.cpu().numpy(), (labels.shape[0]*labels.shape[1], labels.shape[2]))
			elif labels.dim() == 2: labels_reshape = labels.data.cpu().numpy()
			if gt_stack is None: gt_stack = labels_reshape
			else: gt_stack = np.concatenate((gt_stack, labels_reshape))
			# print('gt_stack: ', gt_stack.shape)

			# Stack pred
			pred_reshape = None
			if pred.dim() == 3: pred_reshape = np.reshape(pred.data.cpu().numpy(), (pred.shape[0]*pred.shape[1], pred.shape[2]))
			elif  pred.dim() == 2: pred_reshape = pred.data.cpu().numpy()
			if pred_stack is None: pred_stack = pred_reshape
			else: pred_stack = np.concatenate((pred_stack, pred_reshape))

			# Stack idx_neural_list, idx_coord_list
			idx_neural_batch.append(idx_neural_list)
			idx_coord_batch.append(idx_coord_list)            
			for i in idx_coord_list: idx_coord_stack.extend(i)

			if flag_print:
				if idx_batch_val % 500 == 0: 
					print('idx_batch_val: ', idx_batch_val, ', loss: ', loss.item(), ', inputs: ', inputs.shape, ', labels: ', labels.shape, ', pred: ', pred.shape)
	
	if 'seq2seq' in model_name:
		pred_stack, gt_stack = networks.reverse_mask(pred_stack, gt_stack)
	
	if len(idx_coord_stack) != gt_stack.shape[0]:
		print(len(idx_coord_stack) != gt_stack.shape[0])
		exit()

	# Check NaN
	if np.isnan(pred_stack).any():
		print("NaN in pred_stack")
		if flag_optuna is False:
			exit()
		else:
			raise optuna.TrialPruned()

	# Time
	time_end = time.time()
	time_epoch = time_end-time_start
	if value_saver is not None: value_saver.update_result(kind, 'time', time_epoch)

	# Loss
	val_loss_avg = val_loss_sum/float(num_sample_loss)
	if value_saver is not None: value_saver.update_result(kind, 'loss_epoch', val_loss_avg)

	# Calculate scores
	r2_8 = score_r2(pred_stack, gt_stack)
	r2 = sum(r2_8) / len(r2_8)
	if value_saver is not None:
		value_saver.update_result(kind, 'r2', r2)

	rmse_8 = score_rmse(pred_stack, gt_stack)
	rmse = sum(rmse_8) / len(rmse_8)
	if value_saver is not None:
		value_saver.update_result(kind, 'rmse', rmse)
	
	mse_8 = score_mse(pred_stack, gt_stack)
	mse = sum(mse_8) / len(mse_8)
	if value_saver is not None:
		value_saver.update_result(kind, 'mse', mse)

	if flag_print:
		print("Val is done. r2=", r2, ',rmse=', rmse, ',mse=', mse)
	
	# save output
	if flag_save_output:
		dir_output = os.path.join(dir_result, 'output')
		if os.path.isdir(dir_output) is False: os.mkdir(dir_output)
		
		dir_output_kind = os.path.join(dir_output, kind)
		if os.path.isdir(dir_output_kind) is False: os.mkdir(dir_output_kind)

		np.save(os.path.join(dir_output_kind, 'epoch_' + str(epoch_cur) + '_gt.npy'), np.transpose(gt_stack))
		np.save(os.path.join(dir_output_kind, 'epoch_' + str(epoch_cur) + '_pred.npy'), np.transpose(pred_stack))
		np.save(os.path.join(dir_output_kind, 'epoch_' + str(epoch_cur) + '_idx_neural_batch.npy'), np.array(idx_neural_batch, dtype=object), allow_pickle=True)
		np.save(os.path.join(dir_output_kind, 'epoch_' + str(epoch_cur) + '_idx_coord_batch.npy'), np.array(idx_coord_batch, dtype=object), allow_pickle=True)
		np.save(os.path.join(dir_output_kind, 'epoch_' + str(epoch_cur) + '_idx_coord_stack.npy'), np.array(idx_coord_stack, dtype=object), allow_pickle=True)

	return val_loss_avg, r2, rmse, mse, idx_step

def save_neural_importance(dataloader_ni, device, random_seed, model, model_name, loss_func, epoch_cur, dir_result=None):
	if dir_result is not None:
		dir_neuron_import = os.path.join(dir_result, 'neuron_importance')
		if os.path.isdir(dir_neuron_import) is False: os.mkdir(dir_neuron_import)
	
	# Set random seed
	networks.set_random_seed(random_seed)

	model_copy = deepcopy(model)
	model_copy.train()
	#grd_stack = None
	for idx_batch, data in enumerate(dataloader_ni):
		#print('idx_batch: ', idx_batch, '/', len(dataloader_train))
		if idx_batch>0:
			print('idx_batch>0')
			exit()

		# Load data
		inputs, labels, neural_idx_list, coord_idx_list = data
		inputs = inputs.to(device)
		labels = labels.to(device)
		# print('inputs: ', inputs.shape) #torch.Size([164, 5, 3361])
		# print('labels: ', labels.shape) #torch.Size([164, 20, 8])
		# print('neural_idx_list: ', len(neural_idx_list), neural_idx_list[0])
		# print('coord_idx_list: ', len(coord_idx_list), coord_idx_list[0])

		inputs.requires_grad = True  ### CRUCIAL LINE !!!

		# Train
		if 'seq2seq' in model_name:
			output = model_copy(inputs, labels)
		else:
			output = model_copy(inputs)
		#print('output: ', output.shape) #torch.Size([164, 20, 8])

		if labels.shape != output.shape:
			print('labels and output have different shapes.')
			print('labels: ', labels.shape)
			print('output: ', output.shape)
			exit()

		loss = loss_func(output.float(), labels.float())
		loss.backward() # Calculates x.grad = dloss/dx for every x with x.requires_grad=True

		grad = inputs.grad

		# g_reshape = None
		# if grad.dim() == 3: g_reshape = np.reshape(grad.data.cpu().numpy(), (grad.shape[0]*grad.shape[1], grad.shape[2]))
		# elif  grad.dim() == 2: g_reshape = grad.data.cpu().numpy()
		# if grd_stack is None: grd_stack = g_reshape
		# else: grd_stack = np.concatenate((grd_stack, g_reshape))

	#########################################################################################################################################################

	# Save
	grad = grad.data.cpu().numpy()
	np.save(os.path.join(dir_neuron_import, 'grad_epoch_'+str(epoch_cur)+'.npy'), grad)

def train_one_process(kwargs, flag_ray_tune=False, optuna_trial=None, dir_result=None):
	# print('train_one_process, kwargs: ', kwargs)

	# Data
	dir_data = kwargs['dir_data']
	name_neural = kwargs['name_neural']
	name_coord = kwargs['name_coord']
	#input_size = kwargs['input_size']

	# Train
	random_seed = kwargs['random_seed']
	lr = kwargs['lr']
	batch_size = kwargs['batch_size']
	num_epoch = kwargs['num_epoch']
	optimizer_name = kwargs['optimizer_name']
	loss_func = kwargs['loss_func']
	seq_len = kwargs['seq_len']
	output_idx = kwargs['output_idx']
	#num_output = kwargs['num_output']
	dataset_name = kwargs['dataset_name']
	device = kwargs['device'] #torch.device("cuda:1"), 'DDP'
	flag_train_val = kwargs['flag_train_val']
	flag_save_model = kwargs['flag_save_model']
	flag_save_output = kwargs['flag_save_output']
	flag_save_ni = kwargs['flag_save_ni']
	flag_print = kwargs['flag_print']
	flag_cv = kwargs['flag_cv']
	early_stop_epoch = kwargs['early_stop_epoch']
	early_stop_tol = kwargs['early_stop_tol']
	best_score_name = kwargs['best_score_name'] #'mse','r2'
	best_score_direction = kwargs['best_score_direction'] #'min','max'
  
	# Network
	model_name = kwargs['model_name']
	#n_layer = kwargs['n_layer']
	#n_unit = kwargs['n_unit']

	# log
	if dir_result is not None:
		path_log = os.path.join(dir_result, 'log.txt')
		sys.stdout = Logger(path_log)

		print("### Args")
		print("kwargs_train_process: ")
		for i in kwargs:
			print(i, ': ', kwargs[i])
		print("======================================================================")

	# flag_optuna
	if optuna_trial is None: flag_optuna = False
	else: flag_optuna = True

	value_saver = None
	dir_result_cv = None
	
	#######################################################################################################
	# Set random seed
	networks.set_random_seed(random_seed)

	# Get train-test indices
	if 'dup' in name_neural or 'intp' in name_neural:
		test_idx_cv = np.load(os.path.join(dir_data, 'test_idx_cv_coord.npy'), allow_pickle=True)
	else:
		test_idx_cv = np.load(os.path.join(dir_data, 'test_idx_cv.npy'), allow_pickle=True)
	if flag_print: print('test_idx_cv: ', test_idx_cv)

	run_idx_dict = {}
	if flag_cv is False:
		run_idx_dict[test_idx_cv.shape[0]-1] = test_idx_cv[test_idx_cv.shape[0]-1]
	else:
		for i in range(test_idx_cv.shape[0]):
			run_idx_dict[i] = test_idx_cv[i]
	neural_last_idx = run_idx_dict[test_idx_cv.shape[0]-1][1]
	if flag_print:
		print('run_idx_dict: ', run_idx_dict)
		print('neural_last_idx: ', neural_last_idx)

	# Save excel for Cross validation
	if dir_result is not None:
		path_excel_cv = os.path.join(dir_result, 'results_cv.xlsx') 
		if os.path.isfile(path_excel_cv) is False:
			wb = openpyxl.Workbook()
			worksheet = wb.active
			worksheet.append(['cv', 'r2', 'rmse', 'mse', 'epoch'])
			wb.save(path_excel_cv)

	# best score for cv
	if best_score_direction == 'max':
		score_best_cv = -999999
	elif best_score_direction == 'min':
		score_best_cv = 999999

	# Run cross validation
	for idx_cv in run_idx_dict.keys():
		if flag_print: print('idx_cv=', idx_cv, '/', len(run_idx_dict))
		if dir_result is not None:
			if len(run_idx_dict.keys()) != 1: 
				dir_result_cv = os.path.join(dir_result, 'cv_' + str(idx_cv))
				os.mkdir(dir_result_cv)
			else: 
				dir_result_cv = dir_result

		# Saver
		if dir_result is not None:
			KIND_LIST = ['train', 'train_test', 'val']
			COL_LIST = ['loss_step', 'loss_epoch', 'r2', 'rmse', 'mse', 'time']
			value_saver = ValueSaver(dir_result_cv, KIND_LIST, COL_LIST)

		# Get test_idx, train_idx
		test_idx = run_idx_dict[idx_cv]
		if test_idx[0] == 0: 
			train_idx = [test_idx[1]+1, neural_last_idx]
		elif test_idx[1] == neural_last_idx: 
			train_idx = [0, test_idx[0]-1]
		else: 
			train_idx = [[0, test_idx[0]-1], [test_idx[1]+1, neural_last_idx]]
		if flag_print:
			print('test_idx: ', test_idx)
			print('train_idx: ', train_idx)

		# datasets 
		train_dataset = NNDataset(dir_data, name_neural, name_coord, seq_len, output_idx, dataset_name[0], train_idx)
		test_dataset = NNDataset(dir_data, name_neural, name_coord, seq_len, output_idx, dataset_name[1], test_idx)
		if flag_train_val: train_val_dataset = NNDataset(dir_data, name_neural, name_coord, seq_len, output_idx, dataset_name[1], train_idx)

		# dataloaders
		dataloader_train= DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn_custom)
		dataloader_test = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn_custom)
		if flag_train_val: dataloader_train_val = DataLoader(train_val_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn_custom)
		# print('dataloader_train: ', len(dataloader_train))
		# print('dataloader_test: ', len(dataloader_test))

		if flag_save_ni:
			dataset_ni = NNDataset(dir_data, name_neural, name_coord, seq_len, output_idx, dataset_name[1], test_idx)
			dataloader_ni= DataLoader(dataset_ni, batch_size=99999, pin_memory=True, shuffle=False, collate_fn=collate_fn_custom)

		# Get model
		model = get_model(kwargs)

		# Configure optimizer
		optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

		### Epoch running
		idx_step, idx_step_val = 0, 0
		if flag_train_val: idx_step_train_test = 0

		# best score for optuna
		if best_score_direction == 'max':
			score_best = -999999
			score_prev = -999999
		elif best_score_direction == 'min':
			score_best = 999999
			score_prev = 999999
		# best score
		best_r2, best_rmse, best_mse, best_epoch = None, None, None, None
		# early stop
		epoch_tol = 0

		for epoch_cur in range(1, num_epoch+1):
			# Init value_saver
			if value_saver is not None: value_saver.ready(epoch_cur)

			### Train 
			train_loss_avg, idx_step = train_one_epoch(random_seed, model, model_name, dataloader_train, device, optimizer, loss_func, \
													   value_saver, epoch_cur, idx_step, flag_print=flag_print, flag_optuna=flag_optuna)

			### Validation
			val_loss_avg, r2, rmse, mse, idx_step_val = val_one_epoch(random_seed, model, model_name, dataloader_test, device, loss_func, \
																		value_saver, "val", idx_step_val, \
																		flag_print=flag_print, flag_optuna = flag_optuna, flag_save_output=flag_save_output,\
																		dir_result=dir_result_cv, epoch_cur=epoch_cur)
		   
			if flag_save_ni: save_neural_importance(dataloader_ni, device, random_seed, model, model_name, loss_func, epoch_cur, dir_result_cv)

			### Validation for train data
			if flag_train_val:
				_, _, _, _, idx_step_train_test = val_one_epoch(random_seed, model, model_name, dataloader_train_val, device, loss_func, value_saver, \
																"train_test", idx_step_train_test, flag_print=flag_print, flag_save_output=flag_save_output, \
																	dir_result=dir_result_cv, epoch_cur=epoch_cur) 

			# Best score
			if best_score_name == 'mse':
				score_check = mse
			elif best_score_name == 'r2':
				score_check = r2

			flag_best_update = False
			if best_score_direction == 'max':
				if score_check > score_best: 
					score_best = score_check
					flag_best_update = True
			elif best_score_direction == 'min':
				if score_check < score_best: 
					score_best = score_check
					flag_best_update = True
			
			if flag_best_update:
				best_epoch = epoch_cur
				best_r2 = r2
				best_rmse = rmse
				best_mse = mse

			if flag_print:
				print('Epoch: ', epoch_cur)
				print('score_best: ', score_best, 'r2: ', r2, ', rmse: ', rmse, ', mse: ', mse, ', train_loss: ', train_loss_avg, ', val_loss: ', val_loss_avg)

			# Save overall result
			if value_saver is not None:
				value_saver.save_per_epoch()

				# # Check if loss calculated during training and loss in value_saver are same
				# if train_loss_avg != value_saver.get_epoch_val('train', 'loss'):
				#     print('loss_train != value_saver. ', train_loss_avg, value_saver.get_epoch_val('train', 'loss'))
				#     sys.exit()
				# if val_loss_avg != value_saver.get_epoch_val('val', 'loss'):
				#     print('loss_val != value_saver. ', val_loss_avg, value_saver.get_epoch_val('val', 'loss'))
				#     sys.exit()

			# # Report optuna
			# if flag_ray_tune:
			#     tune.report(score_best=score_best, train_loss=train_loss_avg, val_loss=val_loss_avg, r2=r2, rmse=rmse, mse=mse)
			# if optuna_trial is not None:
			#     optuna_trial.report(score_check, epoch_cur)

			# Save model
			if flag_save_model:
				dir_model = os.path.join(dir_result_cv, 'checkpoint')
				if os.path.isdir(dir_model) is False: os.mkdir(dir_model)
				path_model = os.path.join(dir_model, f'checkpoint_epoch_{epoch_cur}.pth')
				torch.save({'epoch':epoch_cur, 'idx_step':idx_step, 'model_state_dict':model.state_dict()}, path_model)

			# early stop
			if early_stop_epoch is not None:
				if best_score_direction == 'max':
					if score_check-score_prev < early_stop_tol:
						epoch_tol += 1
						if flag_print:
							print('############### Early stop, epoch_tol= ', epoch_tol)
					else:
						epoch_tol = 0
				elif best_score_direction == 'min':
					if score_check-score_prev > early_stop_tol:
						epoch_tol += 1
						if flag_print:
							print('############### Early stop, epoch_tol= ', epoch_tol)
					else:
						epoch_tol = 0

				if epoch_tol == early_stop_epoch:
					print('############### Early stopping!!!. Epoch=', epoch_cur)
					break
				score_prev = score_check
			
		# Save cv result
		if dir_result is not None:
			wb = openpyxl.load_workbook(path_excel_cv)
			ws = wb.active
			ws.append([idx_cv, best_r2, best_rmse, best_mse, best_epoch])
			wb.save(path_excel_cv)
		
		# Get best score for cv
		if best_score_direction == 'max':
			if score_best > score_best_cv: 
				score_best_cv = score_best
			elif best_score_direction == 'min':
				if score_best < score_best_cv: 
					score_best_cv = score_best

	# Close logger
	if dir_result is not None:
		sys.stdout.close_logger()

	# Return for optuna
	if optuna_trial is not None: 
		return score_best_cv


