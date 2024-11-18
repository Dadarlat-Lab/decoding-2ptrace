import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from torch.utils.data import DataLoader
import torch
import pickle

from dataset import NNDataset, collate_fn_custom
import networks
import utils
import plot
from LoggingPrinter import LoggingPrinter
from scores import score_r2, score_rmse

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def plot_output(pred_stack, gt_stack, epoch):
	'''
	pred_stack, gt_stack: (time, coord)
	'''
	
	# if gt_stack.shape[1] == 1: # single output
	#     fig, axes = plt.subplots(1, gt_stack.shape[1], constrained_layout=True)
	#     axes.plot([x for x in range(gt_stack.shape[0])], gt_stack[:,0].tolist(), c='b')
	#     axes.plot([x for x in range(gt_stack.shape[0])], pred_stack[:,0].tolist(), c='r', alpha=0.7)

	alpha = 0.5
	
	if gt_stack.shape[1]==1: # single output
		fig = plt.figure()
		plt.plot([x for x in range(gt_stack.shape[0])], gt_stack[:,0].tolist(), c='b', label='gt')
		plt.plot([x for x in range(gt_stack.shape[0])], pred_stack[:,0].tolist(), c='r', alpha=alpha, label='pred')
	elif gt_stack.shape[1]>1 and gt_stack.shape[1]<8: # multiple outputs
		fig, axes = plt.subplots(gt_stack.shape[1], 1, constrained_layout=True)
		for i in range(gt_stack.shape[1]):
			axes[i].plot([x for x in range(gt_stack.shape[0])], gt_stack[:,i].tolist(), c='b', label='gt')
			axes[i].plot([x for x in range(gt_stack.shape[0])], pred_stack[:,i].tolist(), c='r', alpha=alpha, label='pred')
			axes[i].set_title(str(i))
	elif gt_stack.shape[1] == 8: # 8 outputs
		fig, axes = plt.subplots(2, 4, constrained_layout=True)
		for i in range(8):
			if i < 4:
				axes[0,i].plot([x for x in range(gt_stack.shape[0])], gt_stack[:,i].tolist(), c='b', label='gt')
				axes[0,i].plot([x for x in range(gt_stack.shape[0])], pred_stack[:,i].tolist(), c='r', alpha=alpha, label='pred')
				axes[0,i].set_title(str(i))
			else:
				axes[1,i-4].plot([x for x in range(gt_stack.shape[0])], gt_stack[:,i].tolist(), c='b', label='gt')
				axes[1,i-4].plot([x for x in range(gt_stack.shape[0])], pred_stack[:,i].tolist(), c='r', alpha=alpha, label='pred')
				axes[1,i-4].set_title(str(i))
	elif gt_stack.shape[1]>8: # encoder-decoder neuron
		gt_tr, pred_tr = np.transpose(gt_stack), np.transpose(pred_stack)

		vmin, vmax = 9999, -9999
		for i in [np.min(gt_tr), np.max(gt_tr), np.min(pred_tr), np.max(pred_tr)]:
			if i < vmin: vmin = i
			elif i > vmax: vmax = i

		fig, axes = plt.subplots(3, 1, constrained_layout=True)
		im1 = axes[0].imshow(gt_tr, vmin=vmin, vmax=vmax, aspect='auto')
		axes[0].set_title('groud truth')
		im2 = axes[1].imshow(pred_tr, vmin=vmin, vmax=vmax, aspect='auto')
		axes[1].set_title('output')
		plt.colorbar(im2, ax=axes[0:2])

		gt_avg, pred_avg = np.average(gt_tr, axis=0), np.average(pred_tr, axis=0)
		axes[2].plot([x for x in range(gt_avg.shape[0])], gt_avg.tolist(), c='b', label='gt_avg')
		axes[2].plot([x for x in range(pred_avg.shape[0])], pred_avg.tolist(), c='r', alpha=alpha, label='pred_avg')
		axes[2].set_title('gt_avg vs. pred_avg')
		axes[2].legend()
	
	r2 = score_r2(pred_stack, gt_stack)
	r2 = sum(r2)/len(r2)

	fig.suptitle('Epoch %s, r2=%f' % (epoch, r2))  
	fig.set_size_inches(18, 10)
	fig.legend()

	return fig

def get_best_case(dir_result):
	'''
	Get the best case among all epochs
	returning gt,pred : (8, 1870)
	'''
	# Get best epoch and loss
	path_excel = os.path.join(dir_result, 'results.xlsx')
	df = pd.read_excel(path_excel)
	epoch = df.loc[:, 'epoch'].tolist()
	# train_loss = df.loc[:, 'train__loss'].tolist()
	# val_loss = df.loc[:, 'val__loss'].tolist()
	val_r2 = df.loc[:, 'val__r2'].tolist()

	best_val_idx = np.argmax(val_r2)
	best_epoch = epoch[best_val_idx]
	best_r2 = val_r2[best_val_idx]
	# print('min_epoch: ', best_epoch)
	# print('best_r2: ', best_r2)
	#min_epoch_str = str(min_epoch).zfill(3)

	# final_epoch = epoch[-1]
	# final_r2 = val_r2[-1]

	# Get output
	dir_output_val = os.path.join(dir_result, 'output/val')

	pred = np.load(os.path.join(dir_output_val, 'epoch_' + str(best_epoch) + '_pred.npy'))
	gt = np.load(os.path.join(dir_output_val, 'epoch_' + str(best_epoch) + '_gt.npy'))
	
	# output_val = pd.read_excel(path_output_val)
	# gt = np.transpose(output_val[['gt', 'gt.1', 'gt.2', 'gt.3', 'gt.4', 'gt.5', 'gt.6', 'gt.7']].to_numpy())
	# pred = np.transpose(output_val[['pred', 'pred.1', 'pred.2', 'pred.3', 'pred.4', 'pred.5', 'pred.6', 'pred.7']].to_numpy())


	return best_epoch, best_r2, gt, pred



def plot_output_dir(kwargs=None):
	if kwargs is None:
		dir_result = r"D:\SeungbinPark\Results\Run\NN\Comparison_2024-02-01\NN\20240403_123751_Animal1_demean_lstmencdec"
		best_epoch = 7
	else:
		dir_result = kwargs['dir_result']
		best_epoch = kwargs['best_epoch']
	################################################################################

	name_gt = 'gt_norm_converted_epoch' + str(best_epoch)
	name_pred = 'pred_norm_converted_epoch' + str(best_epoch)
	gt = np.load(os.path.join(dir_result, name_gt+'.npy'))
	pred = np.load(os.path.join(dir_result, name_pred+'.npy'))
	print('gt: ', gt.shape)
	print('pred: ', pred.shape)

	r2 = score_r2(np.transpose(pred), np.transpose(gt))
	print('r2= ', sum(r2)/len(r2))
	print(r2)

	rmse = score_rmse(np.transpose(pred), np.transpose(gt))
	print('rmse= ', sum(rmse)/len(rmse))
	print(rmse)

	plot.plot_output_result(gt, pred, [0,1], dir_result=dir_result, title='plot_output_result'+'_01')
	plot.plot_output_result(gt, pred, [2,3], dir_result=dir_result, title='plot_output_result'+'_23')
	plot.plot_output_result(gt, pred, [4,5], dir_result=dir_result, title='plot_output_result'+'_45')
	plot.plot_output_result(gt, pred, [6,7], dir_result=dir_result, title='plot_output_result'+'_67')

	plot.plot_output_result(gt, pred, [0,1], dir_result=dir_result, title='plot_output_result'+'_01'+'_mean', mean=True)
	plot.plot_output_result(gt, pred, [2,3], dir_result=dir_result, title='plot_output_result'+'_23'+'_mean', mean=True)
	plot.plot_output_result(gt, pred, [4,5], dir_result=dir_result, title='plot_output_result'+'_45'+'_mean', mean=True)
	plot.plot_output_result(gt, pred, [6,7], dir_result=dir_result, title='plot_output_result'+'_67'+'_mean', mean=True)

def convert_norm_coord_dir():
	'''
	Load original data for min and max info, convert gt, pred of the best result in dir_result, and save them into npy files.
	'''
	dir_result = r"E:\tmp\NN\20241118_115845_Animal1_multilimb"
	output_idx = [None] #[None]=multi-limb, [0,1], ...
	dir_data = r"E:\tmp\prepare_data\Animal1-G8_53950_1L_Redo\Animal1_likeli01"
	name_ori = 'behav_coord_likeli_ori'
	
	#######################################
	dir_plot = os.path.join(dir_result, 'convert_norm_coord')
	if os.path.isdir(dir_plot):
		print('Directory exists.')
		exit()
	else: os.mkdir(dir_plot)

	# Load result
	best_epoch, best_r2, gt, pred = get_best_case(dir_result)
	print('gt: ', gt.shape)
	print('pred: ', pred.shape)
	if gt.shape != pred.shape:
		print('gt.shape != pred.shape')
		print(gt.shape, pred.shape)
	
	# Load original data
	if name_ori == 'demean':
		for name, d in zip(['gt', 'pred'], [gt, pred]):
			convert_arr = d
			plot.plot_coord_all(d, title=name+'_before_convert_norm', dir_save=dir_plot)
			plot.plot_coord_all(convert_arr, title=name+'_after_convert_norm', dir_save=dir_plot)
			np.save(os.path.join(dir_result, name + '_norm_converted_epoch' + str(best_epoch) + '.npy'), convert_arr)

	else:
		ori = np.load(os.path.join(dir_data, name_ori+'.npy'))
		if output_idx[0] is not None: ori = ori[output_idx, :]
		print('ori: ', ori.shape)

		convert_all = []
		for name, d in zip(['gt', 'pred'], [gt, pred]):
			convert_arr = utils.convert_norm_coord_func(d,ori)

			convert_all.append(convert_arr)
			plot.plot_coord_all(d, title=name+'_before_convert_norm', dir_save=dir_plot)
			plot.plot_coord_all(convert_arr, title=name+'_after_convert_norm', dir_save=dir_plot)
			np.save(os.path.join(dir_result, name + '_norm_converted_epoch' + str(best_epoch) + '.npy'), convert_arr)

		# Plot results again
		fig = plot_output(np.transpose(pred), np.transpose(gt), best_epoch)
		plt.savefig(os.path.join(dir_plot, 'outputs_before_convert_norm'+'.png'))
		fig = plot_output(np.transpose(convert_all[1]), np.transpose(convert_all[0]), best_epoch)
		plt.savefig(os.path.join(dir_plot, 'outputs_after_convert_norm'+'.png'))
		#plt.show()

	kwargs_plot = {'dir_result':dir_result, 'best_epoch':best_epoch}
	plot_output_dir(kwargs_plot)
	
	return best_epoch

def save_neuron_importance(kwargs=None):
	if kwargs is None:
		### Change here!!!. Reference log.txt in dir_result
		dir_result = r"E:\tmp\NN\20241118_115845_Animal1_multilimb"
		best_epoch = 7
		DIR_DATA = r"E:\tmp\prepare_data\Animal1-G8_53950_1L_Redo\Animal1_likeli01"
		NAME_NEURAL = "spks_z_sel" #"spks_z_sel"
		NAME_COORD = "behav_coord_likeli_norm"
		SEQ_LEN = 5
		OUTPUT_IDX = [None]

		device = torch.device(type='cuda', index=0)
		note_device = "torch.device(type='cuda', index=0)"
		DATASET = 'seq2seq_run_overlapX' #'seq2seq_overlapX', 'seq2seq_prev_overlapX'
		NET_NAME = "seq2seq"
		NET = networks.seq2seq(**{'input_size': 3327, 'hidden_size': 795, 'num_layers': 2, 'output_size': 8, 'device': device})
		note_NET = "networks.seq2seq(**{'input_size': 3327, 'hidden_size': 795, 'num_layers': 2, 'output_size': 8, 'device': device})"
		LOSS_FUNC = networks.loss_seq2seq_mse()
		note_loss = "networks.loss_seq2seq_mse()"
	else:
		dir_result = kwargs['dir_result']
		best_epoch = kwargs['best_epoch']
		DIR_DATA = kwargs['dir_data']
		NAME_NEURAL = kwargs['name_neural']
		NAME_COORD = kwargs['name_coord']
		SEQ_LEN = kwargs['seq_len']
		OUTPUT_IDX = kwargs['output_idx']

		device = torch.device(type='cuda', index=kwargs['device_idx'])
		note_device = "torch.device(type='cuda', index=" + str(kwargs['device_idx']) + ")"
		DATASET = kwargs['dataset']

		if OUTPUT_IDX[0] is None: output_size = 8
		else: output_size = len(OUTPUT_IDX)
		NET_NAME = "seq2seq"
		NET = networks.seq2seq(**{'input_size': kwargs['input_size'], 'hidden_size': kwargs['hidden_size'], 'num_layers': kwargs['num_layers'], 'output_size':output_size, 'device': device})
		note_NET = "networks.seq2seq(**{'input_size': " + str(kwargs['input_size']) + ", 'hidden_size': " + str(kwargs['hidden_size']) + \
			", 'num_layers': " + str(kwargs['num_layers']) + ", 'output_size': " + str(output_size) + ", 'device': "+ str(device) + "})"
		LOSS_FUNC = networks.loss_seq2seq_mse()
		note_loss = "networks.loss_seq2seq_mse()"

	##############################################################################################

	dir_neuron_import = os.path.join(dir_result, 'neuron_importance')
	if os.path.isdir(dir_neuron_import):
		print('Directory exists.')
		exit()
	else: os.mkdir(dir_neuron_import)

	path_log = os.path.join(dir_neuron_import, 'log.txt')
	with LoggingPrinter(path_log):

		print("### Arguments ############################")
		print('dir_result: ', dir_result)
		print('DIR_DATA: ', DIR_DATA)
		print('NAME_NEURAL: ', NAME_NEURAL)
		print('NAME_COORD: ', NAME_COORD)
		print('SEQ_LEN: ', SEQ_LEN)
		print('OUTPUT_IDX: ', OUTPUT_IDX)
		print('device: ', note_device)
		print('DATASET: ', DATASET)
		print('NET_NAME: ', NET_NAME)
		print('NET: ', note_NET)
		print('best_epoch: ', best_epoch)
		print('LOSS_FUNC: ', note_loss)
		print("#########################################")

		### Load data
		# # Get train-test indices
		# train_test_idx = np.load(os.path.join(DIR_DATA, 'train_test_idx.npy'), allow_pickle=True).item()
		# train_idx = train_test_idx['train']
		# test_idx = train_test_idx['test']
		# # print(train_test_idx)
		# # print('train_idx: ', train_idx, train_idx[1]-train_idx[0])
		# # print('test_idx: ', test_idx, test_idx[1]-test_idx[0])
		# size_batch = train_idx[1]-train_idx[0]
		# print('size_batch: ', size_batch)

		# Get train-test indices
		test_idx_cv = np.load(os.path.join(DIR_DATA, 'test_idx_cv.npy'), allow_pickle=True)
		test_idx = test_idx_cv[-1]
		train_idx = [0, test_idx[0]-1]
		size_batch = 99999 #int((test_idx[1]-test_idx[0]).item()/SEQ_LEN)
		print('test_idx: ', test_idx)
		print('train_idx: ', train_idx)
		print('size_batch: ', size_batch)

		# datasets
		dataset_test = NNDataset(DIR_DATA, NAME_NEURAL, NAME_COORD, SEQ_LEN, OUTPUT_IDX, DATASET, test_idx)

		# dataloaders
		dataloader_test= DataLoader(dataset_test, batch_size=size_batch, pin_memory=True, shuffle=False, collate_fn=collate_fn_custom)

		### Calculating gradient version ###############################################################################################################
		NET.to(device)

		### Load model
		model = networks.load_checkpoint(NET, os.path.join(dir_result, 'checkpoint', 'checkpoint_epoch_'+str(best_epoch)+'.pth'))

		### Run
		# https://discuss.pytorch.org/t/newbie-getting-the-gradient-with-respect-to-the-input/12709/8
		grd_stack = None
		for idx_batch, data in enumerate(dataloader_test):
			#print('idx_batch: ', idx_batch, '/', len(dataloader_train))
			if idx_batch>0:
				print('idx_batch>0')
				exit()

			# Load data
			inputs, labels, neural_idx_list, coord_idx_list = data
			inputs = inputs.to(device)
			labels = labels.to(device)
			print('inputs: ', inputs.shape) #torch.Size([164, 5, 3361])
			print('labels: ', labels.shape) #torch.Size([164, 20, 8])
			print('neural_idx_list: ', len(neural_idx_list), neural_idx_list[0])
			print('coord_idx_list: ', len(coord_idx_list), coord_idx_list[0])

			inputs.requires_grad = True  ### CRUCIAL LINE !!!

			# Train
			if 'seq2seq' in NET_NAME:
				output = model(inputs, labels)
			else:
				output = model(inputs)
			print('output: ', output.shape) #torch.Size([164, 20, 8])

			if labels.shape != output.shape:
				print('labels and output have different shapes.')
				print('labels: ', labels.shape)
				print('output: ', output.shape)
				exit()

			loss = LOSS_FUNC(output.float(), labels.float())
			loss.backward() # Calculates x.grad = dloss/dx for every x with x.requires_grad=True

			grad = inputs.grad

			g_reshape = None
			if grad.dim() == 3: g_reshape = np.reshape(grad.data.cpu().numpy(), (grad.shape[0]*grad.shape[1], grad.shape[2]))
			elif  grad.dim() == 2: g_reshape = grad.data.cpu().numpy()
			if grd_stack is None: grd_stack = g_reshape
			else: grd_stack = np.concatenate((grd_stack, g_reshape))

		# print('gt_stack: ', gt_stack.shape)
		# print('pred_stack: ', pred_stack.shape)
		grad = grad.data.cpu().numpy()
		print('grad: ', grad.shape, np.min(grad), np.max(grad)) # torch.Size([164, 5, 3361])
		print('grd_stack: ', grd_stack.shape, np.min(grd_stack), np.max(grd_stack)) #grd_stack:  (3730, 3327)=(frame, neuron) -3.7350696e-06 3.4244154e-06

		grad_abs = np.absolute(grad)
		grd_stack_abs = np.absolute(grd_stack)
		print('grad_abs: ', grad_abs.shape, np.min(grad_abs), np.max(grad_abs))
		print('grd_stack_abs: ', grd_stack_abs.shape, np.min(grd_stack_abs), np.max(grd_stack_abs))

		# plot_output(pred_stack, gt_stack, best_epoch)

		# plt.figure()
		# plt.imshow(grd_stack)
		# plt.show()

		# Get norm of grd_stack over time
		grd_norm_over_time = []
		for i in range(grd_stack.shape[1]):
			grd = grd_stack[:, i]
			l1 = np.linalg.norm(grd, 1)
			# l1_cal = sum([abs(x) for x in grd])
			# print(l1)
			# print(l1_cal)
			grd_norm_over_time.append(l1)
		grd_norm_over_time = np.array(grd_norm_over_time)
		print('grd_norm_over_time: ', grd_norm_over_time.shape) #(3327, )=(neuron, )
		# plt.plot(grd_norm, '-x')
		# plt.show()

		# Normalize
		grd_norm_over_time_norm = (grd_norm_over_time - min(grd_norm_over_time)) / (max(grd_norm_over_time)-min(grd_norm_over_time))
		if np.min(grd_norm_over_time_norm) != 0:
			print('grd_norm_over_time_norm min!=0')
			print(np.min(grd_norm_over_time_norm))
			exit()
		if np.max(grd_norm_over_time_norm) != 1:
			print('grd_norm_over_time_norm max != 1')
			print(np.max(grd_norm_over_time_norm))
			exit()
		print('grd_norm_over_time_norm: ', grd_norm_over_time_norm.shape)

		#########################################################################################################################################################

		# Save
		np.save(os.path.join(dir_neuron_import, 'grad.npy'), grad)
		np.save(os.path.join(dir_neuron_import, 'grd_stack.npy'), grd_stack)
		np.save(os.path.join(dir_neuron_import, 'grad_abs.npy'), grad_abs)
		np.save(os.path.join(dir_neuron_import, 'grd_stack_abs.npy'), grd_stack_abs)
		np.save(os.path.join(dir_neuron_import, 'grd_norm_over_time.npy'), grd_norm_over_time)
		np.save(os.path.join(dir_neuron_import, 'grd_norm_over_time_norm.npy'), grd_norm_over_time_norm)
		with open(os.path.join(dir_neuron_import, 'neural_idx_list'), "wb") as f:
			pickle.dump(neural_idx_list, f)
		with open(os.path.join(dir_neuron_import, 'coord_idx_list'), "wb") as f:
			pickle.dump(coord_idx_list, f)

if __name__ == '__main__':

	### Convert normalization
	convert_norm_coord_dir()

	### Save neural importance
	#save_neuron_importance()
