import os
import numpy as np

import torch
from torch.utils.data import Dataset

'''
https://stackoverflow.com/questions/57893415/pytorch-dataloader-for-time-series-task
'''

def collate_fn_custom(batch):
	'''
	To stack vectors with different lengths by adding zeros
	
	https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3?u=ptrblck
	https://androidkt.com/create-dataloader-with-collate_fn-for-variable-length-input-in-pytorch/
	'''    
	# data = [item[0] for item in batch]
	# target = [item[1] for item in batch]

	data, target = [], []
	len_trg_list = []
	neural_idx = []
	coord_idx = [] 
	
	for item in batch:
		data.append(item[0])

		# check zero in target, to prevent being removed from reverse_mask
		trg = item[1]
		if 0 in trg:
			trg[(trg == 0.0).nonzero(as_tuple=True)] = 1e-6

		target.append(trg)
		len_trg_list.append(len(trg))

		neural_idx.append(item[2])
		coord_idx.append(item[3])

	data = torch.stack(data)
	if len(set(len_trg_list)) == 1:
		target = torch.stack(target)
	else:
		target = torch.nn.utils.rnn.pad_sequence(target, batch_first=True)

		# # sequence heatmap
		# for b in range(target.shape[0]):
		#     trg = np.transpose(target.detach().cpu().numpy()[b])
		#     fig, ax = plt.subplots()
		#     plt.imshow(trg)
		#     plt.xlabel('sequence')
		#     plt.ylabel('limb-8')
		#     for i in range(trg.shape[0]):
		#         for j in range(trg.shape[1]):
		#             text = ax.text(j, i, f"{trg[i, j]:.4f}", ha="center", va="center", color="w")
		#     plt.title('target, Batch=' + str(b) + '/' + str(target.shape[0]))
		#     plt.show()

	# print('data: ', data.shape)
	# print('target: ', target.shape)

	return data, target, neural_idx, coord_idx

class NNDataset(Dataset):
	def __init__(self, dir_data, name_neural, name_coord, sequence_length, output_idx, kind, idx_list):
		'''
		output_idx: List of limb index (0~7). [None]=multi-limb, [0], [0,1], ...
		kind: 
			- original data: 'seq2seq_run_overlapO' for training, 'seq2seq_run_overlapX' for test
			- dup/intp data: 'dupintp_run_overlapO' for training, 'dupintp_run_overlapX' for test
			- match data: 'match_overlapX_run'  for training, 'match_overlapO_run' for test
		idx_list: [start, end(include)] of neural_data. (sequence length, neural_data.shape[1]) or [[start1, end1(include)], [start2, end2(include)], ...]
		'''
		self.kind = kind 
		self.idx_neural_batch_list = []
		self.idx_behav_batch_list = []

		self.kind_option_list = ['seq2seq_run_overlapO', 'seq2seq_run_overlapX', 'dupintp_run_overlapO', 'dupintp_run_overlapX', 'match_overlapX_run', 'match_overlapO_run']

		if self.kind not in self.kind_option_list: 
			print('Weird kind in dataset')
			print(self.kind)
			exit()
		
		if isinstance(idx_list[0], list) is False:
			if idx_list[1] < idx_list[0]:
				print('idx_list is werid. ', idx_list[0], idx_list[1])
				exit()
		else:
			for j in range(len(idx_list)):
				if idx_list[j][1]<idx_list[j][0]:
					print('idx_list[', j, '] is werid. ', idx_list[j])
					exit()
			
		# Load data
		path_data_neural = os.path.join(dir_data, name_neural+'.npy')
		path_data_coord = os.path.join(dir_data, name_coord+'.npy')

		neural_data_load = np.load(path_data_neural) #(neuron, time)
		neural_data_load = np.transpose(neural_data_load) #(time, neuron) 
		self.neural_data_tensor = torch.tensor(neural_data_load, dtype=torch.float32)

		behav_coord_load = np.load(path_data_coord) #(behav, time)
		behav_coord_load = np.transpose(behav_coord_load) #(time, behav)
		if output_idx[0] is not None: behav_coord_load = behav_coord_load[:, output_idx] # Select output_idx
		self.behav_coord_tensor = torch.tensor(behav_coord_load, dtype=torch.float32)

		# print('neural_data_tensor: ', self.neural_data_tensor.shape)
		# print('behav_coord_tensor: ', self.behav_coord_tensor.shape)

		if 'match' in kind or 'dupint' in kind:
			if neural_data_load.shape[0] != behav_coord_load.shape[0]:
				print("neural_data_load.shape[0] != behav_coord_load.shape[0]")
				print(neural_data_load.shape)
				print(behav_coord_load.shape)
				exit()

		if 'seq2seq' in kind: # If seq2seq, load idx_coord_neural
			path_idx = os.path.join(dir_data, 'idx_coord_neural.npy')
			idx_coord_neural = np.load(path_idx).astype('int64')
			#print('idx_coord_neural: ', idx_coord_neural.shape, idx_coord_neural[:30], idx_coord_neural[-30:])

			if idx_coord_neural.shape[0] != behav_coord_load.shape[0]:
				print("idx_coord_neural.shape[0] != behav_coord.shape[0]", idx_coord_neural.shape, behav_coord_load.shape)
				exit()
		
		# Load only running periods
		if kind in ['seq2seq_run_overlapO', 'seq2seq_run_overlapX', 'dupintp_run_overlapO', 'dupintp_run_overlapX']:
			run_windows = np.load(os.path.join(dir_data, 'run_windows_likeli.npy'))
		elif kind in ['match_overlapX_run', 'match_overlapO_run']:
			run_windows = np.load(os.path.join(dir_data, 'run_windows_likeli_match.npy'))
			
		# Get list of possible coord idx
		possible_coord_idx_list = []
		for win in run_windows:
			possible_coord_idx_list.extend(list(range(win[0],win[1]+1,1)))
		if len(possible_coord_idx_list) != len(set(possible_coord_idx_list)):
			print('possible_coord_idx_list has duplicates in dataset.')
			exit()

		# Get idx_neural_batch_list and idx_behav_batch_list from idx_list 
		if self.kind == 'match_overlapO_run':
			if isinstance(idx_list[0], list) is False:
				for i in range(idx_list[0], idx_list[1]-sequence_length+2):
					idx_neural_list = [i+s for s in range(sequence_length)]
					idx_coord_list = idx_neural_list

					# if idx_coord_list includes element in possible_coord_idx_list, don't use it
					if list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))) != idx_coord_list:
						# print('idx_coord_list: ', idx_coord_list)
						# print('list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))): ', list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))))
						# print("idx_coord_list include element not in possible_coord_idx_list")
						continue

					self.idx_neural_batch_list.append(idx_neural_list)
					self.idx_behav_batch_list.append(idx_coord_list)
			else:
				print("Exit at NNDataset match_overlapX_run. Not using cross validation")
				exit()
		
		elif self.kind == 'match_overlapX_run':
			if isinstance(idx_list[0], list) is False:
				for i in range((idx_list[1]-idx_list[0]+1)//sequence_length):
					idx_neural_list = [idx_list[0]+i*sequence_length+s for s in range(sequence_length)]
					idx_coord_list = idx_neural_list
  
					# if idx_coord_list includes element in possible_coord_idx_list, don't use it
					if list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))) != idx_coord_list:
						# print('idx_coord_list: ', idx_coord_list)
						# print('list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))): ', list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))))
						# print("idx_coord_list include element not in possible_coord_idx_list")
						continue

					self.idx_neural_batch_list.append(idx_neural_list)
					self.idx_behav_batch_list.append(idx_coord_list)
			else:
				print("Exit at NNDataset match_overlapX_run. Not using cross validation")
				exit()

		elif self.kind == 'seq2seq_run_overlapO':
			if isinstance(idx_list[0], list) is False:
				for i in range(idx_list[0], idx_list[1]-sequence_length+2):
					idx_neural_list = [i+s for s in range(sequence_length)]

					idx_coord_list = []
					for i in idx_neural_list:
						idx_found = np.where(idx_coord_neural==i)[0]
						idx_coord_list.extend(idx_found)
					if len(idx_coord_list) != len(set(idx_coord_list)):
						print('idx_coord_list has duplicates in dataset.')
						exit()
					
					# if idx_coord_list includes element in possible_coord_idx_list, don't use it
					if list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))) != idx_coord_list:
						# print('idx_coord_list: ', idx_coord_list)
						# print('list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))): ', list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))))
						# print("idx_coord_list include element not in possible_coord_idx_list")
						continue
					
					self.idx_neural_batch_list.append(idx_neural_list)
					self.idx_behav_batch_list.append(idx_coord_list)

			else:
				for idx_list_inside in idx_list:
					for i in range(idx_list_inside[0], idx_list_inside[1]-sequence_length+2):
						idx_neural_list = [i+s for s in range(sequence_length)]
						
						idx_coord_list = []
						for i in idx_neural_list:
							idx_found = np.where(idx_coord_neural==i)[0]
							idx_coord_list.extend(idx_found)
						if len(idx_coord_list) != len(set(idx_coord_list)):
							print('idx_coord_list has duplicates in dataset.')
							exit()

						# if idx_coord_list includes element in possible_coord_idx_list, don't use it
						if list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))) != idx_coord_list:
							# print('idx_coord_list: ', idx_coord_list)
							# print('list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))): ', list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))))
							# print("idx_coord_list include element not in possible_coord_idx_list")
							continue
						
						self.idx_neural_batch_list.append(idx_neural_list)
						self.idx_behav_batch_list.append(idx_coord_list)
		
		elif self.kind == 'seq2seq_run_overlapX':
			if isinstance(idx_list[0], list) is False:
				for i in range((idx_list[1]-idx_list[0]+1)//sequence_length):
					idx_neural_list = [idx_list[0]+i*sequence_length+s for s in range(sequence_length)]

					idx_coord_list = []
					for i in idx_neural_list:
						idx_found = np.where(idx_coord_neural==i)[0]
						idx_coord_list.extend(idx_found)
					if len(idx_coord_list) != len(set(idx_coord_list)):
						print('idx_coord_list has duplicates in dataset.')
						exit()
					
					# if idx_coord_list includes element in possible_coord_idx_list, don't use it
					if list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))) != idx_coord_list:
						# print('idx_coord_list: ', idx_coord_list)
						# print('list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))): ', list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))))
						# print("idx_coord_list include element not in possible_coord_idx_list")
						continue
					
					self.idx_neural_batch_list.append(idx_neural_list)
					self.idx_behav_batch_list.append(idx_coord_list)
			else:
				for idx_list_inside in idx_list:
					for i in range((idx_list_inside[1]-idx_list_inside[0]+1)//sequence_length):
						idx_neural_list = [idx_list_inside[0]+i*sequence_length+s for s in range(sequence_length)]

						idx_coord_list = []
						for i in idx_neural_list:
							idx_found = np.where(idx_coord_neural==i)[0]
							idx_coord_list.extend(idx_found)
						if len(idx_coord_list) != len(set(idx_coord_list)):
							print('idx_coord_list has duplicates in dataset.')
							exit()
						
						# if idx_coord_list includes element in possible_coord_idx_list, don't use it
						if list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))) != idx_coord_list:
							# print('idx_coord_list: ', idx_coord_list)
							# print('list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))): ', list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))))
							# print("idx_coord_list include element not in possible_coord_idx_list")
							continue

						self.idx_neural_batch_list.append(idx_neural_list)
						self.idx_behav_batch_list.append(idx_coord_list)
			
		elif self.kind == 'dupintp_run_overlapO':
			if isinstance(idx_list[0], list) is False:
				for i in range(idx_list[0], idx_list[1]-sequence_length+2):
					idx_neural_list = [i+s for s in range(sequence_length)]
					idx_coord_list = [x for x in idx_neural_list]

					# if idx_coord_list includes element in possible_coord_idx_list, don't use it
					if list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))) != idx_coord_list:
						# print('idx_coord_list: ', idx_coord_list)
						# print('list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))): ', list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))))
						# print("idx_coord_list include element not in possible_coord_idx_list")
						continue
					
					self.idx_neural_batch_list.append(idx_neural_list)
					self.idx_behav_batch_list.append(idx_coord_list)

			else:
				for idx_list_inside in idx_list:
					for i in range(idx_list_inside[0], idx_list_inside[1]-sequence_length+2):
						idx_neural_list = [i+s for s in range(sequence_length)]
						idx_coord_list = [x for x in idx_neural_list]

						# if idx_coord_list includes element in possible_coord_idx_list, don't use it
						if list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))) != idx_coord_list:
							# print('idx_coord_list: ', idx_coord_list)
							# print('list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))): ', list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))))
							# print("idx_coord_list include element not in possible_coord_idx_list")
							continue
						
						self.idx_neural_batch_list.append(idx_neural_list)
						self.idx_behav_batch_list.append(idx_coord_list)
		
		elif self.kind == 'dupintp_run_overlapX':
			if isinstance(idx_list[0], list) is False:
				for i in range((idx_list[1]-idx_list[0]+1)//sequence_length):
					idx_neural_list = [idx_list[0]+i*sequence_length+s for s in range(sequence_length)]
					idx_coord_list = [x for x in idx_neural_list]

					# if idx_coord_list includes element in possible_coord_idx_list, don't use it
					if list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))) != idx_coord_list:
						# print('idx_coord_list: ', idx_coord_list)
						# print('list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))): ', list(sorted(set(possible_coord_idx_list)&set(idx_coord_list))))
						# print("idx_coord_list include element not in possible_coord_idx_list")
						continue
					
					self.idx_neural_batch_list.append(idx_neural_list)
					self.idx_behav_batch_list.append(idx_coord_list)
			else:
				for idx_list_inside in idx_list:
					for i in range((idx_list_inside[1]-idx_list_inside[0]+1)//sequence_length):
						idx_neural_list = [idx_list_inside[0]+i*sequence_length+s for s in range(sequence_length)]
						idx_coord_list = [x for x in idx_neural_list]

						self.idx_neural_batch_list.append(idx_neural_list)
						self.idx_behav_batch_list.append(idx_coord_list)

	def __len__(self):
		if self.kind in self.kind_option_list:
			return len(self.idx_neural_batch_list)
		else:
			print("weird kind.")
			print(self.kind)
			exit()
	  
	def __getitem__(self, idx):
		if self.kind in self.kind_option_list:
			# Get neural_idx_list
			neural_idx_list = self.idx_neural_batch_list[idx]
		   
			# Get coord_idx_list
			coord_idx_list = self.idx_behav_batch_list[idx]
			
			# Get data
			return self.neural_data_tensor[neural_idx_list, :], self.behav_coord_tensor[coord_idx_list, :], neural_idx_list, coord_idx_list
		else:
			print("weird kind.")
			print(self.kind)
			exit()
		   


