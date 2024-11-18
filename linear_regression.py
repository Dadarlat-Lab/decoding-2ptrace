from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
import os
from utils import convert_norm_coord_func
from random import uniform
from dataset import NNDataset, collate_fn_custom
from torch.utils.data import DataLoader
from plot import plot_output_result
from LoggingPrinter import LoggingPrinter
from numpy.linalg import inv as inv #Used in kalman filter


class KalmanFilterDecoder(object):
	'''
	https://github.com/KordingLab/Neural_Decoding/blob/master/Neural_Decoding/decoders.py
	https://proceedings.neurips.cc/paper_files/paper/2002/file/169779d3852b32ce8b1a1724dbf5217d-Paper.pdf
	https://www.dam.brown.edu/people/elie/papers/Wu%20et%20al%20SAB%202002.pdf
	
	'''

	"""
	Class for the Kalman Filter Decoder

	Parameters
	-----------
	C - float, optional, default 1
	This parameter scales the noise matrix associated with the transition in kinematic states.
	It effectively allows changing the weight of the new neural evidence in the current update.

	Our implementation of the Kalman filter for neural decoding is based on that of Wu et al 2003 (https://papers.nips.cc/paper/2178-neural-decoding-of-cursor-motion-using-a-kalman-filter.pdf)
	with the exception of the addition of the parameter C.
	The original implementation has previously been coded in Matlab by Dan Morris (http://dmorris.net/projects/neural_decoding.html#code)

	http://static.cs.brown.edu/people/mjblack/Papers/sab2002.pdf
	"""

	def __init__(self,C=1,C_Q=1,lassoalpha=None):
		self.C=C
		self.C_Q = C_Q
		self.lassoalpha = lassoalpha


	def fit(self,X_kf_train,y_train):

		"""
		Train Kalman Filter Decoder

		Parameters
		----------
		X_kf_train: numpy 2d array of shape [n_samples(i.e. timebins) , n_neurons]
			This is the neural data in Kalman filter format.
			See example file for an example of how to format the neural data correctly

		y_train: numpy 2d array of shape [n_samples(i.e. timebins), n_outputs]
			This is the outputs that are being predicted
		"""

		if X_kf_train.shape[0] != y_train.shape[0]:
			print("X_kf_train.shape[1] != y_train.shape[1]. ", X_kf_train.shape, y_train.shape)
			exit()

		#First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature (specifically that from Wu et al, 2003):
		#xs are the state (here, the variable we're predicting, i.e. y_train)
		#zs are the observed variable (neural data here, i.e. X_kf_train)
		X=np.matrix(y_train.T) #(8, 3495)
		Z=np.matrix(X_kf_train.T) #(3327, 3495)
		# print('X: ', X.shape) #(8, 3490)
		# print('Z: ', Z.shape) #(3327, 3490)

		#number of time bins
		nt = y_train.shape[0]
		#nt=X.shape[1] #slow?

		#Calculate the transition matrix (from x_t to x_t+1) using least-squares, and compute its covariance
		#In our case, this is the transition from one kinematic state to the next
		X2 = X[:,1:]
		X1 = X[:,0:nt-1]

		# plt.plot(X2[0,:100])
		# plt.show()

		# print('X1: ', X1.shape) # (8, 3489)
		# plot_coord_all(X1)

		# print('X2:', X2.shape) #(8, 3489)
		# plot_coord_all(X2)

		### 1. original version
		A=X2*X1.T*inv(X1*X1.T) #Transition matrix
		W=(X2-A*X1)*(X2-A*X1).T/(nt-1)/self.C #Covariance of transition matrix. Note we divide by nt-1 since only nt-1 points were used in the computation (that's the length of X1 and X2). We also introduce the extra parameter C here.
		# print("X2: ", X2.shape) #(8, 3489)
		# print("X1: ", X1.shape) #(8, 3489)
		# print("A: ", A.shape) #(8, 8)
		# print("W: ", W.shape) #(8, 8)

		# ### 2. Fixed version: each limb
		# A = np.zeros((8,8))
		# W = np.zeros((8,8))
		# for i in range(4):
		# 	X2_limb = X[i*2:i*2+2, 1:]
		# 	X1_limb = X[i*2:i*2+2, 0:nt-1]

		# 	A_limb = X2_limb*X1_limb.T*inv(X1_limb*X1_limb.T)
		# 	W_limb = (X2_limb-A_limb*X1_limb)*(X2_limb-A_limb*X1_limb).T/(nt-1)/self.C
			
		# 	A[i*2:i*2+2, i*2:i*2+2] = A_limb
		# 	W[i*2:i*2+2, i*2:i*2+2] = W_limb

		# 	#print(A_limb.shape, W_limb.shape)
		
		# fig = plt.figure(1)
		# plt.matshow(A)

		# fig = plt.figure(2)
		# plt.matshow(W)

		if self.lassoalpha is None:
			# #Calculate the measurement matrix (from x_t to z_t) using least-squares, and compute its covariance
			# #In our case, this is the transformation from kinematics to spikes
			H = Z*X.T*(inv(X*X.T)) #Measurement matrix
			Q = ((Z - H*X)*((Z - H*X).T)) / nt / self.C_Q #Covariance of measurement matrix
			# # print('H: ', H.shape) # (3327, 8)
			# # print('Q: ', Q.shape) #(3327, 3327)
			# # plt.matshow(Q)
			# # plt.show()
		else:
			## lasso version
			model = linear_model.Lasso(alpha=self.lassoalpha)
			model.fit(y_train, X_kf_train)
			H = model.coef_
			Q = ((Z - H*X)*((Z - H*X).T)) / nt / self.C_Q

		### Fixed version: zero out the off-diagonal elements of "emission" covariance matrix
		idx_off_diag = np.where(~np.eye(Q.shape[0],dtype=bool))
		for i,j in zip(idx_off_diag[0], idx_off_diag[1]): Q[i,j]=0.0
		
		# plt.matshow(H)

		# plt.matshow(Q)
		# plt.show()

		params=[A,W,H,Q]
		self.model=params

	def predict(self,X_kf_test,y_test):

		"""
		Predict outcomes using trained Kalman Filter Decoder

		Parameters
		----------
		X_kf_test: numpy 2d array of shape [n_samples(i.e. timebins) , n_neurons]
			This is the neural data in Kalman filter format.

		y_test: numpy 2d array of shape [n_samples(i.e. timebins),n_outputs]
			The actual outputs
			This parameter is necesary for the Kalman filter (unlike other decoders)
			because the first value is nececessary for initialization

		Returns
		-------
		y_test_predicted: numpy 2d array of shape [n_samples(i.e. timebins),n_outputs]
			The predicted outputs
		"""

		if X_kf_test.shape[0] != y_test.shape[0]:
			print("X_kf_test.shape[1] != y_test.shape[1]. ", X_kf_test.shape, y_test.shape)
			exit()

		#Extract parameters
		A,W,H,Q=self.model

		#First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature (specifically that from Wu et al):
		#xs are the state (here, the variable we're predicting, i.e. y_train)
		#zs are the observed variable (neural data here, i.e. X_kf_train)
		X=np.matrix(y_test.T)
		Z=np.matrix(X_kf_test.T)

		#Initializations
		num_states=X.shape[0] #Dimensionality of the state
		states=np.empty(X.shape) #Keep track of states over time (states is what will be returned as y_test_predicted)
		P_m=np.matrix(np.zeros([num_states,num_states]))
		P=np.matrix(np.zeros([num_states,num_states]))
		state=X[:,0] #Initial state

		#Get predicted state for every time bin
		for t in range(X.shape[1]-1):
			if t%100==0: print('t: ', t, '/ ', X.shape[1]-1)
			#Do first part of state update - based on transition matrix
			P_m=A*P*A.T+W
			state_m=A*state

			#Do second part of state update - based on measurement matrix
			K=P_m*H.T*inv(H*P_m*H.T+Q) #Calculate Kalman gain
			P=(np.matrix(np.eye(num_states))-K*H)*P_m
			state=state_m+K*(Z[:,t+1]-H*state_m)
			states[:,t+1]=np.squeeze(state) #Record state at the timestep
		y_test_predicted=states.T

		return y_test_predicted

def saver(dir_data, name_coord_ori, y_test_predicted_limb, y_test_limb, dir_result_par, animal, data_type, model_type, dir_result=None):
	if name_coord_ori is None:
		convert_arr = y_test_predicted_limb
		y_test_convert = y_test_limb
	else:
		# Convert y_test_predicted
		ori_limb = np.load(os.path.join(dir_data, name_coord_ori+'.npy')) #[limb, time]
		convert_arr = convert_norm_coord_func(y_test_predicted_limb,ori_limb)
		#print(convert_arr.shape)

		# Convert y_test
		y_test_convert = convert_norm_coord_func(y_test_limb,ori_limb)

	# Save
	if dir_result is None:
		name_result = animal+'_'+data_type+'_'+model_type #utils.get_name_with_time(animal+'_'+data_type+'_'+model_type)
		dir_result = os.path.join(dir_result_par, name_result)
		if os.path.isdir(dir_result):
			print('Directory exists')
			exit()
		else:
			os.mkdir(dir_result)

	np.save(os.path.join(dir_result, 'pred' + '.npy'), y_test_predicted_limb)
	np.save(os.path.join(dir_result, 'pred_norm_converted' + '.npy'), convert_arr)
	np.save(os.path.join(dir_result, 'gt' + '.npy'), y_test_limb)
	np.save(os.path.join(dir_result, 'gt_norm_converted' + '.npy'), y_test_convert)

	# plot
	plot_output_result(y_test_convert, convert_arr, [0,1], dir_result=dir_result, title='plot_output_result'+'_01')
	plot_output_result(y_test_convert, convert_arr, [2,3], dir_result=dir_result, title='plot_output_result'+'_23')
	plot_output_result(y_test_convert, convert_arr, [4,5], dir_result=dir_result, title='plot_output_result'+'_45')
	plot_output_result(y_test_convert, convert_arr, [6,7], dir_result=dir_result, title='plot_output_result'+'_67')


if __name__ == '__main__':
	### Change here!!!
	dir_result_par = r"E:\tmp\Linear" 
	dir_data_list = [r"E:\tmp\prepare_data\Animal1-G8_53950_1L_Redo\Animal1_likeli01"]
	animal_list = ["MouseA"]

	data_type_list = ['dup', 'intp', 'match'] # ['dup', 'intp', 'match'] 'matchdemean'
	model_type = 'linearlasso' #'linearlasso', 'kalman', 'kalmanlasso', 'kalmanlasso_lag_1'

	#######################################################################################################################
	for idx_case, (animal, dir_data) in enumerate(zip(animal_list, dir_data_list)):
		for data_type in data_type_list:
			print("################", animal, data_type)
			
			### Load data using torch dataset 
			# if data_type == 'ori':
			# 	name_neural = 'spks_z_sel'
			# 	name_coord = 'behav_coord_likeli_norm'
			# 	dataset_name = ["seq2seq_run_overlapO", "seq2seq_run_overlapX"]
			if data_type == 'dup':
				name_neural = 'spkszsel_dup'
				name_coord = 'behav_coord_likeli_norm'
				dataset_name = ["dupintp_run_overlapX", "dupintp_run_overlapX"] # !!! For linear regression, train is also overlapX.
				name_coord_ori = "behav_coord_likeli_ori" 
			elif data_type == 'intp': 
				name_neural = 'spkszsel_intp'
				name_coord = 'behav_coord_likeli_norm'
				dataset_name = ["dupintp_run_overlapX", "dupintp_run_overlapX"]
				name_coord_ori = "behav_coord_likeli_ori"
			elif data_type == 'match':
				name_neural = "spkszsel_match"
				name_coord = "behav_coord_likeli_match_norm"
				dataset_name = ["match_overlapX_run", "match_overlapX_run"]
				name_coord_ori = "behav_coord_likeli_match_ori"
			elif data_type == 'matchdemean':
				name_neural = "spkszsel_match" #"spkszsel_match"
				name_coord = "behav_coord_likeli_demean_match" #"behav_coord_likeli_demean_match"
				dataset_name = ["match_overlapX_run", "match_overlapX_run"]
				name_coord_ori = None

			neural = np.load(os.path.join(dir_data, name_neural+'.npy'))
			# coord = np.load(os.path.join(dir_data, name_coord+'.npy'))
			#print('neural: ', neural.shape)
			# # print('coord: ', coord.shape)

			seq_len = 5
			output_idx = [None]
			batch_size = neural.shape[1]

			# Get train-test indices
			if 'dup' in name_neural or 'intp' in name_neural:
				test_idx_cv = np.load(os.path.join(dir_data, 'test_idx_cv_coord.npy'), allow_pickle=True)
			else:
				test_idx_cv = np.load(os.path.join(dir_data, 'test_idx_cv.npy'), allow_pickle=True)
			test_idx = test_idx_cv[-1]
			train_idx = [0, test_idx[0]-1]
			# print('test_idx: ', test_idx)
			# print('train_idx: ', train_idx)

			# datasets 
			train_dataset = NNDataset(dir_data, name_neural, name_coord, seq_len, output_idx, dataset_name[0], train_idx)
			test_dataset = NNDataset(dir_data, name_neural, name_coord, seq_len, output_idx, dataset_name[1], test_idx)

			# dataloaders
			dataloader_train= DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn_custom)
			dataloader_test = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn_custom)

			# Get data by staking
			X_train = None
			y_train = None
			for idx_batch_train, data_train in enumerate(dataloader_train):
				inputs, labels, idx_neural_list, idx_coord_list = data_train
				#print('inputs: ', inputs.shape, 'labels: ', labels.shape, 'idx_neural_list: ', idx_neural_list, 'idx_coord_list: ', idx_coord_list)

				# Stack gt
				labels_reshape = None
				if labels.dim() == 3: labels_reshape = np.reshape(labels.data.cpu().numpy(), (labels.shape[0]*labels.shape[1], labels.shape[2]))
				elif labels.dim() == 2: labels_reshape = labels.data.cpu().numpy()
				if y_train is None: y_train = labels_reshape
				else: y_train = np.concatenate((y_train, labels_reshape))

				# Stack input
				inputs_reshape = None
				if labels.dim() == 3: inputs_reshape = np.reshape(inputs.data.cpu().numpy(), (inputs.shape[0]*inputs.shape[1], inputs.shape[2]))
				elif labels.dim() == 2: inputs_reshape = inputs.data.cpu().numpy()
				if X_train is None: X_train = inputs_reshape
				else: X_train = np.concatenate((X_train, inputs_reshape))
			#print('X_train: ', X_train.shape, ', y_train: ', y_train.shape) #X_train:  (12755, 3327) , y_train:  (12755, 8)

			X_test = None
			y_test = None
			for idx_batch_val, data_val in enumerate(dataloader_test):
				inputs, labels, idx_neural_list, idx_coord_list = data_val
				#print('inputs: ', inputs.shape, 'labels: ', labels.shape, 'idx_neural_list: ', idx_neural_list, 'idx_coord_list: ', idx_coord_list)

				# Stack gt
				labels_reshape = None
				if labels.dim() == 3: labels_reshape = np.reshape(labels.data.cpu().numpy(), (labels.shape[0]*labels.shape[1], labels.shape[2]))
				elif labels.dim() == 2: labels_reshape = labels.data.cpu().numpy()
				if y_test is None: y_test = labels_reshape
				else: y_test = np.concatenate((y_test, labels_reshape))

				# Stack input
				inputs_reshape = None
				if labels.dim() == 3: inputs_reshape = np.reshape(inputs.data.cpu().numpy(), (inputs.shape[0]*inputs.shape[1], inputs.shape[2]))
				elif labels.dim() == 2: inputs_reshape = inputs.data.cpu().numpy()
				if X_test is None: X_test = inputs_reshape
				else: X_test = np.concatenate((X_test, inputs_reshape))
			#print('X_test: ', X_test.shape, ', y_test: ', y_test.shape) #X_test:  (3220, 3327) , y_test:  (3220, 8) = (time, limb)

			# fig = plt.figure()
			# plt.plot(y_train[:,0])
			# plt.show()
			
			# print(y_train.shape, y_test.shape)
			# fig = plt.figure()
			# plt.plot(y_test[:,0], '-x')
			# plt.show()


			### run
			# if model_type == 'linear':
			# 	model = linear_model.LinearRegression()
			# 	model.fit(X_train, y_train)
			# 	# # X_train: [n_samples, n_features] = [time, neurons]
			# 	# # y_train: [n_samples, n_outputs] = [time, limb]

			# 	y_test_predicted = model.predict(X_test) #[time, limb]
			# 	y_test_predicted_limb = np.transpose(y_test_predicted) #[limb, time]
			# 	#print('y_test_predicted: ', y_test_predicted.shape)

			# 	y_test_limb = np.transpose(y_test) #[limb, time]
			# 	saver(dir_data, name_coord_ori, y_test_predicted_limb, y_test_limb, dir_result_par, animal, data_type, model_type)
				
			if model_type == 'linearlasso':
				# # Multiple random alpha cases version
				# num_run = 500
				# alpha_range = [pow(10,-3), pow(10,-2)] #[pow(10,-3), 1]

				# Single alpha case version 
				num_run = 1
				alpha_range = [0.003] #[pow(10,-3), 1]

				y_test_limb = np.transpose(y_test)

				if len(alpha_range) == 2:
					dir_result = os.path.join(dir_result_par, animal+'_'+data_type+'_'+model_type+'_'+str(alpha_range[0])+'-'+str(alpha_range[1])+'_'+str(num_run)) #dir_result_par, animal+'_'+data_type+'_'+model_type+'_alpha'+str(best_alpha))
				elif len(alpha_range) == 1:
					dir_result = os.path.join(dir_result_par, animal+'_'+data_type+'_'+model_type+'_'+str(alpha_range[0])+'_'+str(num_run)) #dir_result_par, animal+'_'+data_type+'_'+model_type+'_alpha'+str(best_alpha))

				if os.path.isdir(dir_result):
					print('Directory exists')
					exit()
				else:
					os.mkdir(dir_result)
				path_log = os.path.join(dir_result, 'log.txt')
				with LoggingPrinter(path_log):
					best_r2 = -999
					best_alpha = -999
					best_pred, best_pred_normconvert, best_gt, best_gt_normconvert = None, None, None, None
					if name_coord_ori is not None: ori_limb = np.load(os.path.join(dir_data, name_coord_ori+'.npy')) #[limb, time]
					for i in range(num_run):#1000:
						# Random sample alpha
						if len(alpha_range) == 2: alpha_sampled = uniform(alpha_range[0], alpha_range[1]) #uniform(pow(10,-4), pow(10,4))
						elif len(alpha_range) == 1: alpha_sampled = alpha_range[0]

						model = linear_model.Lasso(alpha=alpha_sampled)
						model.fit(X_train, y_train)
						# # X_train: [n_samples, n_features] = [time, neurons]
						# # y_train: [n_samples, n_outputs] = [time, limb]

						y_test_predicted = model.predict(X_test) #[time, limb]
						y_test_predicted_limb = np.transpose(y_test_predicted) #[limb, time]
						#print('y_test_predicted: ', y_test_predicted.shape)
						
						# Convert y_test
						if data_type == 'matchdemean':
							convert_arr = y_test_predicted_limb
							y_test_convert = y_test_limb
						else:
							# Convert y_test_predicted
							convert_arr = convert_norm_coord_func(y_test_predicted_limb,ori_limb) #[limb, time]
							#print(convert_arr.shape)
							y_test_convert = convert_norm_coord_func(y_test_limb,ori_limb) #[limb, time]

						# Get r2
						r2 = r2_score(np.transpose(y_test_convert), np.transpose(convert_arr))
						if r2 > best_r2:
							best_r2 = r2
							best_alpha = alpha_sampled
							best_pred = y_test_predicted_limb
							#best_pred_normconvert = convert_arr
							#best_gt = y_test_limb
							#best_gt_normconvert = y_test_convert
						
						print(i, '/', num_run, ', alpha=', alpha_sampled, ', r2=', r2)

					print('best alpha: ', best_alpha)
					print('best r2: ', best_r2)

					saver(dir_data=dir_data, name_coord_ori=name_coord_ori, y_test_predicted_limb=best_pred, y_test_limb=y_test_limb, 
		   					dir_result_par=None, animal=animal, data_type=data_type, model_type=model_type, dir_result=dir_result)

			elif model_type == 'kalman':
				model_kf = KalmanFilterDecoder(C_Q=2, C=3)
				model_kf.fit(X_train, y_train)
				y_test_predicted = model_kf.predict(X_test, y_test) 
				y_test_predicted_limb = np.transpose(y_test_predicted)
				print(y_test_predicted.shape)

				y_test_limb = np.transpose(y_test)
				saver(dir_data, name_coord_ori, y_test_predicted_limb, y_test_limb, dir_result_par, animal, data_type, model_type)
			
			elif model_type == 'kalmanlasso':
				alpha = 0.00446780992144115 
				model_kf = KalmanFilterDecoder(lassoalpha=alpha)
				model_kf.fit(X_train, y_train)
				y_test_predicted = model_kf.predict(X_test, y_test) 
				y_test_predicted_limb = np.transpose(y_test_predicted)
				print(y_test_predicted.shape)

				y_test_limb = np.transpose(y_test)

				name_result = animal+'_'+data_type+'_'+model_type+'_alpha-'+str(alpha) #utils.get_name_with_time(animal+'_'+data_type+'_'+model_type)
				dir_result = os.path.join(dir_result_par, name_result)
				if os.path.isdir(dir_result):
					print('Directory exists')
					exit()
				else:
					os.mkdir(dir_result)
				saver(dir_data, name_coord_ori, y_test_predicted_limb, y_test_limb, None, animal, data_type, model_type, dir_result)

			elif model_type == 'kalmanlasso_lag_1':
				n_train = X_train.shape[0]
				n_test = X_test.shape[0]

				X_train = X_train[0:n_train-1, :]
				y_train = y_train[1:, :]
				X_test = X_test[0:n_test-1, :]
				y_test = y_test[1:, :]

				alpha = 0.001
				model_kf = KalmanFilterDecoder(lassoalpha=alpha)

				model_kf.fit(X_train, y_train)
				y_test_predicted = model_kf.predict(X_test, y_test) 
				y_test_predicted_limb = np.transpose(y_test_predicted)
				print(y_test_predicted.shape)

				y_test_limb = np.transpose(y_test)
				saver(dir_data, name_coord_ori, y_test_predicted_limb, y_test_limb, dir_result_par, animal, data_type, model_type)

	


















				



		
		

		
		

			
