from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def score_r2(pred_stack, gt_stack):
	'''
	pred_stack, gt_stack: (time, coord)
	'''
	# Check shape
	if pred_stack.shape != gt_stack.shape:
		print("pred_stack.shape != gt_stack.shape", pred_stack.shape, gt_stack.shape)
		exit()
	if pred_stack.shape[1] > 10 or gt_stack.shape[1] > 10:
		print("Somthing is weried in shape for coord.", pred_stack.shape, gt_stack.shape)
		exit()

	r_square = []
	if gt_stack.ndim == 2:
		for i in range(gt_stack.shape[1]):
			r = r2_score(gt_stack[:,i], pred_stack[:,i])
			r_square.append(r)
	elif gt_stack.ndim == 1:
		r_square = r2_score(gt_stack, pred_stack)

	return r_square

def score_r2_manual(pred_stack, gt_stack):
	'''
	pred_stack, gt_stack: (time, coord)
	'''
	# Check shape
	if pred_stack.shape != gt_stack.shape:
		print("pred_stack.shape != gt_stack.shape", pred_stack.shape, gt_stack.shape)
		exit()
	if pred_stack.shape[1] > 10 or gt_stack.shape[1] > 10:
		print("Somthing is weried in shape for coord.", pred_stack.shape, gt_stack.shape)
		exit()

	if gt_stack.ndim == 2:
		r_square = []
		for i in range(gt_stack.shape[1]):
			gt_limb = gt_stack[:,i]
			pred_limb = pred_stack[:,i]
			labels_mean = np.mean(gt_limb)
			ss_tot = np.sum((gt_limb-labels_mean)**2)
			ss_res = np.sum((gt_limb-pred_limb)**2)
			r = 1 - ss_res / ss_tot
			r_square.append(r)
	elif gt_stack.ndim == 1:
		gt_limb = gt_stack[:,i]
		pred_limb = pred_stack[:,i]
		labels_mean = np.mean(gt_limb)
		ss_tot = np.sum((gt_limb-labels_mean)**2)
		ss_res = np.sum((gt_limb-pred_limb)**2)
		r_square = 1 - ss_res / ss_tot

	return r_square

def score_rmse(pred_stack, gt_stack):
	'''
	pred_stack, gt_stack: (time, coord)
	'''

	# Check shape
	if pred_stack.shape != gt_stack.shape:
		print("pred_stack.shape != gt_stack.shape", pred_stack.shape, gt_stack.shape)
		exit()
	if pred_stack.shape[1] > 10 or gt_stack.shape[1] > 10:
		print("Somthing is weried in shape for coord.", pred_stack.shape, gt_stack.shape)
		exit()

	rmse = []
	if gt_stack.ndim == 2:
		for i in range(gt_stack.shape[1]):
			rmse.append(mean_squared_error(gt_stack[:,i], pred_stack[:,i], squared=False))
	elif gt_stack.ndim == 1:
		rmse = mean_squared_error(gt_stack, pred_stack, squared=False)

	return rmse

def score_mse(pred_stack, gt_stack):
	'''
	pred_stack, gt_stack: (time, coord)
	'''
	# Check shape
	if pred_stack.shape != gt_stack.shape:
		print("pred_stack.shape != gt_stack.shape", pred_stack.shape, gt_stack.shape)
		exit()
	if pred_stack.shape[1] > 10 or gt_stack.shape[1] > 10:
		print("Somthing is weried in shape for coord.", pred_stack.shape, gt_stack.shape)
		exit()

	mse = []
	if gt_stack.ndim == 2:
		for i in range(gt_stack.shape[1]):
			mse.append(mean_squared_error(gt_stack[:,i], pred_stack[:,i], squared=True))
	elif gt_stack.ndim == 1:
		mse = mean_squared_error(gt_stack, pred_stack, squared=True)

	return mse