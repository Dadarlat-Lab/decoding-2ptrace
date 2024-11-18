import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from LoggingPrinter import LoggingPrinter
import plot
import utils
import random
import mat73
from scipy.signal import butter,filtfilt
from copy import deepcopy
from suite2p.extraction import dcnv

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def load_suite2p(dir_plane):
	f_output = np.load(os.path.join(dir_plane, "F.npy"), allow_pickle=True)
	fneu_output = np.load(os.path.join(dir_plane, "Fneu.npy"), allow_pickle=True)
	iscell_output = np.load(os.path.join(dir_plane, "iscell.npy"), allow_pickle=True)
	spks_output = np.load(os.path.join(dir_plane, "spks.npy"), allow_pickle=True)
	ops_output = np.load(os.path.join(dir_plane, "ops.npy"), allow_pickle=True).item()
	stat_output = np.load(os.path.join(dir_plane, "stat.npy"), allow_pickle=True)
	print("# Suite2p loaded")
	print('f_output: ', f_output.shape) #(4616, 9420) = (neurons, time)
	print('fneu_output: ', fneu_output.shape)
	print('iscell_output: ', iscell_output.shape)
	print('spks_output: ', spks_output.shape)
	print('stat_output: ', stat_output.shape)
	# iscell_cell = np.array([iscell[i] for i,x in enumerate(iscell[:,1]) if x>=0.4])
	# print(iscell_cell)
	# print(np.min(iscell_cell[:,0]))
	# print(np.min(iscell_cell[:,1]))

	# ops
	ops = {'Lx':ops_output['Lx'], 'Ly':ops_output['Ly']}

	# KEEPING ONLY CELLS CLASSIFIED BY SUITE2P
	f = np.array([f_output[i] for i,x in enumerate(iscell_output[:,0]) if x==1])
	fneu = np.array([fneu_output[i] for i,x in enumerate(iscell_output[:,0]) if x==1])
	spks = np.array([spks_output[i] for i,x in enumerate(iscell_output[:,0]) if x==1])
	stat = np.array([stat_output[i] for i,x in enumerate(iscell_output[:,0]) if x==1])
	print("# Neuron selected")
	print('f: ', f.shape)
	print('fneu: ', fneu.shape)
	print('spks: ', spks.shape)
	print('stat: ', stat.shape)
	# f_prob = np.array([f[i] for i,x in enumerate(iscell[:,1]) if x>=0.5])
	# print((f_cell==f_prob).all())

	# non-cell signals
	f_noncell = np.array([f_output[i] for i,x in enumerate(iscell_output[:,0]) if x==0])
	fneu_noncell = np.array([fneu_output[i] for i,x in enumerate(iscell_output[:,0]) if x==0])
	spks_noncell = np.array([spks_output[i] for i,x in enumerate(iscell_output[:,0]) if x==0])
	stat_noncell = np.array([stat_output[i] for i,x in enumerate(iscell_output[:,0]) if x==0])
	print("# None cell")
	print('f_noncell: ', f_noncell.shape)
	print('fneu_noncell: ', fneu_noncell.shape)
	print('spks_noncell: ', spks_noncell.shape)
	print('stat_noncell: ', stat_noncell.shape)

	if not(f.shape[0]==fneu.shape[0]==spks.shape[0]==stat.shape[0]):
		print("Neuron number does not match!!!")
		exit()

	return f, fneu, spks, stat, ops, f_noncell, fneu_noncell, spks_noncell, stat_noncell

def preprocess_low_likelihood(behavior_coord, likeli, likeli_thrd):
	num_cont = 30*3

	behavior_coord_likeli_processed = np.copy(behavior_coord)
	for idx_limb in range(4):
		idx_limb1 = idx_limb*2
		idx_limb2 = idx_limb*2+1

		idx_low = np.where(likeli[idx_limb,:]<=likeli_thrd)[0]
		#print('idx_limb: ', idx_limb, ', idx_low: ', len(idx_low), idx_low[:20], idx_low[-10:])

		# continuous werid points
		i = 0
		idx_low_cont_list = []
		flag_cont = False
		while (i < len(idx_low)):
			if i != len(idx_low)-1:
				dif = idx_low[i+1]-idx_low[i]
				#print('i:', i, ', idx_low[i]:', idx_low[i], ', idx_low[i+1]:', idx_low[i+1], ', dif=', dif, ', len(idx_low)=', len(idx_low))
				if dif>1:
					if flag_cont == False: #Single point => get the average of two adjacent points
						avg_coord_limb1 = (behavior_coord_likeli_processed[idx_limb1, idx_low[i]-1]+behavior_coord_likeli_processed[idx_limb1, idx_low[i]+1])/2.0
						avg_coord_limb2 = (behavior_coord_likeli_processed[idx_limb2, idx_low[i]-1]+behavior_coord_likeli_processed[idx_limb2, idx_low[i]+1])/2.0
						behavior_coord_likeli_processed[idx_limb1, idx_low[i]] = avg_coord_limb1
						behavior_coord_likeli_processed[idx_limb2, idx_low[i]] = avg_coord_limb2
						#print('Upadated', i, idx_low[i])
					else:
						idx_low_cont_list.append(idx_low[i])
						# print('stacked, ', i, idx_low[i])
						# print('interpol!!!, len=', len(idx_low_cont_list), ', ', idx_low_cont_list)                       
						if len(idx_low_cont_list) > num_cont:
							#print('len(idx_low_cont_list)>num_cont(='+str(num_cont)+'), so get the first previous value.')
							limb1_interp = np.array([behavior_coord_likeli_processed[idx_limb1, idx_low_cont_list[0]-1] for x in range(len(idx_low_cont_list))])
							limb2_interp = np.array([behavior_coord_likeli_processed[idx_limb2, idx_low_cont_list[0]-1] for x in range(len(idx_low_cont_list))])
						else:
							limb1_interp = np.interp(x=idx_low_cont_list, xp=[idx_low_cont_list[0]-1, idx_low_cont_list[-1]+1], \
									fp=[behavior_coord_likeli_processed[idx_limb1, idx_low_cont_list[0]-1], \
										behavior_coord_likeli_processed[idx_limb1, idx_low_cont_list[-1]+1]])
							limb2_interp = np.interp(x=idx_low_cont_list, xp=[idx_low_cont_list[0]-1, idx_low_cont_list[-1]+1], \
									fp=[behavior_coord_likeli_processed[idx_limb2, idx_low_cont_list[0]-1], \
										behavior_coord_likeli_processed[idx_limb2, idx_low_cont_list[-1]+1]])
							# print('xp: ', [idx_low_cont_list[0]-1, idx_low_cont_list[-1]+1])
							# print('limb1_interp: ', limb1_interp, 'fp: ', [behavior_coord_likeli_processed[idx_limb1, idx_low_cont_list[0]-1], \
							#     behavior_coord_likeli_processed[idx_limb1, idx_low_cont_list[-1]+1]])
							# print('limb2_interp: ', limb2_interp, 'fp: ', [behavior_coord_likeli_processed[idx_limb2, idx_low_cont_list[0]-1], \
							#     behavior_coord_likeli_processed[idx_limb2, idx_low_cont_list[-1]+1]])

						# print('limb1_interp: ', limb1_interp.shape)
						# print('limb2_interp: ', limb2_interp.shape)
						behavior_coord_likeli_processed[idx_limb1, idx_low_cont_list] = limb1_interp
						behavior_coord_likeli_processed[idx_limb2, idx_low_cont_list] = limb2_interp

						flag_cont = False
						idx_low_cont_list = []
				elif dif==1:
					idx_low_cont_list.append(idx_low[i])
					flag_cont = True
					#print('stacked, ', i, idx_low[i])
			else: #Last frame
				if idx_low_cont_list == []: #Single point
					if idx_low[i]+1 == behavior_coord_likeli_processed.shape[1]: #when the final point is low likelihood
						#print('the final point is low likelihood, so get the first previous value.')
						limb1_interp = np.array([behavior_coord_likeli_processed[idx_limb1, idx_low[i]-1] for x in range(len(idx_low_cont_list))])
						limb2_interp = np.array([behavior_coord_likeli_processed[idx_limb2, idx_low[i]-1] for x in range(len(idx_low_cont_list))])
					else: #when there are left frames with high likelihood
						avg_coord_limb1 = (behavior_coord_likeli_processed[idx_limb1, idx_low[i]-1]+behavior_coord_likeli_processed[idx_limb1, idx_low[i]+1])/2.0
						avg_coord_limb2 = (behavior_coord_likeli_processed[idx_limb2, idx_low[i]-1]+behavior_coord_likeli_processed[idx_limb2, idx_low[i]+1])/2.0
						behavior_coord_likeli_processed[idx_limb1, idx_low[i]] = avg_coord_limb1
						behavior_coord_likeli_processed[idx_limb2, idx_low[i]] = avg_coord_limb2
						#print('Upadated', i, idx_low[i])
				else:
					if idx_low_cont_list[-1]+1 == behavior_coord_likeli_processed.shape[1]: #when the final point is low likelihood
						#print('the final point is low likelihood, so get the first previous value.')
						limb1_interp = np.array([behavior_coord_likeli_processed[idx_limb1, idx_low_cont_list[0]-1] for x in range(len(idx_low_cont_list))])
						limb2_interp = np.array([behavior_coord_likeli_processed[idx_limb2, idx_low_cont_list[0]-1] for x in range(len(idx_low_cont_list))])
					else: #when there are left frames with high likelihood
						# Do the same interp
						idx_low_cont_list.append(idx_low[i])
						#print('stacked, ', i, idx_low[i])
						#print('interpol!!!, len=', len(idx_low_cont_list), ', ', idx_low_cont_list)                         
						if len(idx_low_cont_list) > num_cont:
							#print('len(idx_low_cont_list)>num_cont(='+str(num_cont)+'), so get the first previous value.')
							limb1_interp = np.array([behavior_coord_likeli_processed[idx_limb1, idx_low_cont_list[0]-1] for x in range(len(idx_low_cont_list))])
							limb2_interp = np.array([behavior_coord_likeli_processed[idx_limb2, idx_low_cont_list[0]-1] for x in range(len(idx_low_cont_list))])
						else:
							limb1_interp = np.interp(x=idx_low_cont_list, xp=[idx_low_cont_list[0]-1, idx_low_cont_list[-1]+1], \
									fp=[behavior_coord_likeli_processed[idx_limb1, idx_low_cont_list[0]-1], \
										behavior_coord_likeli_processed[idx_limb1, idx_low_cont_list[-1]+1]])
							limb2_interp = np.interp(x=idx_low_cont_list, xp=[idx_low_cont_list[0]-1, idx_low_cont_list[-1]+1], \
									fp=[behavior_coord_likeli_processed[idx_limb2, idx_low_cont_list[0]-1], \
										behavior_coord_likeli_processed[idx_limb2, idx_low_cont_list[-1]+1]])
							# print('xp: ', [idx_low_cont_list[0]-1, idx_low_cont_list[-1]+1])
							# print('limb1_interp: ', limb1_interp, 'fp: ', [behavior_coord_likeli_processed[idx_limb1, idx_low_cont_list[0]-1], \
							#     behavior_coord_likeli_processed[idx_limb1, idx_low_cont_list[-1]+1]])
							# print('limb2_interp: ', limb2_interp, 'fp: ', [behavior_coord_likeli_processed[idx_limb2, idx_low_cont_list[0]-1], \
							#     behavior_coord_likeli_processed[idx_limb2, idx_low_cont_list[-1]+1]])
						# print('limb1_interp: ', limb1_interp.shape)
						# print('limb2_interp: ', limb2_interp.shape)                                
					
					behavior_coord_likeli_processed[idx_limb1, idx_low_cont_list] = limb1_interp
					behavior_coord_likeli_processed[idx_limb2, idx_low_cont_list] = limb2_interp

			i += 1

	return behavior_coord_likeli_processed

def smooth(data, window_size):
	if window_size % 2 == 0:
		window_size += 1  # Ensure window size is odd

	half_window = (window_size - 1) // 2
	smoothed_data = np.zeros_like(data, dtype=float)

	for i in range(len(data)):
		if i < half_window: #left edge
			start = max(0, i - half_window)
			end = i*2+1
		elif i > len(data)-half_window-1: #right edge
			start = i-half_window
			end = len(data) 
		else:
			start = i-half_window
			end = i+half_window+1

		#print(i, list(range(start, end)))
		if (start<0) or (start>len(data)-1) or (end-1<0) or (end-1>len(data)-1):
			print("Something's wrong with smooth.")
			print(i, list(range(start, end)))
			exit()
		smoothed_data[i] = np.mean(data[start:end])

	return smoothed_data

def prepare_data():
	'''
	dir_result_parent: Directory to save outputs
	name_result: Directory name to save outputs
	dir_data_working: Directory of the raw data
	dir_site2p_plane: Direcotory of suite2p plane
	path_mat: Path of mat file from scanbox
	path_coord_limb1: Path of csv file for limb1 from Deeplabcut
	path_coord_limb2: Path of csv file for limb2 from Deeplabcut
	path_vid_limb1: Path of video avi file for limb1
	path_vid_limb2: Path of video avi file for limb2
	likeli_thrd: Threshold for the likelihood from Deeplabcut
	limb_to_crop_str: Which limb coordinate to use for run/stand cropping
	flag_calium: Whether to save raw calcium images or not
	'''
	### Change it !!!
	dir_save_par = r"E:\tmp\prepare_data" # Directory to save the outputs 
	animal_name = 'Animal1-G8_53950_1L_Redo' #'Animal1-G8_53950_1L_Redo', "Animal4-G10_55902_1R", "Animal9-G12_55954_1L"
	##################################################################################################################################################

	### Change here too if needed
	dir_result_parent = os.path.join(dir_save_par, animal_name)
	if os.path.isdir(dir_result_parent) is False: os.mkdir(dir_result_parent)

	if animal_name == 'Animal1-G8_53950_1L_Redo':
		#53950_1L_Redo, animal #1
		name_result = 'Animal1_likeli01' 
		dir_data_working = r"E:\SeungbinPark\Decoding_2024-11-04\Data_Working\Data\Animal1-G8_53950_1L_Redo"
		dir_suite2p_plane = os.path.join(dir_data_working, r"suite2p_set\suite2p\plane0") 
		path_mat = os.path.join(dir_data_working, "53950_1L_000_513"+".mat")
		path_coord_limb1 = os.path.join(dir_data_working, "LIMB1_53950_1L_2022-05-13_1DLC_resnet50_53950_1L_m1_limb1May16shuffle1_220000" + ".csv")
		path_coord_limb2 = os.path.join(dir_data_working, "LIMB2_53950_1L_2022-05-13_1DLC_resnet50_53950_2R_m1_limb2May16shuffle1_250000"+".csv")
		path_vid_limb1 = os.path.join(dir_data_working, "LIMB1_53950_1L_2022-05-13_1"+".avi")
		path_vid_limb2 = os.path.join(dir_data_working, "LIMB2_53950_1L_2022-05-13_1"+".avi")
		likeli_thrd = 0.1
		limb_to_crop_str = 'LFX'
		flag_calcium = False #True
	elif animal_name == 'Animal4-G10_55902_1R':
		# Animal4-G10_55902_1R
		name_result = 'Animal4_likeli01' 
		dir_data_working = r"E:\SeungbinPark\Decoding_2024-11-04\Data_Working\Data\Animal4-G10_55902_1R"
		dir_suite2p_plane = os.path.join(dir_data_working, r"suite2p\plane0") 
		path_mat = os.path.join(dir_data_working, "55902_1R_000_627"+".mat")
		path_coord_limb1 = os.path.join(dir_data_working, "LIMB1_55902_1R_2022-06-27_1DLC_resnet50_G10_55902_1R_LIMB1May24shuffle1_200000" + ".csv")
		path_coord_limb2 = os.path.join(dir_data_working, "LIMB2_55902_1R_2022-06-27_1DLC_resnet50_G10_55902_1R_LIMB2May26shuffle1_200000"+".csv")
		path_vid_limb1 = os.path.join(dir_data_working, "LIMB1_55902_1R_2022-06-27_1"+".avi")
		path_vid_limb2 = os.path.join(dir_data_working, "LIMB2_55902_1R_2022-06-27_1"+".avi")
		likeli_thrd = 0.1
		limb_to_crop_str = 'LFX'
		flag_calcium = False
	elif animal_name == 'Animal9-G12_55954_1L':
		# Animal9-G12_55954_1L
		name_result = 'Animal9_likeli095'  
		dir_data_working = r"E:\SeungbinPark\Decoding_2024-11-04\Data_Working\Data\Animal9-G12_55954_1L"
		dir_suite2p_plane = os.path.join(dir_data_working, r"suite2p\plane0") 
		path_mat = os.path.join(dir_data_working, "55954_1L_000_627"+".mat")
		path_coord_limb1 = os.path.join(dir_data_working, "LIMB1_55954_1L_2022-06-27_1DLC_resnet50_G12_55954_1L_LIMB1Jun1shuffle1_200000" + ".csv")
		path_coord_limb2 = os.path.join(dir_data_working, "LIMB2_55954_1L_2022-06-27_1DLC_resnet50_G12_55954_1L_LIMB2Jun1shuffle1_200000"+".csv")
		path_vid_limb1 = os.path.join(dir_data_working, "LIMB1_55954_1L_2022-06-27_1"+".avi")
		path_vid_limb2 = os.path.join(dir_data_working, "LIMB2_55954_1L_2022-06-27_1"+".avi")
		likeli_thrd = 0.95 #None, 0.01
		limb_to_crop_str = 'LFX'
		flag_calcium = False
	##################################################################################################################################################

	### Create directory to save outcomes
	dir_result = os.path.join(dir_result_parent, name_result)
	if os.path.isdir(dir_result):
		print('dir_result already exist.')
		#print('Save anyway')
		exit()
	else:
		os.mkdir(dir_result)

	### log
	path_log = os.path.join(dir_result, 'log.txt')
	with LoggingPrinter(path_log):
		print("### Arguments ############################")
		# print('flag_crop_stil: ', flag_crop_stil)
		# print('flag_displacement: ', flag_displacement)
		print('animal_name: ', animal_name)
		print('dir_suite2p_plane: ', dir_suite2p_plane)
		print('path_mat: ', path_mat)
		print('path_coord_limb1: ', path_coord_limb1)
		print('path_coord_limb2: ', path_coord_limb2)
		print('path_vid_limb1: ', path_vid_limb1)
		print('path_vid_limb2: ', path_vid_limb2)
		print('likeli_thrd: ', likeli_thrd)
		print('limb_to_crop_str: ', limb_to_crop_str)
		print('flag_calcium: ', flag_calcium)
		print("#########################################")

		### 0. Load Scanbox
		print("### 0. Load Scanbox")

		# Load Scanbox
		mat = sio.loadmat(path_mat, squeeze_me = True, struct_as_record=False)
		frame = mat['info'].__dict__['frame'] ###one-based indexing!!!!!
		frame = np.array([x-1 for x in frame]) #change to zero-based indexing
		eventid = mat['info'].__dict__['event_id']  
		print('frame: ', frame.shape, frame[:20], '...', frame[-20:])
		print('eventid: ', eventid.shape, eventid[:20], '...', eventid[-20:], set(eventid))

		# Save raw calcium images
		if flag_calcium:
			calcium = mat73.loadmat(os.path.join(dir_data_working, 'calcium_imaging.mat'))['x']
			#calcium = np.load(os.path.join(dir_data_working, 'calcium_imaging.npy'))
			print('calcium: ', calcium.shape)

		# plt.figure()
		# plt.plot(frame, '-o')
		# plt.title('frame')
		# plt.show()
		# plt.close()

		# plt.figure()
		# plt.plot(eventid, '-o')
		# plt.title('eventid')
		# plt.show()
		# plt.close()

		# Check eventid - depends on experiment setup
		ttl_on = 4
		ttl_off = 8
		for i in range(len(eventid)):
			if i%2==0:
				if eventid[i] != ttl_on:
					print(i)
					exit()
			elif i%2==1:
				if eventid[i] != ttl_off:
					print(i)
					exit()

		### 1. Load neural data from Suite2p #################################################################################################################################
		print("### 1. Load neural data from Suite2p")
		
		# Load suite2p
		f, fneu, spks, stat, ops, f_noncell, fneu_noncell, spks_noncell, stat_noncell = load_suite2p(dir_suite2p_plane)

		ops = np.load(os.path.join(dir_suite2p_plane, "ops.npy"), allow_pickle=True).item()
		xoff = ops['xoff']
		yoff = ops['yoff']
		corrxy = ops['corrXY']

		# Save
		np.save(os.path.join(dir_result, 'stat.npy'), stat)
		np.save(os.path.join(dir_result, 'ops_data.npy'), ops)
		np.save(os.path.join(dir_result, 'stat_noncell.npy'), stat_noncell)
		print("# Save")
		print('stat: ', stat.shape)
		print('ops_data: ', ops)
		print('stat_noncell: ', stat_noncell.shape)

		# Check last neural frame from scanbox mat frame and suite2p outputs length
		print("# Check last neural frame from scanbox mat frame and suite2p outputs length")
		if frame[-1]+1 != spks.shape[1]:
			print("frame[-1]+1 != spks.shape[1]", frame[-1]+1, spks.shape[1])
			dif = spks.shape[1]-(frame[-1]+1)
			f = f[:, dif:]
			fneu = fneu[:, dif:]
			spks = spks[:, dif:]
			print('f: ', f.shape)
			print('fneu: ', fneu.shape)
			print('spks: ', spks.shape)

			if flag_calcium:
				calcium = calcium[:, :, dif:]
				print('calcium: ', calcium.shape)
		else:
			print("frame[-1]+1 = spks.shape[1]")    
			
		# f-fneu
		ffneu_ori = f.copy() - 0.7*fneu.copy()
		ffneu_z = np.zeros(ffneu_ori.shape)
		for i in range(ffneu_ori.shape[0]):
			sig = ffneu_ori[i,:]
			ffneu_z[i,:] = (sig-np.average(sig))/np.std(sig)
		print('ffneu_ori: ', ffneu_ori.shape, np.min(ffneu_ori), np.max(ffneu_ori))
		print('ffneu_z: ', ffneu_z.shape, np.min(ffneu_z), np.max(ffneu_z))

		# fneu-f
		fneuf_ori = fneu.copy() - 0.7*f.copy()
		fneuf_z = np.zeros(fneuf_ori.shape)
		for i in range(fneuf_ori.shape[0]):
			sig = fneuf_ori[i,:]
			fneuf_z[i,:] = (sig-np.average(sig))/np.std(sig)
		print('fneuf_ori: ', fneuf_ori.shape, np.min(fneuf_ori), np.max(fneuf_ori))
		print('fneuf_z: ', fneuf_z.shape, np.min(fneuf_z), np.max(fneuf_z))

		# ffneu noncell
		ffneu_noncell_ori = f_noncell.copy() - 0.7*fneu_noncell.copy()
		ffneu_noncell_z = np.zeros(ffneu_noncell_ori.shape)
		for i in range(ffneu_noncell_ori.shape[0]):
			sig = ffneu_noncell_ori[i,:]
			ffneu_noncell_z[i,:] = (sig-np.average(sig))/np.std(sig)
		print('ffneu_noncell_ori: ', ffneu_noncell_ori.shape, np.min(ffneu_noncell_ori), np.max(ffneu_noncell_ori))
		print('ffneu_noncell_z: ', ffneu_noncell_z.shape, np.min(ffneu_noncell_z), np.max(ffneu_noncell_z))

		# neu
		neu_ori = fneu.copy()
		neu_z = np.zeros(neu_ori.shape)
		for i in range(neu_ori.shape[0]):
			sig = neu_ori[i,:]
			neu_z[i,:] = (sig-np.average(sig))/np.std(sig)
		print('neu_ori: ', neu_ori.shape, np.min(neu_ori), np.max(neu_ori))
		print('neu_z: ', neu_z.shape, np.min(neu_z), np.max(neu_z))

		# neu noncell
		neu_noncell_ori = fneu_noncell.copy()
		neu_noncell_z = np.zeros(neu_noncell_ori.shape)
		for i in range(neu_noncell_ori.shape[0]):
			sig = neu_noncell_ori[i,:]
			neu_noncell_z[i,:] = (sig-np.average(sig))/np.std(sig)
		print('neu_noncell_ori: ', neu_noncell_ori.shape, np.min(neu_noncell_ori), np.max(neu_noncell_ori))
		print('neu_noncell_z: ', neu_noncell_z.shape, np.min(neu_noncell_z), np.max(neu_noncell_z))

		# f
		f_ori = f.copy()
		f_z = np.zeros(f_ori.shape)
		for i in range(f_ori.shape[0]):
			sig = f_ori[i,:]
			if np.std(sig)==0: f_z[i,:] = np.array([0 for _ in range(len(sig))])
			else: f_z[i,:] = (sig-np.average(sig))/np.std(sig)
		print('f_ori: ', f_ori.shape, np.min(f_ori), np.max(f_ori))
		print('f_z: ', f_z.shape, np.min(f_z), np.max(f_z))

		# neudeconv
		baseline = dcnv.preprocess(F=neu_ori, fs=7.8, baseline='maximin', win_baseline=60.0, sig_baseline=10.0)
		neudeconv_ori = dcnv.oasis(F=baseline, tau=1.5, fs=7.8, batch_size=500)
		neudeconv_z = np.zeros(neudeconv_ori.shape)
		for i in range(neudeconv_ori.shape[0]):
			sig = neudeconv_ori[i,:]
			neudeconv_z[i,:] = (sig-np.average(sig))/np.std(sig)
		print('neudeconv_ori: ', neudeconv_ori.shape, np.min(neudeconv_ori), np.max(neudeconv_ori))
		print('neudeconv_z: ', neudeconv_z.shape, np.min(neudeconv_z), np.max(neudeconv_z))

		# fneufdeconv
		baseline = dcnv.preprocess(F=fneuf_ori, fs=7.8, baseline='maximin', win_baseline=60.0, sig_baseline=10.0)
		fneufdeconv_ori = dcnv.oasis(F=baseline, tau=1.5, fs=7.8, batch_size=500)
		fneufdeconv_z = np.zeros(fneufdeconv_ori.shape)
		for i in range(fneufdeconv_ori.shape[0]):
			sig = fneufdeconv_ori[i,:]
			fneufdeconv_z[i,:] = (sig-np.average(sig))/np.std(sig)
		print('fneufdeconv_ori: ', fneufdeconv_ori.shape, np.min(fneufdeconv_ori), np.max(fneufdeconv_ori))
		print('fneufdeconv_z: ', fneufdeconv_z.shape, np.min(fneufdeconv_z), np.max(fneufdeconv_z))

		# spks
		spks_ori = spks.copy()
		spks_z = np.zeros(spks_ori.shape)
		for i in range(spks_ori.shape[0]):
			sig = spks_ori[i,:]
			spks_z[i,:] = (sig-np.average(sig))/np.std(sig)
		print('spks_ori: ', spks_ori.shape, np.min(spks_ori), np.max(spks_ori))
		print('spks_z: ', spks_z.shape, np.min(spks_z), np.max(spks_z))

		# spks noncell
		spks_noncell_ori = spks_noncell.copy()
		spks_noncell_z = np.zeros(spks_noncell_ori.shape)
		for i in range(spks_noncell_ori.shape[0]):
			sig = spks_noncell_ori[i,:]
			spks_noncell_z[i,:] = (sig-np.average(sig))/np.std(sig)
		print('spks_noncell_ori: ', spks_noncell_ori.shape, np.min(spks_noncell_ori), np.max(spks_noncell_ori))
		print('spks_noncell_z: ', spks_noncell_z.shape, np.min(spks_noncell_z), np.max(spks_noncell_z))
			
		####################################################################################################################################################

		### 2. Load behavior coordinates data from Deeplabcut(videos) ##########################################################################################
		print("### 2. Load behavior coordinates data from Deeplabcut(videos)")
		
		coord_right, likeli_right = utils.load_coord_csv(path_coord_limb1, animal_name)
		coord_left, likeli_left = utils.load_coord_csv(path_coord_limb2, animal_name)

		behavior_coord = np.concatenate((coord_right, coord_left))
		likeli = np.concatenate((likeli_right, likeli_left))
		print("# behavior_coord loaded")
		print('behavior_coord: ', behavior_coord.shape) #behavior_coord:  (8, 36731) = (limbs, time)
		print('likeli: ', likeli.shape)

		# Get idx_coord_vid
		vid1_len = utils.get_video_len(path_vid_limb1)
		vid2_len = utils.get_video_len(path_vid_limb2)
		print("# Video loaded")
		print('vid1_len: ', vid1_len)
		print('vid2_len: ', vid2_len)
		if animal_name == "Animal6-G10_55904_1L":
			print("animal_name=Animal6-G10_55904_1L has different lengths in videos.")
		else:
			if vid1_len != vid2_len:
				print('vid1_len != vid2_len')
				exit()
		if vid1_len != behavior_coord.shape[1]:
			print('vid1_len != behavior_coord.shape[1]')
			exit()
		idx_coord_vid = np.array([int(x) for x in range(vid1_len)])
		np.save(os.path.join(dir_result, 'idx_coord_vid.npy'), idx_coord_vid)
		print("# Save")
		print('idx_coord_vid: ', idx_coord_vid.shape, idx_coord_vid[:10], idx_coord_vid[-10:])

		# Selection because Scanbox was turned off before camera
		if animal_name == "Animal1-G8_53950_1L_Redo" or animal_name == "Animal3-G8_53950_2R_Redo" or animal_name == "Animal4-G10_55902_1R" or animal_name == "Animal5-G10_55903_1R" or animal_name == "Animal6-G10_55904_1L" or \
			animal_name == "Animal7-G12_55946_1R" or animal_name == "Animal8-G12_55947_1R" or animal_name == "Animal9-G12_55954_1L":
			# behavior_coord selection for G8_53950_2R_Redo. 
			behavior_coord = behavior_coord[:, :len(frame)*2]
			likeli = likeli[:, :len(frame)*2]
			idx_coord_vid = idx_coord_vid[:len(frame)*2]
		else:
			print("Exit at selection depending on animals")
			exit()

		print("# Selection because Scanbox was turned off before camera")
		print('behavior_coord: ', behavior_coord.shape)
		print('likeli: ', likeli.shape)
		print('idx_coord_vid: ', idx_coord_vid.shape)

		# Save
		np.save(os.path.join(dir_result, 'behav_coord_ori.npy'), behavior_coord)
		np.save(os.path.join(dir_result, 'coord_likelihood.npy'), likeli)
		print("# Save")
		print('behav_coord_ori: ', behavior_coord.shape, np.min(behavior_coord), np.max(behavior_coord))
		print('coord_likelihood: ', likeli.shape)
		plot.plot_coord_all(behavior_coord, 'behav_original', dir_result)
		plot.plot_coord_all(likeli, 'coord_likelihood', dir_result)

		#likelihood histogram
		for i in range(likeli.shape[0]):
			plt.figure()
			values, bins, bars = plt.hist(likeli[i,:], edgecolor='black', bins=[x*0.1 for x in range(11)])
			plt.bar_label(bars, fontsize=10, color='black')
			plt.xticks([x*0.1 for x in range(11)])
			plt.title('likelihood-' + str(i))
			plt.savefig(os.path.join(dir_result, 'likelihood-' + str(i)+'.png'))
			plt.close()

			plt.figure()
			idx_low = np.squeeze(np.argwhere(likeli[i,:]<0.1))
			values, bins, bars = plt.hist(likeli[i,idx_low], edgecolor='black', bins=[x*0.01 for x in range(11)])
			plt.bar_label(bars, fontsize=10, color='black')
			plt.xticks([x*0.01 for x in range(11)])
			plt.title('likelihood_01-' + str(i))
			plt.savefig(os.path.join(dir_result, 'likelihood_01-' + str(i)+'.png'))

		########################################################################################################################################################
			
		### Preprocess low likelihood points
		if likeli_thrd is not None:
			print("### Preprocess low likelihood points")
			behavior_coord_likeli = preprocess_low_likelihood(behavior_coord, likeli, likeli_thrd)
			np.save(os.path.join(dir_result, 'behav_coord_likeli_ori.npy'), behavior_coord_likeli)
			print("# Save")
			print('behav_coord_likeli_ori: ', behavior_coord_likeli.shape, np.min(behavior_coord_likeli), np.max(behavior_coord_likeli))

		### 3. Synchronization ####################################################################################################################################
		print("### 3. Synchronization")

		# Get idx_coord_neural: neural frame number of coord frame
		idx_coord_neural = np.array([float("nan") for x in range(behavior_coord.shape[1])])
		idx_coord_neural_match = np.array([float("nan") for x in range(behavior_coord.shape[1])])
		ttl_on_frame = frame[[i for i,j in enumerate(eventid.tolist()) if j == ttl_on]]
		ttl_off_frame = frame[[i for i,j in enumerate(eventid.tolist()) if j == ttl_off]]
		print('ttl_on_frame: ', ttl_on_frame[:20], ttl_on_frame[-20:])
		print('ttl_off_frame: ', ttl_off_frame[:20], ttl_off_frame[-20:])

		if animal_name == "Animal1-G8_53950_1L_Redo" or animal_name == "Animal3-G8_53950_2R_Redo" or \
			animal_name == "Animal4-G10_55902_1R" or animal_name == "Animal5-G10_55903_1R" or animal_name == "Animal6-G10_55904_1L" or \
			animal_name == "Animal7-G12_55946_1R" or animal_name == "Animal8-G12_55947_1R" or animal_name == "Animal9-G12_55954_1L":
			# %4 version
			for i in range(len(ttl_on_frame)):
				if i*4<idx_coord_neural.shape[0]: idx_coord_neural[i*4] = ttl_on_frame[i] #fake!!!!!!!
				if i*4+1<idx_coord_neural.shape[0]: 
					idx_coord_neural[i*4+1] = ttl_on_frame[i] #real
					idx_coord_neural_match[i*4+1] = ttl_on_frame[i] #real
			for i in range(len(ttl_off_frame)):
				if i*4+2<idx_coord_neural.shape[0]: 
					idx_coord_neural[i*4+2] = ttl_off_frame[i] #real
					idx_coord_neural_match[i*4+2] = ttl_off_frame[i] #real
				if i*4+3<idx_coord_neural.shape[0]: idx_coord_neural[i*4+3] = ttl_off_frame[i] #fake!!!!!!!
		# elif animal_name == "Animal2-G16_55875_1L_2nd":
		# 	# %2 version
		# 	for i in range(len(ttl_on_frame)):
		# 		idx_coord_neural[i*2] = ttl_on_frame[i] 
		# 		idx_coord_neural_match[i*2] = ttl_on_frame[i] #real
		# 	for i in range(len(ttl_off_frame)):
		# 		idx_coord_neural[i*2+1] = ttl_off_frame[i] 
		# 		idx_coord_neural_match[i*2+1] = ttl_on_frame[i] #real
		else:
			print("Exit at synchronization")
			exit()

		# Check nan
		if np.any(np.isnan(idx_coord_neural)):
			print('Nan still exitst in idx_coord_neural.')
			exit()
		if np.isnan(np.min(idx_coord_neural)):
			print('idx_coord_neural has nan.')
			exit()

		# Change to int
		idx_coord_neural = idx_coord_neural.astype('int64')
 
		# Print
		print('idx_coord_neural: ', idx_coord_neural.shape, idx_coord_neural[:20], idx_coord_neural[-20:])
		print('idx_coord_neural_match: ', idx_coord_neural_match.shape, idx_coord_neural_match[:20], idx_coord_neural_match[-20:])
		# [nan, 59, 60, nan, nan, 60, 61, nan, nan, 61, 62, nan, nan, 62, 63, nan, nan, 64, 
		# 64, nan, nan, 65, 65, nan, nan, 66, 66, nan, nan, 67, 67, nan, nan, 68, 68, nan, nan, 69, 
		# 69, nan, nan, 70, 70, nan, nan, 71, 71, nan, nan, 72]
		# plt.plot(idx_coord_neural)
		# plt.show()
		
		# Check if it is right to neural shape
		if spks_z.shape[1] != idx_coord_neural[-1]+1:
			print('spks_z.shape[1] != idx_coord_neural[-1]+1', spks_z.shape[1], idx_coord_neural[-1]+1)
			exit()

		# Check discontinuity in idx_coord_neural
		if np.any(np.diff(idx_coord_neural)>1):
			print("There is a discontinuity in idx_coord_neural.")
			print(np.where(np.any(np.diff(idx_coord_neural)>1)))
			exit()
		
		# Check reverse order in idx_coord_neural
		if np.any(np.diff(idx_coord_neural)<0):
			print("There is a reverse order in idx_coord_neural.")
			print(np.where(np.any(np.diff(idx_coord_neural)<0)))
			exit()

		# # Plot coord vs. neural frame     
		# fig = plt.figure()
		# plt.plot(idx_coord_neural, '-o')
		# plt.xlabel('idx_coord')
		# plt.ylabel('neural frame')
		# plt.title('idx_coord_neural')
		# fig.set_size_inches(20, 7)
		# plt.xlim(0,200)
		# plt.ylim(np.min(idx_coord_neural[:200]), np.max(idx_coord_neural[:200]))
		# plt.savefig(os.path.join(dir_result, 'idx_coord_neural'+'.png'))

		# Select neural_data sequence period considering idx_coord_neural
		ffneuori_sel = ffneu_ori[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		ffneuz_sel = ffneu_z[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		fneufori_sel = fneuf_ori[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		fneufz_sel = fneuf_z[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		spksori_sel = spks_ori[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		spksz_sel = spks_z[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		neuori_sel = neu_ori[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		neuz_sel = neu_z[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		fori_sel = f_ori[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		fz_sel = f_z[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		neudeconvori_sel = neudeconv_ori[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		neudeconvz_sel = neudeconv_z[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		ffneu_noncell_ori_sel = ffneu_noncell_ori[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		ffneu_noncell_z_sel = ffneu_noncell_z[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		neu_noncell_ori_sel = neu_noncell_ori[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		neu_noncell_z_sel = neu_noncell_z[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		spks_noncell_ori_sel = spks_noncell_ori[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		spks_noncell_z_sel = spks_noncell_z[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		fneufdeconvori_sel = fneufdeconv_ori[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]
		fneufdeconvz_sel = fneufdeconv_z[:, idx_coord_neural[0]:idx_coord_neural[-1]+1]

		xoff_sel = xoff[idx_coord_neural[0]:idx_coord_neural[-1]+1]
		yoff_sel = yoff[idx_coord_neural[0]:idx_coord_neural[-1]+1]
		corrxy_sel = corrxy[idx_coord_neural[0]:idx_coord_neural[-1]+1]

		start_idx = idx_coord_neural[0]
		idx_coord_neural = idx_coord_neural - start_idx

		for i in range(len(idx_coord_neural_match)):
			if np.isnan(idx_coord_neural_match[i]) == False:
				idx_coord_neural_match[i] = int(idx_coord_neural_match[i]-start_idx)

		print("# Selection by idx_coord_neural")
		print('ffneuori_sel: ', ffneuori_sel.shape)
		print('ffneuz_sel: ', ffneuz_sel.shape)
		print('fneufori_sel: ', fneufori_sel.shape)
		print('fneufz_sel: ', fneufz_sel.shape)
		print('spksori_sel: ', spksori_sel.shape)
		print('spksz_sel: ', spksz_sel.shape)
		print('neuori_sel: ', neuori_sel.shape)
		print('neuz_sel: ', neuz_sel.shape)
		print('fori_sel: ', fori_sel.shape)
		print('fz_sel: ', fz_sel.shape)
		print('neudeconvori_sel: ', neudeconvori_sel.shape)
		print('neudeconvz_sel: ', neudeconvz_sel.shape)
		print('ffneu_noncell_ori_sel: ', ffneu_noncell_ori_sel.shape)
		print('ffneu_noncell_z_sel: ', ffneu_noncell_z_sel.shape)
		print('neu_noncell_ori_sel: ', neu_noncell_ori_sel.shape)
		print('neu_noncell_z_sel: ', neu_noncell_z_sel.shape)
		print('spks_noncell_ori_sel: ', spks_noncell_ori_sel.shape)
		print('spks_noncell_z_sel: ', spks_noncell_z_sel.shape)
		print('fneufdeconvori_sel: ', fneufdeconvori_sel.shape)
		print('fneufdeconvz_sel: ', fneufdeconvz_sel.shape)
		print('xoff_sel: ', xoff_sel.shape)
		print('yoff_sel: ', yoff_sel.shape)
		print('corrxy_sel: ', corrxy_sel.shape)

		print('idx_coord_neural, changed: ', idx_coord_neural.shape, idx_coord_neural[:20], idx_coord_neural[-20:])
		print('idx_coord_neural_match, changed: ', idx_coord_neural_match.shape, idx_coord_neural_match[:20], idx_coord_neural_match[-20:])
		if flag_calcium:
			calcium_sel = calcium[:,:,idx_coord_neural[0]:idx_coord_neural[-1]+1]
			print('calcium_sel: ', calcium_sel.shape)

		# Save
		np.save(os.path.join(dir_result, 'ffneu_ori_sel.npy'), ffneuori_sel)
		np.save(os.path.join(dir_result, 'ffneu_z_sel.npy'), ffneuz_sel)
		np.save(os.path.join(dir_result, 'fneuf_ori_sel.npy'), fneufori_sel)
		np.save(os.path.join(dir_result, 'fneuf_z_sel.npy'), fneufz_sel)
		np.save(os.path.join(dir_result, 'spks_ori_sel.npy'), spksori_sel)
		np.save(os.path.join(dir_result, 'spks_z_sel.npy'), spksz_sel)
		np.save(os.path.join(dir_result, 'neu_ori_sel.npy'), neuori_sel)
		np.save(os.path.join(dir_result, 'neu_z_sel.npy'), neuz_sel)
		np.save(os.path.join(dir_result, 'fori_sel.npy'), fori_sel)
		np.save(os.path.join(dir_result, 'fz_sel.npy'), fz_sel)
		
		np.save(os.path.join(dir_result, 'ffneu_noncell_ori_sel.npy'), ffneu_noncell_ori_sel)
		np.save(os.path.join(dir_result, 'ffneu_noncell_z_sel.npy'), ffneu_noncell_z_sel)
		np.save(os.path.join(dir_result, 'neu_noncell_ori_sel.npy'), neu_noncell_ori_sel)
		np.save(os.path.join(dir_result, 'neu_noncell_z_sel.npy'), neu_noncell_z_sel)
		np.save(os.path.join(dir_result, 'spks_noncell_ori_sel.npy'), spks_noncell_ori_sel)
		np.save(os.path.join(dir_result, 'spks_noncell_z_sel.npy'), spks_noncell_z_sel)

		np.save(os.path.join(dir_result, 'neudeconv_ori_sel.npy'), neudeconvori_sel)
		np.save(os.path.join(dir_result, 'neudeconv_z_sel.npy'), neudeconvz_sel)

		np.save(os.path.join(dir_result, 'fneufdeconv_ori_sel.npy'), fneufdeconvori_sel)
		np.save(os.path.join(dir_result, 'fneufdeconv_z_sel.npy'), fneufdeconvz_sel)

		np.save(os.path.join(dir_result, 'xoff_sel.npy'), xoff_sel)
		np.save(os.path.join(dir_result, 'yoff_sel.npy'), yoff_sel)
		np.save(os.path.join(dir_result, 'corrxy_sel.npy'), corrxy_sel)

		np.save(os.path.join(dir_result, 'idx_coord_neural.npy'), idx_coord_neural)

		if flag_calcium:
			dir_calcium = os.path.join(dir_result, 'calcium_sel')
			os.mkdir(dir_calcium)
			for i in range(calcium_sel.shape[2]):
				np.save(os.path.join(dir_calcium, str(i)+'.npy'), calcium_sel[:,:,i])

		print("# Save")
		print('ffneu_ori_sel: ', ffneuori_sel.shape)
		print('ffneu_z_sel: ', ffneuz_sel.shape)
		print('fneufori_sel: ', fneufori_sel.shape)
		print('fneufz_sel: ', fneufz_sel.shape)
		print('spks_z_sel: ', spksz_sel.shape)
		print('spksori_sel: ', spksori_sel.shape)
		print('neu_ori_sel: ', neuori_sel.shape)
		print('neu_z_sel: ', neuz_sel.shape)
		print('fori_sel: ', fori_sel.shape)
		print('fz_sel: ', fz_sel.shape)
		print('ffneu_noncell_ori_sel: ', ffneu_noncell_ori_sel.shape)
		print('ffneu_noncell_z_sel: ', ffneu_noncell_z_sel.shape)
		print('neu_noncell_ori_sel: ', neu_noncell_ori_sel.shape)
		print('neu_noncell_z_sel: ', neu_noncell_z_sel.shape)
		print('spks_noncell_ori_sel: ', spks_noncell_ori_sel.shape)
		print('spks_noncell_z_sel: ', spks_noncell_z_sel.shape)
		print('fneufdeconvori_sel: ', fneufdeconvori_sel.shape)
		print('fneufdeconvz_sel: ', fneufdeconvz_sel.shape)
		print('xoff_sel: ', xoff_sel.shape)
		print('yoff_sel: ', yoff_sel.shape)
		print('corrxy_sel: ', corrxy_sel.shape)

		print('idx_coord_neural: ', idx_coord_neural.shape, idx_coord_neural[:10], idx_coord_neural[-10:])
		if flag_calcium:
			print('calcium_sel: ', calcium_sel.shape)
		
		
		plot.plot_neural(ffneuori_sel, dir_save=dir_result, title='neural_ffneu-ori-sel')
		plot.plot_neural_avg(ffneuori_sel, dir_save=dir_result, title='neural_avg_ffneu-ori-sel')

		plot.plot_neural(fneufori_sel, dir_save=dir_result, title='neural_fneuf-ori-sel')
		plot.plot_neural_avg(fneufori_sel, dir_save=dir_result, title='neural_avg_fneuf-ori-sel')

		plot.plot_neural(spksori_sel, dir_save=dir_result, title='neural_spks-ori-sel')
		plot.plot_neural_avg(spksori_sel, dir_save=dir_result, title='neural_avg_spks-ori-sel')

		plot.plot_neural(neuori_sel, dir_save=dir_result, title='neural_neu-ori-sel')
		plot.plot_neural_avg(neuori_sel, dir_save=dir_result, title='neural_avg_neu-ori-sel')

		plot.plot_neural(neudeconvori_sel, dir_save=dir_result, title='neural_neudeconv-ori-sel')
		plot.plot_neural_avg(neudeconvori_sel, dir_save=dir_result, title='neural_avg_neudeconv-ori-sel')

		plot.plot_neural(fori_sel, dir_save=dir_result, title='neural_f-ori')
		plot.plot_neural_avg(fori_sel, dir_save=dir_result, title='neural_f-ori')

		plot.plot_neural(fneufdeconvori_sel, dir_save=dir_result, title='neural_fneufdeconv-ori')
		plot.plot_neural_avg(fneufdeconvori_sel, dir_save=dir_result, title='neural_fneufdeconv-ori')

		plot.plot_neural(ffneuz_sel, dir_save=dir_result, title='neural_ffneu-z-sel')
		plot.plot_neural_avg(ffneuz_sel, dir_save=dir_result, title='neural_avg_ffneu-z-sel')
		plot.plot_trace(seq=np.average(ffneuz_sel, axis=0), color='royalblue', ylim=[-0.6, 1.9], yaxis=True, path_save=os.path.join(dir_result, 'neural_trace_ffneu-z-sel.png'))

		plot.plot_neural(fneufz_sel, dir_save=dir_result, title='neural_fneuf-z-sel')
		plot.plot_neural_avg(fneufz_sel, dir_save=dir_result, title='neural_avg_fneuf-z-sel')
		plot.plot_trace(seq=np.average(fneufz_sel, axis=0), color='royalblue', ylim=[-0.6, 1.9], yaxis=True, path_save=os.path.join(dir_result, 'neural_trace_fneuf-z-sel.png'))

		plot.plot_neural(spksz_sel, dir_save=dir_result, title='neural_spks-z-sel')
		plot.plot_neural_avg(spksz_sel, dir_save=dir_result, title='neural_avg_spks-z-sel')
		plot.plot_trace(seq=np.average(spksz_sel, axis=0), color='royalblue', ylim=[-0.6, 1.9], yaxis=True, path_save=os.path.join(dir_result, 'neural_trace_spks-z-sel.png'))

		plot.plot_neural(neuz_sel, dir_save=dir_result, title='neural_neu-z-sel')
		plot.plot_neural_avg(neuz_sel, dir_save=dir_result, title='neural_avg_neu-z-sel')
		plot.plot_trace(seq=np.average(neuz_sel, axis=0), color='royalblue', yaxis=True, path_save=os.path.join(dir_result, 'neural_trace_neu-z-sel.png'))

		plot.plot_neural(neudeconvz_sel, dir_save=dir_result, title='neural_neudeconv-z-sel')
		plot.plot_neural_avg(neudeconvz_sel, dir_save=dir_result, title='neural_avg_neudeconv-z-sel')
		plot.plot_trace(seq=np.average(neudeconvz_sel, axis=0), color='royalblue', yaxis=True, path_save=os.path.join(dir_result, 'neural_trace_neudeconv-z-sel.png'))

		plot.plot_neural(fz_sel, dir_save=dir_result, title='neural_f-z-sel')
		plot.plot_neural_avg(fz_sel, dir_save=dir_result, title='neural_avg_f-z-sel')
		plot.plot_trace(seq=np.average(fz_sel, axis=0), color='royalblue', yaxis=True, path_save=os.path.join(dir_result, 'neural_trace_f-z-sel.png'))

		plot.plot_neural(ffneu_noncell_z_sel, dir_save=dir_result, title='neural_ffneu-noncell-z-sel')
		plot.plot_neural_avg(ffneu_noncell_z_sel, dir_save=dir_result, title='neural_avg_ffneu-noncell-z-sel')
		plot.plot_trace(seq=np.average(ffneu_noncell_z_sel, axis=0), color='royalblue', yaxis=True, path_save=os.path.join(dir_result, 'neural_trace_ffneu-noncell-z-sel.png'))

		plot.plot_neural(neu_noncell_z_sel, dir_save=dir_result, title='neural_neu-noncell-z-sel')
		plot.plot_neural_avg(neu_noncell_z_sel, dir_save=dir_result, title='neural_avg_neu-noncell-z-sel')
		plot.plot_trace(seq=np.average(neu_noncell_z_sel, axis=0), color='royalblue', yaxis=True, path_save=os.path.join(dir_result, 'neural_trace_neu-noncell-z-sel.png'))

		plot.plot_neural(spks_noncell_z_sel, dir_save=dir_result, title='neural_spks-noncell-z-sel')
		plot.plot_neural_avg(spks_noncell_z_sel, dir_save=dir_result, title='neural_avg_spks-noncell-z-sel')
		plot.plot_trace(seq=np.average(spks_noncell_z_sel, axis=0), color='royalblue', yaxis=True, path_save=os.path.join(dir_result, 'neural_trace_spks-noncell-z-sel.png'))

		plot.plot_neural(fneufdeconvz_sel, dir_save=dir_result, title='neural_fneufdeconv-z-sel')
		plot.plot_neural_avg(fneufdeconvz_sel, dir_save=dir_result, title='neural_avg_fneufdeconv-z-sel')
		plot.plot_trace(seq=np.average(fneufdeconvz_sel, axis=0), color='royalblue', yaxis=True, path_save=os.path.join(dir_result, 'neural_trace_fneufdeconv-z-sel.png'))

		# Count occurrence in idx_coord_neural
		count_list = []
		for i in list(set(idx_coord_neural)):
			count_list.append(np.count_nonzero(idx_coord_neural==i))
		fig = plt.figure()
		values, bins, bars = plt.hist(count_list, edgecolor='black', bins=np.arange(1,max(set(count_list))+2)-0.5)
		plt.bar_label(bars, fontsize=10, color='black')
		plt.xticks(range(1,max(set(count_list))+1))
		plt.title('# of coord frames per one neural frame')
		plt.savefig(os.path.join(dir_result, 'number_frames_hist.png'))
		plt.savefig(os.path.join(dir_result, 'number_frames_hist.pdf'))
		plt.close()
		#####################################################################################################################

		### Get run&stand windows, likelihood
		if likeli_thrd is not None:
			print("### Get run&stand windows, likelihood")
			run_windows_likeli, stand_windows_likeli = utils.get_run_stand_wins_coord(behavior_coord_likeli, flag_figure=True, dir_fig_save=dir_result, title='likeli', limb_to_crop_str=limb_to_crop_str)
			np.save(os.path.join(dir_result, 'run_windows_likeli.npy'), run_windows_likeli)
			np.save(os.path.join(dir_result, 'stand_windows_likeli.npy'), stand_windows_likeli)
			print("# Save")
			print('run_windows_likeli: ', run_windows_likeli.shape)
			print('stand_windows_likeli: ', stand_windows_likeli.shape)
		
		### Get match, neural
		idx_neural_match, idx_coord_match = utils.get_match_idx_only(idx_coord_neural_match)
		ffneuzsel_match = ffneuz_sel[:, idx_neural_match]
		spkszsel_match = spksz_sel[:, idx_neural_match]

		np.save(os.path.join(dir_result, 'ffneuzsel_match.npy'), ffneuzsel_match)
		np.save(os.path.join(dir_result, 'spkszsel_match.npy'), spkszsel_match)
		np.save(os.path.join(dir_result, 'idx_neural_match.npy'), idx_neural_match)
		np.save(os.path.join(dir_result, 'idx_coord_match.npy'), idx_coord_match)

		print("### Get match, neural")
		print("# Save")
		print('ffneuzsel_match: ', ffneuzsel_match.shape)
		print('spkszsel_match: ', spkszsel_match.shape)
		print('idx_neural_match: ', len(idx_neural_match), idx_neural_match[:10], idx_neural_match[-10:])
		print('idx_coord_match: ', len(idx_coord_match), idx_coord_match[:10], idx_coord_match[-10:])

		### Get match, likelihood
		# low-pass filtering
		order = 5
		b_low, a_low = butter(order, 7.8, btype='lowpass', analog=False, fs=30)
		behav_low = np.zeros(behavior_coord_likeli.shape)
		for i in range(8):
			behav_low[i,:] = filtfilt(b_low,a_low,behavior_coord_likeli[i,:])

		behavior_coord_likeli_match = behav_low[:, idx_coord_match]

		# Save
		np.save(os.path.join(dir_result, 'behav_coord_'+'likeli'+'_match_ori.npy'), behavior_coord_likeli_match)

		print("### Get match, behav, "+ "likeli")
		print("# Save")
		print('behav_coord_'+"likeli"+'_match_ori: ', behavior_coord_likeli_match.shape, np.min(behavior_coord_likeli_match), np.max(behavior_coord_likeli_match))
		
		# Plot 
		plot.plot_neural_behav(ffneuzsel_match, behavior_coord_likeli_match, dir_result, 'ffneuzsel_match')
		plot.plot_neural_behav(spkszsel_match, behavior_coord_likeli_match, dir_result, 'spkszsel_match')
		plot.plot_coord_all(behavior_coord_likeli_match, 'behav_coord_'+"likeli"+'_match_ori', dir_result)

		### Get run&stand windows, match
		if likeli_thrd is not None:
			print("### Get run&stand windows, likeli_match")
			run_windows_likeli_match, stand_windows_likeli_match = utils.get_run_stand_wins_coord(behavior_coord_likeli_match, flag_figure=True, dir_fig_save=dir_result, title='likeli_match', limb_to_crop_str=limb_to_crop_str)
			np.save(os.path.join(dir_result, 'run_windows_likeli_match.npy'), run_windows_likeli_match)
			np.save(os.path.join(dir_result, 'stand_windows_likeli_match.npy'), stand_windows_likeli_match)
			print("# Save")
			print('run_windows_likeli_match: ', run_windows_likeli_match.shape)
			print('stand_windows_likeli_match: ', stand_windows_likeli_match.shape)

		### Demean
		if likeli_thrd is not None:
			### Demean
			possible_coord_idx_list = []
			for win in run_windows_likeli:
				possible_coord_idx_list.extend(list(range(win[0],win[1]+1,1)))
			
			behav_likeli_demean = deepcopy(behavior_coord_likeli)
			behav_run = behav_likeli_demean[:, possible_coord_idx_list] #select only run periods
			behav_dm = np.zeros(behav_run.shape) 

			for i in range(behav_dm.shape[0]):
				behav_run_smooth = smooth(behav_run[i,:],50)
				behav_run_demean = behav_run[i,:]-behav_run_smooth
				behav_dm[i,:] = behav_run_demean/np.std(behav_run_demean)

			behav_likeli_demean[:, possible_coord_idx_list] = behav_dm
			
			np.save(os.path.join(dir_result, 'behav_coord_likeli_demean.npy'), behav_likeli_demean)
			np.save(os.path.join(dir_result, 'behav_coord_likeli_demean_onlyrun.npy'), behav_dm)

			plot.plot_coord_all(behav_likeli_demean, 'behav_coord_likeli_demean', dir_result)
			plot.plot_coord_all(behav_dm, 'behav_coord_likeli_demean_onlyrun', dir_result)

			print("### demean")
			print("# Save")
			print("behav_coord_likeli_demean: ", behav_likeli_demean.shape)
			print("behav_coord_likeli_demean_onlyrun: ", behav_dm.shape)
	
			### Demean, match
			possible_coord_idx_list = []
			for win in run_windows_likeli_match:
				possible_coord_idx_list.extend(list(range(win[0],win[1]+1,1)))
			
			behav_likeli_demean_match = deepcopy(behavior_coord_likeli_match)
			behav_run = behav_likeli_demean_match[:, possible_coord_idx_list] #select only run periods
			behav_dm = np.zeros(behav_run.shape) 
			behav_dm_low = np.zeros(behav_run.shape) 

			for i in range(behav_dm.shape[0]):
				behav_run_smooth = smooth(behav_run[i,:],50)
				behav_run_demean = behav_run[i,:]-behav_run_smooth
				behav_dm[i,:] = behav_run_demean/np.std(behav_run_demean)
				behav_dm_low[i,:] = filtfilt(b_low,a_low,behav_dm[i,:])

			behav_likeli_demean_match[:, possible_coord_idx_list] = behav_dm_low
			#behav_likeli_demean_match = behav_likeli_demean_match[:, idx_coord_match]

			np.save(os.path.join(dir_result, 'behav_coord_likeli_demean_match.npy'), behav_likeli_demean_match)
			np.save(os.path.join(dir_result, 'behav_coord_likeli_demean_match_onlyrun.npy'), behav_dm_low)

			plot.plot_coord_all(behav_likeli_demean_match, 'behav_coord_likeli_demean_match', dir_result)
			plot.plot_coord_all(behav_dm_low, 'behav_coord_likeli_demean_match_onlyrun', dir_result)

			print("### demean, match")
			print("# Save")
			print("behav_coord_likeli_demean_match: ", behav_likeli_demean_match.shape)
			print("behav_coord_likeli_demean_match_onlyrun: ", behav_dm_low.shape)


		# # low-pass filtering
		# order = 5
		# b_low, a_low = butter(order, 7.8, btype='lowpass', analog=False, fs=30)
		# behav_likeli_low = np.zeros(behavior_coord_likeli.shape)
		# for i in range(8):
		# 	behav_likeli_low[i,:] = filtfilt(b_low,a_low,behavior_coord_likeli)

		# idx_neural_match, idx_coord_match = utils.get_match_idx_only(idx_coord_neural_match)
		# ffneuzsel_match = ffneuz_sel[:, idx_neural_match]
		# spkszsel_match = spksz_sel[:, idx_neural_match]
		# behavior_coord_likeli_match = behav_likeli_low[:, idx_coord_match]
		# print("### Get match, likelihood")
		# print('idx_coord_neural_match: ', len(idx_coord_neural_match), idx_coord_neural_match[:20], idx_coord_neural_match[-20:], np.max(idx_coord_neural_match))
		# print('idx_neural_match: ', len(idx_neural_match), idx_neural_match[:20], idx_neural_match[-20:], np.max(idx_neural_match))
		# print('idx_coord_match: ', len(idx_coord_match), idx_coord_match[:20], idx_coord_match[-20:])
		# print('ffneuzsel_match: ', ffneuzsel_match.shape)
		# print('spkszsel_match: ', spkszsel_match.shape)
		# print('behavior_coord_likeli_match: ', behavior_coord_likeli_match.shape)

		# # Save
		# np.save(os.path.join(dir_result, 'ffneuzsel_match.npy'), ffneuzsel_match)
		# np.save(os.path.join(dir_result, 'spkszsel_match.npy'), spkszsel_match)
		# np.save(os.path.join(dir_result, 'behavior_coord_likeli_match_ori.npy'), behavior_coord_likeli_match)
		# np.save(os.path.join(dir_result, 'idx_neural_match.npy'), idx_neural_match)
		# np.save(os.path.join(dir_result, 'idx_coord_match.npy'), idx_coord_match)

		# print("# Save")
		# print('ffneuzsel_match: ', ffneuzsel_match.shape)
		# print('spkszsel_match: ', spkszsel_match.shape)
		# print('behavior_coord_likeli_match_ori: ', behavior_coord_likeli_match.shape, np.min(behavior_coord_likeli_match), np.max(behavior_coord_likeli_match))
		# print('idx_neural_match: ', idx_neural_match.shape, idx_neural_match[:10], idx_neural_match[-10:])
		# print('idx_coord_match: ', idx_coord_match.shape, idx_coord_match[:10], idx_coord_match[-10:])

		# # Plot neural_match-behav
		# plot.plot_neural_behav(ffneuzsel_match, behavior_coord_likeli_match, dir_result, 'ffneuzsel_match')
		# plot.plot_neural_behav(spkszsel_match, behavior_coord_likeli_match, dir_result, 'spkszsel_match')
		# plot.plot_coord_all(behavior_coord_likeli_match, 'behavior_coord_likeli_match_ori', dir_result)





		### Duplicate neural signal
		print("### Duplicate neural signal")
		ffneuzsel_dup = np.zeros((ffneuz_sel.shape[0], len(idx_coord_neural))) #(neuron,frame_behav)
		spkszsel_dup = np.zeros((spksz_sel.shape[0], len(idx_coord_neural))) #(neuron,frame_behav)
		for idx_coord in range(len(idx_coord_neural)):
			ffneuzsel_dup[:, idx_coord] = ffneuz_sel[:,idx_coord_neural[idx_coord]]
			spkszsel_dup[:, idx_coord] = spksz_sel[:, idx_coord_neural[idx_coord]]

		# Save
		np.save(os.path.join(dir_result, 'ffneuzsel_dup.npy'), ffneuzsel_dup)
		np.save(os.path.join(dir_result, 'spkszsel_dup.npy'), spkszsel_dup)
		print("# Save")
		print('ffneuzsel_dup: ', ffneuzsel_dup.shape)
		print('spkszsel_dup: ', spkszsel_dup.shape)
	
		fig = plt.figure()
		plt.imshow(ffneuzsel_dup[:,:])
		plt.plot(behavior_coord[0,:],color='red')
		plt.tight_layout()
		fig.set_size_inches(20, 8)
		plt.savefig(os.path.join(dir_result, 'ffneuzsel_dup.png'))
		plt.close()

		fig = plt.figure()
		plt.imshow(spkszsel_dup[:,:])
		plt.plot(behavior_coord[0,:],color='red')
		plt.tight_layout()
		fig.set_size_inches(20, 8)
		plt.savefig(os.path.join(dir_result, 'spkszsel_dup.png'))
		plt.close()

		### Interpolate neural signal
		print("### Interpolate neural signal")
		all_indices_arr = np.arange(len(idx_coord_neural_match))
		known_indices_arr = np.array([i for i, value in enumerate(idx_coord_neural_match) if np.isnan(value)==False])

		known_values_ffneuzsel = np.zeros((ffneuz_sel.shape[0], len(known_indices_arr)))
		for i in range(len(known_indices_arr)):
			known_values_ffneuzsel[:,i] = ffneuz_sel[:,int(idx_coord_neural_match[known_indices_arr[i]])]
		
		known_values_spkszsel = np.zeros((spksz_sel.shape[0], len(known_indices_arr)))
		for i in range(len(known_indices_arr)):
			known_values_spkszsel[:,i] = spksz_sel[:,int(idx_coord_neural_match[known_indices_arr[i]])]

		# print(all_indices_arr.shape)
		# print(known_indices_arr.shape)
		# print(known_values_ffneuzsel.shape)

		ffneuzsel_intp = np.zeros((ffneuz_sel.shape[0], len(idx_coord_neural_match)))
		for i in range(ffneuz_sel.shape[0]):
			ffneuzsel_intp[i,:] = np.interp(all_indices_arr, known_indices_arr, known_values_ffneuzsel[i,:])

		spkszsel_intp = np.zeros((spksz_sel.shape[0], len(idx_coord_neural_match)))
		for i in range(spksz_sel.shape[0]):
			spkszsel_intp[i,:] = np.interp(all_indices_arr, known_indices_arr, known_values_spkszsel[i,:])

		# Save
		np.save(os.path.join(dir_result, 'ffneuzsel_intp.npy'), ffneuzsel_intp)
		np.save(os.path.join(dir_result, 'spkszsel_intp.npy'), spkszsel_intp)
		print("# Save")
		print('ffneuzsel_intp: ', ffneuzsel_intp.shape)
		print('spkszsel_intp: ', spkszsel_intp.shape)

		# plt.figure()
		# plt.scatter(known_indices_arr[:500], known_values_ffneuzsel[0,:500], color='red')
		# plt.plot(ffneuzsel_intp[0,:500], color='blue', marker='x')
		# plt.show()

		### behav normalization
		print("### behav normalization")
		if likeli_thrd is not None:
			behavior_coord_likeli_processed_norm = np.zeros(behavior_coord_likeli.shape)
			for i in range(behavior_coord_likeli_processed_norm.shape[0]):
				row = behavior_coord_likeli[i, :]
				# print(row.shape)
				# print(np.min(row))
				# print(np.max(row))
				behavior_coord_likeli_processed_norm[i, :] = (row - np.min(row)) / (np.max(row) - np.min(row))                
			np.save(os.path.join(dir_result, 'behav_coord_likeli_norm.npy'), behavior_coord_likeli_processed_norm)
			print("### save")
			print('behav_coord_likeli_norm: ', behavior_coord_likeli_processed_norm.shape, np.min(behavior_coord_likeli_processed_norm), np.max(behavior_coord_likeli_processed_norm))
			plot.plot_coord_all(behavior_coord_likeli_processed_norm, 'behavior_coord_likeli_processed_norm', dir_result)

			behavior_coord_likeli_processed_match_norm = np.zeros(behavior_coord_likeli_match.shape)
			for i in range(behavior_coord_likeli_processed_match_norm.shape[0]):
				row = behavior_coord_likeli_match[i, :]
				# print(row.shape)
				# print(np.min(row))
				# print(np.max(row))
				behavior_coord_likeli_processed_match_norm[i, :] = (row - np.min(row)) / (np.max(row) - np.min(row))                
			np.save(os.path.join(dir_result, 'behav_coord_likeli_match_norm.npy'), behavior_coord_likeli_processed_match_norm)
			print('behav_coord_likeli_match_norm: ', behavior_coord_likeli_processed_match_norm.shape, np.min(behavior_coord_likeli_processed_match_norm), np.max(behavior_coord_likeli_processed_match_norm))
			plot.plot_coord_all(behavior_coord_likeli_processed_match_norm, 'behavior_coord_likeli_match_norm', dir_result)

		### test idx for cross validation
		print("### test idx for cross validation")
		ratio_test = 0.2
		print('ratio_test: ', ratio_test)
		test_idx_cv = []
		idx_cv_plot = []
		timelen_neural = spksz_sel.shape[1]
		len_test = int(timelen_neural*ratio_test)
		for i in range(int(1/ratio_test)):
			test_idx = [i*len_test, (i+1)*len_test-1]
			test_idx_cv.append(test_idx)
			idx_coord_end = utils.get_match_idx_coord((i+1)*len_test-1, idx_coord_neural)
			idx_cv_plot.append(idx_coord_end)
	   
		test_idx_cv[-1][-1] = timelen_neural-1 
		test_idx_cv = np.array(test_idx_cv)

		len_sum = 0
		for i in range(test_idx_cv.shape[0]):
			len_test = test_idx_cv[i][1]-test_idx_cv[i][0]+1
			print('test_idx: ', test_idx_cv[i], ', len=', len_test)
			len_sum += len_test
		if len_sum != timelen_neural:
			print('len_sum != timelen_neural')
			exit()
		
		# Plot cross validation
		plot.plot_coord_all(behavior_coord, 'behav_original_cv', dir_result, idx_cv_plot)

		# test_idx coord
		test_idx_cv_coord = []
		for i in range(test_idx_cv.shape[0]):
			test_idx_cv_coord.append([utils.get_match_idx_coord(test_idx_cv[i][0], idx_coord_neural), utils.get_match_idx_coord(test_idx_cv[i][1], idx_coord_neural)])
		test_idx_cv_coord = np.array(test_idx_cv_coord)

		# Save
		np.save(os.path.join(dir_result, 'test_idx_cv.npy'), test_idx_cv)
		np.save(os.path.join(dir_result, 'test_idx_cv_coord.npy'), test_idx_cv_coord)
		print("# Save")
		print('test_idx_cv: ', test_idx_cv)
		print("test_idx_cv_coord: ", test_idx_cv_coord)

def save_skew_neurons():
	'''
	Save only neurons with 10% top and low skewness into a dataset for training.
	'''
	### Change here!!!
	dir_data = r"E:\tmp\prepare_data\Animal1-G8_53950_1L_Redo\Animal1_likeli01"
	#############################################################################################
	### Load data - prepared
	stat = np.load(os.path.join(dir_data, 'stat.npy'), allow_pickle=True)
	spks_z_sel = np.load(os.path.join(dir_data, 'spks_z_sel.npy'))
	num_sig = int(spks_z_sel.shape[0]*0.1)

	### Get skew
	skew_list = []
	skew_abs_list = []
	for i in range(len(stat)):
		skew_list.append(stat[i]['skew'])
		skew_abs_list.append(abs(stat[i]['skew']))
	skewabs_minmax = [(x-min(skew_abs_list)) / (max(skew_abs_list)-min(skew_abs_list)) for x in skew_abs_list]
	#print(skew_list)
	#print('skew_list: ', len(skew_list)) #3327

	### Get high and low skew neurons
	skew_abs_list_low = [x for x in skew_abs_list]
	skew_abs_list_low.sort()
	#print('skew_abs_list_low: ', skew_abs_list_low[:5])

	skew_abs_list_high = [x for x in skew_abs_list]
	skew_abs_list_high.sort(reverse=True)
	#print('skew_abs_list_high: ', skew_abs_list_high[:5])

	### Get high and low skew neuron index
	idx_neuron_skew_min = [skew_abs_list.index(x) for x in skew_abs_list_low[:num_sig]]
	idx_neuron_skew_max = [skew_abs_list.index(x) for x in skew_abs_list_high[:num_sig]]

	### Save data
	spks_z_sel_skewhighneuron =  spks_z_sel[idx_neuron_skew_max,:]
	spks_z_sel_skewlowneuron = spks_z_sel[idx_neuron_skew_min,:]
	np.save(os.path.join(dir_data, 'spks_z_sel_skewhighneuron_'+str(num_sig)+'.npy'),spks_z_sel_skewhighneuron)
	np.save(os.path.join(dir_data, 'spks_z_sel_skewlowneuron_'+str(num_sig)+'.npy'), spks_z_sel_skewlowneuron)
	print("spks_z_sel_skewhighneuron: ", spks_z_sel_skewhighneuron.shape)
	print("spks_z_sel_skewlowneuron: ", spks_z_sel_skewlowneuron.shape)

def save_important_neurons():
	'''
	Only save important neurons as a dataset that can be used to train
	'''
	### Change here
	dir_result = r"E:\tmp\NN\20241118_115845_Animal1_multilimb"
	dir_data = r"E:\tmp\prepare_data\Animal1-G8_53950_1L_Redo\Animal1_likeli01"	
	#####################################################################################

	dir_ni = os.path.join(dir_data, 'save_neuron_importance')
	if os.path.isdir(dir_ni):
		print('directory exists.')
		exit()
	else: os.mkdir(dir_ni)

	# Load max image
	#maximg = loadmat(path_sbx_max_img)['maximg']
	
	path_log = os.path.join(dir_ni, 'log.txt')
	with LoggingPrinter(path_log):
		# Load neuron importance
		impt_norm = np.load(os.path.join(dir_result, 'neuron_importance', 'grd_norm_over_time_norm.npy'))
		print('impt_norm: ', impt_norm.shape, impt_norm[:10])

		impt_sort_arg = np.argsort(-impt_norm)
		print('impt_sort_arg: ', np.argsort(-impt_sort_arg), impt_norm[impt_sort_arg[:10]])
		
		impt_sort_arg_least = np.argsort(impt_norm)
		print('impt_sort_arg_least: ', np.argsort(-impt_sort_arg_least), impt_norm[impt_sort_arg_least[:10]])

		percent_list = [10, 25, 50, 75, 90]
		num_list = []
		for i in percent_list:
			num_list.append(int(impt_norm.shape[0]/100*i))
		print('num_list: ', num_list)

		# p_25 = np.percentile(impt_norm, 25)
		# p_50 = np.percentile(impt_norm, 50)
		# p_75 = np.percentile(impt_norm, 75)

		fig1 = plt.figure(1)
		plt.scatter([x+1 for x in range(impt_norm.shape[0])], impt_norm, edgecolors='black')
		for i, p in zip(num_list, percent_list):
			plt.axhline(y=impt_norm[impt_sort_arg[i-1]], color='r')
			plt.text(-100, impt_norm[impt_sort_arg[i-1]], str(p)+"%")
		plt.xlabel("Neuron")
		plt.ylabel("Importance")
		fig1.set_size_inches(16,5)
		plt.savefig(os.path.join(dir_ni, "neuron-importance.png"))
		plt.savefig(os.path.join(dir_ni, "neuron-importance.pdf"))

		plt.figure(2)
		plt.hist(impt_norm, edgecolor = "black")
		for i, p in zip(num_list, percent_list):
			plt.axvline(x=impt_norm[impt_sort_arg[i-1]], color='r')
			plt.text(impt_norm[impt_sort_arg[i-1]], 50, str(p)+"%")
		plt.savefig(os.path.join(dir_ni, "hist.png"))
		plt.savefig(os.path.join(dir_ni, "hist.pdf"))

		# Load existing dataset
		neural_data = np.load(os.path.join(dir_data, 'spks_z_sel.npy'))
		ops_data = np.load(os.path.join(dir_data, 'ops_data.npy'), allow_pickle=True).item()
		stat = np.load(os.path.join(dir_data, 'stat.npy'), allow_pickle=True)
		print('neural_data: ', neural_data.shape)
		print('stat: ', stat.shape)

		# Index neural_data
		for n,p in zip(num_list, percent_list):
			idx_sel = impt_sort_arg[:n]
			neural_data_select = neural_data[idx_sel, :]
			print('neural_data_select, p=' + str(p) + ': ', neural_data_select.shape)
			np.save(os.path.join(dir_ni, 'impt-' + str(p) + '.npy'), neural_data_select)
			np.save(os.path.join(dir_ni, 'impt-'+str(p)+'_idx.npy'), idx_sel)
		  
			# Plot
			plot.plot_neuron_importance(ops_data, stat[idx_sel], impt_norm[idx_sel], title='p_'+str(p), dir_save=dir_ni, minmax_list=[0,1])
			#plot.plot_neuron_importance(ops_data, stat[impt_sort_arg[:n]], impt_norm[impt_sort_arg[:n]], title='maximg_p_'+str(p), dir_save=dir_ni, minmax_list=[0,1], maximg=maximg)

		for n,p in zip(num_list, percent_list):
			idx_sel = impt_sort_arg_least[:n]
			neural_data_select = neural_data[idx_sel, :]
			print('least neural_data_select, p=' + str(p) + ': ', neural_data_select.shape)
			np.save(os.path.join(dir_ni, 'impt-' + str(p) + '_least.npy'), neural_data_select)
			np.save(os.path.join(dir_ni, 'impt-'+str(p)+'_least_idx.npy'), idx_sel)
		  
			# Plot
			plot.plot_neuron_importance(ops_data, stat[idx_sel], impt_norm[idx_sel], title='least_p_'+str(p), dir_save=dir_ni, minmax_list=[0,1])


def save_random_neurons():
	### Change here!!!
	dir_result = r"E:\tmp\NN\20241118_115845_Animal1_multilimb"
	dir_data = r"E:\tmp\prepare_data\Animal1-G8_53950_1L_Redo\Animal1_likeli01"

	percent_list = [10, 25, 50, 75, 90] #[10, 25, 50, 75, 90]
	num_sample = 50

	#####################################################################################

	dir_save = os.path.join(dir_data, 'save_random_neurons')
	if os.path.isdir(dir_save):
		print('Directory exists.')
		#exit()
	else: os.mkdir(dir_save)

	path_log = os.path.join(dir_save, 'log.txt')
	with LoggingPrinter(path_log):
		# Load existing dataset
		neural_data = np.load(os.path.join(dir_data, 'spks_z_sel.npy'))
		ops_data = np.load(os.path.join(dir_data, 'ops_data.npy'), allow_pickle=True).item()
		stat = np.load(os.path.join(dir_data, 'stat.npy'), allow_pickle=True)
		print('neural_data: ', neural_data.shape)
		print('stat: ', stat.shape)

		# Load neuron importance
		impt_norm = np.load(os.path.join(dir_result, 'neuron_importance', 'grd_norm_over_time_norm.npy'))
		print('impt_norm: ', impt_norm.shape, impt_norm[:10])

		# # Normalize neuron importance
		# impt_norm = (impt - min(impt)) / (max(impt)-min(impt))
		# print('impt_norm: ', impt_norm.shape, impt_norm[:10])

		# Loop in percents
		for p in percent_list:
			num_in_sample = int(neural_data.shape[0] * p / 100)
			
			print('##### percent: ', p)
			print('num_in_sample: ', num_in_sample)

			list_idx_neuron = list(range(neural_data.shape[0]))
			for n in range(num_sample):
				print('num_sample: ', n, '/', num_sample)

				# Get idx list
				idx_list = random.sample(list_idx_neuron, num_in_sample)
				#print('idx_list: ', idx_list)
				np.save(os.path.join(dir_save, 'neural_data_p-' + str(p) + '_sample-' + str(n) + '_idx.npy'), np.array(idx_list))

				# Save
				neural_data_select = neural_data[idx_list, :]
				if np.any(np.isnan(neural_data_select)):
					print('Nan exitst in neural_data_select.')
					exit()

				np.save(os.path.join(dir_save, 'neural_data_p-' + str(p) + '_sample-' + str(n) + '.npy'), neural_data_select)
				
				# Plot
				fig1 = plt.figure(1)
				plt.scatter([x+1 for x in range(impt_norm.shape[0])], impt_norm, edgecolors='black')
				plt.scatter(idx_list, impt_norm[idx_list], edgecolors='black', color='red')
				plt.xlabel("Neuron")
				plt.ylabel("Importance")
				fig1.set_size_inches(16,5)
				plt.savefig(os.path.join(dir_save, "neuron-importance_p-" + str(p) + '_sample-' + str(n) + ".png"))
				plt.savefig(os.path.join(dir_save, "neuron-importance_p-" + str(p) + '_sample-' + str(n) + ".pdf"))
				plt.close()

				plot.plot_neuron_importance(ops_data, stat[idx_list], impt_norm[idx_list], title='p-'+str(p) + '_sample-' + str(n), dir_save=dir_save, minmax_list=[0,1])



if __name__ == '__main__':
	### Prepare data
	prepare_data()
	
	### Save subsets of neurons
	#save_important_neurons()
	#save_random_neurons()
	#save_skew_neurons()







	



