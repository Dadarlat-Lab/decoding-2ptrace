import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import utils
from PIL import Image

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

limb_color_list = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
	(1.0, 0.4980392156862745, 0.054901960784313725),
	(0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
	(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
	(0.5803921568627451, 0.403921568627451, 0.7411764705882353),
	(0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
	(0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
	(0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
	(0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
	(0.09019607843137255, 0.7450980392156863, 0.8117647058823529)]

def plot_neural(neural_data, dir_save=None, title=None, max=None):
	fig, ax = plt.subplots()
	if max is None: im = ax.imshow(neural_data, aspect="auto")
	else: im = ax.imshow(neural_data, aspect="auto", vmax=max)
	plt.colorbar(im, ax=ax)
	# ax.yaxis.set_visible(False)
	fig.tight_layout()
	fig.set_size_inches(17,7)
	ax.set_xlabel('Frame')
	ax.set_ylabel('Neuron')

	if dir_save is None: plt.show()
	else: 
		if title is None: 
			plt.savefig(os.path.join(dir_save, 'neural'+'.png'))
			plt.savefig(os.path.join(dir_save, 'neural'+'.pdf'))
		else: 
			plt.savefig(os.path.join(dir_save, title+'.png'))
			plt.savefig(os.path.join(dir_save, title+'.pdf'))
	plt.close()

def plot_neural_behav(neural_data, behav_coord, dir_save=None, title=None):
	fig, ax = plt.subplots()
	im = ax.imshow(neural_data, aspect="auto")
	plt.colorbar(im, ax=ax)
	for i in range(behav_coord.shape[0]):
		ax.plot(300+behav_coord[i,:]+i*300)
	# ax.yaxis.set_visible(False)
	# plt.subplots_adjust(hspace = .001)
	fig.set_size_inches(17,7)
	ax.set_xlabel('Frame')
	ax.set_ylabel('Neuron')

	if dir_save is None: plt.show()
	else: 
		if title is None: 
			plt.savefig(os.path.join(dir_save, 'neural-behav'+'.png'))
			plt.savefig(os.path.join(dir_save, 'neural-behav'+'.pdf'))
		else: 
			plt.savefig(os.path.join(dir_save, title+'.png'))
			plt.savefig(os.path.join(dir_save, title+'.pdf'))

def plot_coord_all(behavior_coord, title=None, dir_save=None, idx_split_list = None):
	labels = ['rfx','rfy','rhx','rhy','lfx','lfy','lhx','lhy']
	fig, axes = plt.subplots(behavior_coord.shape[0], 1)
	for i in range(behavior_coord.shape[0]):
		axes[i].plot(behavior_coord[i,:], c=limb_color_list[i])
		if i != behavior_coord.shape[0]-1:
			axes[i].xaxis.set_visible(False)
		axes[i].set_ylabel(labels[i])
		if idx_split_list is not None:
			for j in idx_split_list:
				axes[i].axvline(x=j, color='red')
	plt.subplots_adjust(hspace = .001)
	if title is not None: fig.suptitle(title)
	fig.set_size_inches(17,7)

	if dir_save is None: plt.show()
	else: 
		if title is None:
			plt.savefig(os.path.join(dir_save, 'coord'+'.png'))
			plt.savefig(os.path.join(dir_save, 'coord'+'.pdf'))
		else:
			plt.savefig(os.path.join(dir_save, title+'.png'))
			plt.savefig(os.path.join(dir_save, title+'.pdf'))
	plt.close()


def plot_neural_avg(neural_data, title=None, dir_save=None):
	fig = plt.figure()
	avg = np.average(neural_data, axis=0)
	plt.plot(avg)
	plt.xlim([0, neural_data.shape[1]])
	plt.ylim([np.min(avg), np.max(avg)])
	# mng = plt.get_current_fig_manager()
	# mng.full_screen_toggle()
	#plt.show()
	fig.set_size_inches(17,7)
	if dir_save is None: 
		plt.show()
	else: 
		if title is None: title='neural_avg'
		plt.savefig(os.path.join(dir_save, title+'.png'))
		plt.savefig(os.path.join(dir_save, title+'.pdf'))
	plt.close()

def plot_trace(seq, color, ylim=None, x_inch=8, y_inch=2, yaxis=True, path_save=None):
	fig = plt.figure()

	ax1 = plt.axes(frameon=False)
	ax1.set_frame_on(False)
	if yaxis: ax1.get_yaxis().tick_left()
	else: ax1.axes.get_yaxis().set_visible(False)
	ax1.axes.get_xaxis().set_visible(False)
	# xmin, xmax = ax1.get_xaxis().get_view_interval()
	# ymin, ymax = ax1.get_yaxis().get_view_interval()
	#ax1.add_artist(Line2D((0, 0), (ymin, ymax), color='black', linewidth=2))
	# if axis_off:
	#     plt.xticks([]) 
	#     plt.yticks([]) 
	#     plt.box(False)

	ax1.plot(seq, color=color)
	ax1.set_xlim([0, seq.shape[0]])
	if ylim is None: 
		ax1.set_ylim(np.min(seq), np.max(seq))
		print('seq y min,max: ', np.min(seq), np.max(seq))
	else: ax1.set_ylim(ylim)

	fig.set_size_inches(x_inch, y_inch)

	plt.tight_layout(h_pad=0.01, w_pad=0.01)

	if path_save is None: plt.show()
	else:
		plt.savefig(path_save)

def save_vid_frame(frame, dir_save=None, title=None):
	if title is None: path_save = os.path.join(dir_save, 'imshow.png')
	else: path_save = os.path.join(dir_save, title+'.png')
	
	i = Image.fromarray(frame)
	i.save(path_save)  



def plot_coord_frame(fx, fy, hx, hy, frame, path_save=None, title=None):
	# fx = fx_arr[idx_frame]
	# fy = fy_arr[idx_frame]
	# hx = hx_arr[idx_frame]
	# hy = hy_arr[idx_frame]
	# frame = vid[idx_frame]

	fig = plt.figure()
	plt.imshow(frame)
	plt.scatter(fx, fy, label='f')
	plt.scatter(hx, hy, label='h')
	plt.legend()
	if title is not None:
		fig.suptitle(title)
	if path_save is None: plt.show()
	else: plt.savefig(path_save)
	plt.close()

def plot_coord_frame_twolimb(vid_frame_limb1, vid_frame_limb2, fig_size, gt, pred=None, dir_save=None, name_save=None, flag_legend=True, coord_likelihood=None, flag_text=False):
	'''
	vid_frame: (y,x,3)
	fig_size: [x,y], inch
	coord_likelihood: (4)
	'''
	fig, ax = plt.subplots(1,2)

	ax[0].imshow(vid_frame_limb1)
	ax[0].scatter(gt[0], gt[1], label='f_gt', color='blue')
	ax[0].scatter(gt[2], gt[3], label='h_gt', color='dodgerblue')
	if pred is not None:
		ax[0].scatter(pred[0], pred[1], label='f_pred', color='red')
		ax[0].scatter(pred[2], pred[3], label='h_pred', color='tomato')
	if coord_likelihood is not None:
		ax[0].text(gt[0], gt[1], "{:.4f}".format(coord_likelihood[0]), color='orange')
		ax[0].text(gt[2], gt[3], "{:.4f}".format(coord_likelihood[1]), color='orange')
	if flag_text:
		ax[0].text(gt[0], gt[1]+20, "({:.2f},{:.2f})".format(gt[0], gt[1]), color='blue')
		ax[0].text(gt[2], gt[3]+20, "({:.2f},{:.2f})".format(gt[2], gt[3]), color='blue')
	ax[0].set_xlim((0, vid_frame_limb1.shape[1]))
	ax[0].set_ylim((vid_frame_limb1.shape[0], 0))
	ax[0].axis('off')

	ax[1].imshow(vid_frame_limb2)
	ax[1].scatter(gt[4], gt[5], label='f_gt', color='blue')
	ax[1].scatter(gt[6], gt[7], label='h_gt', color='dodgerblue')
	if pred is not None:
		ax[1].scatter(pred[4], pred[5], label='f_pred', color='red')
		ax[1].scatter(pred[6], pred[7], label='h_pred', color='tomato')
	if coord_likelihood is not None:
		ax[1].text(gt[4], gt[5], "{:.4f}".format(coord_likelihood[2]), color='orange')
		ax[1].text(gt[6], gt[7], "{:.4f}".format(coord_likelihood[3]), color='orange')
	if flag_text:
		ax[1].text(gt[4], gt[5]+20, "({:.2f},{:.2f})".format(gt[4], gt[5]), color='blue')
		ax[1].text(gt[6], gt[7]+20, "({:.2f},{:.2f})".format(gt[6], gt[7]), color='blue')
	ax[1].set_xlim((0, vid_frame_limb2.shape[1]))
	ax[1].set_ylim((vid_frame_limb2.shape[0], 0))
	ax[1].axis('off')
	
	pos = ax[1].get_position()
	ax[1].set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
	if flag_legend: ax[1].legend(loc='center right', bbox_to_anchor=(1.3, 0.5)) #(1.25, 0.5)
	plt.subplots_adjust(wspace=0, hspace=0)
	fig.tight_layout()
	fig.set_size_inches(fig_size[0], fig_size[1])

	if dir_save != None and name_save != None:
		plt.savefig(os.path.join(dir_save, name_save+'.png'))
		plt.close()
	else: plt.show()

def plot_coord_frame_twolimb_with_trace(vid_frame_limb1, vid_frame_limb2, gt_total, idx_limb, idx_coord, dir_save=None, name_save=None, coord_likelihood=None, flag_text=False, range_limb=None):
	'''
	vid_frame: (y,x,3)
	fig_size: [x,y], inch
	gt: (8, coord_frame)
	coord_likelihood: (4,)
	'''
	fig_size = (10,8)

	fig = plt.figure()
	gs = fig.add_gridspec(2,2)
	ax1 = fig.add_subplot(gs[0,:])
	ax2 = fig.add_subplot(gs[1,0])
	ax3 = fig.add_subplot(gs[1,1])
	
	len_gt_plot = 90
	gt_plot_start = idx_coord-int(len_gt_plot/2)
	if gt_plot_start < 0: gt_plot_start = 0
	gt_plot_end = idx_coord+int(len_gt_plot/2)
	if gt_plot_end > gt_total.shape[1]-1: gt_plot_end = gt_total.shape[1]-1

	#ax1.plot(gt_total[idx_limb,gt_plot_start:gt_plot_end+1], color=limb_color_list[idx_limb], zorder=10, marker='x')
	ax1.plot(gt_total[idx_limb,:], color=limb_color_list[idx_limb], zorder=10, marker='x')
	ax1.scatter(idx_coord, gt_total[idx_limb,idx_coord], color='red', zorder=100)
	ax1.set_xlim([gt_plot_start, gt_plot_end+1])
	if range_limb is not None:
		ax1.set_xlim([range_limb[0], range_limb[1]+1])       
	#ax1.set_title(name_save)
 
	limb_name_list = ['Right front, x', 'Right front, y','Right hind, x', 'Right hind, y', 'Left front, x', 'Left front, y','Left hind, x', 'Left hind, y'] 
	ax1.set_title(limb_name_list[idx_limb])

	# frames with coordinates
	gt = gt_total[:,idx_coord]
	ax2.imshow(vid_frame_limb1)
	ax2.scatter(gt[0], gt[1], label='f_gt', color='blue')
	ax2.scatter(gt[2], gt[3], label='h_gt', color='dodgerblue')
	if coord_likelihood is not None:
		# ax2.text(gt[0], gt[1], "{:.4f}".format(coord_likelihood[0]), color='orange')
		# ax2.text(gt[2], gt[3], "{:.4f}".format(coord_likelihood[1]), color='orange')
		ax2.text(vid_frame_limb1.shape[1]-150, 20, "{:.4f}".format(coord_likelihood[0]), color='orange')
		ax2.text(20, 20, "{:.4f}".format(coord_likelihood[1]), color='orange')
	if flag_text:
		# ax2.text(gt[0], gt[1]+20, "({:.2f},{:.2f})".format(gt[0], gt[1]), color='blue')
		# ax2.text(gt[2], gt[3]+20, "({:.2f},{:.2f})".format(gt[2], gt[3]), color='blue')
		ax2.text(vid_frame_limb1.shape[1]-150, 40, "({:.2f},{:.2f})".format(gt[0], gt[1]), color='cyan')
		ax2.text(20, 40, "({:.2f},{:.2f})".format(gt[2], gt[3]), color='cyan')
	ax2.set_xlim((0, vid_frame_limb1.shape[1]))
	ax2.set_ylim((vid_frame_limb1.shape[0], 0))
	#ax2.axis('off')

	ax3.imshow(vid_frame_limb2)
	ax3.scatter(gt[4], gt[5], label='f_gt', color='blue')
	ax3.scatter(gt[6], gt[7], label='h_gt', color='dodgerblue')
	if coord_likelihood is not None:
		# ax3.text(gt[4], gt[5], "{:.4f}".format(coord_likelihood[2]), color='orange')
		# ax3.text(gt[6], gt[7], "{:.4f}".format(coord_likelihood[3]), color='orange')
		ax3.text(20, 20, "{:.4f}".format(coord_likelihood[2]), color='orange')
		ax3.text(vid_frame_limb2.shape[1]-150, 20, "{:.4f}".format(coord_likelihood[3]), color='orange')
	if flag_text:
		# ax3.text(gt[4], gt[5]+20, "({:.2f},{:.2f})".format(gt[4], gt[5]), color='cyan')
		# ax3.text(gt[6], gt[7]+20, "({:.2f},{:.2f})".format(gt[6], gt[7]), color='cyan')
		ax3.text(20, 40, "({:.2f},{:.2f})".format(gt[4], gt[5]), color='cyan')
		ax3.text(vid_frame_limb2.shape[1]-150, 40, "({:.2f},{:.2f})".format(gt[6], gt[7]), color='cyan')
	ax3.set_xlim((0, vid_frame_limb2.shape[1]))
	ax3.set_ylim((vid_frame_limb2.shape[0], 0))
	#ax3.axis('off')
	
	# pos = ax[2,1].get_position()
	# ax[2,1].set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
	# if flag_legend: ax[2,1].legend(loc='center right', bbox_to_anchor=(1.3, 0.5)) #(1.25, 0.5)
	fig.set_size_inches(fig_size[0], fig_size[1])
	plt.subplots_adjust(wspace=0, hspace=0)
	fig.tight_layout()

	if dir_save != None: # and name_save != None:
		#plt.savefig(os.path.join(dir_save, name_save+'.png'))
		plt.savefig(os.path.join(dir_save, 'limb_'+str(idx_limb)+'_'+str(idx_coord)+'.png'))
		plt.close()
	else: 
		plt.show()


def plot_two_coords_zoom(coord_single_1, coord_single_2, label_coord, ylim=None, xtick=None, fig_title=None, dir_save=None, file_name=None):
	'''
	plot two single coords, for zooming
	xtick: [start,end(not included),size]
	'''

	# # Editable pdf
	# import matplotlib
	# matplotlib.rcParams['pdf.fonttype'] = 42
	# matplotlib.rcParams['ps.fonttype'] = 42

	if len(coord_single_1) != len(coord_single_2):
		print('len(coord_single_1) != len(coord_single_2)')
		exit()

	fig, ax = plt.subplots()
	plt.plot(range(0,len(coord_single_1),1), coord_single_1, '-x', label=label_coord[0], color='blue')
	plt.plot(range(0,len(coord_single_2),1), coord_single_2, '-x', label=label_coord[1], color='red', alpha=0.7)

	if ylim is not None: plt.ylim(ylim)
	ax.set_xlabel('Frame', fontsize=15)
	ax.set_ylabel('Coordinate', fontsize=15)
	ax.tick_params(axis='x', which='major', labelsize=15)
	ax.tick_params(axis='y', which='major', labelsize=15)
	if xtick is not None: 
		ax.set_xticks(range(0,xtick[1]-xtick[0]+1,xtick[2]))
		ax.set_xticklabels(range(xtick[0], xtick[1]+1, xtick[2]))
	if fig_title is not None: plt.suptitle(fig_title)
	plt.legend()
	fig.set_size_inches(16, 7)
	if dir_save is None: plt.show()
	else:
		if file_name is None:
			print('Input file_name.')
			exit()
		else:
			plt.savefig(os.path.join(dir_save, file_name+'.png'))
			# plt.savefig(os.path.join(dir_save, file_name+'.pdf'))
			plt.close()

def plot_output_result(gt, pred, idx_coord, ylim=None, title=None, dir_result=None, mean=False):

	if isinstance(idx_coord, list):
		# x,y coordinates
		if len(idx_coord)==2:
			fig, axes = plt.subplots(1,2)
			axes[0].plot(gt[idx_coord[0],:], label='gt', color='blue')
			axes[0].plot(pred[idx_coord[0],:], label='pred', color='red', alpha=0.7)
			axes[0].set_title('x', fontsize=20)
			axes[0].legend()
			if mean:
				mean_val = np.mean(gt[idx_coord[0],:])
				axes[0].axhline(y=mean_val, color='green', linestyle='-')

			#axes[0].legend()
			axes[1].plot(gt[idx_coord[1],:], label='gt', color='blue')
			axes[1].plot(pred[idx_coord[1],:], label='pred', color='red', alpha=0.7)
			axes[1].set_title('y', fontsize=20)
			axes[1].legend()
			if mean:
				mean_val = np.mean(gt[idx_coord[1],:])
				axes[1].axhline(y=mean_val, color='green', linestyle='-')

			for i in range(2):
				axes[i].set_xlabel('Coordinate frame', fontsize=15)
				axes[i].set_ylabel('Coordinate', fontsize=15)
				axes[i].tick_params(axis='x', which='major', labelsize=10)
				axes[i].tick_params(axis='y', which='major', labelsize=15)

			fig.set_size_inches(16, 3)
			plt.tight_layout(pad=1)
			
	# one coordinate
	else:
		fig = plt.figure()
		plt.plot(gt[idx_coord,:][0], label='gt', color='blue')
		plt.plot(pred[idx_coord,:][0], label='pred', color='red', alpha=0.7)

		if mean:
			mean_val = np.mean(gt[idx_coord,:])
			plt.axhline(y=mean_val, color='green', linestyle='-')

		if ylim is not None: plt.ylim(ylim)
		plt.xlabel('Coordinate frames')
		plt.ylabel('Coordinate')
		plt.legend()
		fig.set_size_inches(16, 7)

	if dir_result is None: plt.show()
	else:
		if title is None: 
			plt.savefig(os.path.join(dir_result, 'plot_output_result.pdf'))
			plt.savefig(os.path.join(dir_result, 'plot_output_result.png'))
		else:
			plt.savefig(os.path.join(dir_result, title+'.pdf'))
			plt.savefig(os.path.join(dir_result, title+'.png'))
		plt.close()


def plot_neuron_importance(ops_data, stat, importance, title, maximg=None, dir_save=None, minmax_list=None, cortex_map=None):
	'''
	cortex_map: m1_moved, s1fl_moved, s1hl_moved, , w1_moved
	'''
	if importance.shape[0] != stat.shape[0]:
		print('Number of neurons is different between grd_norm and stat')
		exit()
	
	im = np.ones((ops_data['Ly'], ops_data['Lx'])) * -1
	for n in range(stat.shape[0]):
		ypix = stat[n]['ypix'][~stat[n]['overlap']]
		xpix = stat[n]['xpix'][~stat[n]['overlap']]
		im[ypix,xpix] = importance[n]#n+1

	# # save im
	# if dir_save is not None:
	#     np.save(os.path.join(dir_save, title+'_img.npy'), im)

	fig = plt.figure()
	cmap = 'Greens' #'jet'
	# plt.imshow(maximg, cmap='gray')
	if minmax_list is None:
		im_show = plt.imshow(np.flipud(im), cmap=cmap, origin='lower', vmin=0.0, vmax=1.0) #np.flipud(im)
	else:
		im_show = plt.imshow(np.flipud(im), cmap=cmap, vmin=minmax_list[0], vmax=minmax_list[1], origin='lower')
	if maximg is not None:
		plt.imshow(np.flipud(maximg), cmap='gray', alpha=0.4, origin='lower', zorder=100)
	plt.colorbar(im_show, fraction=0.047*im.shape[0]/im.shape[1])   

	# cortex map
	if cortex_map is not None:
		m1_moved = cortex_map[0] 
		s1fl_moved = cortex_map[1] 
		s1hl_moved = cortex_map[2] 
		w1_moved = cortex_map[3] 

		xlim = np.max(w1_moved[:,1])
		ylim = np.max(w1_moved[:,0])

		im_x = im.shape[1]
		im_y = im.shape[0]

		ratio_x = im_x/xlim
		ratio_y = im_y/ylim

		color_list = ['orange', 'cyan', 'lime']
		linewidth = 1.5
		plt.plot(m1_moved[:,0]*ratio_x, m1_moved[:,1]*ratio_y, color=color_list[0], zorder=100, linewidth=linewidth)
		plt.plot(m1_moved[:,0]*ratio_x, m1_moved[:,2]*ratio_y, color=color_list[0], zorder=100, linewidth=linewidth)
		plt.plot(s1fl_moved[:,0]*ratio_x, s1fl_moved[:,1]*ratio_y, color=color_list[1], zorder=100, linewidth=linewidth)
		plt.plot(s1fl_moved[:,0]*ratio_x, s1fl_moved[:,2]*ratio_y, color=color_list[1], zorder=100, linewidth=linewidth)
		plt.plot(s1hl_moved[:,0]*ratio_x, s1hl_moved[:,1]*ratio_y, color=color_list[2], zorder=100, linewidth=linewidth)
		plt.plot(s1hl_moved[:,0]*ratio_x, s1hl_moved[:,2]*ratio_y, color=color_list[2], zorder=100, linewidth=linewidth)

		plt.xlim([0,im_x])
		plt.ylim([0,im_y])

	plt.xticks([])
	plt.yticks([])
	plt.tight_layout(pad=0.1)
	fig.set_size_inches(13,8)
	#plt.show()
	plt.title(title)
	if dir_save is not None:
		plt.savefig(os.path.join(dir_save, title+'.png'))
		plt.savefig(os.path.join(dir_save, title+'.pdf'), dpi=300)
		# plt.savefig(os.path.join(dir_save, title+'.svg'), format='svg', dpi=300)

		plt.close(fig)
	else: plt.show()
	plt.close()


### pca2d-time, Color coding-time
def plot_2d_over_time(dim1, dim2, dir_save=None, name_save=None):
	'''
	X_dimred:  (3, 9361) (n_comp, time)
	'''

	time = np.array([x for x in range(dim1.shape[0])])
	# print('dim1: ', dim1.shape)
	# print('dim2: ', dim2.shape)

	### interpolation
	dim1_interp, dim2_interp = utils.interp_xy_1d(dim1, dim2, len_win=3, num_interp=5)

	time_interp = np.linspace(0, dim1.shape[0], len(dim1_interp))
	#print('time_interp: ', time_interp.shape)

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	# ax.view_init(30, 330)
	# ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 0.3, 0.3, 1]))

	plt.plot(time_interp, dim1_interp, dim2_interp, '-') #, marker='x', markersize=3)
	
	sc = ax.scatter(time, dim1, dim2, edgecolor='black', linewidth=0.1, alpha=0.8, c=time, cmap='jet')
	plt.colorbar(sc, ax=ax, shrink=0.4, pad=0.1)

	ax.set_box_aspect(aspect = (1,0.25,0.25))
	ax.set_xlabel('time', labelpad=25)
	ax.set_ylabel('dim-1')
	ax.set_zlabel('dim-2')
	fig.set_size_inches(10, 10)
	fig.tight_layout()

	if dir_save is None:
		plt.show()
	else:
		if name_save is None:
			plt.savefig(os.path.join(dir_save, 'plot_Xdimred_time.png'))
			plt.savefig(os.path.join(dir_save, 'plot_Xdimred_time.pdf'))
		else:
			plt.title(name_save)
			plt.savefig(os.path.join(dir_save, name_save+'.png'))
			plt.savefig(os.path.join(dir_save, name_save+'.pdf'))
		plt.close()


def plot_vid_BCI_award():
	dir_result = r"C:\Users\park1377\SeungbinPark\Results\20230310_Animal1\NN\Run\20230824_145730_Animal1_likeli01_run"
	best_epoch = '7'
	dir_data = r"C:\Users\park1377\Box\SeungbinPark\Research_DadarlatLab_2\Megan_data_working\Animal1-G8_53950_1L_Redo\prepare_data\Animal1_likeli01"
	dir_data_working = r"E:\Megan_copied\Group 8\M1_Mapping\53950_1L_Redo"
	path_vid_1 = os.path.join(dir_data_working, 'LIMB1_53950_1L_2022-05-13_1.avi') 
	path_vid_2 = os.path.join(dir_data_working, 'LIMB2_53950_1L_2022-05-13_1.avi') 
	idx_coord_test_plot = [1582,2793] #None

	###############################################################################################
	dir_save = os.path.join(dir_result, 'plot_vid_BCI_award')
	if os.path.isdir(dir_save):
		print('Directory exists')
		#exit()
	else:
		os.mkdir(dir_save)

	dir_save_idxcoord = os.path.join(dir_save, 'idx_coord_test_plot_'+str(idx_coord_test_plot[0])+'-'+str(idx_coord_test_plot[1]))
	if os.path.isdir(dir_save_idxcoord):
		print('Directory exists')
		#exit()
	else:
		os.mkdir(dir_save_idxcoord)
	
	# Load train-test indices
	test_idx_cv = np.load(os.path.join(dir_data, 'test_idx_cv.npy'), allow_pickle=True)
	test_idx = test_idx_cv[-1]
	train_idx = [0, test_idx[0]-1]
	test_idx_start_neural = test_idx[0]
	print('test_idx: ', test_idx)
	print('train_idx: ', train_idx)
	print('test_idx_start_neural: ', test_idx_start_neural)

	# Load idx_coord_neural
	idx_coord_neural = np.load(os.path.join(dir_data, 'idx_coord_neural.npy'))
	test_idx_start_coord = np.where(idx_coord_neural == test_idx_start_neural)[0][0]
	print('idx_coord_neural: ', idx_coord_neural.shape, idx_coord_neural[:10], idx_coord_neural[-10:])
	print('test_idx_start_coord: ', test_idx_start_coord)

	# Load idx_coord_vid
	idx_coord_vid = np.load(os.path.join(dir_data, 'idx_coord_vid.npy'))
	print('idx_coord_vid: ', idx_coord_vid.shape, idx_coord_vid[:10], idx_coord_vid[-10:])

	# Load idx_coord_stack
	idx_coord_stack = utils.load_idx_coord_stack(dir_result, best_epoch)
	print('idx_coord_stack: ', idx_coord_stack.shape, idx_coord_stack[:10], idx_coord_stack[-10:])

	# Load limb coordinates
	gt = np.load(os.path.join(dir_result, 'gt_norm_converted_epoch'+str(best_epoch))+'.npy')
	pred = np.load(os.path.join(dir_result, 'pred_norm_converted_epoch'+str(best_epoch))+'.npy')
	print('gt: ', gt.shape)
	print('pred: ', pred.shape)
	
	gt_plot = gt[:, idx_coord_test_plot[0]:idx_coord_test_plot[1]+1]
	pred_plot = pred[:, idx_coord_test_plot[0]:idx_coord_test_plot[1]+1]
	print('gt_plot: ', gt_plot.shape)
	print('pred_plot: ', pred_plot.shape)

	# # Load neural data
	# spks_z_sel = np.load(os.path.join(dir_data, 'spks_z_sel.npy'))
	# ffneu_z_sel = np.load(os.path.join(dir_data, 'ffneu_z_sel.npy'))
	# print('spks_z_sel: ', spks_z_sel.shape)
	# print('ffneu_z_sel: ', ffneu_z_sel.shape)

	# ffneu_z_sel_avg = np.mean(ffneu_z_sel, axis=0)
	# print('ffneu_z_sel_avg: ', ffneu_z_sel_avg.shape)

	idx_neural_real_plot_start = idx_coord_neural[idx_coord_stack[idx_coord_test_plot[0]]]
	idx_neural_real_plot_end = idx_coord_neural[idx_coord_stack[idx_coord_test_plot[-1]]]
	print('idx_neural_real_plot_start: ', idx_neural_real_plot_start)
	print('idx_neural_real_plot_end: ', idx_neural_real_plot_end)

	# # Custom cmap for calcium
	# cval_min, cval_max = 999, -999
	# for i in range(idx_neural_real_plot_start, idx_neural_real_plot_end+1):
	#     calcium_slice = np.load(os.path.join(dir_data, 'calcium_sel', str(i)+'.npy'))
	#     min_val = np.amin(calcium_slice)
	#     max_val = np.amax(calcium_slice)
	#     del calcium_slice
	#     if cval_min > min_val: cval_min = min_val
	#     if cval_max < max_val: cval_max = max_val
	# norm = plt.Normalize(cval_min, cval_max)
	# cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["black","green"])

	# label_coord_x = -0.02
	# label_coord_y = 0.2
	coord_to_show = 25
	vid_crop_top = 100
	color_axvline = 'magenta'
	color_limb_four = ['Red', 'darkorange', 'limegreen', 'darkviolet']
	color_limb_four_gt = ['black', 'black', 'black', 'black']#['lightcoral', 'bisque', 'yellowgreen', 'plum']
	coord_limb_name = ['RFX', 'RFY', 'RHX', 'RHY', 'LFX', 'LFY', 'LHX', 'LHY']
	for idx_coord_plot in range(gt_plot.shape[1]):
		# Get real indices
		idx_coord_test = idx_coord_test_plot[0]+idx_coord_plot
		idx_coord_real = idx_coord_stack[idx_coord_test]
		idx_neural_real = idx_coord_neural[idx_coord_real]

		print('idx_coord_plot: ', idx_coord_plot, gt_plot.shape[1], ', idx_coord_real: ', idx_coord_real, ', idx_neural_real: ', idx_neural_real)

		# Load video frame
		vid_1_sel = utils.load_video_cv2(path_video=path_vid_1, frame_idx=idx_coord_vid[idx_coord_real])[0]
		vid_2_sel = utils.load_video_cv2(path_video=path_vid_2, frame_idx=idx_coord_vid[idx_coord_real])[0]

		# Make figure
		fig = plt.figure()
		gs0 = fig.add_gridspec(2,1, wspace=0, hspace=0, left=0.1)
		gs00 = gridspec.GridSpecFromSubplotSpec(1,4,subplot_spec=gs0[0], wspace=0, width_ratios=[1,0.05,0.8,0.8])
		gs10 = gridspec.GridSpecFromSubplotSpec(4,2,subplot_spec=gs0[1])
		#grid = fig.add_gridspec(9, 4, width_ratios=[1,0.1,1,1], left=0.1, wspace=0, hspace=0)

		# calcium image
		ax1 = fig.add_subplot(gs00[0,0])
		# ax1 = fig.add_subplot(grid[:4, 0])
		#ax1.set_title('Two-photon Calcium Image')
		calcium_slice = np.load(os.path.join(dir_data, 'calcium_sel', str(idx_neural_real)+'.npy'))
		ax1.imshow(calcium_slice, cmap="viridis") #norm=norm, aspect="auto"
		ax1.axis('off')
		ax1.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
		#ax1.text(calcium_slice.shape[1]*0.45, calcium_slice.shape[0]*0.95, 'neural frame # = '+str(idx_neural_real), color='white', fontsize=10)

		# vid_1
		ax2 = fig.add_subplot(gs00[0,2])
		#ax2 = fig.add_subplot(grid[:4, 2])
		#ax2.set_title('Right')
		ax2.imshow(vid_1_sel) #, aspect='auto'
		ax2.scatter(gt_plot[0, idx_coord_plot], gt_plot[1, idx_coord_plot], color=color_limb_four_gt[0])
		ax2.scatter(gt_plot[2, idx_coord_plot], gt_plot[3, idx_coord_plot], color=color_limb_four_gt[1])
		ax2.scatter(pred_plot[0, idx_coord_plot], pred_plot[1, idx_coord_plot], color=color_limb_four[0])
		ax2.scatter(pred_plot[2, idx_coord_plot], pred_plot[3, idx_coord_plot], color=color_limb_four[1])
		ax2.set_xlim([0, vid_1_sel.shape[1]])
		ax2.set_ylim([vid_1_sel.shape[0],vid_crop_top])
		#ax2.axis('off')
		ax2.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

		# vid_2
		ax3 = fig.add_subplot(gs00[0,3])
		#ax3 = fig.add_subplot(grid[:4, 3])
		#ax3.set_title('Left')
		ax3.imshow(vid_2_sel) #aspect='auto'
		ax3.scatter(gt_plot[4, idx_coord_plot], gt_plot[5, idx_coord_plot], color=color_limb_four_gt[2])
		ax3.scatter(gt_plot[6, idx_coord_plot], gt_plot[7, idx_coord_plot], color=color_limb_four_gt[3])
		ax3.scatter(pred_plot[4, idx_coord_plot], pred_plot[5, idx_coord_plot], color=color_limb_four[2])
		ax3.scatter(pred_plot[6, idx_coord_plot], pred_plot[7, idx_coord_plot], color=color_limb_four[3])
		ax3.set_xlim([0, vid_1_sel.shape[1]])
		ax3.set_ylim([vid_1_sel.shape[0],vid_crop_top])
		#ax3.axis('off')
		ax3.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

		# # Average fluorescence
		# ax4 = fig.add_subplot(grid[11, :])
		# ax4.set_ylabel('Avg\ndF/F', rotation=0)
		# ax4.yaxis.set_label_coords(label_coord_x,label_coord_y-0.1)
		# ax4.plot(list(range(idx_neural_real_plot_start, idx_neural_real_plot_end+1)), ffneu_z_sel_avg[idx_neural_real_plot_start:idx_neural_real_plot_end+1], color='black')
		# ax4.axvline(x=idx_neural_real, color=color_axvline)
		# ax4.set_xlim([idx_neural_real_plot_start-5, idx_neural_real_plot_start+gt_plot.shape[1]+5])

		# limbs
		for idx_limb in range(gt.shape[0]):
			ax = fig.add_subplot(gs10[int(idx_limb/2),idx_limb%2])
			#ax5 = fig.add_subplot(grid[5, :2])
			#ax5.set_ylabel('RFX', rotation=0)
			#ax5.yaxis.set_label_coords(label_coord_x,label_coord_y)
			ax.plot(list(range(0,gt_plot.shape[1])), gt_plot[idx_limb,:], color='black')
			ax.plot(list(range(0,gt_plot.shape[1])), pred_plot[idx_limb,:], color=color_limb_four[int(idx_limb/2)], alpha=0.7)
			ax.axvline(x=idx_coord_plot, color=color_axvline)
			ax.text(-0.1, 0.5, coord_limb_name[idx_limb], transform=ax.transAxes)
			ax.set_xlim([idx_coord_plot-coord_to_show, idx_coord_plot+coord_to_show])
			ax.set_ylim([np.min(gt_plot[idx_limb,:]), np.max(gt_plot[idx_limb,:])])
			ax.axis('off')
			ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
		
		# # showing whole sequence version
		# ax6 = fig.add_subplot(grid[5, 2:])
		# ax6.set_ylabel('RFY', rotation=0)
		# ax6.yaxis.set_label_coords(label_coord_x,label_coord_y)
		# ax6.plot(list(range(0,gt_plot.shape[1])), gt_plot[1,:], color='black')
		# ax6.plot(list(range(0,gt_plot.shape[1])), pred_plot[1,:], color=color_limb_four[0], alpha=0.7)
		# ax6.axvline(x=idx_coord_plot, color=color_axvline)
		# ax6.set_xlim([0-5,gt_plot.shape[1]+5])

		# Save figure
		fig.set_size_inches(12,7)
		#plt.tight_layout()
		plt.savefig(os.path.join(dir_save_idxcoord, str(idx_coord_plot).rjust(4, '0')+'.png'))
		plt.close()
		#plt.show()


def plot_pca_2d(pca_1, pca_2, title=None, dir_save=None):
	'''
	plot pca projected in 2d
	'''
	fig = plt.figure()
	pca_interp_1, pca_interp_2 = utils.interp_xy_1d(pca_1, pca_2, len_win=5, num_interp=10)
	plt.plot(pca_interp_1, pca_interp_2, alpha=0.5, zorder=10, color='black')
	plt.scatter(pca_1, pca_2, marker='o', s=10, c=range(pca_1.shape[0]), cmap='jet', zorder=100)
	#plt.plot(hid_pca[:,0], hid_pca[:,1])
	plt.colorbar()
	if title is not None: fig.suptitle(title)
	fig.tight_layout()
	if dir_save is None:
		plt.show()
	else:
		fig.savefig(os.path.join(dir_save, title+'.png'))
		plt.close()



if __name__ == '__main__':
	plot_vid_BCI_award()
