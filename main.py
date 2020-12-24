import open3d as o3d
import numpy as np
from random import randrange
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_shortest_path
from sklearn.neighbors import BallTree
from sklearn import manifold
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from tsp_solver.greedy import solve_tsp
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import rasterfairy
from skimage.restoration import inpaint
import rpack 
from rectpack import newPacker
import math
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sys
import PIL.Image as PImage
import sklearn
# from scipy.optimize import minimize
from scipy import ndimage, optimize

if not sys.warnoptions:
	import warnings
	warnings.simplefilter("ignore")

from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
import statistics

from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import time
from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import SVC
import re
import subprocess
##############################################################################################################################
# 																						 									 #
#											(1) Initialization and Preprocessing											 #
# 																						 									 #
##############################################################################################################################
# Read ply file
def init(filename):
	pcd = o3d.io.read_point_cloud(filename)
	geo_arr = np.asarray(pcd.points)
	rgb_arr = np.asarray(pcd.colors)*255
	normal_arr = np.asarray(pcd.normals)

	point_num = len(geo_arr)
	minPt = [min(geo_arr[:,0]), min(geo_arr[:,1]), min(geo_arr[:,2])]
	maxPt = [max(geo_arr[:,0]), max(geo_arr[:,1]), max(geo_arr[:,2])]
	pc_width = (maxPt[0]-minPt[0])
	return [geo_arr, rgb_arr, normal_arr, point_num, pc_width]

# Read off (simplified point cloud) file
def read_off(off_path, geo_arr, rgb_arr):
	pt_arr = []
	off_rgb_arr = []
	idx = 0
	with open(off_path) as ins:
		for line in ins:
			re2 = line.replace("\n", "").split(" ")
			if idx>1:
				pt = [float(val) for val in re2[0:3]]
				pt_arr.append(pt)
				off_rgb_arr.append([int(val) for val in re2[3:6]])
			idx = idx + 1
	off_pt_num = len(pt_arr)
	return [pt_arr, off_rgb_arr, off_pt_num]

def assign_ply_to_off(off_geo_arr, geo_arr, vis_flag):
	off_pt_assign_dic = dict()

	features = []
	labels = []

	for i in range(len(off_geo_arr)):
		features.append(off_geo_arr[i])
		labels.append(i)
		off_pt_assign_dic[i] = []

	clf = KNeighborsClassifier(n_neighbors = 1, algorithm = 'ball_tree')
	clf.fit(features, labels)
	pre_label = clf.predict(geo_arr)

	for t in range(0, len(pre_label)):
		seg = pre_label[t]
		off_pt_assign_dic[seg].append(t)

	if vis_flag:
		vis_sc_geo = []
		vis_sc_rgb = []
		for off_id in off_pt_assign_dic:
			rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
			for id in off_pt_assign_dic[off_id]:
				vis_sc_geo.append(geo_arr[id])
				vis_sc_rgb.append(rgb)

		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(vis_sc_geo)
		pcd.colors = o3d.utility.Vector3dVector(np.asarray(vis_sc_rgb)/255.0)
		o3d.visualization.draw_geometries([pcd])

	return [off_pt_assign_dic, pre_label]

def get_patch_orig_info(pt_arr, label, geo_arr, rgb_arr, vis_flag):
	X = np.asarray(pt_arr)
	y = label
	clf = KNeighborsClassifier(n_neighbors=3)
	clf.fit(X, y)
	pre_label = clf.predict(geo_arr)

	off_pt_assign_dic = dict()
	for t in range(0, len(pre_label)):
		seg = pre_label[t]
		if not seg in off_pt_assign_dic:
			off_pt_assign_dic[seg] = []
		off_pt_assign_dic[seg].append(t)

	if vis_flag:
		vis_sc_geo = []
		vis_sc_rgb = []
		for off_id in off_pt_assign_dic:
			rgb = [0, randrange(0, 255), randrange(0, 255)]
			for id in off_pt_assign_dic[off_id]:
				vis_sc_geo.append(geo_arr[id])
				vis_sc_rgb.append(rgb)

		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(vis_sc_geo)
		pcd.colors = o3d.utility.Vector3dVector(np.asarray(vis_sc_rgb)/255.0)
		o3d.visualization.draw_geometries([pcd])

	return off_pt_assign_dic

def assign_ply_to_off_patch(all_tf_iso_off_arr, all_untf_iso_off_arr, geo_arr, vis_flag):
	all_off_geo_arr = all_tf_iso_off_arr + all_untf_iso_off_arr

	off_pt_assign_dic = dict()
	features = []
	labels = []

	for i in range(len(all_off_geo_arr)):
		features = features + all_off_geo_arr[i]
		labels = labels + [i for pt in all_off_geo_arr[i]]
		off_pt_assign_dic[i] = []

	# clf = GaussianNB()
	clf = KNeighborsClassifier(n_neighbors = 8, algorithm = 'ball_tree')
	clf.fit(features, labels)
	pre_label = clf.predict(geo_arr)

	for t in range(0, len(pre_label)):
		seg = pre_label[t]
		off_pt_assign_dic[seg].append(t)

	if vis_flag:
		vis_sc_geo = []
		vis_sc_rgb = []
		for off_id in off_pt_assign_dic:
			rgb = [0, randrange(0, 255), randrange(0, 255)]
			if off_id >=len(all_tf_iso_off_arr):
				rgb = [randrange(0, 255), 0, 0]

			for id in off_pt_assign_dic[off_id]:
				vis_sc_geo.append(geo_arr[id])
				vis_sc_rgb.append(rgb)

		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(vis_sc_geo)
		pcd.colors = o3d.utility.Vector3dVector(np.asarray(vis_sc_rgb)/255.0)
		o3d.visualization.draw_geometries([pcd])

	return off_pt_assign_dic

def assign_ply_to_seg(off_seg_dic, all_pt, vis):
	assign_dic = dict()

	features = []
	labels = []

	for seg in off_seg_dic:
		for pt in off_seg_dic[seg]:
			features.append(pt)
			labels.append(seg)
		assign_dic[seg] = []





	clf = KNeighborsClassifier(n_neighbors = 3, algorithm = 'ball_tree')
	clf.fit(features, labels)
	pre_label = clf.predict(all_pt)
	
	for t in range(0, len(pre_label)):
		seg = pre_label[t]
		assign_dic[seg].append(t)

	if vis:
		vis_geo = []
		vis_rgb = []
		for label in assign_dic:
			rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
			for idx in assign_dic[label]:
				vis_geo.append(all_pt[idx])
				vis_rgb.append(rgb)
		pc_vis(vis_geo, vis_rgb)

	return assign_dic

# Point cloud visualization
def pc_vis(pc_geo, pc_rgb):
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(pc_geo)
	pcd.colors = o3d.utility.Vector3dVector(np.asarray(pc_rgb)/255.0)
	o3d.visualization.draw_geometries([pcd])

def get_neighbor_dis(pt_arr):
	tot_num = len(pt_arr)
	X = np.array(pt_arr)
	tree = BallTree(X, leaf_size = 1)
	neighbor_num = 2
	dis_arr = []
	for i in range(0, tot_num, max(int(tot_num/1000), 1)):
		dist, ind = tree.query([pt_arr[0]], k=neighbor_num)
		dis_arr.append(dist[0][1])
	neighbor_dis = min(dis_arr)
	return neighbor_dis

def dbscan_clustering(pt_arr, dbscan_thresh):
	clustering = DBSCAN(eps=dbscan_thresh, min_samples=1).fit(pt_arr)
	label_arr = list(clustering.labels_)
	cluster_dic = dict()
	for i in range(len(pt_arr)):
		label = label_arr[i]
		if not label in cluster_dic:
			cluster_dic[label] = []
		cluster_dic[label].append(i)
	return cluster_dic

def dbscan_clustering_vis(off_pt, seg_thresh, min_pt_num, vis_flag):
	clustering = DBSCAN(eps=seg_thresh, min_samples=1).fit(off_pt)
	label_arr = list(clustering.labels_)

	cluster_dic = dict()
	for i in range(len(off_pt)):
		label = label_arr[i]
		if not label in cluster_dic:
			cluster_dic[label] = []
		cluster_dic[label].append(i)
	new_cluster_dic = dict()
	for cluster_id in cluster_dic:
		if len(cluster_dic[cluster_id]) >= min_pt_num:
			new_cluster_dic[cluster_id] = cluster_dic[cluster_id]
	if vis_flag:
		vis_geo = []
		vis_rgb = []
		for cluster_id in new_cluster_dic:
			rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
			if len(cluster_dic[cluster_id]):
				for idx in cluster_dic[cluster_id]:
					pt = off_pt[idx]
					vis_geo.append([pt[0], pt[1], pt[2]])
					vis_rgb.append(rgb)
		pc_vis(vis_geo, vis_rgb)
	return new_cluster_dic

def draw_line(pt1, pt2):
	vec = [pt2[0]-pt1[0], pt2[1]-pt1[1], pt2[2]-pt1[2]]
	line_pt = []
	for i in range(0, 99):
		line_pt.append([(vec[j]*(i+1)/100.0 + pt1[j]) for j in range(0, 3)])
	return line_pt

def get_off_color(geo_arr, rgb_arr, off_geo_arr):
	tree = BallTree(np.array(geo_arr), leaf_size=1)
	off_rgb_arr = []
	for pt in off_geo_arr:
		dist, ind = tree.query([pt], k = 1)
		off_rgb_arr.append(rgb_arr[ind[0][0]])
	return off_rgb_arr

def get_off_normal(geo_arr, normal_arr, off_geo_arr):
	tree = BallTree(np.array(geo_arr), leaf_size=1)
	off_normal_arr = []
	for pt in off_geo_arr:
		dist, ind = tree.query([pt], k = 1)
		off_normal_arr.append(normal_arr[ind[0][0]])
	return off_normal_arr


# Agglomerative clustering
def aggl_clustering(pt_arr, neighbour_num, vis_flag):
	X = np.asarray(pt_arr)
	knn_graph = kneighbors_graph(X, neighbour_num, include_self = False)
	model = AgglomerativeClustering(linkage = 'ward', connectivity = knn_graph, n_clusters = aggl_cluster_num)
	clustering = model.fit(X)
	label_arr = clustering.labels_
	cluster_dic = dict()
	for idx in range(len(label_arr)):
		label = label_arr[idx]
		if not label in cluster_dic:
			cluster_dic[label] = []
		cluster_dic[label].append(idx)

	if vis_flag:
		vis_geo = []
		vis_rgb = []
		for label in cluster_dic:
			rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
			for idx in cluster_dic[label]:
				vis_geo.append(pt_arr[idx])
				vis_rgb.append(rgb)
		pc_vis(vis_geo, vis_rgb)
	return cluster_dic
# Agglomerative clustering
def aggl_norm_clustering(pt_arr, norm_arr, neighbour_num, vis_flag):
	X = np.asarray(norm_arr)
	knn_graph = kneighbors_graph(X, neighbour_num, include_self = False)
	model = AgglomerativeClustering(linkage = 'ward', connectivity = knn_graph, n_clusters = aggl_cluster_num)
	clustering = model.fit(X)
	label_arr = clustering.labels_
	cluster_dic = dict()
	for idx in range(len(label_arr)):
		label = label_arr[idx]
		if not label in cluster_dic:
			cluster_dic[label] = []
		cluster_dic[label].append(idx)

	if vis_flag:
		vis_geo = []
		vis_rgb = []
		for label in cluster_dic:
			rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
			for idx in cluster_dic[label]:
				vis_geo.append(pt_arr[idx])
				vis_rgb.append(rgb)
		pc_vis(vis_geo, vis_rgb)
	return cluster_dic

def cal_cos_angle(vector_1, vector_2):
	unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
	unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
	dot_product = np.dot(unit_vector_1, unit_vector_2)
	angle = np.arccos(dot_product)
	return angle

def cube_clustering(pt_arr, norm_arr, vis_flag):
	cluster_dic = dict()
	for t in range(0, 6):
		cluster_dic[t] = []

	cube_norm_arr = [[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]]
	for t in range(len(pt_arr)):
		angle_arr = [cal_cos_angle(norm_arr[t], cube_norm) for cube_norm in cube_norm_arr]
		cluster_dic[np.argmin(angle_arr)].append(t)
	if vis_flag:
		vis_geo = []
		vis_rgb = []
		for label in cluster_dic:
			rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
			for idx in cluster_dic[label]:
				vis_geo.append(pt_arr[idx])
				vis_rgb.append(rgb)
		pc_vis(vis_geo, vis_rgb)
	return cluster_dic


##############################################################################################################################
# 																						 									 #
#											(2) Coarse Segmentation															 #
# 																						 									 #
##############################################################################################################################
def detect_boundary_points(pt_arr, label):
	tree = BallTree(np.array(pt_arr), leaf_size=1)
	uni_pt_idx = []
	bi_pt_idx = []
	mul_pt_idx = []

	boundary_label_dic = dict()
	for i in range(len(pt_arr)):
		pt = pt_arr[i]
		dist, ind = tree.query([pt], k = 8)
		nei_label = [label[idx] for idx in ind[0]]
		cnt_arr = [[x, nei_label.count(x)] for x in set(nei_label)]
		if len(cnt_arr) <= 1:
			uni_pt_idx.append(i)
		elif len(cnt_arr) > 2:
			mul_pt_idx.append(i)
			boundary_label_dic[i] = list(set(nei_label))
		else:
			if max([val[1] for val in cnt_arr])*1.0/len(ind[0]) >= 0.1:
				bi_pt_idx.append(i)
				boundary_label_dic[i] = list(set(nei_label))
			else:
				uni_pt_idx.append(i)
	return [uni_pt_idx, bi_pt_idx, mul_pt_idx, boundary_label_dic]

def skeleton_adj_line2(skeleton_adj_dic):
	end_node = []
	dual_node = []
	multi_node = []
	for re in skeleton_adj_dic:
		if len(skeleton_adj_dic[re]) == 1:
			end_node.append(re)
		if len(skeleton_adj_dic[re]) > 2:
			multi_node.append(re)
		if len(skeleton_adj_dic[re]) == 2:
			dual_node.append(re)

	assigned_node = []
	chain_dic = dict()
	for node in end_node:
		chain_dic[node] = []
		chain = [node]
		nex_node = skeleton_adj_dic[node][0]
		# chain.append(nex_node)
		while len(skeleton_adj_dic[nex_node])<=2:
			len_1 = len(chain)
			chain.append(nex_node)
			for nod in skeleton_adj_dic[nex_node]:
				if not nod in chain:
					# print("###", nod)
					nex_node = nod
					# if len(skeleton_adj_dic[nex_node])<=2:
					#     chain.append(nex_node)
			len_2 = len(chain)
			if len_1==len_2:
				break
		assigned_node = assigned_node + chain
		chain_dic[node] = chain



	temp_dual_node = []
	for node in dual_node:

		temp = skeleton_adj_dic[node].copy()
		for nex_node in skeleton_adj_dic[node]:
			if nex_node in multi_node:
				temp.remove(nex_node)
		if len(temp)<=1 and not node in assigned_node:
			temp_dual_node.append(node)

	temp_chain = []
	for node in temp_dual_node:
		chain = [node]
		temp = skeleton_adj_dic[node].copy()
		while True:
			for nex_node in temp:
				if nex_node in multi_node or nex_node in chain:
					temp.remove(nex_node)
			if len(temp)>0:
				nex_node = temp[0]
				if nex_node in dual_node and not nex_node in chain:
					chain.append(nex_node)
					temp = skeleton_adj_dic[nex_node].copy()
				else:
					break
		chain = sorted(chain)

		if not chain in temp_chain:
			temp_chain.append(chain)
			assigned_node = assigned_node + chain
			chain_dic[node] = chain


	multi_node_chain = dict()
	for mn in multi_node:
		multi_node_chain[mn] = [mn]
		nex_node_arr = []
		for nex_node in skeleton_adj_dic[mn]:
			if not nex_node in assigned_node:
				nex_node_arr.append(nex_node)
				multi_node_chain[mn].append(nex_node)
		while len(nex_node_arr):
			temp_nex_node_arr = []
			for nex_node in nex_node_arr:
				for temp_nex_node in skeleton_adj_dic[nex_node]:
					if not temp_nex_node in assigned_node and not temp_nex_node in multi_node_chain[mn]:
						temp_nex_node_arr.append(temp_nex_node)
						multi_node_chain[mn].append(temp_nex_node)
			nex_node_arr = temp_nex_node_arr

		multi_node_chain[mn] = sorted(multi_node_chain[mn])
		multi_node_chain_temp = multi_node_chain.copy()
		for m_n in multi_node_chain_temp:
			if multi_node_chain_temp[m_n] == multi_node_chain_temp[mn] and m_n!=mn:
				multi_node_chain.pop(m_n, None)

	for mn in multi_node_chain:
		chain_dic[mn] = multi_node_chain[mn]


	return [chain_dic, end_node]

def hks_quan(pt_arr, hks_feature_arr, dbscan_thresh, num_cluster, vis):
	all_cluster = []
	min_hks = min(hks_feature_arr)
	max_hks = max(hks_feature_arr)
	seg_len = (max_hks - min_hks)/num_cluster*1.001
	seg_dic = dict()
	for cluster_id in range(num_cluster):
		seg_dic[cluster_id] = []

	for i in range(len(hks_feature_arr)):
		hks_f = hks_feature_arr[i]
		seg_dic[int((hks_f-min_hks)/seg_len)].append(i)
	
	for cluster in seg_dic:
		seg_idx_arr = seg_dic[cluster]
		seg_pt_arr = [pt_arr[idx] for idx in seg_idx_arr]
		dbscan_dic = dbscan_clustering(seg_pt_arr, dbscan_thresh)
		
		for seg_cl in dbscan_dic:
			all_cluster.append([seg_idx_arr[id] for id in dbscan_dic[seg_cl]])

	new_label_arr = [0 for i in range(len(pt_arr))]
	all_cluster_dic = dict()
	for t in range(len(all_cluster)):
		idx_arr = all_cluster[t]
		for idx in idx_arr:
			new_label_arr[idx] = t
		all_cluster_dic[t] = idx_arr

	if vis:
		vis_geo = []
		vis_rgb = []
		for idx_arr in all_cluster:
			rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
			for idx in idx_arr:
				vis_geo.append(pt_arr[idx])
				vis_rgb.append(rgb)
		pc_vis(vis_geo, vis_rgb)

	return [all_cluster_dic, new_label_arr]

	# return seg_dic

def get_hks_feature(geo_arr, rgb_arr, floder, frame_id):
	pt_arr = []
	idx = 0
	with open(floder + "hks/" + frame_id + "_n16.off") as ins:
		for line in ins:
			re2 = line.replace("\n", "").split(" ")
			if idx>1:
				# if len(re2) == 3:
					pt = [float(val) for val in re2[0:3]]
					pt_arr.append(pt)
			idx = idx + 1


	dbscan_thresh = get_neighbor_dis(pt_arr)*2


	hks_feature_arr = []
	with open(floder + "hks/" + frame_id + "_n16_hks.txt") as ins:
		for line in ins:
			re2 = line.replace("\n", "").split("	")
			val = float(re2[0])
			hks_feature_arr.append(val)

	hks_feature_arr = [np.round(val, 4) for val in hks_feature_arr]

	return [pt_arr, hks_feature_arr]

def hks_seg(geo_arr, rgb_arr, floder, frame_id):
	[hks_pt_arr, hks_feature_arr] = get_hks_feature(geo_arr, rgb_arr, floder, frame_id)
	dbscan_thresh = get_neighbor_dis(hks_pt_arr)*2
	[all_cluster_dic, new_label_arr] = hks_quan(hks_pt_arr, hks_feature_arr, dbscan_thresh, num_cluster=18, vis = 1)


	final_all_cluster = []
	for label in all_cluster_dic:
		final_all_cluster = final_all_cluster + [all_cluster_dic[label]]

	new_label_arr = [0 for i in range(len(hks_pt_arr))]

	for t in range(len(final_all_cluster)):
		idx_arr = final_all_cluster[t]
		for idx in idx_arr:
			new_label_arr[idx] = t
	[uni_pt_idx, bi_pt_idx, mul_pt_idx, boundary_label_dic] = detect_boundary_points(hks_pt_arr, new_label_arr)

	
	end_node = []
	bi_node = []
	multi_node = []
	skeleton_adj_dic = dict()
	for t in range(len(final_all_cluster)):
		idx_arr = final_all_cluster[t]
		patch_type = []
		for idx in idx_arr:
			if idx in bi_pt_idx or idx in mul_pt_idx:
				patch_type = patch_type + boundary_label_dic[idx]

		skeleton_adj_dic[t] = []
		for val in list(set(patch_type)):
			if val != t:
				skeleton_adj_dic[t].append(val)

		if len(list(set(patch_type))) == 2:
			end_node.append(t)
		elif len(list(set(patch_type))) == 3:
			bi_node.append(t)
		else:
			multi_node.append(t)

	[chain_dic, end_node2] = skeleton_adj_line2(skeleton_adj_dic)
	print(chain_dic, end_node2)



	chain_seg_dic = dict()
	new_label_arr = [0 for t in hks_pt_arr]
	vis_geo = []
	vis_rgb = []
	for chain in chain_dic:
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		for seg in chain_dic[chain]:
			for idx in all_cluster_dic[seg]:
				vis_geo.append(hks_pt_arr[idx])
				vis_rgb.append(rgb)
				new_label_arr[idx] = chain

		chain_seg_dic[chain] = []
		for seg in chain_dic[chain]:
			for idx in all_cluster_dic[seg]:
				chain_seg_dic[chain].append(idx)

	[uni_pt_idx, bi_pt_idx, mul_pt_idx, boundary_label_dic] = detect_boundary_points(hks_pt_arr, new_label_arr)


	vis_geo = []
	vis_rgb = []
	for seg in chain_seg_dic:
		rgb = [randrange(0, 1), randrange(0, 255), randrange(0, 255)]
		for idx in chain_seg_dic[seg]:
			vis_geo.append(hks_pt_arr[idx])
			vis_rgb.append(rgb)
	pc_vis(vis_geo, vis_rgb)

def hybird_seg(geo_arr, rgb_arr, floder, frame_id, sv_off_geo_arr, spanning_tree_idx_arr):
	node_arr = []
	for path in spanning_tree_idx_arr:
		st_node = path[0]
		end_node = path[-1]
		node_arr.append(st_node)
		node_arr.append(end_node)

	print()
	node_dic = dict((i, node_arr.count(i)) for i in node_arr)

	multinode_arr = []
	for key in node_dic:
		if node_dic[key]>0:
			multinode_arr.append(key)
	print(multinode_arr)
	multinode_pt = [sv_off_geo_arr[idx] for idx in multinode_arr]

	pc_vis(multinode_pt + sv_off_geo_arr, [[231, 0, 0] for pt in multinode_pt] + [[231, 230, 231] for pt in sv_off_geo_arr])

	[hks_pt_arr, hks_feature_arr] = get_hks_feature(geo_arr, rgb_arr, floder, frame_id)

	tree = BallTree(np.asarray(hks_pt_arr), leaf_size=1)
	multinode_hks_arr = []
	for pt in multinode_pt:
		dist, ind = tree.query([pt], k = 1)
		idx = ind[0][0]
		print(idx, hks_feature_arr[idx])
		multinode_hks_arr.append(hks_feature_arr[idx])
	sorted_multinode_hks_arr = sorted(multinode_hks_arr)
	sorted_multinode_hks_arr = sorted_multinode_hks_arr
	print(sorted_multinode_hks_arr)
	diff_thresh = 0.05
	diff_arr = [1]
	for t in range(1, len(sorted_multinode_hks_arr)):
		diff = sorted_multinode_hks_arr[t] - sorted_multinode_hks_arr[t-1]
		if diff < diff_thresh:
			diff_arr.append(0)
		else:
			diff_arr.append(1)
	print(diff_arr)
	nonzeroind = np.nonzero(diff_arr)[0]
	print(nonzeroind)

	threshold_arr = [sorted_multinode_hks_arr[idx] for idx in nonzeroind]
	print(threshold_arr)

	threshold_arr = [0] + threshold_arr + [max(hks_feature_arr)*1.01]
	threshold_arr_len = len(threshold_arr)
	ass_dict = dict()
	for j in range(threshold_arr_len):
		ass_dict[j] = []

	for t in range(len(hks_feature_arr)):
		hks_f = hks_feature_arr[t]
		for j in range(1, threshold_arr_len):
			if hks_f - threshold_arr[j] >= 0:
				continue
			else:
				ass_dict[j].append(t)
				break

	vis_geo = []
	vis_rgb = []
	for label in ass_dict:
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		for idx in ass_dict[label]:
			vis_geo.append(hks_pt_arr[idx])
			vis_rgb.append(rgb)
	pc_vis(vis_geo, vis_rgb)


	dbscan_thresh = get_neighbor_dis(hks_pt_arr)*2
	all_cluster = []
	for cluster in ass_dict:
		if len(ass_dict[cluster]):
			seg_idx_arr = ass_dict[cluster]
			seg_pt_arr = [hks_pt_arr[idx] for idx in seg_idx_arr]
			dbscan_dic = dbscan_clustering(seg_pt_arr, dbscan_thresh)
			
			for seg_cl in dbscan_dic:
				all_cluster.append([seg_idx_arr[id] for id in dbscan_dic[seg_cl]])

	new_label_arr = [0 for i in range(len(hks_pt_arr))]
	for t in range(len(all_cluster)):
		idx_arr = all_cluster[t]
		for idx in idx_arr:
			new_label_arr[idx] = t

	
	vis_geo = []
	vis_rgb = []
	for idx_arr in all_cluster:
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		for idx in idx_arr:
			vis_geo.append(hks_pt_arr[idx])
			vis_rgb.append(rgb)
	print(len(all_cluster))
	pc_vis(vis_geo, vis_rgb)

def patch_seg(patch_geo, hks_pt_arr, hks_feature_arr, vis):
	X = np.array(hks_pt_arr)
	tree = BallTree(X, leaf_size = 1)
	new_hks_feature_arr = []
	for pt in patch_geo:
		dist, ind = tree.query([pt], k=1)
		idx = ind[0][0]
		new_hks_feature_arr.append(hks_feature_arr[idx])
	hks_feature_arr = new_hks_feature_arr

	counts, bins, bars = plt.hist(hks_feature_arr, bins = "doane")
	label_dic = dict()
	for j in range(len(bins)-1):
		label_dic[j] = []
	bins[-1] = bins[-1]*1.001

	for i in range(len(hks_feature_arr)):
		label = 0
		for j in range(len(bins)-1):
			if bins[j]<=hks_feature_arr[i] and hks_feature_arr[i] < bins[j+1]:
				label = j
				break
		if not label in label_dic:
			label_dic[label] = []
		label_dic[label].append(i)
	dbscan_thresh = get_neighbor_dis(patch_geo)*2
	all_cluster = []
	for cluster in label_dic:
		seg_idx_arr = label_dic[cluster]
		seg_pt_arr = [patch_geo[idx] for idx in seg_idx_arr]
		dbscan_dic = dbscan_clustering(seg_pt_arr, dbscan_thresh)
		for seg_cl in dbscan_dic:
			all_cluster.append([seg_idx_arr[id] for id in dbscan_dic[seg_cl]])

	vis_geo = []
	vis_rgb = []
	new_label_arr = [0 for i in range(len(patch_geo))]
	all_cluster_dic = dict()

	for t in range(len(all_cluster)):
		idx_arr = all_cluster[t]
		for idx in idx_arr:
			new_label_arr[idx] = t
		all_cluster_dic[t] = idx_arr

		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		vis_geo = vis_geo + [patch_geo[idx] for idx in idx_arr]
		vis_rgb = vis_rgb + [rgb for idx in idx_arr]

	if vis:
		pc_vis(vis_geo, vis_rgb)

	return []

def fibonacci_sphere(samples, norm_len, center):

	points = []
	phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

	for i in range(samples):
		y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
		radius = math.sqrt(1 - y * y)  # radius at y

		theta = phi * i  # golden angle increment

		x = math.cos(theta) * radius
		z = math.sin(theta) * radius

		points.append((x, y, z))
	points = np.asarray(points)*norm_len + np.asarray(center)

	return points

def hks_raw_seg(hks_feature_arr, sv_off_geo_arr, vis):
	counts, bins, bars = plt.hist(hks_feature_arr, bins = 4)
	# print(len(bins))
	# plt.show()
	# plt.close()
	label_dic = dict()
	for j in range(len(bins)-1):
		label_dic[j] = []
	bins[-1] = bins[-1]*1.001

	for i in range(len(hks_feature_arr)):
		label = 0
		for j in range(len(bins)-1):
			if bins[j]<=hks_feature_arr[i] and hks_feature_arr[i] < bins[j+1]:
				label = j
				break
		if not label in label_dic:
			label_dic[label] = []
		label_dic[label].append(i)

	dbscan_thresh = get_neighbor_dis(sv_off_geo_arr)*2

	all_cluster = []
	for cluster in label_dic:
		seg_idx_arr = label_dic[cluster]
		seg_pt_arr = [sv_off_geo_arr[idx] for idx in seg_idx_arr]
		if len(seg_pt_arr):
			dbscan_dic = dbscan_clustering(seg_pt_arr, dbscan_thresh)
			for seg_cl in dbscan_dic:
				all_cluster.append([seg_idx_arr[id] for id in dbscan_dic[seg_cl]])

	vis_geo = []
	vis_rgb = []
	new_label_arr = [0 for i in range(len(sv_off_geo_arr))]
	all_cluster_dic = dict()

	for t in range(len(all_cluster)):
		idx_arr = all_cluster[t]
		for idx in idx_arr:
			new_label_arr[idx] = t
		all_cluster_dic[t] = idx_arr

		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		vis_geo = vis_geo + [sv_off_geo_arr[idx] for idx in idx_arr]
		vis_rgb = vis_rgb + [rgb for idx in idx_arr]

	if vis:
		pc_vis(vis_geo, vis_rgb)

def hybird_seg2(geo_arr, rgb_arr, floder, frame_id, sv_off_geo_arr, hks_rgb, vis):
	# s_pts = fibonacci_sphere(samples=1000)
	# pc_vis(s_pts, [[255, 0, 0] for pt in s_pts])
	[hks_pt_arr, hks_feature_arr] = get_hks_feature(geo_arr, rgb_arr, floder, frame_id)
	X = np.array(hks_pt_arr)
	tree = BallTree(X, leaf_size = 1)
	new_hks_feature_arr = []
	for pt in sv_off_geo_arr:
		dist, ind = tree.query([pt], k=1)
		idx = ind[0][0]
		new_hks_feature_arr.append(hks_feature_arr[idx])
	hks_feature_arr = new_hks_feature_arr

	#doane
	counts, bins, bars = plt.hist(hks_feature_arr, bins = "doane")
	# print(len(bins))
	# plt.show()
	# plt.close()
	label_dic = dict()
	for j in range(len(bins)-1):
		label_dic[j] = []
	bins[-1] = bins[-1]*1.001

	for i in range(len(hks_feature_arr)):
		label = 0
		for j in range(len(bins)-1):
			if bins[j]<=hks_feature_arr[i] and hks_feature_arr[i] < bins[j+1]:
				label = j
				break
		if not label in label_dic:
			label_dic[label] = []
		label_dic[label].append(i)

	dbscan_thresh = get_neighbor_dis(sv_off_geo_arr)*2

	all_cluster = []
	for cluster in label_dic:
		seg_idx_arr = label_dic[cluster]
		seg_pt_arr = [sv_off_geo_arr[idx] for idx in seg_idx_arr]
		if len(seg_pt_arr):
			dbscan_dic = dbscan_clustering(seg_pt_arr, dbscan_thresh)
			for seg_cl in dbscan_dic:
				all_cluster.append([seg_idx_arr[id] for id in dbscan_dic[seg_cl]])

	vis_geo = []
	vis_rgb = []
	new_label_arr = [0 for i in range(len(sv_off_geo_arr))]
	all_cluster_dic = dict()

	for t in range(len(all_cluster)):
		idx_arr = all_cluster[t]
		for idx in idx_arr:
			new_label_arr[idx] = t
		all_cluster_dic[t] = idx_arr

		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		vis_geo = vis_geo + [sv_off_geo_arr[idx] for idx in idx_arr]
		vis_rgb = vis_rgb + [rgb for idx in idx_arr]

	if vis:
		pc_vis(vis_geo, vis_rgb)


	[uni_pt_idx, bi_pt_idx, mul_pt_idx, boundary_label_dic] = detect_boundary_points(sv_off_geo_arr, new_label_arr)

	
	end_node = []
	bi_node = []
	multi_node = []
	skeleton_adj_dic = dict()
	for t in range(len(all_cluster)):
		idx_arr = all_cluster[t]
		patch_type = []
		for idx in idx_arr:
			if idx in bi_pt_idx or idx in mul_pt_idx:
				patch_type = patch_type + boundary_label_dic[idx]

		skeleton_adj_dic[t] = []
		# for val in list(set(patch_type)):
		# 	if val != t:
		# 		skeleton_adj_dic[t].append(val)


		patch_type_dic = dict((i, patch_type.count(i)) for i in patch_type)
		print(patch_type_dic)
		p_type_cnt = 0
		for p_type in patch_type_dic:
			if patch_type_dic[p_type] > 5:
				p_type_cnt = p_type_cnt + 1
				if p_type != t:
					skeleton_adj_dic[t].append(p_type)

		if p_type_cnt <= 2:
			end_node.append(t)
		elif p_type_cnt == 3:
			bi_node.append(t)
		else:
			multi_node.append(t)

		# if len(list(set(patch_type))) == 2:
		# 	end_node.append(t)
		# elif len(list(set(patch_type))) == 3:
		# 	bi_node.append(t)
		# else:
		# 	multi_node.append(t)

	[chain_dic, end_node2] = skeleton_adj_line2(skeleton_adj_dic)
	print(chain_dic, end_node2)



	chain_seg_dic = dict()
	new_label_arr = [0 for t in sv_off_geo_arr]
	# vis_geo = []
	# vis_rgb = []
	for chain in chain_dic:
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		for seg in chain_dic[chain]:
			for idx in all_cluster_dic[seg]:
				# vis_geo.append(sv_off_geo_arr[idx])
				# vis_rgb.append(rgb)
				new_label_arr[idx] = chain

		chain_seg_dic[chain] = []
		for seg in chain_dic[chain]:
			for idx in all_cluster_dic[seg]:
				chain_seg_dic[chain].append(idx)

	# [uni_pt_idx, bi_pt_idx, mul_pt_idx, boundary_label_dic] = detect_boundary_points(sv_off_geo_arr, new_label_arr)
	norm_len = get_neighbor_dis(sv_off_geo_arr)*0.75
	if vis:
		vis_geo = []
		vis_rgb = []
		for seg in chain_seg_dic:
			rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
			rgb =np.mean(np.asarray([hks_rgb[idx] for idx in chain_seg_dic[seg]]), axis=0)
			# max_id = np.argmax([hks_feature_arr[idx] for idx in chain_seg_dic[seg]])
			
			# rgb = hks_rgb[chain_seg_dic[seg][max_id]]
			for idx in chain_seg_dic[seg]:
				vis_geo.append(sv_off_geo_arr[idx])
				vis_rgb.append(rgb)
		pc_vis(vis_geo, vis_rgb)

	center_dic = dict()
	vis_geo = []
	vis_rgb = []
	for seg in all_cluster_dic:
		patch_geo = [sv_off_geo_arr[idx] for idx in all_cluster_dic[seg]]
		patch_rgb = [hks_rgb[idx] for idx in all_cluster_dic[seg]]
		center_dic[seg] = np.mean(np.asarray(patch_geo), axis=0)

		vis_geo.append(np.mean(np.asarray(patch_geo), axis=0))
		vis_rgb.append(np.mean(np.asarray(patch_rgb), axis=0))

		rgb = np.mean(np.asarray(patch_rgb), axis=0)
		s_pts = fibonacci_sphere(100, norm_len, np.mean(np.asarray(patch_geo), axis=0))
		vis_geo = vis_geo + list(s_pts)
		vis_rgb = vis_rgb + [rgb for pt in s_pts]
	line_pt = []
	for node in skeleton_adj_dic:
		for adj_node in skeleton_adj_dic[node]:
			line_pt = line_pt + draw_line(center_dic[node], center_dic[adj_node])
	line_color = [[210, 210, 210] for pt in line_pt]
	if vis:
		pc_vis(vis_geo + line_pt, vis_rgb + line_color)
	# node_arr = []
	# for path in spanning_tree_idx_arr:
	# 	st_node = path[0]
	# 	end_node = path[-1]
	# 	node_arr.append(st_node)
	# 	node_arr.append(end_node)

	
	# node_dic = dict((i, node_arr.count(i)) for i in node_arr)

	# multinode_arr = []
	# for key in node_dic:
	# 	if node_dic[key]>2:
	# 		multinode_arr.append(key)
	# print(multinode_arr)
	# multinode_pt = [sv_off_geo_arr[idx] for idx in multinode_arr]

	# X = np.array(sv_off_geo_arr)
	# tree = BallTree(X, leaf_size = 1)
	# neigh_arr = []
	# nei_hks_pt = []
	# nei_hks_rgb = []
	# for pt in multinode_pt:
	# 	dist, ind = tree.query([pt], k=10)
	# 	neigh_arr = neigh_arr + [sv_off_geo_arr[idx] for idx in ind[0]]

	# 	nei_hks = [hks_feature_arr[idx] for idx in ind[0]]
	# 	max_hks = max(nei_hks)
	# 	min_hks = min(nei_hks)
	# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
	# 	# for t in range(len(hks_feature_arr)):
	# 	# 	if hks_feature_arr[t] >= min_hks and hks_feature_arr[t] <= max_hks:
	# 	# 		nei_hks_pt.append(sv_off_geo_arr[t])
	# 	# 		nei_hks_rgb.append(rgb)


	# pc_vis(nei_hks_pt + neigh_arr + sv_off_geo_arr, nei_hks_rgb + [[231, 0, 0] for pt in neigh_arr] + [[231, 230, 231] for pt in sv_off_geo_arr])

	return [skeleton_adj_dic, chain_seg_dic, chain_dic, all_cluster, end_node, bi_node, multi_node]

def quan_hks_seg(err_arr, sv_off_geo_arr, dis_pt_idx):
	[hks_pt_arr, hks_feature_arr] = get_hks_feature(geo_arr, rgb_arr, floder, frame_id)
	X = np.array(hks_pt_arr)
	tree = BallTree(X, leaf_size = 1)
	new_hks_feature_arr = []
	for pt in sv_off_geo_arr:
		dist, ind = tree.query([pt], k=1)
		idx = ind[0][0]
		new_hks_feature_arr.append(hks_feature_arr[idx])

	threshold_arr = [new_hks_feature_arr[idx] for idx in dis_pt_idx]
	threshold_arr = sorted(threshold_arr)

	print(threshold_arr)

	threshold_arr_len = len(threshold_arr)
	ass_dict = dict()
	for j in range(threshold_arr_len):
		ass_dict[j] = []

	for t in range(len(new_hks_feature_arr)):
		hks_f = new_hks_feature_arr[t]
		for j in range(1, threshold_arr_len):
			if hks_f - threshold_arr[j] >= 0:
				continue
			else:
				ass_dict[j].append(t)
				break

	vis_geo = []
	vis_rgb = []
	for label in ass_dict:
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		for idx in ass_dict[label]:
			vis_geo.append(sv_off_geo_arr[idx])
			vis_rgb.append(rgb)
	pc_vis(vis_geo, vis_rgb)


	# all_cluster_idx_arr = []
	# np.median(hks_feature_arr)

def km_hks_seg(pt_arr, hks_feature_arr, dbscan_thresh, num_cluster):
	all_cluster_idx_arr = []
	km = KMeans(n_clusters=num_cluster)

	label = km.fit_predict(np.asarray(hks_feature_arr).reshape(-1,1))

	seg_dic = dict()
	for cluster_id in range(num_cluster):
		seg_dic[cluster_id] = []

	for i in range(len(label)):
		seg_dic[label[i]].append(i)

	for cluster in seg_dic:
		seg_idx_arr = seg_dic[cluster]
		seg_pt_arr = [pt_arr[idx] for idx in seg_idx_arr]
		dbscan_dic = dbscan_clustering(seg_pt_arr, dbscan_thresh)
		
		for seg_cl in dbscan_dic:
			all_cluster_idx_arr.append([seg_idx_arr[id] for id in dbscan_dic[seg_cl]])

	new_label_arr = [0 for i in range(len(pt_arr))]
	for t in range(len(all_cluster_idx_arr)):
		idx_arr = all_cluster_idx_arr[t]
		for idx in idx_arr:
			new_label_arr[idx] = t

	all_cluster_pt_arr = []
	for idx_arr in all_cluster_idx_arr:
		all_cluster_pt_arr.append([pt_arr[idx] for idx in idx_arr])



	# vis_geo = []
	# vis_rgb = []
	# for idx_arr in all_cluster_idx_arr:
	# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
	# 	for idx in idx_arr:
	# 		vis_geo.append(pt_arr[idx])
	# 		vis_rgb.append(rgb)

	# pc_vis(vis_geo, vis_rgb)

	return [all_cluster_pt_arr, all_cluster_idx_arr, new_label_arr]

def detect_boundary_points(pt_arr, label):
	tree = BallTree(np.array(pt_arr), leaf_size=1)
	uni_pt_idx = []
	bi_pt_idx = []
	mul_pt_idx = []

	boundary_label_dic = dict()
	for i in range(len(pt_arr)):
		pt = pt_arr[i]
		dist, ind = tree.query([pt], k = 5)
		nei_label = [label[idx] for idx in ind[0]]
		cnt_arr = [[x, nei_label.count(x)] for x in set(nei_label)]
		if len(cnt_arr) <= 1:
			uni_pt_idx.append(i)
			boundary_label_dic[i] = list(set(nei_label))
		elif len(cnt_arr) > 2:
			mul_pt_idx.append(i)
			boundary_label_dic[i] = list(set(nei_label))
		else:
			if max([val[1] for val in cnt_arr])*1.0/len(ind[0]) >= 0.4:
				bi_pt_idx.append(i)
				boundary_label_dic[i] = list(set(nei_label))
			else:
				uni_pt_idx.append(i)
	return [uni_pt_idx, bi_pt_idx, mul_pt_idx, boundary_label_dic]

def bi_hks_seg(geo_arr, rgb_arr, floder, frame_id_head, sv_off_geo_arr, dis_pt_idx):
	[hks_pt_arr, hks_feature_arr] = get_hks_feature(geo_arr, rgb_arr, floder, frame_id)
	dis_pt = [sv_off_geo_arr[idx] for idx in dis_pt_idx]

	dbscan_thresh = get_neighbor_dis(sv_off_geo_arr)*2

	tot_num = len(hks_pt_arr)
	X = np.array(hks_pt_arr)
	tree = BallTree(X, leaf_size = 1)

	new_hks_feature_arr = []
	for pt in sv_off_geo_arr:
		dist, ind = tree.query([pt], k=1)
		idx = ind[0][0]
		new_hks_feature_arr.append(hks_feature_arr[idx])

	X = np.array(sv_off_geo_arr)
	tree = BallTree(X, leaf_size = 1)

	final_all_cluster = []
	temp_pt_arr = [sv_off_geo_arr]
	while len(temp_pt_arr):
		temp = []
		for sub_pt_arr in temp_pt_arr:
			dis_num = 0
			for pt in sub_pt_arr:
				dist, ind = tree.query([pt], k=1)
				if ind[0][0] in dis_pt_idx:
					dis_num = dis_num + 1
			print("dis_num: ", dis_num)

			if dis_num > 3:
				sub_hks_feature_arr = []
				for pt in sub_pt_arr:
					dist, ind = tree.query([pt], k=1)
					idx = ind[0][0]
					sub_hks_feature_arr.append(new_hks_feature_arr[idx])
				[all_cluster_pt_arr, all_cluster_idx_arr, new_label_arr] = km_hks_seg(sub_pt_arr, sub_hks_feature_arr, dbscan_thresh, num_cluster=2)
				temp = temp + all_cluster_pt_arr
			else:
				final_all_cluster.append(sub_pt_arr)

		temp_pt_arr = temp

		print(len(temp_pt_arr), len(final_all_cluster))

	vis_geo = []
	vis_rgb = []
	pt_arr = []
	label_arr = []
	label = 0
	pt_idx = 0
	final_all_cluster_dic = dict()
	for cluster_pt in final_all_cluster:
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		patch_idx_arr = []
		for pt in cluster_pt:
			vis_geo.append(pt)
			vis_rgb.append(rgb)
			pt_arr.append(pt)
			label_arr.append(label)
			patch_idx_arr.append(pt_idx)
			pt_idx = pt_idx + 1
		final_all_cluster_dic[label] = patch_idx_arr

		label = label + 1
	pc_vis(vis_geo, vis_rgb)

	return [pt_arr, label_arr, final_all_cluster_dic]

def hks_seg4(geo_arr, rgb_arr, floder, frame_id_head):
	[hks_pt_arr, hks_feature_arr] = get_hks_feature(geo_arr, rgb_arr, floder, frame_id)

	dbscan_thresh = get_neighbor_dis(hks_pt_arr)*2
	for num_c in range(2, 10, 1):
		# [all_cluster_dic, new_label_arr] = hks_quan(hks_pt_arr, hks_feature_arr, dbscan_thresh, num_cluster=num_c, vis = 0)
		# print(num_c, len(all_cluster_dic))
		# vis_geo = []
		# vis_rgb = []
		# for la in all_cluster_dic:
		# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		# 	vis_geo = vis_geo + [hks_pt_arr[idx] for idx in all_cluster_dic[la]]
		# 	vis_rgb = vis_rgb + [rgb for idx in all_cluster_dic[la]]
		# pc_vis(vis_geo, vis_rgb)

		[all_cluster, new_label_arr] = km_hks_seg(hks_pt_arr, hks_feature_arr, dbscan_thresh, num_cluster=num_c)

def hks_seg3(geo_arr, rgb_arr, floder, frame_id_head):
	[hks_pt_arr, hks_feature_arr] = get_hks_feature(geo_arr, rgb_arr, floder, frame_id)
	max_hks = max(hks_feature_arr)
	min_hks = min(hks_feature_arr)
	vis_geo = []
	vis_rgb = []
	colors = cm.bwr(np.linspace(0, 1, 256)) # other color schemes: gist_rainbow, nipy_spectral, plasma, inferno, magma, cividis
	for t in range(len(hks_pt_arr)):
		pt = hks_pt_arr[t]
		rgb = colors[int((hks_feature_arr[t]-min_hks)/max_hks*255)][0:3]*255
		vis_geo.append(pt)
		vis_rgb.append(rgb)
	pc_vis(vis_geo, vis_rgb)

	dbscan_thresh = get_neighbor_dis(hks_pt_arr)*2

	for num_c in range(20, 5, -2):
		# try:
			[all_cluster_dic, new_label_arr] = hks_quan(hks_pt_arr, hks_feature_arr, dbscan_thresh, num_cluster=num_c, vis = 0)
			# print(time.time() - st)
			# vis_geo = []
			# vis_rgb = []
			# for la in all_cluster_dic:
			# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
			# 	vis_geo = vis_geo + [hks_pt_arr[idx] for idx in all_cluster_dic[la]]
			# 	vis_rgb = vis_rgb + [rgb for idx in all_cluster_dic[la]]

			# pc_vis(vis_geo, vis_rgb)

			[uni_pt_idx, bi_pt_idx, mul_pt_idx, boundary_label_dic] = detect_boundary_points(hks_pt_arr, new_label_arr)
			print(num_c, len(mul_pt_idx))
			if len(mul_pt_idx) == 0:
				vis_geo = []
				vis_rgb = []
				# for idx in uni_pt_idx:
				# 	vis_geo.append(hks_pt_arr[idx])
				# 	vis_rgb.append([128, 128, 128])

				# for idx in bi_pt_idx:
				# 	vis_geo.append(hks_pt_arr[idx])
				# 	vis_rgb.append([255, 0, 0])

				# for idx in mul_pt_idx:
				# 	vis_geo.append(hks_pt_arr[idx])
				# 	vis_rgb.append([0, 255, 0])

				# pcd = o3d.geometry.PointCloud()
				# pcd.points = o3d.utility.Vector3dVector(vis_geo)
				# pcd.colors = o3d.utility.Vector3dVector(np.asarray(vis_rgb)/255.0)
				# o3d.visualization.draw_geometries([pcd])

				final_all_cluster = []
				for label in all_cluster_dic:
					final_all_cluster = final_all_cluster + [all_cluster_dic[label]]


				# vis_geo = []
				# vis_rgb = []
				new_label_arr = [0 for i in range(len(hks_pt_arr))]

				for t in range(len(final_all_cluster)):
					idx_arr = final_all_cluster[t]
					for idx in idx_arr:
						new_label_arr[idx] = t
				[uni_pt_idx, bi_pt_idx, mul_pt_idx, boundary_label_dic] = detect_boundary_points(hks_pt_arr, new_label_arr)

				
				end_node = []
				bi_node = []
				multi_node = []
				skeleton_adj_dic = dict()
				for t in range(len(final_all_cluster)):
					idx_arr = final_all_cluster[t]
					patch_type = []
					for idx in idx_arr:
						vis_geo.append(hks_pt_arr[idx])
						if idx in bi_pt_idx or idx in mul_pt_idx:
							vis_rgb.append(rgb)
							patch_type = patch_type + boundary_label_dic[idx]
						else:
							vis_rgb.append([rgb[0]/4, rgb[1]/4, rgb[2]/4])

					# print(t, "patch_type:", list(set(patch_type)), len(list(set(patch_type))))
					skeleton_adj_dic[t] = []
					for val in list(set(patch_type)):
						if val != t:
							skeleton_adj_dic[t].append(val)

					if len(list(set(patch_type))) == 2:
						end_node.append(t)
					elif len(list(set(patch_type))) == 3:
						bi_node.append(t)
					else:
						multi_node.append(t)

				[chain_dic, end_node2] = skeleton_adj_line2(skeleton_adj_dic)
				print(chain_dic, end_node2)

				vis_geo = []
				vis_rgb = []

				for cluster_idx in end_node:
					rgb = [0, 255, 0]
					for idx in all_cluster_dic[cluster_idx]:
						vis_geo.append(hks_pt_arr[idx])
						vis_rgb.append(rgb)

				for cluster_idx in bi_node:
					rgb = [0, 0, 255]
					for idx in all_cluster_dic[cluster_idx]:
						vis_geo.append(hks_pt_arr[idx])
						vis_rgb.append(rgb)

				for cluster_idx in multi_node:
					rgb = [255, 0, 0]
					for idx in all_cluster_dic[cluster_idx]:
						vis_geo.append(hks_pt_arr[idx])
						vis_rgb.append(rgb)

				print("end_node: ", end_node)
				print("bi_node: ", bi_node)
				print("multi_node: ", multi_node)
				# pc_vis(vis_geo, vis_rgb)


				print("chain_dic: ", chain_dic)
				vis_geo = []
				vis_rgb = []
				for chain in chain_dic:
					rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
					print("chain: ", chain)
					print(chain, chain_dic[chain], end_node)
					for node in chain_dic[chain]:
						for idx in all_cluster_dic[node]:
							vis_geo.append(hks_pt_arr[idx])
							vis_rgb.append(rgb)
				
				pc_vis(vis_geo, vis_rgb)


				# 	for cluster_idx in chain_dic[chain]:
				# 		idx_arr = final_all_cluster[cluster_idx]
				# 		if cluster_idx in end_node:
				# 			rgb = [0, 255, 0]
				# 			for idx in idx_arr:
				# 				vis_geo.append(hks_pt_arr[idx])
				# 				vis_rgb.append(rgb)
				# 		else:
				# 			for idx in idx_arr:
				# 				vis_geo.append(hks_pt_arr[idx])
				# 				vis_rgb.append(rgb1)

				# pcd = o3d.geometry.PointCloud()
				# pcd.points = o3d.utility.Vector3dVector(vis_geo)
				# pcd.colors = o3d.utility.Vector3dVector(np.asarray(vis_rgb)/255.0)
				# o3d.visualization.draw_geometries([pcd])

				break
		# except:
		# 	print(num_c)

##############################################################################################################################
# 																						 									 #
#											(3) Complex point detection 													 #
# 																						 									 #
##############################################################################################################################
def compute_distortion(sv_off_geo_arr, seg_thresh, vis):
	n_neigh = 6
	n_components = 2

	
	
	tot_num = len(sv_off_geo_arr)
	X = np.array(sv_off_geo_arr)
	tree = BallTree(X, leaf_size=1)
	err_arr = []

	for pt in sv_off_geo_arr:
		dist, ind = tree.query([pt], k = n_neigh)
		patch_geo_arr = [sv_off_geo_arr[idx] for idx in ind[0]]
		embedding = manifold.Isomap(n_neigh-1, n_components, eigen_solver='dense')
		X = np.matrix(patch_geo_arr)
		Y = embedding.fit_transform(X)
		reconstruction_err = embedding.reconstruction_error()
		d2_geo = [[val[0], val[1], 0] for val in Y]
		if np.isnan(reconstruction_err):
			err_arr.append(0)
		else:
			err_arr.append(reconstruction_err)
	err_orig_err = err_arr
	new_err_arr = []
	for pt in sv_off_geo_arr:
		dist, ind = tree.query([pt], k = n_neigh)
		new_err_arr.append(np.mean([err_arr[idx] for idx in ind[0]]))
		# new_err_arr.append(err_arr[ind[0][0]]*0.75 + 0.25*np.mean([err_arr[idx] for idx in ind[0][1:]]))
	err_arr = new_err_arr

	max_err = np.max(err_arr)
	min_err = np.min(err_arr)
	dis_arr = max_err - min_err
	err_arr = [(val-min_err)/dis_arr for val in err_arr]

	
	if vis:
		max_err = np.max(err_arr)
		colors = cm.OrRd(np.linspace(0, 1, 256))*256 # other color schemes: gist_rainbow, nipy_spectral, plasma, inferno, magma, cividis

		# for colors in [cm.coolwarm(np.linspace(0, 1, 256)), cm.bwr(np.linspace(0, 1, 256)), cm.RdYlBu(np.linspace(0, 1, 256)), cm.Spectral(np.linspace(0, 1, 256)), cm.OrRd(np.linspace(0, 1, 256)), cm.YlOrRd(np.linspace(0, 1, 256)), cm.PuRd(np.linspace(0, 1, 256)), cm.gist_heat(np.linspace(0, 1, 256)), cm.BrBG(np.linspace(0, 1, 256)), cm.RdBu(np.linspace(0, 1, 256)), cm.magma(np.linspace(0, 1, 256))]:
		# for colors in [cm.coolwarm(np.linspace(0, 1, 256)), cm.RdYlBu(np.linspace(0, 1, 256))]:

		# colors = cm.coolwarm(np.linspace(0, 1, 256))*256
		colors = cm.RdYlBu(np.linspace(0, 1, 256))*256
		colors = cm.coolwarm(np.linspace(0, 1, 256))*256
		# colors = colors[::-1]

		vis_geo = []
		vis_rgb = []
		for t in range(len(sv_off_geo_arr)):
			pt = sv_off_geo_arr[t]
			rgb = colors[int(err_arr[t]/max_err*255)][0:3]
			vis_geo.append(pt)
			vis_rgb.append(rgb)
		pc_vis(vis_geo, vis_rgb)
	return [err_orig_err, err_arr]

def mini_spanning_tree(sv_off_geo_arr, sparse_sv_off_geo_arr, dis_pt_idx, vis):
	line_pt = []
	line_color = []
	G = nx.Graph()
	# dist_m = graph_shortest_path(kng, method='auto', directed=False)

	distortion_pts = [sv_off_geo_arr[idx] for idx in dis_pt_idx]
	sparse_dis_pts = distortion_pts + sparse_sv_off_geo_arr
	neigh = NearestNeighbors(n_neighbors=7)
	neigh.fit(sparse_dis_pts)
	kng = neigh.kneighbors_graph(sparse_dis_pts)
	kng = kng.toarray()
	dist_m = graph_shortest_path(kng, method='auto', directed=False)

	G_dis = nx.Graph()
	for i in range(len(dis_pt_idx) - 1):
		for j in range(i+1, len(dis_pt_idx)):
			G_dis.add_edge(i, j, weight = dist_m[i][j])

	tree = BallTree(np.asarray(sv_off_geo_arr), leaf_size=1)
	for i in range(len(sv_off_geo_arr)):
		pt = sv_off_geo_arr[i]
		dist, ind = tree.query([pt], k = 7)
		for idx in ind[0][1:]:
			if i > idx:
				G.add_edge(i, idx, weight = 1)

	# dis_pt_idx = dis_pt_idx
	# G_dis = nx.Graph()
	# for i in range(len(dis_pt_idx) - 1):
	# 	print(i)
	# 	for j in range(i+1, len(dis_pt_idx)):
	# 		path_arr = nx.shortest_path(G, source=dis_pt_idx[i],target=dis_pt_idx[j], weight='weight')
	# 		dis = len(path_arr)
	# 		G_dis.add_edge(i, j, weight = dis)

	spanning_tree_idx_arr = []

	T = nx.minimum_spanning_tree(G_dis)
	for rec in sorted(T.edges(data=False)):
		src_idx = dis_pt_idx[rec[0]]
		dst_idx = dis_pt_idx[rec[1]]
		path_arr = nx.shortest_path(G, source=src_idx,target=dst_idx, weight='weight')
		# print(src_idx, dst_idx, path_arr)
		spanning_tree_idx_arr.append(path_arr)
		temp_pt = []
		for t in range(0, len(path_arr)-1):
			temp_pt = temp_pt + draw_line(sv_off_geo_arr[path_arr[t]], sv_off_geo_arr[path_arr[t+1]])
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		temp_color = [rgb for pt in temp_pt]
		line_pt = line_pt + temp_pt
		line_color= line_color + temp_color

	# for rec in sorted(T.edges(data=False)):
	# 	line_pt = line_pt + draw_line(sv_off_geo_arr[dis_pt_idx[rec[0]]], sv_off_geo_arr[dis_pt_idx[rec[1]]])
	# pc_vis(sv_off_geo_arr + line_pt, [[128, 128, 128] for pt in sv_off_geo_arr] + line_color)

	# graph = csr_matrix(dist_m)
	# n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
	# print(n_components)
	# Tcsr = minimum_spanning_tree(dist_m)
	# csr_matrix = Tcsr.toarray()
   
	# for i in range(len(off_geo_arr)):
	#     for j in range(len(off_geo_arr)):
	#         if csr_matrix[i][j]:
	#             line_pt = line_pt + draw_line(off_geo_arr[i], off_geo_arr[j])

	# print(nx.shortest_path(G, source=131,target=96, weight='weight'))

	# T=nx.minimum_spanning_tree(G)
	# # print(sorted(T.edges(data=False)))
	# for rec in sorted(T.edges(data=False)):
	# 	line_pt = line_pt + draw_line(sv_off_geo_arr[rec[0]], sv_off_geo_arr[rec[1]])

	if vis:
		pc_vis(sv_off_geo_arr + line_pt, [[228, 228, 228] for pt in sv_off_geo_arr] +  [[255, 0, 0] for pt in line_pt])

	return spanning_tree_idx_arr

def remove_similar_distortion_pts(sv_off_geo_arr, sparse_sv_off_geo_arr, dis_pt_idx, err_arr):
	spanning_tree_idx_arr = mini_spanning_tree(sv_off_geo_arr, sparse_sv_off_geo_arr, dis_pt_idx, 0)
	short_node_arr = []
	short_dic = dict()
	for leaf in spanning_tree_idx_arr:
		if len(leaf)<5:
			st_node = leaf[0]
			end_node = leaf[-1]
			if not st_node in short_dic:
				short_dic[st_node] = []
			if not end_node in short_dic:
				short_dic[end_node] = []
			short_dic[st_node].append(st_node)
			short_dic[st_node].append(end_node)
			short_dic[end_node].append(st_node)
			short_dic[end_node].append(end_node)
	
	node_str_arr = []
	for node in short_dic:
		node_str = ""
		for val in sorted(list(set(short_dic[node]))):
			node_str = node_str + "_" + str(val)
		node_str_arr.append(node_str)


	remove_pt_idx = []
	for val in list(set(node_str_arr)):
		idx_arr = [int(idx) for idx in val.split("_")[1:]]
		max_idx = np.argmax([err_arr[idx] for idx in idx_arr])
		for i in range(len(idx_arr)):
			if i != max_idx:
				remove_pt_idx.append(idx_arr[i])
	remove_pt_idx = list(set(remove_pt_idx))
	dis_pt_idx = list(set(dis_pt_idx) - set(remove_pt_idx))

	return [dis_pt_idx, remove_pt_idx]
			
def detect_distortion_pts(boundary_pt_idx, sv_off_geo_arr, sparse_sv_off_geo_arr, err_arr, seg_thresh, N_r, vis):
	print(statistics.median(err_arr), np.min(err_arr), np.max(err_arr))
	sec_thresh = statistics.median(err_arr)*1.5
	sec_thresh = 0.1
	high_distortion_pt_idx = []
	high_distortion_pt = []
	high_distortion_err = []
	for t in range(len(sv_off_geo_arr)):
		if err_arr[t] > sec_thresh:
			high_distortion_pt.append(sv_off_geo_arr[t])
			high_distortion_err.append(err_arr[t])
			high_distortion_pt_idx.append(t)

	if vis:
		colors = cm.RdYlBu(np.linspace(0, 1, 256))*256
		colors = cm.coolwarm(np.linspace(0, 1, 256))*256
		# colors = colors[::-1]
		max_err = np.max(err_arr)


		vis_geo = []
		vis_rgb = []
		for t in range(len(sv_off_geo_arr)):
			pt = sv_off_geo_arr[t]
			if err_arr[t] > sec_thresh:
				vis_geo.append([pt[0] + 0, pt[1], pt[2]])
				
				rgb = colors[int(err_arr[t]/max_err*255)][0:3]
				vis_rgb.append(rgb)
			else:
				vis_geo.append([pt[0] + 0, pt[1], pt[2]])
				vis_rgb.append([221, 222, 223])
		pc_vis(vis_geo, vis_rgb)

	tree = BallTree(np.asarray(sv_off_geo_arr), leaf_size=1)

	final_distortion_pt_idx = []
	temp_area_arr = [high_distortion_pt]
	temp_err_arr = [high_distortion_err]
	while len(temp_area_arr):
		new_temp_area_arr = []
		new_temp_err_arr = []
		for i in range(len(temp_area_arr)):
			cluster_pt = temp_area_arr[i]
			cluster_err = temp_err_arr[i]
			if len(cluster_pt) >= N_r:
				max_pt = cluster_pt[np.argmax(cluster_err)]
				dist, ind = tree.query([max_pt], k = 1)
				max_idx = ind[0][0]
				if not max_idx in final_distortion_pt_idx:
					final_distortion_pt_idx.append(max_idx)

				median_err = statistics.median(cluster_err)*1
				high_distortion_pt = []
				high_distortion_err = []
				for t in range(len(cluster_pt)):
					if cluster_err[t] > median_err:
						high_distortion_pt.append(cluster_pt[t])
						high_distortion_err.append(cluster_err[t])

				if len(high_distortion_pt):
					cluster_dic = dbscan_clustering_vis(high_distortion_pt, seg_thresh, N_r, 0)
					if len(cluster_dic):
						for label in cluster_dic:
							new_temp_area_arr.append([high_distortion_pt[idx] for idx in cluster_dic[label]])
							new_temp_err_arr.append([cluster_err[idx] for idx in cluster_dic[label]])
		temp_area_arr = new_temp_area_arr
		temp_err_arr = new_temp_err_arr
		print(len(final_distortion_pt_idx))




	final_distortion_pt_idx = list(set(boundary_pt_idx + final_distortion_pt_idx))
	dis_pt_arr = [sv_off_geo_arr[idx] for idx in final_distortion_pt_idx]

	if vis:
		dis_pt = [sv_off_geo_arr[idx] for idx in final_distortion_pt_idx]	
		pc_vis(dis_pt + sparse_sv_off_geo_arr, [[200, 53, 55] for pt in dis_pt] + [[221, 222, 223] for pt in sparse_sv_off_geo_arr])

		# tree = BallTree(np.asarray(sv_off_geo_arr), leaf_size=1)
		# norm_len = get_neighbor_dis(sv_off_geo_arr)*0.25
		# vis_geo = []
		# vis_rgb = []
		# neigh_idx = []
		# if t in final_distortion_pt_idx:
		# 	pt = sv_off_geo_arr[t]
		# 	dist, ind = tree.query([pt], k = 5)
		# 	for idx in ind[1:]:
		# 		if not idx in final_distortion_pt_idx:
		# 			neigh_idx.append(idx)
		# rest_idx = list(set([t for t in range(len(sv_off_geo_arr))]) - set(final_distortion_pt_idx + neigh_idx))

		# for t in range(len(sv_off_geo_arr)):
		# 	if t in final_distortion_pt_idx:
		# 		vis_geo.append(sv_off_geo_arr[t])
		# 		vis_rgb.append([179, 2, 32])
				
		# 		s_pts = fibonacci_sphere(100, norm_len, np.asarray(sv_off_geo_arr[t]))
		# 		vis_geo = vis_geo + list(s_pts)
		# 		vis_rgb = vis_rgb + [[179, 2, 32] for pt in s_pts]

		# for idx in rest_idx:
		# 	# else:
		# 		vis_geo.append(sv_off_geo_arr[idx])
		# 		vis_rgb.append([221, 222, 223])


		# pc_vis(vis_geo, vis_rgb)

	dis_pt_idx = final_distortion_pt_idx
	print("dis_pt_idx1: ", len(dis_pt_idx))
	flag = 1
	while flag:
		[dis_pt_idx, remove_pt_idx] = remove_similar_distortion_pts(sv_off_geo_arr, sparse_sv_off_geo_arr, dis_pt_idx, err_arr)
		print(len(remove_pt_idx))
		if len(remove_pt_idx) == 0:
			flag = 0

	print("dis_pt_idx2: ", len(dis_pt_idx))
	if vis:
		dis_pt = [sv_off_geo_arr[idx] for idx in dis_pt_idx]	
		pc_vis(dis_pt + sparse_sv_off_geo_arr, [[200, 53, 55] for pt in dis_pt] + [[221, 222, 223] for pt in sparse_sv_off_geo_arr])
	
		spanning_tree_idx_arr = mini_spanning_tree(sv_off_geo_arr, sparse_sv_off_geo_arr, dis_pt_idx, 1)


		line_pt = []
		line_color = []
		for leaf in spanning_tree_idx_arr:
				temp_pt = []
				for t in range(0, len(leaf)-1):
					temp_pt = temp_pt + draw_line(sv_off_geo_arr[leaf[t]], sv_off_geo_arr[leaf[t+1]])
				# rgb = [randrange(254, 255), randrange(0, 255), randrange(0, 255)]
				rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
				temp_color = [rgb for pt in temp_pt]
				line_pt = line_pt + temp_pt
				line_color= line_color + temp_color

		pc_vis(line_pt + sparse_sv_off_geo_arr, line_color + [[221, 222, 223] for pt in sparse_sv_off_geo_arr])


			

	return dis_pt_idx


##############################################################################################################################
# 																						 									 #
#											(2) Image Generation															 #
# 																						 									 #
##############################################################################################################################
def isomap_based_dimension_reduction(patch_geo_arr, patch_off_arr, landmark_flag, n_nei):
	if landmark_flag:
		n_neighbors = min(n_nei, len(patch_off_arr)-1)
		n_components = 2
		X_off8 = np.matrix(patch_off_arr)
		embedding = manifold.Isomap(n_neighbors, n_components, eigen_solver='dense')
		Y_off8 = embedding.fit_transform(X_off8)
		reconstruction_err = embedding.reconstruction_error()
		X = np.matrix(patch_geo_arr)
		Y = []
		if len(patch_geo_arr)>5000:
			for t in range(0, len(patch_geo_arr), 5000):
				sub_non_smooth_sc_geo_arr = patch_geo_arr[t:t+5000]
				sub_X = np.matrix(sub_non_smooth_sc_geo_arr)
				sub_Y = embedding.transform(sub_X)
				for val in sub_Y:
					Y.append(val)
		else:
			Y = embedding.transform(X)
		# cal_isomap_local_distortion2(patch_geo_arr, embedding, X, Y)
		d2_geo = [[val[0], val[1], 0] for val in Y]

		return [d2_geo, reconstruction_err, embedding]
	else:
		n_neighbors = min(n_nei, len(patch_geo_arr)-1)
		n_components = 2
		embedding = manifold.Isomap(n_neighbors, n_components, eigen_solver='dense')
		X = np.matrix(patch_geo_arr)
		Y = embedding.fit_transform(X)
		reconstruction_err = embedding.reconstruction_error()
		d2_geo = [[val[0], val[1], 0] for val in Y]
		# cal_isomap_local_distortion2(patch_geo_arr, embedding, X, Y)

		return [d2_geo, reconstruction_err, embedding]

def cal_isomap_local_distortion2(sub_geo, embedding, X, Y):
	n_neighbors = 9


	X = np.array(sub_geo)
	tree = BallTree(X, leaf_size=1)

	diff_arr = []
	for t in range(len(sub_geo)):
		pt = sub_geo[t]
		dist, ind = tree.query([pt], k = n_neighbors)
		# print(t, ind, dist[0][1:])
		ori_dis = dist[0][1:]
		new_dis = [np.linalg.norm(np.asarray(Y[t]) - np.asarray(Y[idx])) for idx in ind[0][1:]]
		

		diff = np.sum(np.abs(np.asarray(ori_dis)-np.asarray(new_dis)))
		# print(t, diff, np.asarray(ori_dis)-np.asarray(new_dis))
		# diff = np.log(diff/10)
		diff_arr.append(diff)

	print(max(diff_arr), min(diff_arr))
	max_diff = max(diff_arr)
	d2_rgb1 = cm.jet(np.asarray(diff_arr)/max_diff)
	# print(d2_rgb1)
	d2_rgb = [val[0:3] for val in d2_rgb1]
	# print(d2_rgb)

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector((sub_geo))
	pcd.colors = o3d.utility.Vector3dVector(d2_rgb)
	o3d.visualization.draw_geometries([pcd])

	med_diff = np.median(diff_arr) 

	partition_rgb = []
	for dif in diff_arr:
		if dif > med_diff:
			partition_rgb.append([1, 0, 0])
		else:
			partition_rgb.append([0.5, 0.5, 0.5])
			
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector((sub_geo))
	pcd.colors = o3d.utility.Vector3dVector(partition_rgb)
	o3d.visualization.draw_geometries([pcd])
	# center = np.mean(sub_geo, axis=0)
	# dist1, ind1 = tree.query([center], k=tot_num)
	# farthest_pt1 = sub_geo[ind1[0][-1]]

 #    distances, indices = embedding.nbrs_.kneighbors(X, return_distance=True)
 #    for t in range(len(sub_geo)):
 #        new_dis = [np.linalg.norm(np.asarray(Y[t]) - np.asarray(Y[idx])) for idx in indices[t]]
 #        print(t, new_dis, indices[t])
		# diff = np.sum(np.abs(np.asarray(distances[t]-np.asarray(new_dis))))/8

def cal_isomap_local_distortion(sub_geo, embedding, X, Y):
	thresh = get_neighbor_dis(sub_geo)
	n_neighbors = 8
	distances, indices = embedding.nbrs_.kneighbors(X, return_distance=True)
	# print(distances, indices, len(distances[0]))
	valid_cnt = 0
	d2_rgb = []
	diff_arr = []
	for t in range(len(sub_geo)):
		# vec = embedding.kernel_pca_.alphas_[t]
		new_dis = [np.linalg.norm(np.asarray(Y[t]) - np.asarray(Y[idx])) for idx in indices[t]]
		diff = np.sum(np.abs(np.asarray(distances[t]-np.asarray(new_dis))))/8
		diff = np.log(diff)
		diff_arr.append(diff)
		if diff>thresh*0.4:
			d2_rgb.append([1, 0, 0])
		else:
			d2_rgb.append([0, 0, 1])
			valid_cnt = valid_cnt + 1
	
	d2_rgb = cm.viridis(diff_arr)
	d2_rgb = [val[0:3] for val in d2_rgb]
	# print(d2_rgb)
	max_dif_idx = np.argmax(diff_arr)
	print(max_dif_idx)

	d2_geo = [[val[0], val[1], 0] for val in Y]
	if 1:
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(([sub_geo[max_dif_idx]] + sub_geo + d2_geo))
		pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]] + d2_rgb + d2_rgb)
		o3d.visualization.draw_geometries([pcd])

   

	print("valid_cnt: ", valid_cnt)
	return valid_cnt/len(d2_geo)

def crop_img(img):
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	img = np.asarray(img)

	image_data_bw = img.max(axis=2)
	non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
	non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
	cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
	img_new = img[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]
	img_new = cv2.cvtColor(img_new, cv2.COLOR_RGB2GRAY)

	return img_new

def img_inpaint(image_orig, mask):
	image_defect = image_orig.copy()
	for layer in range(image_defect.shape[-1]):
		   image_defect[np.where(mask)] = 0

	image_result = inpaint.inpaint_biharmonic(image_defect, mask, multichannel=True)
	return image_result

def gen_virtual_point(patch_geo_arr_d2, patch_rgb_arr, target_num):
	patch_pt_num = len(patch_geo_arr_d2)
	tree = BallTree(np.array(patch_geo_arr_d2), leaf_size=1)
	center = np.mean(patch_geo_arr_d2, axis=0)
	dist, ind = tree.query([center], k=patch_pt_num)
	far_idx = ind[0][-1]
	maxPt = patch_geo_arr_d2[far_idx]
	new_patch_geo_arr_d2 = patch_geo_arr_d2[0:far_idx] + [maxPt for t in range(target_num - patch_pt_num)] + patch_geo_arr_d2[far_idx:] 
	new_patch_rgb_arr = patch_rgb_arr[0:far_idx] + [patch_rgb_arr[far_idx] for t in range(target_num - patch_pt_num)] + patch_rgb_arr[far_idx:] 
	mask_val = [255 for t in range(far_idx)] + [0 for t in range(target_num - patch_pt_num)] + [255 for t in range(patch_pt_num - far_idx)]
	pt_idx =  [t for t in range(far_idx)] + [-1 for t in range(target_num - patch_pt_num)] + [t for t in range(far_idx, patch_pt_num)]
	return [new_patch_geo_arr_d2, new_patch_rgb_arr, mask_val]
	
def estimate_mask_blk(patch_geo_arr_d2, patch_rgb_arr, blk_size):
	tot_num = len(patch_geo_arr_d2)
	patch_geo_arr_d2_arr = np.asarray(patch_geo_arr_d2)
	min_x =  min(patch_geo_arr_d2_arr[:, 0])
	min_y =  min(patch_geo_arr_d2_arr[:, 1])

	height = max(patch_geo_arr_d2_arr[:, 1]) - min(patch_geo_arr_d2_arr[:, 1])
	width = max(patch_geo_arr_d2_arr[:, 0]) - min(patch_geo_arr_d2_arr[:, 0])
	aspect_ratio = width/height

	img_height = np.sqrt(tot_num/aspect_ratio)
	img_weight = img_height*aspect_ratio

	mask = np.zeros((0, 0), np.uint8)
	start_s = 100
	step = 1
	mask_valid_cnt = 0

	

	for scale_f in range(start_s, 301, step):
		scale_f = scale_f/100.0
		img_w = int(np.round(img_weight*scale_f, 0))
		img_h = int(np.round(img_height*scale_f, 0))
		img_w = int(np.ceil(img_w/blk_size)*blk_size)
		img_h = int(np.ceil(img_h/blk_size)*blk_size)
		mask = np.ones((img_h, img_w), np.uint8)*0
		for i in range(tot_num):
			p_x = patch_geo_arr_d2[i][0]
			p_y = patch_geo_arr_d2[i][1]
			y_pix = int(np.floor((p_y - min_y)/height*0.9999*img_h))
			x_pix = int(np.floor((p_x - min_x)/width*0.9999*img_w))
			mask[y_pix][x_pix] = 255
		kernel = np.ones((3, 3), np.uint8)
		mask = cv2.dilate(mask, kernel, iterations = 1)
		
		# print(scale_f, np.sum(mask)/255, tot_num)
		for v_stp in range(0, img_h, blk_size):
			for h_stp in range(0, img_w, blk_size):
				sub_mask = mask[v_stp:v_stp + blk_size, h_stp:h_stp + blk_size]
				sub_h, sub_w = sub_mask.shape
				if np.sum(mask[v_stp:v_stp + blk_size, h_stp:h_stp + blk_size])/255 < sub_h*sub_w*0.5 and sub_h*sub_w>=0.5*blk_size*blk_size:
					mask[v_stp:v_stp + blk_size, h_stp:h_stp + blk_size] = 0
				elif np.sum(mask[v_stp:v_stp + blk_size, h_stp:h_stp + blk_size])/255 >= sub_h*sub_w*0.5 and sub_h*sub_w>=0.5*blk_size*blk_size:
					mask[v_stp:v_stp + blk_size, h_stp:h_stp + blk_size] = 255
				else:
					mask[v_stp:v_stp + blk_size, h_stp:h_stp + blk_size] = 0
		if np.sum(mask)/255>=tot_num:
			mask_valid_cnt = int(np.sum(mask)/255)
			break

	# cv2.imshow("mask", mask)
	# cv2.imshow("mask_rect", mask_rect)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# cv2.imshow("mask", mask)
	mask = crop_img(mask)
	# cv2.imshow("crop_img", mask)
	# cv2.waitKey(0)

	return [255 - mask, mask_valid_cnt]

def estimate_mask(patch_geo_arr_d2, patch_rgb_arr, blk_size):
	tot_num = len(patch_geo_arr_d2)
	patch_geo_arr_d2_arr = np.asarray(patch_geo_arr_d2)
	# bl = [min(patch_geo_arr_d2_arr[:,0]), min(patch_geo_arr_d2_arr[:,1]), 0]
	# tr = [max(patch_geo_arr_d2_arr[:,0]), max(patch_geo_arr_d2_arr[:,1]), 0]
	# tl = [min(patch_geo_arr_d2_arr[:,0]), max(patch_geo_arr_d2_arr[:,1]), 0]
	# br = [max(patch_geo_arr_d2_arr[:,0]), min(patch_geo_arr_d2_arr[:,1]), 0]
	# line_pt = draw_line(bl, tl) + draw_line(bl, br) + draw_line(br, tr) + draw_line(tl, tr)
	# line_color = [[255, 0, 0] for pt in line_pt]
	# pc_vis(patch_geo_arr_d2 + line_pt, patch_rgb_arr + line_color)


	min_x =  min(patch_geo_arr_d2_arr[:, 0])
	min_y =  min(patch_geo_arr_d2_arr[:, 1])

	height = max(patch_geo_arr_d2_arr[:, 1]) - min(patch_geo_arr_d2_arr[:, 1])
	width = max(patch_geo_arr_d2_arr[:, 0]) - min(patch_geo_arr_d2_arr[:, 0])
	aspect_ratio = width/height

	img_height = np.sqrt(tot_num/aspect_ratio)
	img_weight = img_height*aspect_ratio

	print(img_height, img_weight)


	mask = np.zeros((0, 0), np.uint8)
	start_s = 100
	step = 1
	mask_valid_cnt = 0
	for scale_f in range(start_s, 301, step):
		scale_f = scale_f/100.0
		img_w = int(np.round(img_weight*scale_f, 0))
		img_h = int(np.round(img_height*scale_f, 0))
		mask = np.ones((img_h, img_w), np.uint8)*0
		for i in range(tot_num):
			p_x = patch_geo_arr_d2[i][0]
			p_y = patch_geo_arr_d2[i][1]
			y_pix = int(np.floor((p_y - min_y)/height*0.9999*img_h))
			x_pix = int(np.floor((p_x - min_x)/width*0.9999*img_w))
			mask[y_pix][x_pix] = 255
		kernel = np.ones((3, 3), np.uint8)
		mask = cv2.dilate(mask, kernel, iterations = 1)
		
		print(scale_f, np.sum(mask)/255, tot_num)
		
		
		if np.sum(mask)/255>=tot_num:
			mask_valid_cnt = int(np.sum(mask)/255)
			break

	

	
	img_height = int(img_height)
	img_weight = int(img_weight)

	if img_height*img_weight<tot_num:
		img_height = img_height + 1
	if img_height*img_weight<tot_num:
		img_weight = img_weight + 1

	img_height = int(np.ceil(img_height/blk_size)*blk_size)
	img_weight = int(np.ceil(img_weight/blk_size)*blk_size)
	mask_rect = np.zeros((img_height, img_weight), np.uint8)
	# cv2.imshow("mask", mask)
	# cv2.imshow("mask_rect", mask_rect)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()		
	return [255 - mask, mask_valid_cnt, mask_rect, img_weight*img_height]

def test_rf_with_ip(patch_geo_arr_d2, patch_rgb_arr):
	patch_pt_num = len(patch_geo_arr_d2)
	# img_w = int(np.ceil(np.sqrt(patch_pt_num)))+50
	# img_h = img_w+50
	# mask_img = np.ones((img_h, img_w), np.uint8)*255
	# mask_img_rgb = np.ones((img_h, img_w, 3), np.uint8)*255
	# while int(np.sum(mask_img)/255) != img_w*img_h-patch_pt_num:
	# 	rand_x = randrange(50, img_w)
	# 	rand_y = randrange(50, img_h)
	# 	mask_img[rand_y][rand_x] = 0
	# 	mask_img_rgb[rand_y][rand_x] = [0, 0, 0]
	# cv2.imshow("mask_img", mask_img)
	# cv2.waitKey(0)
	# mask_img = PImage.fromarray(mask_img)
	# mask_img = PImage.convert("RGB")
	# mask_img = PImage.fromarray(mask_img_rgb)
	print("estimate_mask")
	[mask_img, mask_valid_cnt, mask_rect, mask_rect_valid_cnt] = estimate_mask(patch_geo_arr_d2, patch_rgb_arr)
	print("estimate_mask_blk")
	[mask_blk, mask_blk_valid_cnt] = estimate_mask_blk(patch_geo_arr_d2, patch_rgb_arr)
	# cv2.imshow("mask_img", mask_img)
	# cv2.imshow("mask_blk", mask_blk)
	# cv2.waitKey(0)
	rasterMask = rasterfairy.getRasterMaskFromImage(PImage.fromarray(mask_img))
	print("gen_virtual_point")
	[new_patch_geo_arr_d2, new_patch_rgb_arr, mask_val] = gen_virtual_point(patch_geo_arr_d2, patch_rgb_arr, mask_valid_cnt)
	grid_xy,(width,height) = rasterfairy.transformPointCloud2D(np.asarray(new_patch_geo_arr_d2)[:, 0:2], target=rasterMask, autoAdjustCount = False)
	grid_img = np.ones((height, width, 3), np.uint8)*255
	grid_mask = np.ones((height, width), np.uint8)*0
	for i in range(0, len(new_patch_geo_arr_d2)):
		grid_img[int(grid_xy[i][1])][int(grid_xy[i][0])] = new_patch_rgb_arr[i][::-1]
		grid_mask[int(grid_xy[i][1])][int(grid_xy[i][0])] = mask_val[i]
	ip_grid_img = img_inpaint(grid_img, 1-grid_mask/255)
	ip_grid_img = np.uint8(ip_grid_img*255.999)


	print("gen_virtual_point")
	rasterMask_rect = rasterfairy.getRasterMaskFromImage(PImage.fromarray(mask_rect))
	[new_patch_geo_arr_d2_rect, new_patch_rgb_arr_rect, mask_val_rect] = gen_virtual_point(patch_geo_arr_d2, patch_rgb_arr, mask_rect_valid_cnt)
	grid_xy,(width,height) = rasterfairy.transformPointCloud2D(np.asarray(new_patch_geo_arr_d2_rect)[:, 0:2], target=rasterMask_rect, autoAdjustCount = False)
	grid_img_rect = np.ones((height, width, 3), np.uint8)*255
	grid_mask_rect = np.ones((height, width), np.uint8)*0
	for i in range(0, len(new_patch_geo_arr_d2_rect)):
		grid_img_rect[int(grid_xy[i][1])][int(grid_xy[i][0])] = new_patch_rgb_arr_rect[i][::-1]
		grid_mask_rect[int(grid_xy[i][1])][int(grid_xy[i][0])] = mask_val_rect[i]
	
	print("grid_img_rect")

	ip_grid_img_rect = img_inpaint(grid_img_rect, 1-grid_mask_rect/255)
	ip_grid_img_rect = np.uint8(ip_grid_img_rect*255.999)
	# ip_grid_img = grid_img_rect
	# cv2.imshow("ip_grid_img", ip_grid_img)
	# cv2.imshow("grid_img", grid_img)
	# cv2.imshow("grid_img_rect", grid_img_rect)

	# cv2.imshow("grid_mask", grid_mask)
	# cv2.imshow("grid_mask_rect", grid_mask_rect)

	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	rasterMask = rasterfairy.getRasterMaskFromImage(PImage.fromarray(mask_blk))
	[new_patch_geo_arr_d2, new_patch_rgb_arr, mask_val] = gen_virtual_point(patch_geo_arr_d2, patch_rgb_arr, mask_blk_valid_cnt)
	print("grid_blk_img")
	grid_xy,(width,height) = rasterfairy.transformPointCloud2D(np.asarray(new_patch_geo_arr_d2)[:, 0:2], target=rasterMask, autoAdjustCount = False)
	grid_blk_img = np.ones((height, width, 3), np.uint8)*255
	grid_blk_mask = np.ones((height, width), np.uint8)*0
	for i in range(0, len(new_patch_geo_arr_d2)):
		grid_blk_img[int(grid_xy[i][1])][int(grid_xy[i][0])] = new_patch_rgb_arr[i][::-1]
		# grid_blk_img[int(grid_xy[i][1])][int(grid_xy[i][0])] = [170,175,209][::-1]
		grid_blk_mask[int(grid_xy[i][1])][int(grid_xy[i][0])] = mask_val[i]
	# cv2.imshow("grid_mask", grid_mask)
	# cv2.imshow("grid_blk_img", grid_blk_img)

	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	ip_grid_blk_img = img_inpaint(grid_blk_img, 1-grid_blk_mask/255)
	ip_grid_blk_img = np.uint8(ip_grid_blk_img*255.999)
	# ip_grid_blk_img = grid_blk_img
	# return [ip_grid_img, grid_img, grid_mask, grid_img_rect, grid_mask_rect, grid_blk_img, grid_blk_mask, ip_grid_blk_img]
	return [ip_grid_img, grid_img, grid_mask, grid_img_rect, grid_mask_rect, ip_grid_img_rect, grid_blk_img, grid_blk_mask, ip_grid_blk_img]

def test_rf_new(patch_geo_arr_d2, patch_rgb_arr):
	patch_pt_num = len(patch_geo_arr_d2)
	print("estimate_mask...")
	[mask_img, mask_valid_cnt, mask_rect, mask_rect_valid_cnt] = estimate_mask(patch_geo_arr_d2, patch_rgb_arr, 8)
	
	rasterMask = rasterfairy.getRasterMaskFromImage(PImage.fromarray(mask_img))
	[new_patch_geo_arr_d2, new_patch_rgb_arr, mask_val] = gen_virtual_point(patch_geo_arr_d2, patch_rgb_arr, mask_valid_cnt)
	st = time.time()
	grid_xy,(width,height) = rasterfairy.transformPointCloud2D(np.asarray(new_patch_geo_arr_d2)[:, 0:2], target=rasterMask, autoAdjustCount = False)
	print("orig time:", time.time()-st)
	grid_img = np.ones((height, width, 3), np.uint8)*255
	grid_mask = np.ones((height, width), np.uint8)*0
	for i in range(0, len(new_patch_geo_arr_d2)):
		grid_img[int(grid_xy[i][1])][int(grid_xy[i][0])] = new_patch_rgb_arr[i][::-1]
		grid_mask[int(grid_xy[i][1])][int(grid_xy[i][0])] = mask_val[i]
	
	print("grid_img_rect...")
	rasterMask_rect = rasterfairy.getRasterMaskFromImage(PImage.fromarray(mask_rect))
	[new_patch_geo_arr_d2_rect, new_patch_rgb_arr_rect, mask_val_rect] = gen_virtual_point(patch_geo_arr_d2, patch_rgb_arr, mask_rect_valid_cnt)
	st = time.time()
	grid_xy,(width,height) = rasterfairy.transformPointCloud2D(np.asarray(new_patch_geo_arr_d2_rect)[:, 0:2], target=rasterMask_rect, autoAdjustCount = False)
	print("rect time:", time.time()-st)
	grid_img_rect = np.ones((height, width, 3), np.uint8)*255
	grid_mask_rect = np.ones((height, width), np.uint8)*0
	for i in range(0, len(new_patch_geo_arr_d2_rect)):
		grid_img_rect[int(grid_xy[i][1])][int(grid_xy[i][0])] = new_patch_rgb_arr_rect[i][::-1]
		grid_mask_rect[int(grid_xy[i][1])][int(grid_xy[i][0])] = mask_val_rect[i]
	
	print("estimate_mask_blk8...")
	[mask_blk, mask_blk_valid_cnt] = estimate_mask_blk(patch_geo_arr_d2, patch_rgb_arr, blk_size=8)
	print("grid_blk_img8...")
	rasterMask = rasterfairy.getRasterMaskFromImage(PImage.fromarray(mask_blk))
	[new_patch_geo_arr_d2, new_patch_rgb_arr, mask_val] = gen_virtual_point(patch_geo_arr_d2, patch_rgb_arr, mask_blk_valid_cnt)
	st = time.time()
	grid_xy,(width,height) = rasterfairy.transformPointCloud2D(np.asarray(new_patch_geo_arr_d2)[:, 0:2], target=rasterMask, autoAdjustCount = False)
	print("grid_blk_img8 time:", time.time()-st)
	grid_blk_img8 = np.ones((height, width, 3), np.uint8)*255
	grid_blk_mask8 = np.ones((height, width), np.uint8)*0
	for i in range(0, len(new_patch_geo_arr_d2)):
		grid_blk_img8[int(grid_xy[i][1])][int(grid_xy[i][0])] = new_patch_rgb_arr[i][::-1]
		grid_blk_mask8[int(grid_xy[i][1])][int(grid_xy[i][0])] = mask_val[i]
	
	print("estimate_mask_blk16...")
	[mask_blk, mask_blk_valid_cnt] = estimate_mask_blk(patch_geo_arr_d2, patch_rgb_arr, blk_size=16)
	print("grid_blk_img16...")
	rasterMask = rasterfairy.getRasterMaskFromImage(PImage.fromarray(mask_blk))
	[new_patch_geo_arr_d2, new_patch_rgb_arr, mask_val] = gen_virtual_point(patch_geo_arr_d2, patch_rgb_arr, mask_blk_valid_cnt)
	grid_xy,(width,height) = rasterfairy.transformPointCloud2D(np.asarray(new_patch_geo_arr_d2)[:, 0:2], target=rasterMask, autoAdjustCount = False)
	grid_blk_img16 = np.ones((height, width, 3), np.uint8)*255
	grid_blk_mask16 = np.ones((height, width), np.uint8)*0
	for i in range(0, len(new_patch_geo_arr_d2)):
		grid_blk_img16[int(grid_xy[i][1])][int(grid_xy[i][0])] = new_patch_rgb_arr[i][::-1]
		grid_blk_mask16[int(grid_xy[i][1])][int(grid_xy[i][0])] = mask_val[i]

	print("estimate_mask_blk32...")
	[mask_blk, mask_blk_valid_cnt] = estimate_mask_blk(patch_geo_arr_d2, patch_rgb_arr, blk_size=32)
	print("grid_blk_img32...")
	rasterMask = rasterfairy.getRasterMaskFromImage(PImage.fromarray(mask_blk))
	[new_patch_geo_arr_d2, new_patch_rgb_arr, mask_val] = gen_virtual_point(patch_geo_arr_d2, patch_rgb_arr, mask_blk_valid_cnt)
	grid_xy,(width,height) = rasterfairy.transformPointCloud2D(np.asarray(new_patch_geo_arr_d2)[:, 0:2], target=rasterMask, autoAdjustCount = False)
	grid_blk_img32 = np.ones((height, width, 3), np.uint8)*255
	grid_blk_mask32 = np.ones((height, width), np.uint8)*0
	for i in range(0, len(new_patch_geo_arr_d2)):
		grid_blk_img32[int(grid_xy[i][1])][int(grid_xy[i][0])] = new_patch_rgb_arr[i][::-1]
		grid_blk_mask32[int(grid_xy[i][1])][int(grid_xy[i][0])] = mask_val[i]

	return [grid_img, grid_mask, grid_img_rect, grid_mask_rect, grid_blk_img8, grid_blk_mask8, grid_blk_img16, grid_blk_mask16, grid_blk_img32, grid_blk_mask32]

def test_rf(patch_geo_arr_d2, patch_rgb_arr):
	patch_pt_num = len(patch_geo_arr_d2)

	print("estimate_mask...")

	[mask_img, mask_valid_cnt, mask_rect, mask_rect_valid_cnt] = estimate_mask(patch_geo_arr_d2, patch_rgb_arr)
	print("estimate_mask_blk...")
	[mask_blk, mask_blk_valid_cnt] = estimate_mask_blk(patch_geo_arr_d2, patch_rgb_arr)

	rasterMask = rasterfairy.getRasterMaskFromImage(PImage.fromarray(mask_img))
	[new_patch_geo_arr_d2, new_patch_rgb_arr, mask_val] = gen_virtual_point(patch_geo_arr_d2, patch_rgb_arr, mask_valid_cnt)
	grid_xy,(width,height) = rasterfairy.transformPointCloud2D(np.asarray(new_patch_geo_arr_d2)[:, 0:2], target=rasterMask, autoAdjustCount = False)
	grid_img = np.ones((height, width, 3), np.uint8)*255
	grid_mask = np.ones((height, width), np.uint8)*0
	for i in range(0, len(new_patch_geo_arr_d2)):
		grid_img[int(grid_xy[i][1])][int(grid_xy[i][0])] = new_patch_rgb_arr[i][::-1]
		grid_mask[int(grid_xy[i][1])][int(grid_xy[i][0])] = mask_val[i]
	# ip_grid_img = img_inpaint(grid_img, 1-grid_mask/255)
	# ip_grid_img = np.uint8(ip_grid_img*255.999)

	print("grid_img_rect...")
	rasterMask_rect = rasterfairy.getRasterMaskFromImage(PImage.fromarray(mask_rect))
	[new_patch_geo_arr_d2_rect, new_patch_rgb_arr_rect, mask_val_rect] = gen_virtual_point(patch_geo_arr_d2, patch_rgb_arr, mask_rect_valid_cnt)
	grid_xy,(width,height) = rasterfairy.transformPointCloud2D(np.asarray(new_patch_geo_arr_d2_rect)[:, 0:2], target=rasterMask_rect, autoAdjustCount = False)
	grid_img_rect = np.ones((height, width, 3), np.uint8)*255
	grid_mask_rect = np.ones((height, width), np.uint8)*0
	for i in range(0, len(new_patch_geo_arr_d2_rect)):
		grid_img_rect[int(grid_xy[i][1])][int(grid_xy[i][0])] = new_patch_rgb_arr_rect[i][::-1]
		grid_mask_rect[int(grid_xy[i][1])][int(grid_xy[i][0])] = mask_val_rect[i]

	# ip_grid_img_rect = img_inpaint(grid_img_rect, 1-grid_mask_rect/255)
	# ip_grid_img_rect = np.uint8(ip_grid_img_rect*255.999)

	print("grid_blk_img...")
	rasterMask = rasterfairy.getRasterMaskFromImage(PImage.fromarray(mask_blk))
	[new_patch_geo_arr_d2, new_patch_rgb_arr, mask_val] = gen_virtual_point(patch_geo_arr_d2, patch_rgb_arr, mask_blk_valid_cnt)

	grid_xy,(width,height) = rasterfairy.transformPointCloud2D(np.asarray(new_patch_geo_arr_d2)[:, 0:2], target=rasterMask, autoAdjustCount = False)
	grid_blk_img = np.ones((height, width, 3), np.uint8)*255
	grid_blk_mask = np.ones((height, width), np.uint8)*0
	for i in range(0, len(new_patch_geo_arr_d2)):
		grid_blk_img[int(grid_xy[i][1])][int(grid_xy[i][0])] = new_patch_rgb_arr[i][::-1]
		grid_blk_mask[int(grid_xy[i][1])][int(grid_xy[i][0])] = mask_val[i]

	# ip_grid_blk_img = img_inpaint(grid_blk_img, 1-grid_blk_mask/255)
	# ip_grid_blk_img = np.uint8(ip_grid_blk_img*255.999)

	return [grid_img, grid_mask, grid_img_rect, grid_mask_rect, grid_blk_img, grid_blk_mask]

def isomap_based_aig(patch_off_geo, patch_geo, patch_rgb):
	[patch_geo_arr_d2, recon_err, off_embedding] = isomap_based_dimension_reduction(patch_geo, patch_off_geo, len(patch_off_geo)>16)
	pc_vis(patch_off_geo, [[170,175,209] for pt in patch_off_geo])
	pc_vis(patch_geo, [[170,175,209] for pt in patch_geo])
	pc_vis(patch_geo_arr_d2, [[170,175,209] for pt in patch_geo])
	test_rf_mask(patch_geo_arr_d2, patch_rgb)

##############################################################################################################################
# 																						 									 #
#											(4) Fine-Grained Segmentation								     				 #
# 																						 									 #
##############################################################################################################################
def rotate(origin, point, angle):
	"""
	Rotate a point counterclockwise by a given angle around a given origin.

	The angle should be given in radians.
	"""
	[ox, oy] = origin
	[px, py] = point

	qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
	qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
	return [qx, qy]

def rot_pt(pt_arr, angle):
	origin = np.mean(pt_arr, axis=0)
	rot_pt_arr = []
	for pt in pt_arr:
		[x, y] = rotate(origin[0:2], pt[0:2], angle)
		rot_pt_arr.append([x, y, 0])
	return rot_pt_arr

def img_compression(attr_img, pt_num, mb_size, cmp_method, extra_bit):
	mode = 0
	if cmp_method == "jpg":
		mode = int(cv2.IMWRITE_JPEG_QUALITY)
	elif cmp_method == "webp":
		mode = int(cv2.IMWRITE_WEBP_QUALITY)
	quality_arr = [20, 50, 80, 90]
		
	blk_size = int(np.floor(16376.0/mb_size)*mb_size) # The maximum pixel dimensions of a WebP image is 16383 x 16383.
	
	img_h, img_w, c = attr_img.shape
	attr_img_yuv = cv2.cvtColor(attr_img, cv2.COLOR_BGR2YUV)
	attr_img_y = attr_img_yuv[:,:,0]
	
	bpp_arr = []
	psnr_arr = []
	# size_arr = []
	# diff_arr = []
	for quality in quality_arr:
		yuv_size = 0
		tot_diff = 0.0
		for i in range(0, int(np.ceil(img_w/blk_size))):
			compressed_img_path =  'D:\\' + dataset + '\\' + str(i) + '_' + str(quality) + "." + cmp_method
			sub_attr_img = attr_img[:, i*blk_size:(i+1)*blk_size]
			cv2.imwrite(compressed_img_path, sub_attr_img, [mode, quality])
			sub_attr_img_y = attr_img_y[:, i*blk_size:(i+1)*blk_size]
			sub_attr_img_yuv_size = os.stat(compressed_img_path).st_size
			yuv_size = yuv_size + sub_attr_img_yuv_size
			compressed_yuv_img = cv2.imread(compressed_img_path)
			compressed_yuv_img_y = cv2.cvtColor(compressed_yuv_img, cv2.COLOR_BGR2YUV)[:,:,0]
			# print(compressed_yuv_img.shape, sub_attr_img_y.shape)
			for s in range(0, compressed_yuv_img_y.shape[0]):
				for t in range(0, compressed_yuv_img_y.shape[1]):
					dif = int(sub_attr_img_y[s][t]) - int(compressed_yuv_img_y[s][t])
					dif = dif*dif
					tot_diff = tot_diff + dif
		mse = tot_diff/pt_num
		psnr = 20*np.log10(255.0/np.sqrt(mse))
		bpp = yuv_size*8.0/pt_num
		if extra_bit:
			bpp = (yuv_size*8.0 + extra_bit)/pt_num
		bpp_arr.append(bpp)
		psnr_arr.append(psnr)
		# diff_arr.append(tot_diff)
		# size_arr.append(yuv_size)

	return [bpp_arr, psnr_arr]

def img_compression_with_mask(attr_img, rect_mask_all, pt_num):
	quality_arr = [20, 50, 80, 90]
	bpp_arr = []
	psnr_arr = []

	mb_size, img_w, ch = attr_img.shape
	blk_size = int(np.floor(16376.0/mb_size)*mb_size)

	img_h, img_w, c = attr_img.shape
	attr_img_yuv = cv2.cvtColor(attr_img, cv2.COLOR_BGR2YUV)
	attr_img_y = attr_img_yuv[:,:,0]
	
	bpp_arr = []
	psnr_arr = []
	# size_arr = []
	# diff_arr = []
	for quality in quality_arr:
		yuv_size = 0
		tot_diff = 0.0
		for i in range(0, int(np.ceil(img_w/blk_size))):
			compressed_img_path = 'D:\\' + frame_id + '\\' + str(i) + '_' + str(quality) + "_2.webp"
			sub_attr_img = attr_img[:, i*blk_size:(i+1)*blk_size]
			sub_rect_mask_all = rect_mask_all[:, i*blk_size:(i+1)*blk_size]


			cv2.imwrite(compressed_img_path, sub_attr_img, [int(cv2.IMWRITE_WEBP_QUALITY), quality])
			sub_attr_img_y = attr_img_y[:, i*blk_size:(i+1)*blk_size]
			sub_attr_img_yuv_size = os.stat(compressed_img_path).st_size
			yuv_size = yuv_size + sub_attr_img_yuv_size
			compressed_yuv_img = cv2.imread(compressed_img_path)
			compressed_yuv_img_y = cv2.cvtColor(compressed_yuv_img, cv2.COLOR_BGR2YUV)[:,:,0]
			# print(sub_rect_mask_all.shape, compressed_yuv_img_y.shape)
			for s in range(0, compressed_yuv_img_y.shape[0]):
				for t in range(0, compressed_yuv_img_y.shape[1]):
					if sub_rect_mask_all[s][t]>0:
						dif = int(sub_attr_img_y[s][t]) - int(compressed_yuv_img_y[s][t])
						dif = dif*dif
						tot_diff = tot_diff + dif
		mse = tot_diff/pt_num
		psnr = 20*np.log10(255.0/np.sqrt(mse))
		bpp = yuv_size*8.0/pt_num
		# if extra_bit:
		# 	bpp = (yuv_size*8.0 + extra_bit)/pt_num
		bpp_arr.append(bpp)
		psnr_arr.append(psnr)
		# diff_arr.append(tot_diff)
		# size_arr.append(yuv_size)

	return [bpp_arr, psnr_arr]




	# for quality in quality_arr:
	# 	yuv_size = 0
	# 	tot_diff = 0.0
	# 	attr_img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	# 	attr_img_y = attr_img_yuv[:,:,0]

	# 	compressed_img_path = '' + str(quality) + ".webp"
	# 	mode =  int(cv2.IMWRITE_WEBP_QUALITY)
	# 	cv2.imwrite(compressed_img_path, img, [mode, quality])
	# 	sub_attr_img_y = attr_img_y
	# 	sub_attr_img_yuv_size = os.stat(compressed_img_path).st_size
	# 	yuv_size = yuv_size + sub_attr_img_yuv_size
	# 	compressed_yuv_img = cv2.imread(compressed_img_path)
	# 	compressed_yuv_img_y = cv2.cvtColor(compressed_yuv_img, cv2.COLOR_BGR2YUV)[:,:,0]
	# 	print("rect_mask_all:", rect_mask_all.shape)
	# 	for s in range(0, compressed_yuv_img_y.shape[0]):
	# 		for t in range(0, compressed_yuv_img_y.shape[1]):
	# 			if rect_mask_all[s][t]>0:
	# 				dif = int(sub_attr_img_y[s][t]) - int(compressed_yuv_img_y[s][t])
	# 				dif = dif*dif
	# 				tot_diff = tot_diff + dif

	# 	mse = tot_diff/pt_num
	# 	psnr = 20*np.log10(255.0/np.sqrt(mse))
	# 	bpp = yuv_size*8.0/pt_num
	# 	if 0:
	# 		bpp = (yuv_size*8.0 + 0)/pt_num
	# 	print(bpp, psnr)
	# 	bpp_arr.append(bpp)
	# 	psnr_arr.append(psnr)
	# return [bpp_arr, psnr_arr]

def peer_method(dataset):
	chain_bpp_arr_all = []
	chain_psnr_arr_all = []
	chain_legend_arr_all = []
	if dataset == "soldier":
		chain_bpp_arr_all.append([0.42, 0.9, 1.75, 2.92])
		chain_psnr_arr_all.append([35.5, 39.1, 43.2, 47.8])
		chain_legend_arr_all.append("mm2018")

		chain_bpp_arr_all.append([0.48, 1.05, 1.9, 3.05])
		chain_psnr_arr_all.append([33.8, 37.7, 42, 47.2])
		chain_legend_arr_all.append("MPEG TMC1")

	if dataset == "andrew9_frame0027":
		chain_bpp_arr_all.append([0.2562620423892099, 0.5838150289017339, 1.5394990366088632, 3.3468208092485554])
		chain_psnr_arr_all.append([27.747143433257158, 31.448626607518932, 36.343460142492276, 41.480754581709014])
		chain_legend_arr_all.append("GSR")

		chain_bpp_arr_all.append([0.33718689788053946, 0.8535645472061657, 1.8169556840077075, 3.3815028901734108])
		chain_psnr_arr_all.append([27.44575883855357, 31.381995787964335, 36.25366312676435, 41.38813460590581])
		chain_legend_arr_all.append("G-PCC")

		chain_bpp_arr_all.append([0.37186897880539493, 0.9922928709055878, 1.9710982658959537, 3.481695568400771])
		chain_psnr_arr_all.append([27.446162118564324, 31.406864721960837, 36.25545548236771, 41.41255545100148])
		chain_legend_arr_all.append("RAHT")   

	if dataset == "David_frame0000":
		chain_bpp_arr_all.append([0.1127819548872181, 0.2135338345864662, 0.4120300751879699, 0.9112781954887217])
		chain_psnr_arr_all.append([33.353191489361706, 36.229787234042554, 39.00425531914894, 42.45957446808511])
		chain_legend_arr_all.append("GSR")

		chain_bpp_arr_all.append([0.12932330827067673, 0.2706766917293233, 0.58796992481203, 1.3052631578947367])
		chain_psnr_arr_all.append([33.45531914893617, 36.04255319148936, 39.07234042553192, 42.47659574468085])
		chain_legend_arr_all.append("G-PCC")

		chain_bpp_arr_all.append([0.14436090225563913, 0.30977443609022565, 0.6511278195488721, 1.3609022556390977])
		chain_psnr_arr_all.append([33.45531914893617, 36.12765957446808, 39.1063829787234, 42.52765957446809])
		chain_legend_arr_all.append("RAHT")

	if dataset == "ricardo9_frame0039":
		chain_bpp_arr_all.append([0.0884476534296029, 0.14368231046931407, 0.2996389891696752, 0.6148014440433215])
		chain_psnr_arr_all.append([34.8275920965801, 37.59590078024921, 41.049493420286474, 44.550133923372535])
		chain_legend_arr_all.append("GSR")

		chain_bpp_arr_all.append([0.1068592057761733, 0.21299638989169678, 0.4296028880866427, 0.8866425992779786])
		chain_psnr_arr_all.append([34.89588137106478, 37.86918209696828, 41.062971157951935, 44.47352199060595])
		chain_legend_arr_all.append("G-PCC")

		chain_bpp_arr_all.append([0.1317689530685921, 0.2638989169675091, 0.5054151624548738, 0.9624548736462096])
		chain_psnr_arr_all.append([34.877962812002636, 37.91933542952525, 41.07800163037149, 44.522961065176034])
		chain_legend_arr_all.append("RAHT")

	if dataset == "phil9_frame0244":
		chain_bpp_arr_all.append([0.24630862329803316, 0.6269818456883509, 1.3968759455370652, 3.2635249621785176])
		chain_psnr_arr_all.append([28.74, 32.24, 36.64, 41.22])
		chain_legend_arr_all.append("GSR")

		chain_bpp_arr_all.append([0.32194402420574875, 0.8009606656580939, 1.7600075642965207, 3.3618078668683813])
		chain_psnr_arr_all.append([28.759999999999998, 32.24, 36.52, 41.36])
		chain_legend_arr_all.append("G-PCC")

		chain_bpp_arr_all.append([0.378661119515885, 0.8992813918305596, 1.8507866868381242, 3.399621785173979])
		chain_psnr_arr_all.append([28.8, 32.28, 36.5, 41.38])
		chain_legend_arr_all.append("RAHT")

	if dataset == "sarah9_frame0018":
		chain_bpp_arr_all.append([0.10734094616639478, 0.21044045676998369, 0.36704730831973903, 0.8199021207177815])
		chain_psnr_arr_all.append([34.33118279569892, 37.9268817204301, 40.920430107526876, 44.894623655913975])
		chain_legend_arr_all.append("GSR")

		chain_bpp_arr_all.append([0.13213703099510604, 0.2665579119086461, 0.5275693311582383, 1.039151712887439])
		chain_psnr_arr_all.append([34.4, 37.445161290322574, 40.920430107526876, 44.63655913978494])
		chain_legend_arr_all.append("G-PCC")

		chain_bpp_arr_all.append([0.16215334420880917, 0.3187601957585645, 0.6163132137030998, 1.1135399673735729])
		chain_psnr_arr_all.append([34.382795698924724, 37.427956989247306, 40.98924731182795, 44.6021505376344])
		chain_legend_arr_all.append("RAHT")

	if dataset == "longdress_vox10_1051":
		chain_bpp_arr_all.append([0.3745935567963643, 0.8042203159952792, 1.7209870889331198, 3.3760519965476554])
		chain_psnr_arr_all.append([28.5004843851831, 31.902524087154106, 36.256389480915225, 41.22776672009582])
		chain_legend_arr_all.append("GSR")

		chain_bpp_arr_all.append([0.4440068342345833, 1.0501787821675796, 2.143437901819527, 3.8733905201416174])
		chain_psnr_arr_all.append([28.30331319465239, 31.844855829355506, 36.239409578496826, 41.25080584080461])
		chain_legend_arr_all.append("G-PCC")

		chain_bpp_arr_all.append([0.508188751695348, 1.2320081728990895, 2.46963169112078, 4.354686206471386])
		chain_psnr_arr_all.append([28.32349884628257, 31.865816497278637, 36.221795571838726, 41.273739277473446])
		chain_legend_arr_all.append("RAHT")

	if dataset == "loot_vox10_1000":
		chain_bpp_arr_all.append([0.09545454545454551, 0.18333333333333335, 0.4106060606060606, 0.9424242424242426])
		chain_psnr_arr_all.append([32.59227467811159, 35.442060085836914, 39.081545064377686, 43.03004291845494])
		chain_legend_arr_all.append("GSR")

		chain_bpp_arr_all.append([0.10909090909090913, 0.2575757575757576, 0.5833333333333333, 1.1651515151515153])
		chain_psnr_arr_all.append([32.866952789699575, 35.562231759656655, 38.97854077253219, 42.9442060085837])
		chain_legend_arr_all.append("G-PCC")

		chain_bpp_arr_all.append([0.11515151515151517, 0.29090909090909095, 0.6545454545454545, 1.2878787878787878])
		chain_psnr_arr_all.append([32.84978540772532, 35.596566523605155, 38.97854077253219, 42.995708154506445])
		chain_legend_arr_all.append("RAHT")

	if dataset == "soldier_vox10_0536":
		chain_bpp_arr_all.append([0.154, 0.3456, 0.6945, 1.42])
		chain_psnr_arr_all.append([30.257, 33.36, 37.194, 41.638])
		chain_legend_arr_all.append("GSR")

		chain_bpp_arr_all.append([0.196, 0.4615, 0.992, 1.831])
		chain_psnr_arr_all.append([30.257, 33.342, 37.211, 41.778])
		chain_legend_arr_all.append("G-PCC")

		chain_bpp_arr_all.append([0.20904783017023987, 0.5140444215614766, 1.09715488432183, 1.9911924409929302])
		chain_psnr_arr_all.append([30.25708061002179, 33.32461873638344, 37.1764705882353, 41.77777777777778])
		chain_legend_arr_all.append("RAHT")

	if dataset == "frame_0200":
		chain_bpp_arr_all.append([0.2210879605566034, 0.40688477360143754, 0.7622206216493161, 1.4878803629505324])
		chain_psnr_arr_all.append([33.53648068669528, 36.66094420600858, 39.73390557939914, 43.18454935622317])
		chain_legend_arr_all.append("GSR")

		chain_bpp_arr_all.append([0.1588450628926148, 0.3238249402260273, 0.6428286083431101, 1.3148588443203588])
		chain_psnr_arr_all.append([33.72532188841201, 36.592274678111586, 39.66523605150214, 43.133047210300425])
		chain_legend_arr_all.append("G-PCC")

		chain_bpp_arr_all.append([0.19171332254184176, 0.3878276430491409, 0.7552779304097301, 1.4878766502814205])
		chain_psnr_arr_all.append([33.708154506437765, 36.54077253218884, 39.63090128755365, 43.167381974248926])
		chain_legend_arr_all.append("RAHT")

	if dataset == "redandblack_vox10_1550":
		chain_bpp_arr_all.append([0.1883116883116882, 0.4253246753246752, 0.9253246753246752, 2.00974025974026])
		chain_psnr_arr_all.append([32.820174828740136, 35.4952584888345, 38.97848475578668, 43.22213261399557])
		chain_legend_arr_all.append("GSR")

		chain_bpp_arr_all.append([0.22727272727272718, 0.5324675324675324, 1.1623376623376624, 2.272727272727273])
		chain_psnr_arr_all.append([32.82061977993455, 35.513612725604155, 38.96406092123436, 43.00243796175273])
		chain_legend_arr_all.append("G-PCC")

		chain_bpp_arr_all.append([0.30194805194805185, 0.6720779220779219, 1.3863636363636362, 2.577922077922078])
		chain_psnr_arr_all.append([32.78721136108716, 35.48094589208079, 38.915227527647225, 43.04018465474568])
		chain_legend_arr_all.append("RAHT")

	if dataset == "Egyptian_mask":
		chain_bpp_arr_all.append([0.1908602150537635, 0.3602150537634409, 0.9327956989247312, 1.9220430107526882])
		chain_psnr_arr_all.append([31.008490573247435, 33.701660667476744, 37.942484199585785, 41.87153327066663])
		chain_legend_arr_all.append("GSR")

		chain_bpp_arr_all.append([0.1827956989247312, 0.403225806451613, 0.900537634408602, 1.8440860215053763])
		chain_psnr_arr_all.append([31.844403176131628, 34.554050026365864, 37.85756318255114, 41.87241977516411])
		chain_legend_arr_all.append("G-PCC")

		chain_bpp_arr_all.append([0.20967741935483875, 0.4946236559139784, 1.0833333333333333, 2.172043010752688])
		chain_psnr_arr_all.append([31.87821262351835, 34.75770150782187, 37.90665719023927, 41.85163277315419])
		chain_legend_arr_all.append("RAHT")

	if dataset == "Shiva_00035":
		chain_bpp_arr_all.append([0.8641509433962262, 1.815094339622641, 3.4226415094339613, 6.396226415094338])
		chain_psnr_arr_all.append([24.623469352557688, 29.569224931875404, 34.979876513400704, 40.8343692870201])
		chain_legend_arr_all.append("GSR")

		chain_bpp_arr_all.append([0.8716981132075472, 1.8452830188679246, 3.3471698113207538, 5.603773584905659])
		chain_psnr_arr_all.append([25.16099479148701, 29.799765444448276, 35.1841468041806, 40.854892897795864])
		chain_legend_arr_all.append("G-PCC")

		chain_bpp_arr_all.append([0.9245283018867925, 1.9811320754716981, 3.558490566037735, 5.852830188679245])
		chain_psnr_arr_all.append([25.289303576972163, 29.877417129453967, 35.13431064813217, 40.77970404608326])
		chain_legend_arr_all.append("RAHT")

	if dataset == "Facade_00009":
		chain_bpp_arr_all.append([0.5326086956521741, 1.3043478260869565, 2.217391304347826, 3.8858695652173907])
		chain_psnr_arr_all.append([26.639596868232555, 31.30034982508746, 35.93703148425788, 40.95872896884892])
		chain_legend_arr_all.append("GSR")

		chain_bpp_arr_all.append([0.625, 1.2391304347826086, 2.2228260869565215, 3.7771739130434785])
		chain_psnr_arr_all.append([26.77681992337165, 31.001999000499758, 35.70710478094287, 40.890596368482434])
		chain_legend_arr_all.append("G-PCC")

		chain_bpp_arr_all.append([0.7010869565217392, 1.3858695652173916, 2.440217391304348, 4.08695652173913])
		chain_psnr_arr_all.append([26.845202398800605, 31.046851574212898, 35.68245044144595, 40.865234049641856])
		chain_legend_arr_all.append("RAHT")

	if dataset == "House_without_roof_00057":
		chain_bpp_arr_all.append([0.28911418029842956, 0.8106670584778137, 1.527407189734548, 2.852453717308257])
		chain_psnr_arr_all.append([29.51807228915662, 33.29317269076304, 37.26907630522088, 41.787148594377506])
		chain_legend_arr_all.append("GSR")

		chain_bpp_arr_all.append([0.289101119926862, 0.6642211120906392, 1.3224899598393574, 2.458970842720475])
		chain_psnr_arr_all.append([29.477911646586342, 32.9718875502008, 37.14859437751004, 41.827309236947784])
		chain_legend_arr_all.append("G-PCC")

		chain_bpp_arr_all.append([0.3151173800894637, 0.7455219250987692, 1.4948476834165931, 2.7549058020700685])
		chain_psnr_arr_all.append([29.477911646586342, 32.9718875502008, 37.14859437751004, 41.827309236947784])
		chain_legend_arr_all.append("RAHT")

	if dataset == "Frog_00067":
		chain_bpp_arr_all.append([0.19838420107719923, 0.4245960502692998, 1.0816876122082584, 2.1427289048473974])
		chain_psnr_arr_all.append([30.564102564102566, 32.32478632478633, 36.90598290598291, 41.58974358974359])
		chain_legend_arr_all.append("GSR")

		chain_bpp_arr_all.append([0.1903052064631956, 0.3815080789946139, 1.0062836624775584, 1.9111310592459612])
		chain_psnr_arr_all.append([30.273504273504276, 32.25641025641026, 36.991452991452995, 41.43589743589744])
		chain_legend_arr_all.append("G-PCC")

		chain_bpp_arr_all.append([0.2037701974865348, 0.3868940754039497, 1.0924596050269302, 2.056552962298025])
		chain_psnr_arr_all.append([30.273504273504276, 32.25641025641026, 36.95726495726496, 41.401709401709404])
		chain_legend_arr_all.append("RAHT")

	if dataset == "Arco_Valentino_Dense":
		chain_bpp_arr_all.append([])
		chain_psnr_arr_all.append([])
		chain_legend_arr_all.append("GSR (N.A.)")

		chain_bpp_arr_all.append([2.448, 3.72, 6.264, 10.584])
		chain_psnr_arr_all.append([31.51, 36.486, 42.727, 49.892])
		chain_legend_arr_all.append("G-PCC")
		
		chain_bpp_arr_all.append([2.16, 3.672, 6.264, 11.04])
		chain_psnr_arr_all.append([30.84, 36.62, 42.93, 50.359])
		chain_legend_arr_all.append("RAHT")

	if dataset == "Staue_Klimt":
		chain_bpp_arr_all.append([])
		chain_psnr_arr_all.append([])
		chain_legend_arr_all.append("GSR (N.A.)")

		chain_bpp_arr_all.append([1.355, 2.433, 3.253, 5.729])
		chain_psnr_arr_all.append([32.06, 36.9, 39.52, 46.184])
		chain_legend_arr_all.append("G-PCC")
		
		chain_bpp_arr_all.append([1.3887, 2.654, 3.815, 5.4195])
		chain_psnr_arr_all.append([31.32, 36.094, 39.59, 43.49])
		chain_legend_arr_all.append("RAHT")


	return [chain_bpp_arr_all, chain_psnr_arr_all, chain_legend_arr_all]

def ball_tree_partition(seg_geo_arr, seg_color_arr):
	tot_num = len(seg_geo_arr)
	X = np.array(seg_geo_arr)
	tree = BallTree(X, leaf_size=1)
	center = np.mean(seg_geo_arr, axis=0)
	dist1, ind1 = tree.query([center], k=tot_num)
	farthest_pt1 = seg_geo_arr[ind1[0][-1]]
	dist2, ind2 = tree.query([farthest_pt1], k=tot_num)
	farthest_pt2 = seg_geo_arr[ind2[0][-1]]

	partition_arr_idx = [list(range(tot_num))]
	flag = 1
	iteration = 0
	while flag:
		temp_partition_arr = []
		for t in range(0, len(partition_arr_idx)):
			sub_seg_geo_idx_arr = partition_arr_idx[t]
			sub_seg_geo_arr = [seg_geo_arr[id] for id in sub_seg_geo_idx_arr]
			sub_tot_num = len(sub_seg_geo_arr)
			if sub_tot_num>1:
				X = np.array(sub_seg_geo_arr)
				tree = BallTree(X, leaf_size=1)
				center = np.mean(sub_seg_geo_arr, axis=0)
				dist1, ind1 = tree.query([center], k=sub_tot_num)
				farthest_pt1 = sub_seg_geo_arr[ind1[0][-1]]
				dist2, ind2 = tree.query([farthest_pt1], k=sub_tot_num)
				farthest_pt2 = sub_seg_geo_arr[ind2[0][-1]]
				temp_pt_idx_arr1 = []
				temp_pt_idx_arr2 = []
				for i in range(0, sub_tot_num):
					pt = sub_seg_geo_arr[i]
					vec1 = np.asarray(farthest_pt1) - np.asarray(pt)
					vec2 = np.asarray(farthest_pt2) - np.asarray(pt)
					
					if np.linalg.norm(vec1) < np.linalg.norm(vec2):
						temp_pt_idx_arr1.append(sub_seg_geo_idx_arr[i])
					else:
						temp_pt_idx_arr2.append(sub_seg_geo_idx_arr[i])
				
				temp_pt_arr1 = []
				temp_pt_arr2 = []
				for idx in temp_pt_idx_arr1:
					temp_pt_arr1.append(seg_geo_arr[idx])
				for idx in temp_pt_idx_arr2:
					temp_pt_arr2.append(seg_geo_arr[idx])

				temp_center1 = np.mean(temp_pt_arr1, axis=0)
				temp_center2 = np.mean(temp_pt_arr2, axis=0)

				if t == 0:
					if len(partition_arr_idx)>1:
						parent_sub_seg_geo_idx_arr = partition_arr_idx[t+1]
						parent_sub_seg_geo_arr = [seg_geo_arr[id] for id in parent_sub_seg_geo_idx_arr]
						parent_center = np.mean(parent_sub_seg_geo_arr, axis=0)

						parent_vec1 = np.asarray(temp_center1) - np.asarray(parent_center)
						parent_vec2 = np.asarray(temp_center2) - np.asarray(parent_center)
						if np.linalg.norm(parent_vec1) < np.linalg.norm(parent_vec2):
							temp_partition_arr.append(temp_pt_idx_arr2)
							temp_partition_arr.append(temp_pt_idx_arr1)
						else:
							temp_partition_arr.append(temp_pt_idx_arr1)
							temp_partition_arr.append(temp_pt_idx_arr2)
					else:
						temp_partition_arr.append(temp_pt_idx_arr1)
						temp_partition_arr.append(temp_pt_idx_arr2)
				else:
					parent_sub_seg_geo_idx_arr = temp_partition_arr[-1]
					parent_sub_seg_geo_arr = [seg_geo_arr[id] for id in parent_sub_seg_geo_idx_arr]
					parent_center = np.mean(parent_sub_seg_geo_arr, axis=0)
					parent_vec1 = np.asarray(temp_center1) - np.asarray(parent_center)
					parent_vec2 = np.asarray(temp_center2) - np.asarray(parent_center)
					if np.linalg.norm(parent_vec1) > np.linalg.norm(parent_vec2):
						temp_partition_arr.append(temp_pt_idx_arr2)
						temp_partition_arr.append(temp_pt_idx_arr1)
					else:
						temp_partition_arr.append(temp_pt_idx_arr1)
						temp_partition_arr.append(temp_pt_idx_arr2)
			else:
				temp_partition_arr.append(sub_seg_geo_idx_arr)

		partition_arr_idx = temp_partition_arr

		flag = 0
		for sub_seg_geo_idx_arr in partition_arr_idx:
			if len(sub_seg_geo_idx_arr)>1:
				flag = 1
				break

		iteration = iteration + 1
		# print("iteration: ", iteration)

	idx_arr = []
	ball_tree_traveral_color = []
	for s in range(0, len(partition_arr_idx)):
		re2 = partition_arr_idx[s]
		for idx in re2:
			ball_tree_traveral_color.append(seg_color_arr[idx])
			idx_arr.append(idx)
	# print("ball_tree_traveral_color_len: ", len(ball_tree_traveral_color))
	return [ball_tree_traveral_color, idx_arr]

### Attribute image generation using hybrid space-filling pattern (1D to 2D)
def hori_snake_curve_single(blk_color_arr, hori_snake_b):
	blk_img = np.ones((hori_snake_b, hori_snake_b), np.uint8)*0
	for t in range(len(blk_color_arr)):
		y_pos = int(t/hori_snake_b)
		x_pos = ((-1)**(y_pos%2))*(t%hori_snake_b) + (hori_snake_b-1)*(y_pos%2)
		blk_img[y_pos][x_pos] = blk_color_arr[t]
	return blk_img

def hori_snake_curve(blk_color_arr, hori_snake_b):
	blk_img = np.ones((hori_snake_b, hori_snake_b, 3), np.uint8)*0
	for t in range(len(blk_color_arr)):
		y_pos = int(t/hori_snake_b)
		x_pos = ((-1)**(y_pos%2))*(t%hori_snake_b) + (hori_snake_b-1)*(y_pos%2)
		blk_img[y_pos][x_pos] = blk_color_arr[t]
	return blk_img

def rot(n, x, y, rx, ry):
	if (ry == 0):
		if (rx == 1):
			x = n-1 - x
			y = n-1 - y
		t = x
		x = y
		y = t
	return [x,y]

def get_hilbert_pos(hilbert_b, hilbert_idx):
	rx = 0
	ry = 0
	t = hilbert_idx
	x = 0
	y = 0
	s = 1
	while s<hilbert_b:
		rx = 1 & int(t/2)
		ry = 1 & (int(t) ^ rx)
		[x,y] = rot(s, x, y, rx, ry)
		x = x + s * rx
		y = y + s * ry
		t = t/4
		s = s*2
	return [x, y]

def hybrid_space_filling3(color_arr, img_w, img_h):
	mb_size = 16
	hori_snake_b = 4
	hilbert_b = 4 # mb_size = 2^n1, 	hori_snake_b = 2^n2,		hori_snake_b < mb_size
	tot_pt_num = len(color_arr)
	mask_rgb = [255 for t in range(tot_pt_num)]
	last_rgb = color_arr[-1]
	color_arr = color_arr + [last_rgb for t in range(img_w*img_h-tot_pt_num)]
	mask_rgb = mask_rgb + [0 for t in range(img_w*img_h-tot_pt_num)]
	sub_blk_num = int(np.ceil(img_w*img_h/(hori_snake_b*hori_snake_b)))

	attr_img = np.ones((img_h, img_w, 3), np.uint8)*0
	attr_mask = np.ones((img_h, img_w), np.uint8)*0

	w_blk_num = int(img_w/hori_snake_b)
	h_blk_num = int(img_h/hori_snake_b)

	grid_xy = np.ones((h_blk_num, w_blk_num), np.uint8)*0

	for blk_idx in range(sub_blk_num):
		blk_color_arr = color_arr[blk_idx*hori_snake_b*hori_snake_b:(blk_idx+1)*hori_snake_b*hori_snake_b]
		blk_img = hori_snake_curve(blk_color_arr, hori_snake_b)
		mask_blk_color_arr = mask_rgb[blk_idx*hori_snake_b*hori_snake_b:(blk_idx+1)*hori_snake_b*hori_snake_b]
		mask_blk_img = hori_snake_curve_single(mask_blk_color_arr, hori_snake_b)

		temp_blk_idx = blk_idx
		flag = 1
		while flag:
			[hil_x, hil_y] = get_hilbert_pos(hilbert_b, temp_blk_idx%(hilbert_b*hilbert_b))
			hil_x = hil_x + int(temp_blk_idx/(hilbert_b*hilbert_b))*hilbert_b
			if hil_x < w_blk_num and hil_y < h_blk_num:
				if grid_xy[hil_y][hil_x] == 0:
					attr_img[hil_y*hori_snake_b:(hil_y+1)*hori_snake_b, hil_x*hori_snake_b:(hil_x+1)*hori_snake_b] = blk_img
					attr_mask[hil_y*hori_snake_b:(hil_y+1)*hori_snake_b, hil_x*hori_snake_b:(hil_x+1)*hori_snake_b] = mask_blk_img
					# attr_img[hil_y*hori_snake_b:(hil_y+1)*hori_snake_b, hil_x*hori_snake_b + int(temp_blk_idx/(hilbert_b*hilbert_b))*mb_size:(hil_x+1)*hori_snake_b + int(temp_blk_idx/(hilbert_b*hilbert_b))*mb_size] = blk_img
					# attr_mask[hil_y*hori_snake_b:(hil_y+1)*hori_snake_b, hil_x*hori_snake_b + int(temp_blk_idx/(hilbert_b*hilbert_b))*mb_size:(hil_x+1)*hori_snake_b + int(temp_blk_idx/(hilbert_b*hilbert_b))*mb_size] = mask_blk_img
					grid_xy[hil_y][hil_x] = 1
					flag = 0
				else:
					temp_blk_idx = temp_blk_idx + 1
			else:
				temp_blk_idx = temp_blk_idx + 1

	return [attr_img, attr_mask]

def hybrid_space_filling2(color_arr, mb_size, hori_snake_b):
	hilbert_b = int(mb_size/hori_snake_b) # mb_size = 2^n1, 	hori_snake_b = 2^n2,		hori_snake_b < mb_size
	tot_pt_num = len(color_arr)
	blk_num = int(np.ceil(tot_pt_num/(mb_size*mb_size)))
	sub_blk_num = int(np.ceil(tot_pt_num/(hori_snake_b*hori_snake_b)))
	attr_img = np.ones((mb_size, blk_num*mb_size, 3), np.uint8)*0
	
	attr_mask = np.ones((mb_size, blk_num*mb_size), np.uint8)*0
	mask_rgb = [255 for t in range(tot_pt_num)]
	last_rgb = color_arr[-1]
	# color_arr
	# for t in range(mb_size*blk_num*mb_size-tot_pt_num):
	# 	color_arr.append(last_rgb)
	
	color_arr = color_arr + [last_rgb for t in range(mb_size*blk_num*mb_size-tot_pt_num)]
	mask_rgb = mask_rgb + [0 for t in range(mb_size*blk_num*mb_size-tot_pt_num)]
	# print(tot_pt_num, mb_size*blk_num*mb_size, len(color_arr), last_rgb)
	sub_blk_num = int(np.ceil(len(color_arr)/(hori_snake_b*hori_snake_b)))
	for blk_idx in range(sub_blk_num):
		blk_color_arr = color_arr[blk_idx*hori_snake_b*hori_snake_b:(blk_idx+1)*hori_snake_b*hori_snake_b]
		blk_img = hori_snake_curve(blk_color_arr, hori_snake_b)
		[hil_x, hil_y] = get_hilbert_pos(hilbert_b, blk_idx%(hilbert_b*hilbert_b))
		attr_img[hil_y*hori_snake_b:(hil_y+1)*hori_snake_b, hil_x*hori_snake_b + int(blk_idx/(hilbert_b*hilbert_b))*mb_size:(hil_x+1)*hori_snake_b + int(blk_idx/(hilbert_b*hilbert_b))*mb_size] = blk_img

		mask_blk_color_arr = mask_rgb[blk_idx*hori_snake_b*hori_snake_b:(blk_idx+1)*hori_snake_b*hori_snake_b]
		mask_blk_img = hori_snake_curve_single(mask_blk_color_arr, hori_snake_b)
		[hil_x, hil_y] = get_hilbert_pos(hilbert_b, blk_idx%(hilbert_b*hilbert_b))
		attr_mask[hil_y*hori_snake_b:(hil_y+1)*hori_snake_b, hil_x*hori_snake_b + int(blk_idx/(hilbert_b*hilbert_b))*mb_size:(hil_x+1)*hori_snake_b + int(blk_idx/(hilbert_b*hilbert_b))*mb_size] = mask_blk_img
	return [attr_img, attr_mask]

def test_rf3(patch_geo_arr, patch_geo_arr_d2, patch_rgb_arr, file_id):
	patch_pt_num = len(patch_geo_arr_d2)

	chain_bpp_arr_all = []
	chain_psnr_arr_all = []
	chain_legend_arr_all = []
	[peer_chain_bpp_arr, peer_chain_psnr_arr, peer_chain_legend_arr] = peer_method(frame_id)
	chain_bpp_arr_all = chain_bpp_arr_all + peer_chain_bpp_arr
	chain_psnr_arr_all = chain_psnr_arr_all + peer_chain_psnr_arr
	chain_legend_arr_all = chain_legend_arr_all + peer_chain_legend_arr
	# print("bsp...")
	# [bt_color, bt_idx_arr] = ball_tree_partition(patch_geo_arr, patch_rgb_arr)
	# [sfc_based_attr_img, sfc_based_attr_mask_img] = hybrid_space_filling2([patch_rgb_arr[idx][::-1] for idx in bt_idx_arr], mb_size = 16, hori_snake_b = 4)
	# [ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(sfc_based_attr_img, sfc_based_attr_mask_img, patch_pt_num)
	# chain_bpp_arr_all.append(ball_tree_bpp_arr)
	# chain_psnr_arr_all.append(ball_tree_psnr_arr)
	# chain_legend_arr_all.append("bsp_" + str(patch_pt_num))

	[grid_img, grid_mask, grid_img_rect, grid_mask_rect, grid_blk_img8, grid_blk_mask8, grid_blk_img16, grid_blk_mask16, grid_blk_img32, grid_blk_mask32] = test_rf_new(patch_geo_arr_d2, patch_rgb_arr)

	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(grid_img, grid_mask, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("orig_" + str(patch_pt_num))
	cv2.imwrite(file_id + "_orig.png", grid_img)
	cv2.imwrite(file_id + "_orig_mask.png", grid_mask)

	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(grid_img_rect, grid_mask_rect, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("rect_" + str(patch_pt_num))
	cv2.imwrite(file_id + "_rect.png", grid_img_rect)
	cv2.imwrite(file_id + "_rect_mask.png", grid_mask_rect)

	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(grid_blk_img8, grid_blk_mask8, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("rect_blk8_" + str(patch_pt_num))
	cv2.imwrite(file_id + "_rect_blk8.png", grid_blk_img8)
	cv2.imwrite(file_id + "_rect_blk_mask8.png", grid_blk_mask8)


	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(grid_blk_img16, grid_blk_mask16, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("rect_blk16_" + str(patch_pt_num))
	cv2.imwrite(file_id + "_rect_blk16.png", grid_blk_img16)
	cv2.imwrite(file_id + "_rect_blk_mask16.png", grid_blk_mask16)


	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(grid_blk_img32, grid_blk_mask32, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("rect_blk32_" + str(patch_pt_num))
	cv2.imwrite(file_id + "_rect_blk32.png", grid_blk_img32)
	cv2.imwrite(file_id + "_rect_blk_mask32.png", grid_blk_mask32)

	plt.figure()
	for ttt in range(0,len(chain_bpp_arr_all)):
		plt.plot(chain_bpp_arr_all[ttt], chain_psnr_arr_all[ttt], color = plt_color_arr[ttt], marker = plt_marker_arr[ttt], markerfacecolor="None", linewidth=1)
	plt.xlabel('bpp', fontsize=12)
	plt.ylabel('PSNR', fontsize=12)
	plt.title("all", fontsize=12)
	plt.legend(chain_legend_arr_all, loc="lower right")
	plt.savefig(file_id + '_new.png')
	plt.close()

def test_rf2(patch_geo_arr, patch_geo_arr_d2, patch_rgb_arr, file_id):
	patch_pt_num = len(patch_geo_arr_d2)

	chain_bpp_arr_all = []
	chain_psnr_arr_all = []
	chain_legend_arr_all = []
	[peer_chain_bpp_arr, peer_chain_psnr_arr, peer_chain_legend_arr] = peer_method(frame_id)
	chain_bpp_arr_all = chain_bpp_arr_all + peer_chain_bpp_arr
	chain_psnr_arr_all = chain_psnr_arr_all + peer_chain_psnr_arr
	chain_legend_arr_all = chain_legend_arr_all + peer_chain_legend_arr

	# print("bsp_")

	[bt_color, bt_idx_arr] = ball_tree_partition(patch_geo_arr, patch_rgb_arr)
	[sfc_based_attr_img, sfc_based_attr_mask_img] = hybrid_space_filling2([patch_rgb_arr[idx][::-1] for idx in bt_idx_arr], mb_size = 16, hori_snake_b = 4)
	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(sfc_based_attr_img, sfc_based_attr_mask_img, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("bsp_" + str(patch_pt_num))



	# for angle in range(0, 91, 15):
	for angle in [0]:
		rot_patch_geo_arr_d2 = rot_pt(patch_geo_arr_d2, angle)
		print("test_rf_mask")
		# [ip_grid_img, grid_img, grid_mask, grid_img_rect, grid_mask_rect, grid_blk_img, grid_blk_mask, ip_grid_blk_img] = test_rf_mask(rot_patch_geo_arr_d2, patch_rgb_arr)
		[ip_grid_img, grid_img, grid_mask, grid_img_rect, grid_mask_rect, ip_grid_img_rect, grid_blk_img, grid_blk_mask, ip_grid_blk_img] = test_rf_with_ip(rot_patch_geo_arr_d2, patch_rgb_arr)
		# [grid_img, grid_mask, grid_img_rect, grid_mask_rect, grid_blk_img, grid_blk_mask] = test_rf(rot_patch_geo_arr_d2, patch_rgb_arr)
		print("img_compression_with_mask")


		[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(grid_img, grid_mask, patch_pt_num)
		chain_bpp_arr_all.append(ball_tree_bpp_arr)
		chain_psnr_arr_all.append(ball_tree_psnr_arr)
		chain_legend_arr_all.append("orig_" + str(angle))
		cv2.imwrite(file_id + "_orig.png", grid_img)
		cv2.imwrite(file_id + "_orig_mask.png", grid_mask)

		[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(ip_grid_img, grid_mask, patch_pt_num)
		chain_bpp_arr_all.append(ball_tree_bpp_arr)
		chain_psnr_arr_all.append(ball_tree_psnr_arr)
		chain_legend_arr_all.append("orig_ip" + str(angle))
		cv2.imwrite(file_id + "_orig_ip.png", ip_grid_img)

		[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(grid_img_rect, grid_mask_rect, patch_pt_num)
		chain_bpp_arr_all.append(ball_tree_bpp_arr)
		chain_psnr_arr_all.append(ball_tree_psnr_arr)
		chain_legend_arr_all.append("rect_" + str(angle))
		cv2.imwrite(file_id + "_rect.png", grid_img_rect)
		cv2.imwrite(file_id + "_rect_mask.png", grid_mask_rect)

		[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(ip_grid_img_rect, grid_mask_rect, patch_pt_num)
		chain_bpp_arr_all.append(ball_tree_bpp_arr)
		chain_psnr_arr_all.append(ball_tree_psnr_arr)
		chain_legend_arr_all.append("rect_ip_" + str(angle))
		cv2.imwrite(file_id + "_rect_ip.png", ip_grid_img)

		[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(grid_blk_img, grid_blk_mask, patch_pt_num)
		chain_bpp_arr_all.append(ball_tree_bpp_arr)
		chain_psnr_arr_all.append(ball_tree_psnr_arr)
		chain_legend_arr_all.append("rect_blk_" + str(angle))
		cv2.imwrite(file_id + "_rect_blk.png", grid_blk_img)
		cv2.imwrite(file_id + "_rect_blk_mask.png", grid_blk_mask)

		[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(ip_grid_blk_img, grid_blk_mask, patch_pt_num)
		chain_bpp_arr_all.append(ball_tree_bpp_arr)
		chain_psnr_arr_all.append(ball_tree_psnr_arr)
		chain_legend_arr_all.append("rect_blk_ip_" + str(angle))
		cv2.imwrite(file_id + "_rect_blk_ip.png", ip_grid_blk_img)

	plt.figure()
	for ttt in range(0,len(chain_bpp_arr_all)):
		plt.plot(chain_bpp_arr_all[ttt], chain_psnr_arr_all[ttt], color = plt_color_arr[ttt], marker = plt_marker_arr[ttt], markerfacecolor="None", linewidth=1)
	plt.xlabel('bpp', fontsize=12)
	plt.ylabel('PSNR', fontsize=12)
	plt.title("all", fontsize=12)
	plt.legend(chain_legend_arr_all, loc="lower right")
	plt.savefig(file_id + '_new.png')
	plt.close()

def patch_generation2(geo_arr, rgb_arr, floder, frame_id, frame_id_head, iso_sv_pt_num):
	svoff_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(128) + ".off"
	[sv_off_geo_arr, sv_off_rgb_arr, sv_off_pt_num] = read_off(svoff_path, geo_arr, rgb_arr)

	[hks_pt_arr, hks_feature_arr] = get_hks_feature(geo_arr, rgb_arr, floder, frame_id)

	tot_num = len(hks_pt_arr)
	X = np.array(hks_pt_arr)
	tree = BallTree(X, leaf_size = 1)

	new_hks_feature_arr = []
	for pt in sv_off_geo_arr:
		dist, ind = tree.query([pt], k=1)
		idx = ind[0][0]
		new_hks_feature_arr.append(hks_feature_arr[idx])

	hks_pt_arr = sv_off_geo_arr
	hks_feature_arr = new_hks_feature_arr

	dbscan_thresh = get_neighbor_dis(hks_pt_arr)*2
	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]

	for num_c in range(40, 15, -2):
		# try:
			[all_cluster_dic, new_label_arr] = hks_quan(hks_pt_arr, hks_feature_arr, dbscan_thresh, num_cluster=num_c, vis = 0)
			

			[uni_pt_idx, bi_pt_idx, mul_pt_idx, boundary_label_dic] = detect_boundary_points(hks_pt_arr, new_label_arr)
			print(num_c, len(mul_pt_idx))
			if len(mul_pt_idx) == 0 or num_c == 16:
				vis_geo = []
				vis_rgb = []

				final_all_cluster = []
				for label in all_cluster_dic:
					final_all_cluster = final_all_cluster + [all_cluster_dic[label]]


				# vis_geo = []
				# vis_rgb = []
				new_label_arr = [0 for i in range(len(hks_pt_arr))]

				for t in range(len(final_all_cluster)):
					idx_arr = final_all_cluster[t]
					for idx in idx_arr:
						new_label_arr[idx] = t
				[uni_pt_idx, bi_pt_idx, mul_pt_idx, boundary_label_dic] = detect_boundary_points(hks_pt_arr, new_label_arr)

				
				end_node = []
				bi_node = []
				multi_node = []
				skeleton_adj_dic = dict()
				for t in range(len(final_all_cluster)):
					idx_arr = final_all_cluster[t]
					patch_type = []
					for idx in idx_arr:
						vis_geo.append(hks_pt_arr[idx])
						if idx in bi_pt_idx or idx in mul_pt_idx:
							vis_rgb.append(rgb)
							patch_type = patch_type + boundary_label_dic[idx]
						else:
							vis_rgb.append([rgb[0]/4, rgb[1]/4, rgb[2]/4])

					# print(t, "patch_type:", list(set(patch_type)), len(list(set(patch_type))))
					skeleton_adj_dic[t] = []
					for val in list(set(patch_type)):
						if val != t:
							skeleton_adj_dic[t].append(val)

					if len(list(set(patch_type))) == 2:
						end_node.append(t)
					elif len(list(set(patch_type))) == 3:
						bi_node.append(t)
					else:
						multi_node.append(t)

				[chain_dic, end_node2] = skeleton_adj_line2(skeleton_adj_dic)
				print(chain_dic, end_node2)

				vis_geo = []
				vis_rgb = []

				for cluster_idx in end_node:
					rgb = [0, 255, 0]
					for idx in all_cluster_dic[cluster_idx]:
						vis_geo.append(hks_pt_arr[idx])
						vis_rgb.append(rgb)

				for cluster_idx in bi_node:
					rgb = [0, 0, 255]
					for idx in all_cluster_dic[cluster_idx]:
						vis_geo.append(hks_pt_arr[idx])
						vis_rgb.append(rgb)

				for cluster_idx in multi_node:
					rgb = [255, 0, 0]
					for idx in all_cluster_dic[cluster_idx]:
						vis_geo.append(hks_pt_arr[idx])
						vis_rgb.append(rgb)

				print("end_node: ", end_node)
				print("bi_node: ", bi_node)
				print("multi_node: ", multi_node)
				# pc_vis(vis_geo, vis_rgb)


				print("chain_dic: ", chain_dic)
				vis_geo = []
				vis_rgb = []
				for chain in chain_dic:
					rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
					print("chain: ", chain)
					print(chain, chain_dic[chain], end_node)
					for node in chain_dic[chain]:
						for idx in all_cluster_dic[node]:
							vis_geo.append(hks_pt_arr[idx])
							vis_rgb.append(rgb)
				
				pc_vis(vis_geo, vis_rgb)


				# 	for cluster_idx in chain_dic[chain]:
				# 		idx_arr = final_all_cluster[cluster_idx]
				# 		if cluster_idx in end_node:
				# 			rgb = [0, 255, 0]
				# 			for idx in idx_arr:
				# 				vis_geo.append(hks_pt_arr[idx])
				# 				vis_rgb.append(rgb)
				# 		else:
				# 			for idx in idx_arr:
				# 				vis_geo.append(hks_pt_arr[idx])
				# 				vis_rgb.append(rgb1)

				# pcd = o3d.geometry.PointCloud()
				# pcd.points = o3d.utility.Vector3dVector(vis_geo)
				# pcd.colors = o3d.utility.Vector3dVector(np.asarray(vis_rgb)/255.0)
				# o3d.visualization.draw_geometries([pcd])

				break

def merge_multinode(skeleton_adj_dic, end_node, bi_node, multi_node, all_cluster, sv_off_geo_arr):
	new_multi_node = []
	for node in multi_node:
		print(node, skeleton_adj_dic[node])
		valid_adj_node = []
		for adj_node in skeleton_adj_dic[node]:
			if adj_node in end_node + bi_node:
				valid_adj_node.append(adj_node)
		print(valid_adj_node)
		# if len(valid_adj_node) == 1:
		# 	all_cluster[valid_adj_node[0]] = all_cluster[valid_adj_node[0]] + all_cluster[node]
		if len(valid_adj_node) >= 1:
			new_assign_dic = region_growing2(sv_off_geo_arr, all_cluster, node, valid_adj_node)
			for new_node in new_assign_dic:
				all_cluster[new_node] = new_assign_dic[new_node]
			all_cluster[node] = []

	return all_cluster
	# dbscan_thresh = get_neighbor_dis(sv_off_geo_arr)*2
	# all_pt = []
	# for node in multi_node:
	# 	all_pt = all_pt + [sv_off_geo_arr[idx] for idx in all_cluster[node]]

	# dbscan_dic = dbscan_clustering(all_pt, dbscan_thresh)


	# X = np.array(sv_off_geo_arr)
	# tree = BallTree(X, leaf_size = 1)
	# new_multi_node_arr = []
	# for label in dbscan_dic:
	# 	temp_idx = []
	# 	for idx in dbscan_dic[label]:
	# 		dist, ind = tree.query([all_pt[idx]], k=1)
	# 		idx = ind[0][0]
	# 		temp_idx.append(idx)
	# 	new_multi_node_arr.append(temp_idx)
	# return new_multi_node_arr
		
def region_growing2(sv_off_geo_arr, all_cluster, node, adj_node_arr):
	off_seg_dic = dict()
	all_pt = []
	all_pt_idx = []
	for nd in adj_node_arr:
		off_seg_dic[nd] = [sv_off_geo_arr[idx] for idx in all_cluster[nd]]
		all_pt = all_pt + [sv_off_geo_arr[idx] for idx in all_cluster[nd]]
		all_pt_idx = all_pt_idx + all_cluster[nd]
	all_pt = all_pt + [sv_off_geo_arr[idx] for idx in all_cluster[node]]
	all_pt_idx = all_pt_idx + all_cluster[node]

	assign_dic = assign_ply_to_seg(off_seg_dic, all_pt, vis=0)
	new_assign_dic = dict()
	vis_geo = []
	vis_rgb = []
	for node in assign_dic:
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		for idx in assign_dic[node]:
			vis_geo.append(all_pt[idx])
			vis_rgb.append(rgb)

		new_assign_dic[node] = [all_pt_idx[idx] for idx in assign_dic[node]]
	return new_assign_dic

def region_growing(skeleton_adj_dic, chain_seg_dic, chain_dic, sv_off_geo_arr, all_cluster, end_node, bi_node, multi_node):
	vis_geo = []
	vis_rgb = []

	for cluster_idx in end_node:
		rgb = [randrange(100, 255), randrange(0, 1), randrange(0, 1)]
		for idx in all_cluster[cluster_idx]:
			vis_geo.append(sv_off_geo_arr[idx])
			vis_rgb.append(rgb)

	for cluster_idx in bi_node:
		rgb = [randrange(0, 1), randrange(100, 255), randrange(0, 1)]
		for idx in all_cluster[cluster_idx]:
			vis_geo.append(sv_off_geo_arr[idx])
			vis_rgb.append(rgb)

	for cluster_idx in multi_node:
		rgb = [randrange(0, 1), randrange(0, 1), randrange(100, 255)]
		for idx in all_cluster[cluster_idx]:
			vis_geo.append(sv_off_geo_arr[idx])
			vis_rgb.append(rgb)
	pc_vis(vis_geo, vis_rgb)

	print("end_node: ", end_node)
	print("bi_node: ", bi_node)
	print("multi_node: ", multi_node)
	# print("chain_seg_dic: ", chain_seg_dic)
	print("chain_dic: ", chain_dic)

	new_multi_node = []
	for node in multi_node:
		print(node, skeleton_adj_dic[node])
		valid_adj_node = []
		for adj_node in skeleton_adj_dic[node]:
			if adj_node in end_node + bi_node:
				valid_adj_node.append(adj_node)
		print(valid_adj_node)
		if len(valid_adj_node) == 1:
			all_cluster[valid_adj_node[0]] = all_cluster[valid_adj_node[0]] + all_cluster[node]
		elif len(valid_adj_node) > 1:
			region_growing2(sv_off_geo_arr, all_cluster, node, valid_adj_node)


			new_multi_node.append(node)
	multi_node = new_multi_node

	vis_geo = []
	vis_rgb = []

	for cluster_idx in end_node:
		rgb = [randrange(100, 255), randrange(0, 1), randrange(0, 1)]
		for idx in all_cluster[cluster_idx]:
			vis_geo.append(sv_off_geo_arr[idx])
			vis_rgb.append(rgb)

	for cluster_idx in bi_node:
		rgb = [randrange(0, 1), randrange(100, 255), randrange(0, 1)]
		for idx in all_cluster[cluster_idx]:
			vis_geo.append(sv_off_geo_arr[idx])
			vis_rgb.append(rgb)

	for cluster_idx in multi_node:
		rgb = [randrange(0, 1), randrange(0, 1), randrange(100, 255)]
		for idx in all_cluster[cluster_idx]:
			vis_geo.append(sv_off_geo_arr[idx])
			vis_rgb.append(rgb)
	pc_vis(vis_geo, vis_rgb)
	
	
	# pc_vis(vis_geo, vis_rgb)

def patch_unfold(chain_dic, all_cluster, sv_off_geo_arr, geo_arr, rgb_arr, floder, frame_id, end_node, bi_node, multi_node):
	seg_path = floder  + "seg_assign//" + frame_id + "_seg.txt"
	[all_final_seg_arr, all_final_seg_label] = [[], []]
	if not os.path.exists(seg_path):
		print("end_node: ", end_node)
		print("bi_node: ", bi_node)
		print("multi_node: ", multi_node)
		seg_arr_dic = dict()
		seg_idx_arr_dic = dict()
		label = 0
		new_label_arr = [0 for t in range(len(sv_off_geo_arr))]
		valid_chain = []
		for chain in chain_dic:
			print("chain:", chain, chain_dic[chain])
			patch_geo = []
			patch_idx = []
			for seg in chain_dic[chain]:
				patch_idx = patch_idx + all_cluster[seg]
				for idx in all_cluster[seg]:
					patch_geo.append(sv_off_geo_arr[idx])
					new_label_arr[idx] = label

			print(len(patch_geo))
			if len(patch_geo):
				seg_arr_dic[label] = patch_geo
				seg_idx_arr_dic[label] = patch_idx
				label = label + 1
				valid_chain.append(chain)


		[uni_pt_idx, bi_pt_idx, mul_pt_idx, boundary_label_dic] = detect_boundary_points(sv_off_geo_arr, new_label_arr)

		valid_chain_dic = dict()

		all_final_seg_arr = []
		all_final_seg_label = []
		for chain in valid_chain:
			vis_geo = []
			vis_rgb = []
			boundary_pt_idx = []
			patch_idx_arr = []
			patch_geo = []
			cnt = 0
			for seg in chain_dic[chain]:
				rgb = [randrange(0, 1), randrange(0, 255), randrange(0, 255)]
				for idx in all_cluster[seg]:
					vis_geo.append(sv_off_geo_arr[idx])
					vis_rgb.append(rgb)
					patch_geo.append(sv_off_geo_arr[idx])
					patch_idx_arr.append(idx)

					if idx in bi_pt_idx + mul_pt_idx:
						boundary_pt_idx.append(cnt)
					cnt = cnt + 1

			seg_thresh = get_neighbor_dis(patch_geo)*2
			err_arr = compute_distortion(patch_geo, seg_thresh, vis = 0)
			dis_pt_idx = detect_distortion_pts(boundary_pt_idx, patch_geo, patch_geo, err_arr, seg_thresh, N_r = 6, vis = 0)
			spanning_tree_idx_arr = mini_spanning_tree(patch_geo, patch_geo, dis_pt_idx)

			leaf_node_arr = []
			for leaf in spanning_tree_idx_arr:
				leaf_node_arr = leaf_node_arr + leaf
			leaf_node_dic = dict((i, leaf_node_arr.count(i)) for i in leaf_node_arr)
			multi_leaf_node = []
			for node in leaf_node_dic:
				if leaf_node_dic[node] > 2 and not node in boundary_pt_idx:
					multi_leaf_node.append(node)

			multi_leaf_node_idx = [patch_idx_arr[idx] for idx in multi_leaf_node]

			# pc_vis([sv_off_geo_arr[idx] for idx in multi_leaf_node_idx] + vis_geo, [[255, 0, 0] for idx in multi_leaf_node_idx] + vis_rgb) 

			# for seg in chain_dic[chain]:



			multi_seg_arr = []
			for idx in multi_leaf_node_idx:
				for seg in chain_dic[chain]:
					if idx in all_cluster[seg]:
						multi_seg_arr.append(seg)
						break
			print("chain_dic[chain]: ", chain_dic[chain])
			print("multi_seg_arr:", multi_seg_arr)

			chain_seg_arr = chain_dic[chain]
			
			seg_label = [0 for t in range(len(chain_seg_arr))]
			
			for t in range(len(chain_seg_arr)):
				seg = chain_seg_arr[t]
				if seg in multi_seg_arr:
					seg_label[t] = 2 
				if seg in end_node:
					seg_label[t] = 1
			print(seg_label)
			final_seg_arr = []
			final_seg_label = []
			temp = []
			for t in range(len(chain_seg_arr)):

				if seg_label[t] == 1:
					if len(temp):
						final_seg_arr.append(temp)
						final_seg_label.append(0)
						temp = []

					final_seg_arr.append([chain_seg_arr[t]])
					final_seg_label.append(1)
				elif seg_label[t] == 2:
					if len(temp):
						final_seg_arr.append(temp)
						final_seg_label.append(0)
						temp = []
					final_seg_arr.append([chain_seg_arr[t]])
					final_seg_label.append(2)
				else:
					temp.append(chain_seg_arr[t])
			if len(temp):
				final_seg_arr.append(temp)
				final_seg_label.append(0)
				temp = []
			print(final_seg_arr, final_seg_label)

			all_final_seg_arr = all_final_seg_arr + final_seg_arr
			all_final_seg_label = all_final_seg_label + final_seg_label
			# break

		print(all_final_seg_arr)
		print(all_final_seg_label)



		f = open(seg_path, "w")
		for t in range(len(all_final_seg_arr)):
			f.write(str(all_final_seg_label[t]) + "\t")
			for seg in all_final_seg_arr[t]:
				f.write(str(seg) + " ")
			f.write("\n")
		f.close()
	else:
		with open(seg_path) as ins:
			for line in ins:
				rec = line.replace(" \n", "").split("\t")
				seg = int(rec[0])
				all_final_seg_label.append(seg)
				idx_arr = [int(val) for val in rec[1].split(" ")]
				all_final_seg_arr.append(idx_arr)

	return [all_final_seg_arr, all_final_seg_label]

	# 



	# [hks_pt_arr, hks_feature_arr] = get_hks_feature(geo_arr, rgb_arr, floder, frame_id)




	# assign_dic = assign_ply_to_seg(seg_arr_dic, hks_pt_arr)
	# vis_geo = []
	# vis_rgb = []
	# for label in assign_dic:
	# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
	# 	for idx in assign_dic[label]:
	# 		vis_geo.append(hks_pt_arr[idx])
	# 		vis_rgb.append(rgb)
	# pc_vis(vis_geo, vis_rgb) 

	# for label in assign_dic:
	# 	patch_geo = [hks_pt_arr[idx] for idx in assign_dic[label]]
	# 	vis_geo = []
	# 	vis_rgb = []
	# 	boundary_pt_idx = []
	# 	cnt = 0
	# 	for idx in seg_idx_arr_dic[label]:
	# 		vis_geo.append(sv_off_geo_arr[idx])
	# 		if not idx in bi_pt_idx + mul_pt_idx:
	# 			vis_rgb.append([121, 220, 120])
	# 		else:
	# 			vis_rgb.append([255, 0, 255])
	# 			boundary_pt_idx.append(cnt)

	# 		cnt = cnt + 1
		# pc_vis(vis_geo, vis_rgb) 



		# sparse_sv_off_geo_arr = seg_arr_dic[label]
		# seg_thresh = get_neighbor_dis(sparse_sv_off_geo_arr)*2
		# err_arr = compute_distortion(sparse_sv_off_geo_arr, seg_thresh, vis = 0)
		# dis_pt_idx = detect_distortion_pts(boundary_pt_idx, sparse_sv_off_geo_arr, sparse_sv_off_geo_arr, err_arr, seg_thresh, N_r = 6, vis = 0)

		# spanning_tree_idx_arr = mini_spanning_tree(sparse_sv_off_geo_arr, sparse_sv_off_geo_arr, dis_pt_idx)


		# line_pt = []
		# line_color = []
		# for leaf in spanning_tree_idx_arr:
		# 	st_node = leaf[0]
		# 	end_node = leaf[-1]
		# 	if not (st_node in boundary_pt_idx and end_node in boundary_pt_idx):
		# 		temp_pt = []
		# 		for t in range(0, len(leaf)-1):
		# 			temp_pt = temp_pt + draw_line(sparse_sv_off_geo_arr[leaf[t]], sparse_sv_off_geo_arr[leaf[t+1]])
		# 		rgb = [randrange(254, 255), randrange(0, 255), randrange(0, 255)]
		# 		temp_color = [rgb for pt in temp_pt]
		# 		line_pt = line_pt + temp_pt
		# 		line_color= line_color + temp_color

		# pc_vis(line_pt + sparse_sv_off_geo_arr, line_color + [[222, 220, 221] for pt in sparse_sv_off_geo_arr])





		# patch_seg(patch_geo, hks_pt_arr, hks_feature_arr, vis = 1)

		# spanning_tree_idx_arr = mini_spanning_tree(sparse_sv_off_geo_arr, sparse_sv_off_geo_arr, dis_pt_idx)
		# leaf_node_arr = []
		# for leaf in spanning_tree_idx_arr:
		# 	leaf_node_arr = leaf_node_arr + leaf
		# leaf_node_dic = dict((i, leaf_node_arr.count(i)) for i in leaf_node_arr)
		# multi_leaf_node = []
		# for node in leaf_node_dic:
		# 	if leaf_node_dic[node] > 2:
		# 		multi_leaf_node.append(node)
		# rest_pt_idx = list(set([t for t in range(len(sparse_sv_off_geo_arr))]) -  set(multi_leaf_node))
		# pc_vis([sparse_sv_off_geo_arr[idx] for idx in rest_pt_idx] + [sparse_sv_off_geo_arr[idx] for idx in multi_leaf_node], [[200, 200, 200] for pt in rest_pt_idx] + [[200, 0, 0] for idx in multi_leaf_node])

def unfold_patch(bi_node, end_node, all_cluster, sv_off_geo_arr, geo_arr, rgb_arr):
	vis_geo = []
	vis_rgb = []
	off_seg_dic = dict()

	for t in range(len(all_cluster)):
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		patch_geo = []
		for idx in all_cluster[t]:
			vis_geo.append(sv_off_geo_arr[idx])
			vis_rgb.append(rgb)
			patch_geo.append(sv_off_geo_arr[idx])
		if len(patch_geo):
			off_seg_dic[t] = patch_geo

	assign_dic = dict()
	assign_path = floder  + "seg_assign//" + frame_id + "_assign.txt"
	if not os.path.exists(assign_path):
		assign_dic = assign_ply_to_seg(off_seg_dic, geo_arr)
		f = open(assign_path, "w")
		for seg in assign_dic:
			f.write(str(seg) + "\t")
			for idx in assign_dic[seg]:
				f.write(str(idx) + " ")
			f.write("\n")
		f.close()
	else:
		with open(assign_path) as ins:
			for line in ins:
				rec = line.replace(" \n", "").split("\t")
				seg = int(rec[0])
				idx_arr = [int(val) for val in rec[1].split(" ")]
				assign_dic[seg] = idx_arr
	return assign_dic

	# for soi in bi_node:
	# 	patch_geo_arr = []
	# 	patch_rgb_arr = []
	# 	for idx in assign_dic[soi]:
	# 		patch_geo_arr.append(geo_arr[idx])
	# 		patch_rgb_arr.append(rgb_arr[idx])

	# 	patch_off_arr = []
	# 	for idx in all_cluster[soi]:
	# 		patch_off_arr.append(sv_off_geo_arr[idx])

	# 	patch_off_color_arr = get_off_color(patch_geo_arr, patch_rgb_arr, patch_off_arr)

	# 	# patch_geo_arr = patch_off_arr

	# 	[d2_geo, reconstruction_err, embedding] = isomap_based_dimension_reduction(patch_off_arr, patch_off_arr, len(patch_off_arr)> 8)

		# pc_vis(d2_geo + [[pt[0]+ pc_width, pt[1], pt[2]] for pt in patch_geo_arr], patch_off_color_arr + patch_off_color_arr)

def coarse_seg(all_cluster_init, spanning_tree_idx_arr, sv_off_geo_arr):
	vis_geo = []
	vis_rgb = []
	# for cluster in all_cluster_init:
	# 	rgb = [randrange(0, 1), randrange(0, 255), randrange(0, 255)]
	# 	for idx in cluster:
	# 		vis_geo.append(sv_off_geo_arr[idx])
	# 		vis_rgb.append(rgb)
	# pc_vis(vis_geo, vis_rgb)
	node_arr = []
	for path in spanning_tree_idx_arr:
		st_node = path[0]
		end_node = path[-1]
		node_arr.append(st_node)
		node_arr.append(end_node)
	
	node_dic = dict((i, node_arr.count(i)) for i in node_arr)

	multinode_arr = []
	for key in node_dic:
		if node_dic[key]>2:
			multinode_arr.append(key)
	print(multinode_arr)
	multinode_pt = [sv_off_geo_arr[idx] for idx in multinode_arr]

	for cluster in all_cluster_init:
		flag = 0
		for idx in multinode_arr:
			if idx in cluster:
				flag = 1
				break

		rgb = [randrange(0, 1), randrange(0, 255), randrange(0, 255)]
		if flag:
			for idx in cluster:
				vis_geo.append(sv_off_geo_arr[idx])
				vis_rgb.append(rgb)
		else:
			for idx in cluster:
				pt = sv_off_geo_arr[idx]
				vis_geo.append([pt[0] + pc_width, pt[1], pt[2]])
				vis_rgb.append(rgb)


	pc_vis(vis_geo, vis_rgb)


	# pc_vis(multinode_pt + vis_geo, [[255, 0, 0] for pt in multinode_pt] + vis_rgb)

	# X = np.array(sv_off_geo_arr)
	# tree = BallTree(X, leaf_size = 1)
	# neigh_arr = []
	# nei_hks_pt = []
	# nei_hks_rgb = []
	# for pt in multinode_pt:
	# 	dist, ind = tree.query([pt], k=10)
	# 	neigh_arr = neigh_arr + [sv_off_geo_arr[idx] for idx in ind[0]]

def find_cut_edge(cluster_idx, bound_dic, inner_pt_idx, sv_off_geo_arr, err_orig_err):
	vis_geo = []
	vis_rgb = []
	for idx in inner_pt_idx:
		vis_geo.append(sv_off_geo_arr[idx])
		vis_rgb.append([228, 229, 220])

	for label in bound_dic:
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		for idx in bound_dic[label]:
			vis_geo.append(sv_off_geo_arr[idx])
			vis_rgb.append(rgb)

	inner_err =  [err_orig_err[idx] for idx in inner_pt_idx]
	max_dis_idx = inner_pt_idx[np.argmax(inner_err)]
	max_dis_pt = sv_off_geo_arr[max_dis_idx]

	# pc_vis([max_dis_pt] + vis_geo, [[228, 0, 0]] + vis_rgb)

	bound_pt_arr = []
	for label in bound_dic:
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		bound_pt = []
		for idx in bound_dic[label]:
			vis_geo.append(sv_off_geo_arr[idx])
			vis_rgb.append(rgb)
			bound_pt.append(sv_off_geo_arr[idx])
		bound_pt_arr.append(bound_pt)



	cluster_pt = [sv_off_geo_arr[idx] for idx in cluster_idx]
	neigh = NearestNeighbors(n_neighbors=7)
	neigh.fit(cluster_pt)
	kng = neigh.kneighbors_graph(cluster_pt)
	kng = kng.toarray()
	dist_m = graph_shortest_path(kng, method='auto', directed=False)

	X = np.array(cluster_pt)
	tree = BallTree(X, leaf_size = 1)

	dist, ind = tree.query([max_dis_pt], k=1)
	max_dis_idx = ind[0][0]

	min_bound_pt = []
	min_bound_pt_idx = []
	for bound_pt in bound_pt_arr:
		temp_dis = []
		temp_idx = []
		for pt in bound_pt:
			dist, ind = tree.query([pt], k=1)
			temp_idx.append(ind[0][0])
			temp_dis.append(dist_m[max_dis_idx][ind[0][0]])
		min_idx = temp_idx[np.argmin(temp_dis)]
		min_bound_pt.append(cluster_pt[min_idx])
		min_bound_pt_idx.append(min_idx)
	# pc_vis(min_bound_pt + [max_dis_pt] + vis_geo, [[228, 0, 255] for pt in min_bound_pt] + [[228, 0, 0]] + vis_rgb)


	G = nx.Graph()
	tree = BallTree(np.asarray(cluster_pt), leaf_size=1)
	for i in range(len(cluster_pt)):
		pt = cluster_pt[i]
		dist, ind = tree.query([pt], k = 7)
		idx_arr = ind[0]
		for t in range(1, len(idx_arr)):
			idx = idx_arr[t]
		# for idx in ind[0][1:]:

			if i > idx:
				G.add_edge(i, idx, weight = dist[0][t])

	line_pt = []
	line_color = []
	for idx in min_bound_pt_idx:
		path_arr = nx.shortest_path(G, source=max_dis_idx,target=idx, weight='weight')
		print(path_arr)

		temp_pt = []
		for t in range(0, len(path_arr)-1):
			temp_pt = temp_pt + draw_line(cluster_pt[path_arr[t]], cluster_pt[path_arr[t+1]])
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		temp_color = [rgb for pt in temp_pt]
		line_pt = line_pt + temp_pt
		line_color = line_color + temp_color

	pc_vis(min_bound_pt + [max_dis_pt] + vis_geo + line_pt, [[228, 0, 255] for pt in min_bound_pt] + [[228, 0, 0]] + vis_rgb + line_color)

def normal_based_cut(geo_arr, rgb_arr, normal_arr, sv_off_geo_arr, cluster_idx, seg_thresh, patch_assign_idx):
	cluster_pt = [sv_off_geo_arr[idx] for idx in cluster_idx]
	cluster_normal_arr = get_off_normal(geo_arr, normal_arr, cluster_pt)
	cluster_dic = cube_clustering(cluster_pt, cluster_normal_arr, vis_flag=0)

	all_max_cluster = []
	for cluster_label in cluster_dic:
		patch_geo = [cluster_pt[idx] for idx in cluster_dic[cluster_label]]
		print("patch_geo_len: ", len(patch_geo))
		if len(patch_geo):
			dbscan_dic = dbscan_clustering(patch_geo, seg_thresh)
			temp = []
			for dbscan_label in dbscan_dic:
				temp.append([patch_geo[idx] for idx in dbscan_dic[dbscan_label]])
			max_db_cluster = temp[np.argmax([len(arr) for arr in temp])]
			all_max_cluster.append(max_db_cluster)

	all_max_cluster_cnt_arr = [len(arr) for arr in all_max_cluster]
	max_idx1 = np.argmax(all_max_cluster_cnt_arr)
	all_max_cluster_cnt_arr[max_idx1] = -1
	max_idx2 = np.argmax(all_max_cluster_cnt_arr)

	X = np.array(cluster_pt)
	tree = BallTree(X, leaf_size = 1)
	cluster1 = all_max_cluster[max_idx1]
	cluster2 = all_max_cluster[max_idx2]
	
	cluster1_idx = []
	cluster2_idx = []
	for pt in cluster1:
		dist, ind = tree.query([pt], k = 1)
		cluster1_idx.append(ind[0][0])

	for pt in cluster2:
		dist, ind = tree.query([pt], k = 1)
		cluster2_idx.append(ind[0][0])

	cluster_rest_idx = list(set([t for t in range(len(cluster_pt))]) - set(cluster1_idx + cluster2_idx))

	vis_geo = []
	vis_rgb = []

	for idx in cluster1_idx:
		vis_geo.append(cluster_pt[idx])
		vis_rgb.append([255, 0, 0])
	for idx in cluster2_idx:
		vis_geo.append(cluster_pt[idx])
		vis_rgb.append([0, 255, 0])
	for idx in cluster_rest_idx:
		vis_geo.append(cluster_pt[idx])
		vis_rgb.append([0, 0, 255])
	# pc_vis(vis_geo, vis_rgb)


	features = []
	labels = []
	features = cluster1 + cluster2
	labels = [0 for pt in cluster1] + [1 for pt in cluster2]

	clf = KNeighborsClassifier(n_neighbors = 3, algorithm = 'ball_tree')
	clf.fit(features, labels)
	pre_label = clf.predict(cluster_pt)

	cluster1 = []
	cluster2 = []
	for t in range(0, len(pre_label)):
		if pre_label[t] == 0:
			cluster1.append(cluster_pt[t])
		else:
			cluster2.append(cluster_pt[t])

	# pc_vis(cluster1 + cluster2, [[255, 0, 0] for pt in cluster1] + [[0, 255, 0] for pt in cluster2])

	features = []
	labels = []
	features = cluster1 + cluster2
	labels = [0 for pt in cluster1] + [1 for pt in cluster2]

	clf = KNeighborsClassifier(n_neighbors = 16, algorithm = 'ball_tree')
	clf.fit(features, labels)
	pre_label = clf.predict(cluster_pt)

	cluster1 = []
	cluster2 = []
	for t in range(0, len(pre_label)):
		if pre_label[t] == 0:
			cluster1.append(cluster_pt[t])
		else:
			cluster2.append(cluster_pt[t])


	# pc_vis(cluster1 + cluster2, [[255, 0, 0] for pt in cluster1] + [[0, 255, 0] for pt in cluster2])

	patch_geo = [geo_arr[idx] for idx in patch_assign_idx]
	patch_rgb = [rgb_arr[idx] for idx in patch_assign_idx]

	features = []
	labels = []
	features = cluster1 + cluster2
	labels = [0 for pt in cluster1] + [1 for pt in cluster2]
	
	clf = KNeighborsClassifier(n_neighbors = 16, algorithm = 'ball_tree')
	clf.fit(features, labels)
	pre_label = clf.predict(patch_geo)


	cluster1_geo = []
	cluster1_rgb = []

	cluster2_geo = []
	cluster2_rgb = []

	for t in range(0, len(pre_label)):
		if pre_label[t] == 0:
			cluster1_geo.append(patch_geo[t])
			cluster1_rgb.append(patch_rgb[t])
		else:
			cluster2_geo.append(patch_geo[t])
			cluster2_rgb.append(patch_rgb[t])

	# pc_vis(cluster1_geo, cluster1_rgb)
	[d2_geo1, reconstruction_err1, embedding1] = isomap_based_dimension_reduction(cluster1_geo, cluster1, len(cluster1) > 8)
	pc_vis(cluster1_geo + d2_geo1, cluster1_rgb + cluster1_rgb)


	[d2_geo2, reconstruction_err2, embedding2] = isomap_based_dimension_reduction(cluster2_geo, cluster2, len(cluster2) > 8)
	pc_vis(cluster2_geo + d2_geo2, cluster2_rgb + cluster2_rgb)

def unfold_patch2(all_final_seg_arr, all_final_seg_label, all_cluster, sv_off_geo_arr, sparse_sv_off_geo_arr, err_orig_err, sv_off_normal_arr):
	off_seg_dic = dict()
	vis_geo = []
	vis_rgb = []
	new_label_arr = [0 for pt in sv_off_geo_arr]
	for t in range(len(all_final_seg_arr)):
		rgb = [67, 89, 203]
		rgb =  [randrange(37, 97), randrange(59, 119), randrange(173, 233)]
		if all_final_seg_label[t] == 1:
			rgb =  [randrange(150, 255), randrange(0, 1), randrange(0, 1)]
			rgb = [221, 210, 219]
			rgb =  [randrange(201, 241), randrange(190, 230), randrange(199, 229)]
		if all_final_seg_label[t] == 2:
			rgb =  [randrange(0, 1), randrange(0, 1), randrange(150, 255)]
			rgb = [205, 65, 58]
			rgb =  [randrange(185, 225), randrange(45, 85), randrange(38, 78)]
		
		patch_geo = []
		for seg in all_final_seg_arr[t]:
			for idx in all_cluster[seg]:
				vis_geo.append(sv_off_geo_arr[idx])
				vis_rgb.append(rgb)
				new_label_arr[idx] = t
				patch_geo.append(sv_off_geo_arr[idx])
		off_seg_dic[t] = patch_geo

	sparse_assign_dic = assign_ply_to_seg(off_seg_dic, sparse_sv_off_geo_arr, vis=0)

	[uni_pt_idx, bi_pt_idx, mul_pt_idx, boundary_label_dic] = detect_boundary_points(sv_off_geo_arr, new_label_arr)

	# for t in range(len(all_final_seg_arr)):
	# 	cluster_idx = []
	# 	for seg in all_final_seg_arr[t]:
	# 		cluster_idx = cluster_idx + all_cluster[seg]
	# 	cluster_geo = [sv_off_geo_arr[idx] for idx in cluster_idx]
	# 	cluster_norm = [sv_off_normal_arr[idx] for idx in cluster_idx]
	# 	cluster_rgb = [[205, 65, 58] for idx in cluster_idx]
		# pc_vis(cluster_geo, cluster_rgb)
		# pc_vis(cluster_norm, cluster_rgb)
		# cube_clustering(cluster_geo, cluster_norm, vis_flag=1)
		# aggl_clustering(pt_arr, 2, vis_flag)

	sv_off_geo_arr_assign_dic = dict()
	for t in range(len(all_final_seg_arr)):
		cluster_idx = []
		for seg in all_final_seg_arr[t]:
			cluster_idx = cluster_idx + all_cluster[seg]
		sv_off_geo_arr_assign_dic[t] = [sv_off_geo_arr[idx] for idx in cluster_idx]

	sv_off_geo_dic = assign_ply_to_seg(sv_off_geo_arr_assign_dic, geo_arr, vis=1)

	seg_thresh = get_neighbor_dis(sv_off_geo_arr)*2
	for t in range(len(all_final_seg_arr)):
	# for t in [0, 13, 11]:
	# for t in [13, 14]:
		bound_dic = dict()
		cluster_idx = []
		for seg in all_final_seg_arr[t]:
			cluster_idx = cluster_idx + all_cluster[seg]

		inner_pt_idx = []
		inner_err_idx = []
		for idx in cluster_idx:
			if idx in bi_pt_idx:
				for label in boundary_label_dic[idx]:
					if label != t:
						if not label in bound_dic:
							bound_dic[label] = []
						bound_dic[label].append(idx)
			else:
				inner_pt_idx.append(idx)
				inner_err_idx.append(err_orig_err[idx])

		print(t, len(inner_pt_idx))

		patch_sparse_geo = []
		for s_id in sparse_assign_dic[t]:
			pt = sparse_sv_off_geo_arr[s_id]
			patch_sparse_geo.append([pt[0], pt[1], pt[2]])
		
		# find_cut_edge(cluster_idx, bound_dic, inner_pt_idx, sv_off_geo_arr, err_orig_err)
		normal_based_cut(geo_arr, rgb_arr, normal_arr, sv_off_geo_arr, cluster_idx, seg_thresh, sv_off_geo_dic[t])
		# vis_geo = []
		# vis_rgb = []
		# for idx in inner_pt_idx:
		# 	vis_geo.append(sv_off_geo_arr[idx])
		# 	vis_rgb.append([228, 229, 220])

		# for label in bound_dic:
		# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		# 	for idx in bound_dic[label]:
		# 		vis_geo.append(sv_off_geo_arr[idx])
		# 		vis_rgb.append(rgb)

		# max_dis_idx = inner_pt_idx[np.argmax(inner_err_idx)]

		# pc_vis(patch_sparse_geo + vis_geo, [[228, 0, 0] for pt in patch_sparse_geo] + vis_rgb)

		# bound_pt = []
		# for label in bound_dic:
		# 	bound_idx_arr = bound_dic[label]
		# 	dis_arr = []
		# 	path_arr_all = []
		# 	for bound_idx in bound_idx_arr:
		# 		bound_pt.append(sv_off_geo_arr[bound_idx])

		# poi = [sv_off_geo_arr[max_dis_idx]] + patch_sparse_geo 

		# pc_vis([sv_off_geo_arr[max_dis_idx]] + patch_sparse_geo + bound_pt, [[255, 0, 0]] + [[0, 255, 0] for pt in patch_sparse_geo] + [[0, 0, 255] for pt in bound_pt])

		# neigh = NearestNeighbors(n_neighbors=5)
		# neigh.fit(cluster_pt)
		# kng = neigh.kneighbors_graph(cluster_pt)
		# kng = kng.toarray()
		# dist_m = graph_shortest_path(kng, method='auto', directed=False)

		# G_dis = nx.Graph()
		# for i in range(len(cluster_idx) - 1):
		# 	for j in range(i+1, len(cluster_idx)):
		# 		G_dis.add_edge(i, j, weight = dist_m[i][j])




		# max_dis_idx = inner_pt_idx[np.argmax(inner_err_idx)]

		# cluster_pt = [sv_off_geo_arr[idx] for idx in cluster_idx]
		# neigh = NearestNeighbors(n_neighbors=5)
		# neigh.fit(cluster_pt)
		# kng = neigh.kneighbors_graph(cluster_pt)
		# kng = kng.toarray()
		# dist_m = graph_shortest_path(kng, method='auto', directed=False)

		# G_dis = nx.Graph()
		# for i in range(len(cluster_idx) - 1):
		# 	for j in range(i+1, len(cluster_idx)):
		# 		G_dis.add_edge(i, j, weight = dist_m[i][j])


		# line_pt = []
		# line_color = []
		# for label in bound_dic:
		# 	bound_idx_arr = bound_dic[label]
		# 	dis_arr = []
		# 	path_arr_all = []
		# 	for bound_idx in bound_idx_arr:
		# 		path_arr = nx.shortest_path(G_dis, source=cluster_idx.index(max_dis_idx),target=cluster_idx.index(bound_idx), weight='weight')
		# 		dis = 0
		# 		for t in range(1, len(path_arr)):
		# 			dis = dis + np.linalg.norm(np.asarray(sv_off_geo_arr[cluster_idx[path_arr[t-1]]])-np.asarray(sv_off_geo_arr[cluster_idx[path_arr[t]]]))
		# 		dis_arr.append(dis)
		# 		path_arr_all.append(path_arr)
			
		# 	path_arr = path_arr_all[np.argmin(dis_arr)]
		# 	temp_pt = []
		# 	for t in range(0, len(path_arr)-1):
		# 		temp_pt = temp_pt + draw_line(cluster_pt[path_arr[t]], cluster_pt[path_arr[t+1]])
		# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		# 	temp_color = [rgb for pt in temp_pt]
		# 	line_pt = line_pt + temp_pt
		# 	line_color= line_color + temp_color

		# 	print(label, np.argmin(dis_arr), dis_arr)
		# pc_vis([sv_off_geo_arr[max_dis_idx]] + line_pt + vis_geo, [[255, 0, 0]] + line_color + vis_rgb)



				



	# vis_geo = []
	# vis_rgb = []
	# for idx in uni_pt_idx:
	# 	vis_geo.append(sv_off_geo_arr[idx])
	# 	vis_rgb.append([128, 128, 128])

	# for idx in bi_pt_idx:
	# 	vis_geo.append(sv_off_geo_arr[idx])
	# 	vis_rgb.append([255, 0, 0])

	# for idx in mul_pt_idx:
	# 	vis_geo.append(sv_off_geo_arr[idx])
	# 	vis_rgb.append([0, 255, 0])
	# pc_vis(vis_geo, vis_rgb)


	# for t in range(len(all_final_seg_arr)):

def patch_generation(geo_arr, rgb_arr, normal_arr, floder, frame_id, frame_id_head, iso_sv_pt_num):
	svoff_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(128) + ".off"
	[sv_off_geo_arr, sv_off_rgb_arr, sv_off_pt_num] = read_off(svoff_path, geo_arr, rgb_arr)

	sparse_sv_pt_num = 512
	sparse_svoff_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(sparse_sv_pt_num) + ".off"
	[sparse_sv_off_geo_arr, sparse_sv_off_rgb_arr, sparse_sv_off_pt_num] = read_off(sparse_svoff_path, geo_arr, rgb_arr)
	# pc_vis(sv_off_geo_arr, sv_off_rgb_arr)

	sv_off_normal_arr = get_off_normal(geo_arr, normal_arr, sv_off_geo_arr)
	sparse_sv_off_normal_arr = get_off_normal(geo_arr, normal_arr, sparse_sv_off_geo_arr)

	[hks_pt_arr, hks_feature_arr] = get_hks_feature(geo_arr, rgb_arr, floder, frame_id)
	tot_num = len(hks_pt_arr)
	X = np.array(hks_pt_arr)
	tree = BallTree(X, leaf_size = 1)
	new_hks_feature_arr = []
	for pt in sv_off_geo_arr:
		dist, ind = tree.query([pt], k=1)
		idx = ind[0][0]
		new_hks_feature_arr.append(hks_feature_arr[idx])
	hks_feature_arr = new_hks_feature_arr

	colors = cm.coolwarm(np.linspace(0, 1, 256))
	colors = [rgb[0:3] for rgb in colors]
	colors = colors[::-1]
	min_hks = min(hks_feature_arr)
	max_hks = max(hks_feature_arr)
	hks_rgb = [colors[int(np.round((val-min_hks)/(max_hks-min_hks)*255))][0:3]*255 for val in hks_feature_arr]

	seg_thresh = get_neighbor_dis(sv_off_geo_arr)*2

	[err_orig_err, err_arr] = compute_distortion(sv_off_geo_arr, seg_thresh, vis = 0)
	dis_pt_idx = detect_distortion_pts([], sv_off_geo_arr, sparse_sv_off_geo_arr, err_arr, seg_thresh, N_r = 6, vis = 0)
	spanning_tree_idx_arr = mini_spanning_tree(sv_off_geo_arr, sparse_sv_off_geo_arr, dis_pt_idx, vis = 0)
	
	[skeleton_adj_dic, chain_seg_dic, chain_dic, all_cluster_init, end_node, bi_node, multi_node] = hybird_seg2(geo_arr, rgb_arr, floder, frame_id, sv_off_geo_arr, hks_rgb, vis = 0)
	all_cluster = merge_multinode(skeleton_adj_dic, end_node, bi_node, multi_node, all_cluster_init, sv_off_geo_arr)

	# coarse_seg(all_cluster_init, spanning_tree_idx_arr, sv_off_geo_arr)

	vis_geo = []
	vis_rgb = []
	for cluster in all_cluster:
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		for idx in cluster:
			vis_geo.append(sv_off_geo_arr[idx])
			vis_rgb.append(rgb)
	pc_vis(vis_geo, vis_rgb)


	###########
	assign_dic = unfold_patch(bi_node, end_node, all_cluster, sv_off_geo_arr, geo_arr, rgb_arr)
	[all_final_seg_arr, all_final_seg_label] = patch_unfold(chain_dic, all_cluster, sv_off_geo_arr, geo_arr, rgb_arr, floder, frame_id, end_node, bi_node, multi_node)
	###########
	# unfold_patch2(all_final_seg_arr, all_final_seg_label, all_cluster, sv_off_geo_arr, sparse_sv_off_geo_arr, err_orig_err, sv_off_normal_arr)
	vis_geo = []
	vis_rgb = []
	for t in range(len(all_final_seg_arr)):
		rgb = [67, 89, 203]
		rgb =  [randrange(37, 97), randrange(59, 119), randrange(173, 233)]
		if all_final_seg_label[t] == 1:
			rgb =  [randrange(150, 255), randrange(0, 1), randrange(0, 1)]
			rgb = [221, 210, 219]
			rgb =  [randrange(201, 241), randrange(190, 230), randrange(199, 229)]
		if all_final_seg_label[t] == 2:
			rgb =  [randrange(0, 1), randrange(0, 1), randrange(150, 255)]
			rgb = [205, 65, 58]
			rgb =  [randrange(185, 225), randrange(45, 85), randrange(38, 78)]
		for seg in all_final_seg_arr[t]:
			for idx in all_cluster[seg]:
				vis_geo.append(sv_off_geo_arr[idx])
				vis_rgb.append(rgb)
	pc_vis(vis_geo, vis_rgb)

	return [all_final_seg_arr, all_final_seg_label, all_cluster, sv_off_geo_arr]





	# for t in range(len(all_final_seg_arr)):
	# 	patch_geo = []
	# 	patch_rgb = []
	# 	patch_off_geo = []
	# 	for seg in all_final_seg_arr[t]:
	# 		for idx in assign_dic[seg]:
	# 			patch_geo.append(geo_arr[idx])
	# 			patch_rgb.append(rgb_arr[idx])


	# 		for idx in all_cluster[seg]:
	# 			patch_off_geo.append(sv_off_geo_arr[idx])
	# 	print(t, len(patch_geo), len(patch_off_geo))


	# 	pc_vis(patch_off_geo + patch_geo, [[255, 0, 0] for pt in patch_off_geo] + patch_rgb)
	# 	patch_off_rgb = get_off_color(patch_geo, patch_rgb, patch_off_geo)
	# 	patch_geo = patch_off_geo
	# 	patch_rgb = patch_off_rgb
		# projection_based_aig(patch_geo, patch_rgb, patch_off_geo, all_cluster_init, skeleton_adj_dic, sv_off_geo_arr, all_final_seg_arr[t])


	# for t in [0]:
	# 	print(t)
	# 	patch_geo = []
	# 	patch_rgb = []
	# 	patch_off_geo = []
	# 	for seg in all_final_seg_arr[t]:
	# 		for idx in assign_dic[seg]:
	# 			patch_geo.append(geo_arr[idx])
	# 			patch_rgb.append(rgb_arr[idx])
	# 		for idx in all_cluster[seg]:
	# 			patch_off_geo.append(sv_off_geo_arr[idx])
	# 	pc_vis(patch_geo, patch_rgb)
	# 	isomap_based_aig(patch_off_geo, patch_geo, patch_rgb)






	# line_pt = []
	# line_color = []
	# for path_arr in spanning_tree_idx_arr:
	# 	temp_pt = []
	# 	for t in range(0, len(path_arr)-1):
	# 		temp_pt = temp_pt + draw_line(sv_off_geo_arr[path_arr[t]], sv_off_geo_arr[path_arr[t+1]])
	# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
	# 	temp_color = [rgb for pt in temp_pt]
	# 	line_pt = line_pt + temp_pt
	# 	line_color= line_color + temp_color
	# pc_vis(line_pt + vis_geo, line_color + vis_rgb)



	# vis_geo = []
	# vis_rgb = []
	# for node in end_node:
	# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
	# 	for idx in all_cluster[node]:
	# 		vis_geo.append(sv_off_geo_arr[idx])
	# 		vis_rgb.append(rgb)
	# pc_vis(vis_geo, vis_rgb)


	# vis_geo = []
	# vis_rgb = []
	# for chain in chain_dic:
	# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
	# 	for seg in chain_dic[chain]:
	# 		for idx in all_cluster[seg]:
	# 			vis_geo.append(sv_off_geo_arr[idx])
	# 			vis_rgb.append(rgb)
	# pc_vis(vis_geo, vis_rgb)


	# for chain in chain_dic:
	# 	patch_geo = []
	# 	for seg in chain_dic[chain]:
	# 		for idx in all_cluster[seg]:
	# 			patch_geo.append(sv_off_geo_arr[idx])
	# 	if len(patch_geo):
	# 		seg_thresh = get_neighbor_dis(patch_geo)*2
	# 		err_arr = compute_distortion(patch_geo, seg_thresh, vis = 1)
	# 		dis_pt_idx = detect_distortion_pts(patch_geo, sparse_sv_off_geo_arr, err_arr, seg_thresh, N_r = 6, vis = 1)
	# 		spanning_tree_idx_arr = mini_spanning_tree(patch_geo, sparse_sv_off_geo_arr, dis_pt_idx)
			
def hks_vis(geo_arr, rgb_arr, floder, frame_id):
	pt_arr = []
	idx = 0
	with open(floder + "hks/" + frame_id + "_n16.off") as ins:
		for line in ins:
			re2 = line.replace("\n", "").split(" ")
			if idx>1:
				# if len(re2) == 3:
					pt = [float(val) for val in re2[0:3]]
					pt_arr.append(pt)
			idx = idx + 1


	dbscan_thresh = get_neighbor_dis(pt_arr)*2


	hks_feature_arr = []
	with open(floder + "hks/" + frame_id + "_n16_hks.txt") as ins:
		for line in ins:
			re2 = line.replace("\n", "").split("	")
			val = float(re2[0])
			hks_feature_arr.append(val)

	[hks_pt_arr, hks_feature_arr] = get_hks_feature(geo_arr, rgb_arr, floder, frame_id)
	

	# for colors in [cm.jet(np.linspace(0, 1, 256)), cm.viridis(np.linspace(0, 1, 256)), cm.plasma(np.linspace(0, 1, 256)), cm.winter(np.linspace(0, 1, 256))]:
	# for colors in [cm.coolwarm(np.linspace(0, 1, 256))]:
	colors = cm.coolwarm(np.linspace(0, 1, 256))
	# colors = cm.jet(np.linspace(0, 1, 256))
	colors = [rgb[0:3] for rgb in colors]
	colors = colors[::-1]
	min_hks = min(hks_feature_arr)
	max_hks = max(hks_feature_arr)
	d2_rgb = [colors[int(np.round((val-min_hks)/(max_hks-min_hks)*255))][0:3] for val in hks_feature_arr]
	# pc_vis(pt_arr, np.asarray(d2_rgb)*255)

	print(min_hks, max_hks)

	n, bins, patches = plt.hist(hks_feature_arr, bins = "doane", edgecolor='black')
	bin_val_color_arr = []
	for j in range(len(bins)-1):
		bin_val = (bins[j] + bins[j+1])/2.0
		rgb = colors[int(np.round((bin_val-min_hks)/(max_hks-min_hks)*255))][0:4]
		bin_val_color_arr.append([rgb[0]*255, rgb[1]*255, rgb[2]*255])
	# for c, p in zip(bin_val_color_arr, patches):
	#     plt.setp(p, 'facecolor', c)
	# plt.xlabel("Heat Kernel Signature", {'fontname':'Arial', 'size':'14'})
	# plt.ylabel("Number of Points", {'fontname':'Arial', 'size':'14'})
	# plt.show()

	label_dic = dict()
	for j in range(len(bins)-1):
		label_dic[j] = []
	bins[-1] = bins[-1]*1.001

	for i in range(len(hks_feature_arr)):
		label = 0
		for j in range(len(bins)-1):
			if bins[j]<=hks_feature_arr[i] and hks_feature_arr[i] < bins[j+1]:
				label = j
				break
		if not label in label_dic:
			label_dic[label] = []
		label_dic[label].append(i)

	vis_geo = []
	vis_rgb = []
	for label in label_dic:
		for idx in label_dic[label]:
			vis_geo.append(hks_pt_arr[idx])
			vis_rgb.append(bin_val_color_arr[label])

	dbscan_thresh = get_neighbor_dis(hks_pt_arr)*2
	all_cluster = []
	for cluster in label_dic:
		seg_idx_arr = label_dic[cluster]
		seg_pt_arr = [hks_pt_arr[idx] for idx in seg_idx_arr]
		dbscan_dic = dbscan_clustering(seg_pt_arr, dbscan_thresh)
		for seg_cl in dbscan_dic:
			all_cluster.append([seg_idx_arr[id] for id in dbscan_dic[seg_cl]])




	new_label_arr = [0 for i in range(len(hks_pt_arr))]
	all_cluster_dic = dict()

	for t in range(len(all_cluster)):
		idx_arr = all_cluster[t]
		for idx in idx_arr:
			new_label_arr[idx] = t
		all_cluster_dic[t] = idx_arr

		# rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		# vis_geo = vis_geo + [hks_pt_arr[idx] for idx in idx_arr]
		# vis_rgb = vis_rgb + [rgb for idx in idx_arr]

	[uni_pt_idx, bi_pt_idx, mul_pt_idx, boundary_label_dic] = detect_boundary_points(hks_pt_arr, new_label_arr)
	print(len(uni_pt_idx), len(bi_pt_idx), len(mul_pt_idx))

	boundary_pt = [hks_pt_arr[idx] for idx in bi_pt_idx + mul_pt_idx]
	boundary_dbscan_dic = dbscan_clustering(boundary_pt, get_neighbor_dis(hks_pt_arr)*2.5)

	lines_geo = []
	lines_rgb = []
	for cluster in all_cluster:
		
		boundary_pt = []
		for idx in cluster:
			if idx in  bi_pt_idx + mul_pt_idx:
				boundary_pt.append(hks_pt_arr[idx])

		boundary_dbscan_dic = dbscan_clustering(boundary_pt, get_neighbor_dis(hks_pt_arr)*4)
		for label in boundary_dbscan_dic:
			sub_geo_arr = []
			rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
			for idx in boundary_dbscan_dic[label]:
				# vis_geo.append(boundary_pt[idx])
				# vis_rgb.append(rgb)
				sub_geo_arr.append(boundary_pt[idx])

			[line_pt_all, curve_info, new_merged_center_arr] = generate_circle(sub_geo_arr,  get_neighbor_dis(hks_pt_arr)*4)
			lines_geo = lines_geo + [pt[0:3] for pt in line_pt_all]
			rgb = [0, 0, 0]
			lines_rgb = lines_rgb + [rgb for pt in line_pt_all]
	pc_vis(lines_geo + vis_geo, lines_rgb + vis_rgb)
	vis_geo = []
	vis_rgb = []
	for cluster in all_cluster:
		boundary_pt = []
		for idx in cluster:
			if idx in  bi_pt_idx + mul_pt_idx:
				boundary_pt.append(hks_pt_arr[idx])
		boundary_dbscan_dic = dbscan_clustering(boundary_pt, get_neighbor_dis(hks_pt_arr)*4)
		for label in boundary_dbscan_dic:
			sub_geo_arr = []
			rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]






	# vis_geo = []
	# vis_rgb = []
	# seg_thresh = get_neighbor_dis(hks_pt_arr)*1
	# for label in boundary_dbscan_dic:
	# 	sub_geo_arr = []
	# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
	# 	for idx in boundary_dbscan_dic[label]:
	# 		vis_geo.append(boundary_pt[idx])
	# 		vis_rgb.append(rgb)
	# 		sub_geo_arr.append(boundary_pt[idx])

	# 	[line_pt_all, curve_info, new_merged_center_arr] = generate_circle(sub_geo_arr, seg_thresh)
	# 	vis_geo = vis_geo + [pt[0:3] for pt in line_pt_all]
	# 	# rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
	# 	rgb = [0, 0, 0]
	# 	vis_rgb = vis_rgb + [rgb for pt in line_pt_all]
	# 	pc_vis(vis_geo, vis_rgb)

	# pc_vis([hks_pt_arr[idx] for idx in bi_pt_idx + mul_pt_idx] + vis_geo, [[0, 0, 0] for idx in bi_pt_idx + mul_pt_idx] +  vis_rgb)	

	# vis_geo = []
	# vis_rgb = []
	# for label in label_dic:
	# 	for idx in label_dic[label]:
	# 		vis_geo.append(hks_pt_arr[idx])
	# 		vis_rgb.append(bin_val_color_arr[label])
	# pc_vis(vis_geo, np.asarray(vis_rgb)*255)

def normal_clustering2_patch_off(geo_arr, rgb_arr, normal_arr, floder, frame_id, frame_id_head):
	iso_num = 16
	min_cluster_num = int(64*128/iso_num)
	print("min_cluster_num: ", min_cluster_num)
	svoff_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(iso_num) + ".off"
	[sv_off_geo_arr, sv_off_rgb_arr, sv_off_pt_num] = read_off(svoff_path, geo_arr, rgb_arr)
	sv_off_normal_arr = get_off_normal(geo_arr, normal_arr, sv_off_geo_arr)
	cube_clustering_dic = cube_clustering(sv_off_geo_arr, sv_off_normal_arr, vis_flag=0)

	seg_thresh = get_neighbor_dis(sv_off_geo_arr)*1.8

	all_cluster = []
	all_cluster_idx = []
	for label in cube_clustering_dic:
		cube_geo = [sv_off_geo_arr[idx] for idx in cube_clustering_dic[label]]

		dbscan_dic = dbscan_clustering(cube_geo, seg_thresh)
		temp = []
		temp_idx = []
		for dbscan_label in dbscan_dic:
			if len(dbscan_dic[dbscan_label])>0:
				temp.append([cube_geo[idx] for idx in dbscan_dic[dbscan_label]])
				temp_idx.append(label)
				# print(dbscan_label, len(dbscan_dic[dbscan_label]))
		all_cluster = all_cluster + temp
		all_cluster_idx = all_cluster_idx + temp_idx


	features = []
	labels = []
	cluster_dic = dict()
	cluster_label_dic = dict()
	rest_pt = []

	cnt = 0
	for t in range(len(all_cluster)):
		cluster = all_cluster[t]
		if len(cluster) > 128:
			features = features + cluster 
			labels = labels + [cnt for pt in cluster]
			cluster_dic[cnt] = cluster
			cluster_label_dic[cnt] = all_cluster_idx[t]
			cnt = cnt + 1
		else:
			rest_pt = rest_pt + cluster


	clf = KNeighborsClassifier(n_neighbors = 3, algorithm = 'ball_tree')
	clf.fit(features, labels)
	pre_label = clf.predict(rest_pt)
	for t in range(0, len(pre_label)):
		cluster_dic[pre_label[t]].append(rest_pt[t])
	vis_geo = []
	vis_rgb = []
	for label in cluster_dic:
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		# vis_geo = vis_geo + [sv_off_geo_arr[idx] for idx in cluster_dic[label]]
		vis_geo = vis_geo + cluster_dic[label]
		vis_rgb = vis_rgb + [rgb for idx in cluster_dic[label]]
	# pc_vis(vis_geo, vis_rgb)


	all_cluster = []
	all_cluster_label = []
	for label in cluster_dic:
		all_cluster.append(cluster_dic[label])
		all_cluster_label = all_cluster_label + [cluster_label_dic[label]]

	features = []
	labels = []
	cluster_dic = dict()
	cluster_label_dic = dict()
	for t in range(len(all_cluster)):
		if len(all_cluster)>0:
			cluster = all_cluster[t]
			features = features + cluster 
			labels = labels + [t for pt in cluster]
			cluster_dic[t] = []
			cluster_label_dic[t] = all_cluster_label[t]

	sv_off_geo_arr = sv_off_geo_arr
	clf = KNeighborsClassifier(n_neighbors = 16, algorithm = 'ball_tree')
	clf.fit(features, labels)
	pre_label = clf.predict(sv_off_geo_arr)
	for t in range(0, len(pre_label)):
		cluster_dic[pre_label[t]].append(t)

	# vis_geo = []
	# vis_rgb = []
	# for label in cluster_dic:
	# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
	# 	vis_geo = vis_geo + [sv_off_geo_arr[idx] for idx in cluster_dic[label]]
	# 	vis_rgb = vis_rgb + [rgb for idx in cluster_dic[label]]
	# pc_vis(vis_geo, vis_rgb)

	return [cluster_dic, sv_off_geo_arr]

def normal_clustering2(geo_arr, rgb_arr, normal_arr, floder, frame_id, frame_id_head):
	iso_num = 16
	min_cluster_num = int(64*128/iso_num)
	print("min_cluster_num: ", min_cluster_num)
	svoff_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(iso_num) + ".off"
	[sv_off_geo_arr, sv_off_rgb_arr, sv_off_pt_num] = read_off(svoff_path, geo_arr, rgb_arr)
	sv_off_normal_arr = get_off_normal(geo_arr, normal_arr, sv_off_geo_arr)
	cube_clustering_dic = cube_clustering(sv_off_geo_arr, sv_off_normal_arr, vis_flag=0)

	seg_thresh = get_neighbor_dis(sv_off_geo_arr)*1.8

	all_cluster = []
	all_cluster_idx = []
	for label in cube_clustering_dic:
		cube_geo = [sv_off_geo_arr[idx] for idx in cube_clustering_dic[label]]

		dbscan_dic = dbscan_clustering(cube_geo, seg_thresh)
		temp = []
		temp_idx = []
		for dbscan_label in dbscan_dic:
			if len(dbscan_dic[dbscan_label])>0:
				temp.append([cube_geo[idx] for idx in dbscan_dic[dbscan_label]])
				temp_idx.append(label)
				# print(dbscan_label, len(dbscan_dic[dbscan_label]))
		all_cluster = all_cluster + temp
		all_cluster_idx = all_cluster_idx + temp_idx


	features = []
	labels = []
	cluster_dic = dict()
	cluster_label_dic = dict()
	rest_pt = []

	cnt = 0
	for t in range(len(all_cluster)):
		cluster = all_cluster[t]
		if len(cluster) > 128:
			features = features + cluster 
			labels = labels + [cnt for pt in cluster]
			cluster_dic[cnt] = cluster
			cluster_label_dic[cnt] = all_cluster_idx[t]
			cnt = cnt + 1
		else:
			rest_pt = rest_pt + cluster


	clf = KNeighborsClassifier(n_neighbors = 3, algorithm = 'ball_tree')
	clf.fit(features, labels)
	pre_label = clf.predict(rest_pt)
	for t in range(0, len(pre_label)):
		cluster_dic[pre_label[t]].append(rest_pt[t])
	vis_geo = []
	vis_rgb = []
	for label in cluster_dic:
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		# vis_geo = vis_geo + [sv_off_geo_arr[idx] for idx in cluster_dic[label]]
		vis_geo = vis_geo + cluster_dic[label]
		vis_rgb = vis_rgb + [rgb for idx in cluster_dic[label]]
	# pc_vis(vis_geo, vis_rgb)


	all_cluster = []
	all_cluster_label = []
	for label in cluster_dic:
		all_cluster.append(cluster_dic[label])
		all_cluster_label = all_cluster_label + [cluster_label_dic[label]]

	features = []
	labels = []
	cluster_dic = dict()
	cluster_label_dic = dict()
	for t in range(len(all_cluster)):
		if len(all_cluster)>0:
			cluster = all_cluster[t]
			features = features + cluster 
			labels = labels + [t for pt in cluster]
			cluster_dic[t] = []
			cluster_label_dic[t] = all_cluster_label[t]

	sv_off_geo_arr = geo_arr
	clf = KNeighborsClassifier(n_neighbors = 16, algorithm = 'ball_tree')
	clf.fit(features, labels)
	pre_label = clf.predict(sv_off_geo_arr)
	for t in range(0, len(pre_label)):
		cluster_dic[pre_label[t]].append(t)

	vis_geo = []
	vis_rgb = []
	for label in cluster_dic:
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		vis_geo = vis_geo + [sv_off_geo_arr[idx] for idx in cluster_dic[label]]
		vis_rgb = vis_rgb + [rgb for idx in cluster_dic[label]]
	# pc_vis(vis_geo, vis_rgb)

	return [cluster_dic, cluster_label_dic, sv_off_geo_arr]

def normal_clustering(geo_arr, rgb_arr, normal_arr, floder, frame_id, frame_id_head):
	iso_num = 16
	min_cluster_num = int(64*128/iso_num)
	print("min_cluster_num: ", min_cluster_num)
	svoff_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(iso_num) + ".off"
	[sv_off_geo_arr, sv_off_rgb_arr, sv_off_pt_num] = read_off(svoff_path, geo_arr, rgb_arr)
	sv_off_normal_arr = get_off_normal(geo_arr, normal_arr, sv_off_geo_arr)
	cube_clustering_dic = cube_clustering(sv_off_geo_arr, sv_off_normal_arr, vis_flag=0)

	seg_thresh = get_neighbor_dis(sv_off_geo_arr)*1.8

	all_cluster = []
	for label in cube_clustering_dic:
		cube_geo = [sv_off_geo_arr[idx] for idx in cube_clustering_dic[label]]

		dbscan_dic = dbscan_clustering(cube_geo, seg_thresh)
		temp = []
		for dbscan_label in dbscan_dic:
			if len(dbscan_dic[dbscan_label])>0:
				temp.append([cube_geo[idx] for idx in dbscan_dic[dbscan_label]])
				# print(dbscan_label, len(dbscan_dic[dbscan_label]))
		all_cluster = all_cluster + temp


	features = []
	labels = []
	cluster_dic = dict()
	rest_pt = []
	cnt = 0
	for t in range(len(all_cluster)):
		cluster = all_cluster[t]
		if len(cluster) > 128:
			features = features + cluster 
			labels = labels + [cnt for pt in cluster]
			cluster_dic[cnt] = cluster
			cnt = cnt + 1
		else:
			rest_pt = rest_pt + cluster

	clf = KNeighborsClassifier(n_neighbors = 3, algorithm = 'ball_tree')
	clf.fit(features, labels)
	pre_label = clf.predict(rest_pt)
	for t in range(0, len(pre_label)):
		cluster_dic[pre_label[t]].append(rest_pt[t])
	vis_geo = []
	vis_rgb = []
	for label in cluster_dic:
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		# vis_geo = vis_geo + [sv_off_geo_arr[idx] for idx in cluster_dic[label]]
		vis_geo = vis_geo + cluster_dic[label]
		vis_rgb = vis_rgb + [rgb for idx in cluster_dic[label]]
	# pc_vis(vis_geo, vis_rgb)


	all_cluster = []
	for label in cluster_dic:
		all_cluster.append(cluster_dic[label])

	features = []
	labels = []
	cluster_dic = dict()
	for t in range(len(all_cluster)):
		if len(all_cluster)>0:
			cluster = all_cluster[t]
			features = features + cluster 
			labels = labels + [t for pt in cluster]
			cluster_dic[t] = []

	clf = KNeighborsClassifier(n_neighbors = 16, algorithm = 'ball_tree')
	clf.fit(features, labels)
	pre_label = clf.predict(sv_off_geo_arr)
	for t in range(0, len(pre_label)):
		cluster_dic[pre_label[t]].append(t)

	vis_geo = []
	vis_rgb = []
	for label in cluster_dic:
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		vis_geo = vis_geo + [sv_off_geo_arr[idx] for idx in cluster_dic[label]]
		vis_rgb = vis_rgb + [rgb for idx in cluster_dic[label]]
	pc_vis(vis_geo, vis_rgb)

	return [cluster_dic, sv_off_geo_arr]

def get_hks_map(patch_off):
	hks_feature_arr = []
	with open(floder + "hks/" + frame_id + "_n16_hks.txt") as ins:
		for line in ins:
			re2 = line.replace("\n", "").split("	")
			val = float(re2[0])
			hks_feature_arr.append(val)
	[hks_pt_arr, hks_feature_arr] = get_hks_feature(geo_arr, rgb_arr, floder, frame_id)

	X = np.array(hks_pt_arr)
	tree = BallTree(X, leaf_size = 1)

	patch_hks = []
	for pt in patch_off:
		dist, ind = tree.query([pt], k = 1)
		patch_hks.append(hks_feature_arr[ind[0][0]])
	return patch_hks

def pca_rot(sub_patch_d2_geo, sub_patch_rgb):
	sub_patch_d2 = [pt[0:2] for pt in sub_patch_d2_geo]
	pca = PCA(n_components=2)
	pca.fit(np.array(sub_patch_d2))
	p = pca.components_
	
	center = np.mean(sub_patch_d2, axis=0)

	cos_len_arr = []
	cos_len_dic = dict()
	sin_len_arr = []
	slice_geo_dic = dict()
	vis_row_pts = []
	rot_pt = []
	for i in range(0, len(sub_patch_d2)):
		pt = sub_patch_d2[i]
		[x, y] = point_projectto_line2(pt, center, p[0], p[1])
		rot_pt.append([x, y, 20])

	
	pc_vis(sub_patch_d2_geo + rot_pt, sub_patch_rgb + sub_patch_rgb)

def normal_patch_unfold2(geo_arr, rgb_arr, cluster_dic, sv_off_geo_arr, res_folder):
	sv_off_geo_arr_assign_dic = dict()
	for label in cluster_dic:
		patch_off = [sv_off_geo_arr[idx] for idx in cluster_dic[label]]
		sv_off_geo_arr_assign_dic[label] = patch_off
	sv_off_geo_dic = assign_ply_to_seg(sv_off_geo_arr_assign_dic, geo_arr, vis=0)

	for label in cluster_dic:
	# for label in [1]:
		patch_off = [sv_off_geo_arr[idx] for idx in cluster_dic[label]]
		patch_off_rgb = [[255, 0, 0] for pt in patch_off]
		print(label, len(patch_off))

		patch_geo = [geo_arr[idx] for idx in sv_off_geo_dic[label]]
		patch_rgb = [rgb_arr[idx] for idx in sv_off_geo_dic[label]]
		# pc_vis(patch_off + patch_geo, patch_off_rgb + patch_rgb)

		[patch_cluster_pt_dic, tf_iso_off_arr, untf_iso_off_arr, off_embedding_arr] = distortion_based_sv_segmentation(patch_off, 0)

		patch_sv_off_geo_dic = assign_ply_to_seg(patch_cluster_pt_dic, patch_geo, vis=0)

		for sub_label in patch_sv_off_geo_dic:
		# for sub_label in [0]:
			sub_patch_geo = [patch_geo[idx] for idx in patch_sv_off_geo_dic[sub_label]]
			sub_patch_rgb = [patch_rgb[idx] for idx in patch_sv_off_geo_dic[sub_label]]
			sub_patch_off = patch_cluster_pt_dic[sub_label]
			# pc_vis(sub_patch_off + sub_patch_geo, [[255, 0, 0] for pt in sub_patch_off] + sub_patch_rgb)

			# [sub_patch_d2_geo, reconstruction_err, embedding] = isomap_based_dimension_reduction_new(sub_patch_geo, sub_patch_off, len(sub_patch_off) > 8)


			# file_id = res_folder + "//" + str(label) + "_" + str(sub_label)
			
			# try:
			# 	test_rf2(sub_patch_geo, sub_patch_d2_geo, sub_patch_rgb, file_id)
			# except:
			# 	print()



def normal_patch_unfold(geo_arr, rgb_arr, cluster_dic, sv_off_geo_arr, res_folder):
	seg_thresh = get_neighbor_dis(sv_off_geo_arr)*2
	dbscan_thresh = seg_thresh
	vis_geo = []
	vis_rgb = []
	for label in cluster_dic:
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		vis_geo = vis_geo + [sv_off_geo_arr[idx] for idx in cluster_dic[label]]
		vis_rgb = vis_rgb + [rgb for idx in cluster_dic[label]]
	pc_vis(vis_geo, vis_rgb)

	sv_off_geo_arr_assign_dic = dict()
	for label in cluster_dic:
		patch_off = [sv_off_geo_arr[idx] for idx in cluster_dic[label]]
		sv_off_geo_arr_assign_dic[label] = patch_off

	# sv_off_geo_dic = assign_ply_to_seg(sv_off_geo_arr_assign_dic, geo_arr, vis=1)

	for label in cluster_dic:
		patch_off = [sv_off_geo_arr[idx] for idx in cluster_dic[label]]
		patch_off_rgb = [[255, 0, 0] for pt in patch_off]
		print(label, len(patch_off))

		distortion_based_sv_segmentation(patch_off, 1)

		# patch_geo = [geo_arr[idx] for idx in sv_off_geo_dic[label]]
		# patch_rgb = [rgb_arr[idx] for idx in sv_off_geo_dic[label]]
		# pc_vis(patch_off + patch_geo, patch_off_rgb + patch_rgb)

		# pc_vis(patch_off, patch_off_rgb)

		# [d2_geo1, reconstruction_err1, embedding1] = isomap_based_dimension_reduction(patch_off, patch_off, len(patch_off) > 8,  32)
		# patch_off_color = get_off_color(geo_arr, rgb_arr, patch_off)
		# pc_vis(patch_off + d2_geo1, patch_off_color + patch_off_color)


		# n_neighbors = 16
		# n_components = 2
		# st = time.time()
		# embedding = manifold.LocallyLinearEmbedding(n_neighbors, n_components, eigen_solver='auto', method="ltsa")
		# Y_off = embedding.fit_transform(np.matrix(patch_off))
		# print(time.time()-st, len(patch_off))

		# for md in ['standard', 'ltsa', 'hessian', 'modified']:
		# 	print(md)
		# 	st = time.time()
		# 	embedding = manifold.LocallyLinearEmbedding(n_neighbors, n_components, eigen_solver='auto', method=md)
		# 	Y_off = embedding.fit_transform(np.matrix(patch_off))
		# 	print(time.time()-st)
		# 	st = time.time()
		# 	d2_geo = embedding.transform(np.matrix(patch_off))
		# 	print(time.time()-st)
		# 	d2_geo = [[val[0], val[1], 0] for val in d2_geo]
		# 	pc_vis(d2_geo, patch_off_color)


		
		# pc_vis([[pt[0], pt[1], 0] for pt in Y_off], patch_off_color)

		# [d2_geo1, reconstruction_err1, embedding1] = isomap_based_dimension_reduction(patch_geo, patch_off, len(patch_off) > 8)
		# pc_vis(patch_geo + d2_geo1, patch_rgb + patch_rgb)
		# if len(patch_off) < 1000:
		# 	file_id = res_folder + "//" + str(label)
			# [d2_geo1, reconstruction_err1, embedding1] = isomap_based_dimension_reduction(patch_geo, patch_off, len(patch_off) > 8)
			# try:
			# 	test_rf2(patch_geo, d2_geo1, patch_rgb, file_id)
			# except:
			# 	print()
			# pc_vis(patch_geo + d2_geo1, patch_rgb + patch_rgb)















		# patch_hks = get_hks_map(patch_off)

		# [err_orig_err, err_arr] = compute_distortion(patch_off, seg_thresh, vis = 1)
		# dis_pt_idx = detect_distortion_pts([], patch_off, patch_off, err_arr, seg_thresh, N_r = 6, vis = 1)




		#doane
		# counts, bins, bars = plt.hist(patch_hks, bins = "doane")

		# label_dic = dict()
		# for j in range(len(bins)-1):
		# 	label_dic[j] = []
		# bins[-1] = bins[-1]*1.001

		# for i in range(len(patch_hks)):
		# 	label = 0
		# 	for j in range(len(bins)-1):
		# 		if bins[j]<=patch_hks[i] and patch_hks[i] < bins[j+1]:
		# 			label = j
		# 			break
		# 	if not label in label_dic:
		# 		label_dic[label] = []
		# 	label_dic[label].append(i)


		# all_cluster = []
		# for cluster in label_dic:
		# 	seg_idx_arr = label_dic[cluster]
		# 	seg_pt_arr = [patch_off[idx] for idx in seg_idx_arr]
		# 	if len(seg_pt_arr):
		# 		dbscan_dic = dbscan_clustering(seg_pt_arr, dbscan_thresh)
		# 		for seg_cl in dbscan_dic:
		# 			all_cluster.append([seg_idx_arr[id] for id in dbscan_dic[seg_cl]])

		# vis_geo = []
		# vis_rgb = []
		# new_label_arr = [0 for i in range(len(patch_off))]
		# all_cluster_dic = dict()

		# for t in range(len(all_cluster)):
		# 	idx_arr = all_cluster[t]
		# 	for idx in idx_arr:
		# 		new_label_arr[idx] = t
		# 	all_cluster_dic[t] = idx_arr

		# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		# 	vis_geo = vis_geo + [patch_off[idx] for idx in idx_arr]
		# 	vis_rgb = vis_rgb + [rgb for idx in idx_arr]

		# pc_vis(vis_geo, vis_rgb)

##############################################
def isomap_based_dimension_reduction_new(patch_geo_arr, patch_off_arr, landmark_flag):
	if landmark_flag:
		n_neighbors = min(16, len(patch_off_arr)-1)
		n_components = 2
		X_off8 = np.matrix(patch_off_arr)
		embedding = manifold.Isomap(n_neighbors, n_components, eigen_solver='dense')
		Y_off8 = embedding.fit_transform(X_off8)
		reconstruction_err = embedding.reconstruction_error()
		Y = []
		if len(patch_geo_arr)>5000:
			for t in range(0, len(patch_geo_arr), 5000):
				sub_non_smooth_sc_geo_arr = patch_geo_arr[t:t+5000]
				sub_X = np.matrix(sub_non_smooth_sc_geo_arr)
				sub_Y = embedding.transform(sub_X)
				for val in sub_Y:
					Y.append(val)
		else:
			X = np.matrix(patch_geo_arr)
			Y = embedding.transform(X)
		d2_geo = [[val[0], val[1], 0] for val in Y]
		return [d2_geo, reconstruction_err, embedding]
	else:
		n_neighbors = min(16, len(patch_geo_arr)-1)
		n_components = 2
		embedding = manifold.Isomap(n_neighbors, n_components, eigen_solver='dense')
		X = np.matrix(patch_geo_arr)
		Y = embedding.fit_transform(X)
		reconstruction_err = embedding.reconstruction_error()
		d2_geo = [[val[0], val[1], 0] for val in Y]
		return [d2_geo, reconstruction_err, embedding]

def vis_off_distortion(sub_patch_off_geo_arr_d2, sub_patch_off_geo_arr, pc_width):
	X_3d = np.array(sub_patch_off_geo_arr)
	tree_3d = BallTree(X_3d, leaf_size=1)
	X_2d = np.array(sub_patch_off_geo_arr_d2)
	tree_2d = BallTree(X_2d, leaf_size=1)
	nei_num = min(len(X_2d)-1, 8)
	inter_cnt_arr = []
	colors = []
	valid_index_arr = []
	non_valid_index_arr = []
	for idx in range(len(sub_patch_off_geo_arr)):
		pt_3d = sub_patch_off_geo_arr[idx]
		dist_3d, ind_3d = tree_3d.query([pt_3d], k=nei_num)
		pt_2d = sub_patch_off_geo_arr_d2[idx]
		dist_2d, ind_2d = tree_2d.query([pt_2d], k=nei_num)
		cnt = len(list(set(ind_3d[0]) & set(ind_2d[0])))
		inter_cnt_arr.append(cnt)
		if cnt<nei_num*0.75:
			colors.append([255, 0, 0])
			non_valid_index_arr.append(idx)
		else:
			colors.append([0, 0, 255])
			valid_index_arr.append(idx)
	return [valid_index_arr, non_valid_index_arr, colors]

def distortion_based_sv_segmentation(sv_iso_off_arr, sv_seg_thresh, vis_flag):
	dbscan_dic = dbscan_clustering(sv_iso_off_arr, sv_seg_thresh)
	temp_non_valid_off_geo_all = []
	for label in dbscan_dic:
		temp_non_valid_off_geo_all.append([sv_iso_off_arr[idx] for idx in dbscan_dic[label]])
	# temp_non_valid_off_geo_all = [sv_iso_off_arr]
	tf_iso_off_arr = []
	untf_iso_off_arr = []
	off_embedding_arr = []
	flag = 1
	while flag:
		new_geo_temp = []
		for t in range(len(temp_non_valid_off_geo_all)):
			sub_geo = temp_non_valid_off_geo_all[t]
			if len(sub_geo) < min_patch_pt_num:
				untf_iso_off_arr.append(sub_geo)
			else:
				[sub_patch_off_geo_arr_d2, sub_off_recon_err, off_embedding] = isomap_based_dimension_reduction_new(sub_geo, sub_geo, len(sub_geo)>8)
				[valid_index_arr, non_valid_index_arr, off_distortion_color] = vis_off_distortion(sub_patch_off_geo_arr_d2, sub_geo, pc_width)
				if len(non_valid_index_arr) < ut_iso_th:
					tf_iso_off_arr.append(sub_geo)
					off_embedding_arr.append(off_embedding)
				else:
					sub_cluster_dic = aggl_clustering(sub_geo, neighbour_num = 5, vis_flag = 0)
					for off_id in sub_cluster_dic:
						subsub_geo = [sub_geo[idx] for idx in sub_cluster_dic[off_id]]
						if len(subsub_geo) < min_patch_pt_num:
							untf_iso_off_arr.append(subsub_geo)
							off_embedding_arr.append(None)
						else:
							new_geo_temp.append(subsub_geo)
		temp_non_valid_off_geo_all = new_geo_temp
		if len(temp_non_valid_off_geo_all)==0:
			flag = 0


	cluster_pt_dic = dict()
	cnt = 0
	for patch_off in tf_iso_off_arr:
		if len(patch_off)>3:
				cluster_pt_dic[cnt] = patch_off
				cnt = cnt + 1

	for patch_off in untf_iso_off_arr:
		if len(patch_off)>3:
			cluster_pt_dic[cnt] = patch_off
			cnt = cnt + 1

	if vis_flag:
		vis_geo = []
		vis_rgb = []
		for patch_off in tf_iso_off_arr:
			rgb = [0, randrange(0, 255), randrange(0, 255)]
			vis_geo = vis_geo + patch_off
			vis_rgb = vis_rgb + [rgb for pt in patch_off]
			
		for patch_off in untf_iso_off_arr:
			rgb = [randrange(0, 255), 0, 0]
			vis_geo = vis_geo + patch_off
			vis_rgb = vis_rgb + [rgb for pt in patch_off]
			
		pc_vis(vis_geo, vis_rgb)
	return [cluster_pt_dic, tf_iso_off_arr, untf_iso_off_arr, off_embedding_arr]

def distortion_based_unfold(all_final_seg_arr, all_final_seg_label, all_cluster, sv_off_geo_arr):
	vis_geo = []
	vis_rgb = []
	for t in range(len(all_final_seg_arr)):

		rgb = [67, 89, 203]
		rgb =  [randrange(37, 97), randrange(59, 119), randrange(173, 233)]
		if all_final_seg_label[t] == 1:
			rgb =  [randrange(150, 255), randrange(0, 1), randrange(0, 1)]
			rgb = [221, 210, 219]
			rgb =  [randrange(201, 241), randrange(190, 230), randrange(199, 229)]
		if all_final_seg_label[t] == 2:
			rgb =  [randrange(0, 1), randrange(0, 1), randrange(150, 255)]
			rgb = [205, 65, 58]
			rgb =  [randrange(185, 225), randrange(45, 85), randrange(38, 78)]

		temp_pt = []
		for seg in all_final_seg_arr[t]:
			for idx in all_cluster[seg]:
				vis_geo.append(sv_off_geo_arr[idx])
				vis_rgb.append(rgb)
				temp_pt.append(sv_off_geo_arr[idx])
		distortion_based_sv_segmentation(temp_pt, 1)

	# pc_vis(vis_geo, vis_rgb)

def normal_based_biseg(geo_arr, rgb_arr, normal_arr):
	cluster_dic = dict()
	biseg_path = floder + "seg_assign//" + frame_id + "_biseg.txt"
	if not os.path.exists(biseg_path):
		mid_iso_num = 16
		mid_svoff_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(mid_iso_num) + ".off"
		[mid_sv_off_geo_arr, mid_sv_off_rgb_arr, mid_sv_off_pt_num] = read_off(mid_svoff_path, geo_arr, rgb_arr)

		mid_normal_arr = get_off_normal(geo_arr, normal_arr, mid_sv_off_geo_arr)
		cube_clustering_dic = cube_clustering(mid_sv_off_geo_arr, mid_normal_arr, vis_flag = 0)

		seg_thresh = get_neighbor_dis(mid_sv_off_geo_arr) * 1.5

		patch0 = []
		patch1 = []
		for label in cube_clustering_dic:
			if label < 3:
				for idx in cube_clustering_dic[label]:
					patch0.append(mid_sv_off_geo_arr[idx])
			else:
				for idx in cube_clustering_dic[label]:
					patch1.append(mid_sv_off_geo_arr[idx])
		
		dbscan_dic0 = dbscan_clustering(patch0, seg_thresh)
		dbscan_dic1 = dbscan_clustering(patch1, seg_thresh)
		
		features = []
		labels = []
		cluster_dic = dict()
		cluster_dic[0] = []
		cluster_dic[1] = []
		for seg_cl in dbscan_dic0:
			sub_patch = [patch0[id] for id in dbscan_dic0[seg_cl]]
			if len(sub_patch) > 64:
				features = features + sub_patch
				labels = labels + [0 for t in range(len(sub_patch))]
				
		for seg_cl in dbscan_dic1:
			sub_patch = [patch1[id] for id in dbscan_dic1[seg_cl]]
			if len(sub_patch) > 64:
				features = features + sub_patch
				labels = labels + [1 for t in range(len(sub_patch))]
				
		clf = KNeighborsClassifier(n_neighbors = 3, algorithm = 'ball_tree')
		clf.fit(features, labels)
		pre_label = clf.predict(mid_sv_off_geo_arr)
		for t in range(0, len(pre_label)):
			cluster_dic[pre_label[t]].append(t)

		# vis_geo = []
		# vis_rgb = []
		# for label in cluster_dic:
		# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		# 	vis_geo = vis_geo + [mid_sv_off_geo_arr[idx] for idx in cluster_dic[label]]
		# 	vis_rgb = vis_rgb + [rgb for idx in cluster_dic[label]]
		# pc_vis(vis_geo, vis_rgb)

		patch0 = [mid_sv_off_geo_arr[idx] for idx in cluster_dic[0]]
		patch1 = [mid_sv_off_geo_arr[idx] for idx in cluster_dic[1]]
		dbscan_dic0 = dbscan_clustering(patch0, seg_thresh)
		dbscan_dic1 = dbscan_clustering(patch1, seg_thresh)
		features = []
		labels = []
		cluster_dic = dict()
		cluster_dic[0] = []
		cluster_dic[1] = []
		for seg_cl in dbscan_dic0:
			sub_patch = [patch0[id] for id in dbscan_dic0[seg_cl]]
			if len(sub_patch) > 64:
				features = features + sub_patch
				labels = labels + [0 for t in range(len(sub_patch))]
				
		for seg_cl in dbscan_dic1:
			sub_patch = [patch1[id] for id in dbscan_dic1[seg_cl]]
			if len(sub_patch) > 64:
				features = features + sub_patch
				labels = labels + [1 for t in range(len(sub_patch))]
				
		clf = KNeighborsClassifier(n_neighbors = 16, algorithm = 'ball_tree')
		clf.fit(features, labels)
		pre_label = clf.predict(mid_sv_off_geo_arr)
		for t in range(0, len(pre_label)):
			cluster_dic[pre_label[t]].append(t)

		# vis_geo = []
		# vis_rgb = []
		# for label in cluster_dic:
		# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		# 	vis_geo = vis_geo + [mid_sv_off_geo_arr[idx] for idx in cluster_dic[label]]
		# 	vis_rgb = vis_rgb + [rgb for idx in cluster_dic[label]]
		# pc_vis(vis_geo, vis_rgb)

		patch0 = [mid_sv_off_geo_arr[idx] for idx in cluster_dic[0]]
		patch1 = [mid_sv_off_geo_arr[idx] for idx in cluster_dic[1]]
		dbscan_dic0 = dbscan_clustering(patch0, seg_thresh)
		dbscan_dic1 = dbscan_clustering(patch1, seg_thresh)
		features = []
		labels = []
		cluster_dic = dict()
		cluster_dic[0] = []
		cluster_dic[1] = []
		for seg_cl in dbscan_dic0:
			sub_patch = [patch0[id] for id in dbscan_dic0[seg_cl]]
			if len(sub_patch) > 64:
				features = features + sub_patch
				labels = labels + [0 for t in range(len(sub_patch))]
				
		for seg_cl in dbscan_dic1:
			sub_patch = [patch1[id] for id in dbscan_dic1[seg_cl]]
			if len(sub_patch) > 64:
				features = features + sub_patch
				labels = labels + [1 for t in range(len(sub_patch))]
				
		clf = KNeighborsClassifier(n_neighbors = 16, algorithm = 'ball_tree')
		clf.fit(features, labels)
		pre_label = clf.predict(mid_sv_off_geo_arr)

		
		for t in range(0, len(pre_label)):
			cluster_dic[pre_label[t]].append(t)
			
		f = open(biseg_path, "w")
		for label in cluster_dic:
			f.write(str(label) + "\t")
			for idx in cluster_dic[label]:
				f.write(str(idx) + " ")
			f.write("\n")
		f.close()


		# vis_geo = []
		# vis_rgb = []
		# for label in cluster_dic:
		# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		# 	vis_geo = vis_geo + [mid_sv_off_geo_arr[idx] for idx in cluster_dic[label]]
		# 	vis_rgb = vis_rgb + [rgb for idx in cluster_dic[label]]
		# pc_vis(vis_geo, vis_rgb)
	else:
		with open(biseg_path) as ins:
			for line in ins:
				rec = line.split("\t")
				label = int(rec[0])
				cluster_dic[label] = []
				for idx in rec[1].replace(" \n", "").split(" "):
					cluster_dic[label].append(int(idx))

	return cluster_dic


def normal_based_sampling(geo_arr, rgb_arr, normal_arr, bi_cluster_dic):
	iso_num = 128
	svoff_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(iso_num) + ".off"
	[sv_off_geo_arr, sv_off_rgb_arr, sv_off_pt_num] = read_off(svoff_path, geo_arr, rgb_arr)

	mid_iso_num = 64
	mid_svoff_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(mid_iso_num) + ".off"
	[mid_sv_off_geo_arr, mid_sv_off_rgb_arr, mid_sv_off_pt_num] = read_off(mid_svoff_path, geo_arr, rgb_arr)
	
	mini_iso_num = 16
	mini_svoff_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(mini_iso_num) + ".off"
	[mini_sv_off_geo_arr, mini_sv_off_rgb_arr, mini_sv_off_pt_num] = read_off(mini_svoff_path, geo_arr, rgb_arr)

	sv_seg_thresh = get_neighbor_dis(sv_off_geo_arr) * 1.5

	# vis_geo = []
	# vis_rgb = []
	# for label in bi_cluster_dic:
	# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
	# 	vis_geo = vis_geo + [mini_sv_off_geo_arr[idx] for idx in bi_cluster_dic[label]]
	# 	vis_rgb = vis_rgb + [rgb for idx in bi_cluster_dic[label]]
	# pc_vis(vis_geo, vis_rgb)

	
	mini_patch0_off =  [mini_sv_off_geo_arr[idx] for idx in bi_cluster_dic[0]]
	mini_patch1_off =  [mini_sv_off_geo_arr[idx] for idx in bi_cluster_dic[1]]

	mini_normal_arr0 = get_off_normal(geo_arr, normal_arr, mini_patch0_off)
	mini_normal_arr1 = get_off_normal(geo_arr, normal_arr, mini_patch1_off)

	mini_sv_off_geo_arr_assign_dic = dict()
	mini_sv_off_geo_arr_assign_dic[0] = mini_patch0_off
	mini_sv_off_geo_arr_assign_dic[1] = mini_patch1_off

	mini_sv_off_geo_dic = assign_ply_to_seg(mini_sv_off_geo_arr_assign_dic, sv_off_geo_arr, vis=0)
	sv_patch0_off = [sv_off_geo_arr[idx] for idx in mini_sv_off_geo_dic[0]]
	sv_patch1_off = [sv_off_geo_arr[idx] for idx in mini_sv_off_geo_dic[1]]

	geo_dic = assign_ply_to_seg(mini_sv_off_geo_arr_assign_dic, geo_arr, vis=0)
	geo_arr0 = [geo_arr[idx] for idx in geo_dic[0]]
	geo_arr1 = [geo_arr[idx] for idx in geo_dic[1]]
	rgb_arr0 = [rgb_arr[idx] for idx in geo_dic[0]]
	rgb_arr1 = [rgb_arr[idx] for idx in geo_dic[1]]


	mini_mid_sv_off_geo_dic = assign_ply_to_seg(mini_sv_off_geo_arr_assign_dic, mid_sv_off_geo_arr, vis=0)
	mid_sv_patch0_off = [mid_sv_off_geo_arr[idx] for idx in mini_mid_sv_off_geo_dic[0]]
	mid_sv_patch1_off = [mid_sv_off_geo_arr[idx] for idx in mini_mid_sv_off_geo_dic[1]]

	[sv_mini_geo_assign0, pre_label] = assign_ply_to_off(sv_patch0_off, mini_patch0_off, vis_flag = 0)
	[sv_mid_geo_assign0, pre_label] = assign_ply_to_off(sv_patch0_off, mid_sv_patch0_off, vis_flag = 0)
	
	[sv_mini_geo_assign1, pre_label] = assign_ply_to_off(sv_patch1_off, mini_patch1_off, vis_flag = 0)
	[sv_mid_geo_assign1, pre_label] = assign_ply_to_off(sv_patch1_off, mid_sv_patch1_off, vis_flag = 0)	

	vis_geo = []
	vis_rgb = []
	for off_id in sv_mini_geo_assign0:
		sv_normal = np.asarray([mini_normal_arr0[idx] for idx in sv_mini_geo_assign0[off_id]])
		nor_std = np.max([np.std(sv_normal[:,0]), np.std(sv_normal[:,1]), np.std(sv_normal[:,2])])
		if nor_std < 0.3:
			vis_geo.append(sv_patch0_off[off_id])
			vis_rgb.append([180, 180, 180])
		elif nor_std >= 0.3 and nor_std < 0.5:
			vis_geo = vis_geo + [mid_sv_patch0_off[idx] for idx in sv_mid_geo_assign0[off_id]]
			vis_rgb = vis_rgb + [[0, 0, 220] for idx in sv_mid_geo_assign0[off_id]]
		else:
			vis_geo = vis_geo + [mini_patch0_off[idx] for idx in sv_mini_geo_assign0[off_id]]
			vis_rgb = vis_rgb + [[0, 220, 0] for idx in sv_mini_geo_assign0[off_id]]
	# pc_vis(vis_geo, vis_rgb)
	land_mark0 = vis_geo


	vis_geo = []
	vis_rgb = []
	for off_id in sv_mini_geo_assign1:
		sv_normal = np.asarray([mini_normal_arr1[idx] for idx in sv_mini_geo_assign1[off_id]])
		if len(sv_normal):
			# print(off_id, len(sv_normal))
			nor_std = np.max([np.std(sv_normal[:,0]), np.std(sv_normal[:,1]), np.std(sv_normal[:,2])])
			if nor_std < 0.3:
				vis_geo.append(sv_patch1_off[off_id])
				vis_rgb.append([180, 180, 180])
			elif nor_std >= 0.3 and nor_std < 0.5:
				vis_geo = vis_geo + [mid_sv_patch1_off[idx] for idx in sv_mid_geo_assign1[off_id]]
				vis_rgb = vis_rgb + [[0, 0, 220] for idx in sv_mid_geo_assign1[off_id]]
			else:
				vis_geo = vis_geo + [mini_patch1_off[idx] for idx in sv_mini_geo_assign1[off_id]]
				vis_rgb = vis_rgb + [[0, 220, 0] for idx in sv_mini_geo_assign1[off_id]]
	# pc_vis(vis_geo, vis_rgb)
	land_mark1 = vis_geo

	print(len(land_mark0), len(land_mark1))

	[cluster_pt_dic0, tf_iso_off_arr0, untf_iso_off_arr0, off_embedding_arr0] = distortion_based_sv_segmentation(land_mark0, sv_seg_thresh, vis_flag = 0)
	cluster_geo_dic0 = assign_ply_to_seg(cluster_pt_dic0, geo_arr0, vis=0)
	for label in cluster_pt_dic0:
	# for label in [19]:
		cluster_off = cluster_pt_dic0[label]
		cluster_off_color = get_off_color(geo_arr, rgb_arr, cluster_off)
		cluster_geo = [geo_arr0[idx] for idx in cluster_geo_dic0[label]]
		cluster_rgb = [rgb_arr0[idx] for idx in cluster_geo_dic0[label]]
		# embedding = off_embedding_arr0[label]
		# print(label, embedding)
		# d2_geo = []
		# if embedding == None:
		# 	n_neighbors = min(16, len(cluster_geo)-1)
		# 	n_components = 2
		# 	embedding = manifold.Isomap(n_neighbors, n_components, eigen_solver='dense')
		# 	X = np.matrix(cluster_geo)
		# 	Y = embedding.fit_transform(X)
		# 	reconstruction_err = embedding.reconstruction_error()
		# 	d2_geo = [[val[0], val[1], 0] for val in Y]
		# else:
		# 	X = np.matrix(cluster_geo)
		# 	Y = embedding.transform(X)
		# 	d2_geo = [[val[0], val[1], 0] for val in Y]

		[d2_geo, sub_off_recon_err, off_embedding] = isomap_based_dimension_reduction_new(cluster_geo, cluster_off, len(cluster_off)>8)
		file_id = res_folder + "//" + str(0) + "_" + str(label)
		try:
			test_rf2(cluster_geo, d2_geo, cluster_rgb, file_id)
		except:
			print()




	[cluster_pt_dic1, tf_iso_off_arr1, untf_iso_off_arr1, off_embedding_arr1] = distortion_based_sv_segmentation(land_mark1, sv_seg_thresh, vis_flag = 0)
	cluster_geo_dic1 = assign_ply_to_seg(cluster_pt_dic1, geo_arr1, vis=0)
	for label in cluster_pt_dic1:
		cluster_off = cluster_pt_dic1[label]
		cluster_off_color = get_off_color(geo_arr, rgb_arr, cluster_off)
		cluster_geo = [geo_arr1[idx] for idx in cluster_geo_dic1[label]]
		cluster_rgb = [rgb_arr1[idx] for idx in cluster_geo_dic1[label]]
		# embedding = off_embedding_arr1[label]
		# d2_geo = []
		# if embedding == None:
		# 	n_neighbors = min(16, len(cluster_geo)-1)
		# 	n_components = 2
		# 	embedding = manifold.Isomap(n_neighbors, n_components, eigen_solver='dense')
		# 	X = np.matrix(cluster_geo)
		# 	Y = embedding.fit_transform(X)
		# 	reconstruction_err = embedding.reconstruction_error()
		# 	d2_geo = [[val[0], val[1], 0] for val in Y]
		# else:
		# 	X = np.matrix(cluster_geo)
		# 	Y = embedding.transform(X)
		# 	d2_geo = [[val[0], val[1], 0] for val in Y]
		# pc_vis(cluster_off + d2_geo, cluster_off_color + cluster_rgb)
		file_id = res_folder + "//" + str(1) + "_" + str(label)
			
		[d2_geo, sub_off_recon_err, off_embedding] = isomap_based_dimension_reduction_new(cluster_geo, cluster_off, len(cluster_off)>8)
	
		try:
			test_rf2(cluster_geo, d2_geo, cluster_rgb, file_id)
		except:
			print()


		# pc_vis(cluster_off + d2_geo, cluster_off_color + cluster_rgb)



	# mini_sv_off_geo_arr_assign_dic = dict()
	# mini_sv_off_geo_arr_assign_dic[0] = patch0_off
	# mini_sv_off_geo_arr_assign_dic[1] = patch1_off
	# sv_off_geo_dic = assign_ply_to_seg(mini_sv_off_geo_arr_assign_dic, geo_arr, vis=0)

	# vis_geo = []
	# vis_rgb = []
	# for idx in sv_off_geo_dic[0]:
	# 	pt = geo_arr[idx]
	# 	vis_geo.append([pt[0] - pc_width/2.0, pt[1], pt[2]])
	# 	vis_rgb.append(rgb_arr[idx])

	# for idx in sv_off_geo_dic[1]:
	# 	pt = geo_arr[idx]
	# 	vis_geo.append([pt[0] + pc_width/2.0, pt[1], pt[2]])
	# 	vis_rgb.append(rgb_arr[idx])
	# pc_vis(vis_geo, vis_rgb)

	# features = patch0_off + patch1_off
	# labels = [0 for pt in patch0_off] + [1 for pt in patch1_off]

	# clf = KNeighborsClassifier(n_neighbors = 3, algorithm = 'ball_tree')
	# clf.fit(features, labels)
	# pre_label = clf.predict(geo_arr)

	# geo_cluster_dic = dict()
	# for t in range(0, len(pre_label)):
	# 	geo_cluster_dic[pre_label[t]].append(t)



	# mid_normal_arr = get_off_normal(geo_arr, normal_arr, mid_sv_off_geo_arr)
	# mini_normal_arr = get_off_normal(geo_arr, normal_arr, mini_sv_off_geo_arr)


	# [mini_geo_assign, pre_label] = assign_ply_to_off(mini_sv_off_geo_arr, geo_arr, vis_flag = 0)

	# [svoff_midi_assign, pre_label] = assign_ply_to_off(sv_off_geo_arr, mid_sv_off_geo_arr, vis_flag = 0)
	# [svoff_mini_assign, pre_label] = assign_ply_to_off(sv_off_geo_arr, mini_sv_off_geo_arr, vis_flag = 0)

	# vis_geo = []
	# vis_rgb = []
	# for off_id in svoff_mini_assign:
	# 	sv_normal = np.asarray([mini_normal_arr[idx] for idx in svoff_mini_assign[off_id]])
	# 	nor_std = np.max([np.std(sv_normal[:,0]), np.std(sv_normal[:,1]), np.std(sv_normal[:,2])])
	# 	print(nor_std)
	# 	if nor_std < 0.3:
	# 		vis_geo.append(sv_off_geo_arr[off_id])
	# 		vis_rgb.append([180, 180, 180])
	# 	elif nor_std >= 0.3 and nor_std < 0.5:
	# 		vis_geo = vis_geo + [mid_sv_off_geo_arr[idx] for idx in svoff_midi_assign[off_id]]
	# 		vis_rgb = vis_rgb + [[0, 0, 220] for idx in svoff_midi_assign[off_id]]
	# 	else:
	# 		vis_geo = vis_geo + [mini_sv_off_geo_arr[idx] for idx in svoff_mini_assign[off_id]]
	# 		vis_rgb = vis_rgb + [[0, 220, 0] for idx in svoff_mini_assign[off_id]]
	# pc_vis(vis_geo, vis_rgb)

def point_projectto_plane(p, center, norm):
	p = np.asarray(p)
	center = np.asarray(center)
	norm = np.asarray(norm)
	v = p-center
	dist = np.dot(v, norm)
	projected_point = p - np.asarray(dist*norm)
	return projected_point

def normal_based_simplification(geo_arr, rgb_arr, normal_arr):
	cube_norm_arr = [[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]]
	normal_cube_path = floder + "seg_assign//" + frame_id + "_normal_cube_assign.txt"
	cluster_dic = dict()
	cluster_label_dic = dict()
	sv_off_geo_arr = geo_arr
	if not os.path.exists(normal_cube_path):
		[cluster_dic, cluster_label_dic, geo_arr2] = normal_clustering2(geo_arr, rgb_arr, normal_arr, floder, frame_id, frame_id_head)
		f = open(normal_cube_path, "w")
		for label in cluster_dic:
			f.write(str(label) + "\t" + str(cluster_label_dic[label]) + "\t")
			for idx in cluster_dic[label]:
				f.write(str(idx) + " ")
			f.write("\n")
	else:
		with open(normal_cube_path) as ins:
			for line in ins:
				rec = line.replace(" \n", "").split("\t")
				label = int(rec[0])
				cube_label = int(rec[1])
				idx_arr = [int(idx) for idx in rec[2].split(" ")]
				cluster_dic[label] = idx_arr
				cluster_label_dic[label] = cube_label

	for label in cluster_dic:
		try:
			cube_label = cluster_label_dic[label]
			norm = cube_norm_arr[cube_label]
			patch_pt = [geo_arr[idx] for idx in cluster_dic[label]]

			patch_rgb = get_off_color(geo_arr, rgb_arr, patch_pt)
			center = np.mean(patch_pt, axis=0)
			proj_p_arr = []
			for idx in cluster_dic[label]:
				p = geo_arr[idx]
				proj_p = point_projectto_plane(p, center, norm)
				proj_p_arr.append(proj_p)

			d2_geo = []
			if cube_label in [0, 3]:
				d2_geo = [[pt[1], pt[2], 0] for pt in proj_p_arr]
				# pc_vis([[0, pt[1], pt[2]] for pt in proj_p_arr], patch_rgb)
			elif cube_label in [1, 4]:
				d2_geo = [[pt[0], pt[2], 0] for pt in proj_p_arr]
				# pc_vis([[pt[0], 0, pt[2]] for pt in proj_p_arr], patch_rgb)
			else:
				d2_geo = [[pt[0], pt[1], 0] for pt in proj_p_arr]
				# pc_vis([[pt[0], pt[1], 0] for pt in proj_p_arr], patch_rgb)

			file_id = res_folder + "//" + str(label)
			print(label, len(d2_geo))


			[d2_geo, sub_off_recon_err, off_embedding] = isomap_based_dimension_reduction_new(cluster_geo, cluster_off, len(cluster_off)>8)


			# test_rf3(patch_pt, d2_geo, patch_rgb, file_id)
		except:
			print("!!!!!!!!!!!!!!!!!!!!!!!!", file_id)

		# [d2_geo1, reconstruction_err1, embedding1] = isomap_based_dimension_reduction(patch_geo, patch_off, len(patch_off) > 8)
		# try:
		# 	test_rf2(patch_geo, d2_geo1, patch_rgb, file_id)
		# except:
		# 	print()
		




	# iso_num = 64
	# svoff_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(iso_num) + ".off"
	# [sv_off_geo_arr, sv_off_rgb_arr, sv_off_pt_num] = read_off(svoff_path, geo_arr, rgb_arr)
	# print(len(sv_off_rgb_arr))
	# # pc_vis(sv_off_geo_arr, sv_off_rgb_arr)
	
	# sparse_sv_pt_num = 512
	# sparse_svoff_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(sparse_sv_pt_num) + ".off"
	# [sparse_sv_off_geo_arr, sparse_sv_off_rgb_arr, sparse_sv_off_pt_num] = read_off(sparse_svoff_path, geo_arr, rgb_arr)

	


	# sv_off_normal_arr = get_off_normal(geo_arr, normal_arr, sv_off_geo_arr)
	# sparse_sv_off_normal_arr = get_off_normal(geo_arr, normal_arr, sparse_sv_off_geo_arr)

	# [hks_pt_arr, hks_feature_arr] = get_hks_feature(geo_arr, rgb_arr, floder, frame_id)
	# tot_num = len(hks_pt_arr)
	# X = np.array(hks_pt_arr)
	# tree = BallTree(X, leaf_size = 1)
	# new_hks_feature_arr = []
	# for pt in sv_off_geo_arr:
	# 	dist, ind = tree.query([pt], k=1)
	# 	idx = ind[0][0]
	# 	new_hks_feature_arr.append(hks_feature_arr[idx])
	# hks_feature_arr = new_hks_feature_arr



	# colors = cm.coolwarm(np.linspace(0, 1, 256))
	# colors = [rgb[0:3] for rgb in colors]
	# colors = colors[::-1]
	# min_hks = min(hks_feature_arr)
	# max_hks = max(hks_feature_arr)
	# hks_rgb = [colors[int(np.round((val-min_hks)/(max_hks-min_hks)*255))][0:3]*255 for val in hks_feature_arr]

	# seg_thresh = get_neighbor_dis(sv_off_geo_arr)*2

	# cluster_pt = [sv_off_geo_arr[idx] for idx in cluster_idx]
	# cluster_normal_arr = get_off_normal(geo_arr, normal_arr, sv_off_geo_arr)
	# cluster_dic = cube_clustering(sv_off_geo_arr, cluster_normal_arr, vis_flag=1)

	# pc_vis(sv_off_geo_arr, hks_rgb)

	# hks_raw_seg(hks_feature_arr, sv_off_geo_arr, vis=1)
	
	# [skeleton_adj_dic, chain_seg_dic, chain_dic, all_cluster_init, end_node, bi_node, multi_node] = hybird_seg2(geo_arr, rgb_arr, floder, frame_id, sv_off_geo_arr, hks_rgb, vis = 1)
	

	# [err_orig_err, err_arr] = compute_distortion(sv_off_geo_arr, seg_thresh, vis = 1)
	# dis_pt_idx = detect_distortion_pts([], sv_off_geo_arr, sparse_sv_off_geo_arr, err_arr, seg_thresh, N_r = 6, vis = 1)
	# spanning_tree_idx_arr = mini_spanning_tree(sv_off_geo_arr, sparse_sv_off_geo_arr, dis_pt_idx, vis = 1)

	# all_cluster = merge_multinode(skeleton_adj_dic, end_node, bi_node, multi_node, all_cluster_init, sv_off_geo_arr)

	# vis_geo = []
	# vis_rgb = []
	# for cluster in all_cluster:
	# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
	# 	for idx in cluster:
	# 		vis_geo.append(sv_off_geo_arr[idx])
	# 		vis_rgb.append(rgb)
	# pc_vis(vis_geo, vis_rgb)

	# for iso_num in [512, 256, 128, 64, 32, 16]:
	# 	svoff_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(iso_num) + ".off"
	# 	[sv_off_geo_arr, sv_off_rgb_arr, sv_off_pt_num] = read_off(svoff_path, geo_arr, rgb_arr)
	# 	print(iso_num, len(sv_off_rgb_arr))
	# 	# pc_vis(sv_off_geo_arr, sv_off_rgb_arr)
	# 	assign_ply_to_off(sv_off_geo_arr, geo_arr, vis_flag=1)


	# mid_iso_num = 64
	# mid_svoff_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(mid_iso_num) + ".off"
	# [mid_sv_off_geo_arr, mid_sv_off_rgb_arr, mid_sv_off_pt_num] = read_off(mid_svoff_path, geo_arr, rgb_arr)
	
	# mini_iso_num = 16
	# mini_svoff_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(mini_iso_num) + ".off"
	# [mini_sv_off_geo_arr, mini_sv_off_rgb_arr, mini_sv_off_pt_num] = read_off(mini_svoff_path, geo_arr, rgb_arr)

	# sv_seg_thresh = get_neighbor_dis(sv_off_geo_arr) * 1.5

def normal_based_simplification2(geo_arr, rgb_arr, normal_arr):
	cube_norm_arr = [[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]]
	normal_cube_path = floder + "seg_assign//" + frame_id + "_normal_cube_assign2.txt"
	cluster_dic = dict()
	cluster_label_dic = dict()
	# sv_off_geo_arr = geo_arr
	if not os.path.exists(normal_cube_path):
	# if 1:
		[cluster_dic, cluster_label_dic, geo_arr2] = normal_clustering2(geo_arr, rgb_arr, normal_arr, floder, frame_id, frame_id_head)
		f = open(normal_cube_path, "w")
		for label in cluster_dic:
			f.write(str(label) + "\t" + str(cluster_label_dic[label]) + "\t")
			for idx in cluster_dic[label]:
				f.write(str(idx) + " ")
			f.write("\n")
	else:
		with open(normal_cube_path) as ins:
			for line in ins:
				rec = line.replace(" \n", "").split("\t")
				label = int(rec[0])
				cube_label = int(rec[1])
				idx_arr = [int(idx) for idx in rec[2].split(" ")]
				cluster_dic[label] = idx_arr
				cluster_label_dic[label] = cube_label


	[sv_off_cluster_dic, sv_off_geo_arr] = normal_clustering2_patch_off(geo_arr, rgb_arr, normal_arr, floder, frame_id, frame_id_head)
	# vis_geo = []
	# vis_rgb = []
	# for label in cluster_dic:
	# 	patch_pt = [sv_off_geo_arr[idx] for idx in sv_off_cluster_dic[label]]
	# 	vis_geo = vis_geo + patch_pt
	# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
	# 	vis_rgb = vis_rgb + [rgb for pt in patch_pt]
	# pc_vis(vis_geo, vis_rgb)

	
	for label in cluster_dic:
		file_id = res_folder + "//" + str(label)
	# for label in [17]:
		try:
			if not os.path.exists(file_id + "_rect_blk_mask32.png"):
				cube_label = cluster_label_dic[label]
				norm = cube_norm_arr[cube_label]
				patch_pt = [geo_arr[idx] for idx in cluster_dic[label]]
				patch_off = [sv_off_geo_arr[idx] for idx in sv_off_cluster_dic[label]]
				patch_rgb = get_off_color(geo_arr, rgb_arr, patch_pt)
				center = np.mean(patch_pt, axis=0)
				proj_p_arr = []
				for idx in cluster_dic[label]:
					p = geo_arr[idx]
					proj_p = point_projectto_plane(p, center, norm)
					proj_p_arr.append(proj_p)

				d2_geo = []
				if cube_label in [0, 3]:
					d2_geo = [[pt[1], pt[2], 0] for pt in proj_p_arr]
				elif cube_label in [1, 4]:
					d2_geo = [[pt[0], pt[2], 0] for pt in proj_p_arr]
				else:
					d2_geo = [[pt[0], pt[1], 0] for pt in proj_p_arr]

				
				print(label, len(patch_off), len(patch_pt))
				# if len(patch_off)<5000:
					# [d2_geo, sub_off_recon_err, off_embedding] = isomap_based_dimension_reduction_new(patch_pt, patch_off, len(patch_off)>8)
				# pc_vis(patch_off + d2_geo + patch_pt, [[255, 0, 0] for pt in patch_off] + patch_rgb + patch_rgb)

			
				test_rf3(patch_pt, d2_geo, patch_rgb, file_id)
		except:
			print("!!!!!!!!!!!!!!!!!!!!!!!!", file_id)

def blk_based_ip(all_img, all_mask, blk_size):
	img_h, img_w = all_mask.shape
	 # = 16
	blk16_ip_grid_img = all_img.copy()
	for h in range(0, img_h, blk_size):
		for w in range(0, img_w, blk_size):
			# print(h, w, img_h, img_w )
			sub_all_img = all_img[h:h+blk_size, w:w+blk_size, :]
			sub_all_mask = all_mask[h:h+blk_size, w:w+blk_size]
			
			# if np.sum(sub_all_mask)>0:
			if not (np.all((sub_all_mask == 0)) or  np.all((sub_all_mask== 255))):
				try:
					sub_ip_grid_img = img_inpaint(sub_all_img, 1-sub_all_mask/255)
					sub_ip_grid_img = np.uint8(sub_ip_grid_img*255.999)
					blk16_ip_grid_img[h:h+blk_size, w:w+blk_size, :] = sub_ip_grid_img
				except:
					continue
			# else:
			# 	blk16_ip_grid_img[h:h+blk_size, w:w+blk_size, :] = sub_all_img
	return blk16_ip_grid_img

def bpg_compression_with_mask(ori_file_path, attr_img, rect_mask_all, pt_num):
	quality_arr = [40, 34, 28, 23]
	ori_yuv_img_y = cv2.cvtColor(attr_img, cv2.COLOR_BGR2YUV)[:,:,0]

	com_file_path = 'out.bpg'
	de_file_path = 'out.png'



	bpp_arr = []
	psnr_arr = []
	for qp in quality_arr:
		yuv_size = 0
		tot_diff = 0.0
		subprocess.call(['E:\\bpg-0.9.8-win64\\bpgenc.exe', ori_file_path, '-o', com_file_path, '-q', str(qp)])
		time.sleep(1)
		subprocess.call(['E:\\bpg-0.9.8-win64\\bpgdec.exe', com_file_path, '-i -o', de_file_path])
		time.sleep(1)
		sub_attr_img_yuv_size = os.stat(com_file_path).st_size
		com_img = cv2.imread(de_file_path)
		compressed_yuv_img_y = cv2.cvtColor(com_img, cv2.COLOR_BGR2YUV)[:,:,0]

		for s in range(0, compressed_yuv_img_y.shape[0]):
			for t in range(0, compressed_yuv_img_y.shape[1]):
				if rect_mask_all[s][t]:
					dif = int(ori_yuv_img_y[s][t]) - int(compressed_yuv_img_y[s][t])
					dif = dif*dif
					tot_diff = tot_diff + dif
		mse = tot_diff/pt_num
		psnr = 20*np.log10(255.0/np.sqrt(mse))
		bpp = sub_attr_img_yuv_size*8.0/pt_num
		bpp_arr.append(bpp)
		psnr_arr.append(psnr)
	return [bpp_arr, psnr_arr]

def blk_rect_packing_bsp(geo_arr, rgb_arr, normal_arr, blk):
	cube_norm_arr = [[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]]
	normal_cube_path = floder + "seg_assign//" + frame_id + "_normal_cube_assign2.txt"
	cluster_dic = dict()
	cluster_label_dic = dict()
	if not os.path.exists(normal_cube_path):
		[cluster_dic, cluster_label_dic, geo_arr2] = normal_clustering2(geo_arr, rgb_arr, normal_arr, floder, frame_id, frame_id_head)
		f = open(normal_cube_path, "w")
		for label in cluster_dic:
			f.write(str(label) + "\t" + str(cluster_label_dic[label]) + "\t")
			for idx in cluster_dic[label]:
				f.write(str(idx) + " ")
			f.write("\n")
	else:
		with open(normal_cube_path) as ins:
			for line in ins:
				rec = line.replace(" \n", "").split("\t")
				label = int(rec[0])
				cube_label = int(rec[1])
				idx_arr = [int(idx) for idx in rec[2].split(" ")]
				cluster_dic[label] = idx_arr
				cluster_label_dic[label] = cube_label
	rectangles = []
	img_info_arr = []
	
	for label in cluster_dic:
			file_id = res_folder + "//" + str(label)
			print(label)
			rect_img = cv2.imread(file_id + "_rect_blk" + str(blk) + ".png")
			rect_mask = cv2.imread(file_id + "_rect_blk_mask" + str(blk) + ".png", 0)
			img_h, img_w, ch = rect_img.shape
			print(img_h, img_w)
			# if label == len(cluster_dic)-1:
			# 	rect_mask = np.ones((img_h, img_w), np.uint8)*255
			img_info_arr.append([img_w, img_h, rect_img, rect_mask, label])
			print(label, img_h, img_w)
			rectangles.append((img_w, img_h))

	positions = []
	w_arr = []
	h_arr = []
	positions = rpack.pack(rectangles)
	for i in range(0, len(positions)):
		w_l = positions[i][0]
		h_t = positions[i][1]
		w_r = w_l + img_info_arr[i][0]
		h_b = h_t + img_info_arr[i][1]
		w_arr.append(w_r)
		h_arr.append(h_b)
	max_w = max(w_arr)
	max_h = max(h_arr)

	print(positions)
	img_all = np.ones((max_h, max_w, 3), np.uint8)*255
	mask_all = np.ones((max_h, max_w), np.uint8)*0
	mask_all2 = np.ones((max_h, max_w), np.uint8)*0
	for i in range(len(positions)):
		w_l = positions[i][0]
		h_t = positions[i][1]
		w_r = w_l + img_info_arr[i][0]
		h_b = h_t + img_info_arr[i][1]
		img_all[h_t:h_b, w_l:w_r] = img_info_arr[i][2]
		mask_all[h_t:h_b, w_l:w_r] = np.ones((img_info_arr[i][1], img_info_arr[i][0]), np.uint8)*255
		mask_all2[h_t:h_b, w_l:w_r] = img_info_arr[i][3]
	blk16_ip_grid_img = blk_based_ip(img_all, mask_all2, blk_size=16)

	type_id = "17_blk" + str(blk) + "_"
	cv2.imwrite("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_all.png", img_all)
	cv2.imwrite("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_mask_all.png", mask_all)
	cv2.imwrite("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_mask_all2.png", mask_all2)
	cv2.imwrite("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_all_ip16.png", blk16_ip_grid_img)

	patch_pt_num = len(geo_arr)

	chain_bpp_arr_all = []
	chain_psnr_arr_all = []
	chain_legend_arr_all = []
	[peer_chain_bpp_arr, peer_chain_psnr_arr, peer_chain_legend_arr] = peer_method(frame_id)
	chain_bpp_arr_all = chain_bpp_arr_all + peer_chain_bpp_arr
	chain_psnr_arr_all = chain_psnr_arr_all + peer_chain_psnr_arr
	chain_legend_arr_all = chain_legend_arr_all + peer_chain_legend_arr

	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(img_all, mask_all, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("rect_mask_" + str(patch_pt_num))

	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(img_all, mask_all2, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("rect_mask2_" + str(patch_pt_num))

	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(blk16_ip_grid_img, mask_all2, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("rect_mask_ip16_" + str(patch_pt_num))

	blk16_ip_grid_img = cv2.imread("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_all.png")
	mask_all2 = cv2.imread("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_mask_all2.png", 0)
	[ball_tree_bpp_arr, ball_tree_psnr_arr] = bpg_compression_with_mask("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_all.png", img_all, mask_all2, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("bpg_" + type_id + str(patch_pt_num))

	blk16_ip_grid_img = cv2.imread("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_all_ip16.png")
	mask_all2 = cv2.imread("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_mask_all2.png", 0)
	[ball_tree_bpp_arr, ball_tree_psnr_arr] = bpg_compression_with_mask("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_all_ip16.png", blk16_ip_grid_img, mask_all2, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("bpg_" + type_id + "ip_" + str(patch_pt_num))

	plt.figure()
	for ttt in range(0,len(chain_bpp_arr_all)):
		plt.plot(chain_bpp_arr_all[ttt], chain_psnr_arr_all[ttt], color = plt_color_arr[ttt], marker = plt_marker_arr[ttt], markerfacecolor="None", linewidth=1)
	plt.xlabel('bpp', fontsize=12)
	plt.ylabel('PSNR', fontsize=12)
	plt.title("all", fontsize=12)
	plt.legend(chain_legend_arr_all, loc="lower right")
	plt.savefig("D://final_res//" + type_id + frame_id + "_" + 'new_final_all.png')
	plt.close()


def orig_rect_packing_bsp(geo_arr, rgb_arr, normal_arr):
	cube_norm_arr = [[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]]
	normal_cube_path = floder + "seg_assign//" + frame_id + "_normal_cube_assign2.txt"
	cluster_dic = dict()
	cluster_label_dic = dict()
	if not os.path.exists(normal_cube_path):
		[cluster_dic, cluster_label_dic, geo_arr2] = normal_clustering2(geo_arr, rgb_arr, normal_arr, floder, frame_id, frame_id_head)
		f = open(normal_cube_path, "w")
		for label in cluster_dic:
			f.write(str(label) + "\t" + str(cluster_label_dic[label]) + "\t")
			for idx in cluster_dic[label]:
				f.write(str(idx) + " ")
			f.write("\n")
	else:
		with open(normal_cube_path) as ins:
			for line in ins:
				rec = line.replace(" \n", "").split("\t")
				label = int(rec[0])
				cube_label = int(rec[1])
				idx_arr = [int(idx) for idx in rec[2].split(" ")]
				cluster_dic[label] = idx_arr
				cluster_label_dic[label] = cube_label
	rectangles = []
	img_info_arr = []
	
	for label in cluster_dic:
			file_id = res_folder + "//" + str(label)
			print(label)
			rect_img = cv2.imread(file_id + "_orig.png")
			rect_mask = cv2.imread(file_id + "_orig_mask.png", 0)
			img_h, img_w, ch = rect_img.shape
			print(img_h, img_w)
			# if label == len(cluster_dic)-1:
			# 	rect_mask = np.ones((img_h, img_w), np.uint8)*255
			img_info_arr.append([img_w, img_h, rect_img, rect_mask, label])
			print(label, img_h, img_w)
			rectangles.append((img_w, img_h))

	positions = []
	w_arr = []
	h_arr = []
	positions = rpack.pack(rectangles)
	for i in range(0, len(positions)):
		w_l = positions[i][0]
		h_t = positions[i][1]
		w_r = w_l + img_info_arr[i][0]
		h_b = h_t + img_info_arr[i][1]
		w_arr.append(w_r)
		h_arr.append(h_b)
	max_w = max(w_arr)
	max_h = max(h_arr)

	print(positions)
	img_all = np.ones((max_h, max_w, 3), np.uint8)*255
	mask_all = np.ones((max_h, max_w), np.uint8)*0
	mask_all2 = np.ones((max_h, max_w), np.uint8)*0
	for i in range(len(positions)):
		w_l = positions[i][0]
		h_t = positions[i][1]
		w_r = w_l + img_info_arr[i][0]
		h_b = h_t + img_info_arr[i][1]
		img_all[h_t:h_b, w_l:w_r] = img_info_arr[i][2]
		mask_all[h_t:h_b, w_l:w_r] = np.ones((img_info_arr[i][1], img_info_arr[i][0]), np.uint8)*255
		mask_all2[h_t:h_b, w_l:w_r] = img_info_arr[i][3]
	blk16_ip_grid_img = blk_based_ip(img_all, mask_all2, blk_size=16)

	type_id = "17_orig_"
	cv2.imwrite("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_all.png", img_all)
	cv2.imwrite("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_mask_all.png", mask_all)
	cv2.imwrite("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_mask_all2.png", mask_all2)
	cv2.imwrite("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_all_ip16.png", blk16_ip_grid_img)

	patch_pt_num = len(geo_arr)

	chain_bpp_arr_all = []
	chain_psnr_arr_all = []
	chain_legend_arr_all = []
	[peer_chain_bpp_arr, peer_chain_psnr_arr, peer_chain_legend_arr] = peer_method(frame_id)
	chain_bpp_arr_all = chain_bpp_arr_all + peer_chain_bpp_arr
	chain_psnr_arr_all = chain_psnr_arr_all + peer_chain_psnr_arr
	chain_legend_arr_all = chain_legend_arr_all + peer_chain_legend_arr

	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(img_all, mask_all, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("rect_mask_" + str(patch_pt_num))

	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(img_all, mask_all2, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("rect_mask2_" + str(patch_pt_num))

	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(blk16_ip_grid_img, mask_all2, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("rect_mask_ip16_" + str(patch_pt_num))

	blk16_ip_grid_img = cv2.imread("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_all.png")
	mask_all2 = cv2.imread("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_mask_all2.png", 0)
	[ball_tree_bpp_arr, ball_tree_psnr_arr] = bpg_compression_with_mask("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_all.png", img_all, mask_all2, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("bpg_" + type_id + str(patch_pt_num))

	blk16_ip_grid_img = cv2.imread("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_all_ip16.png")
	mask_all2 = cv2.imread("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_mask_all2.png", 0)
	[ball_tree_bpp_arr, ball_tree_psnr_arr] = bpg_compression_with_mask("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_all_ip16.png", blk16_ip_grid_img, mask_all2, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("bpg_" + type_id + "ip_" + str(patch_pt_num))

	plt.figure()
	for ttt in range(0,len(chain_bpp_arr_all)):
		plt.plot(chain_bpp_arr_all[ttt], chain_psnr_arr_all[ttt], color = plt_color_arr[ttt], marker = plt_marker_arr[ttt], markerfacecolor="None", linewidth=1)
	plt.xlabel('bpp', fontsize=12)
	plt.ylabel('PSNR', fontsize=12)
	plt.title("all", fontsize=12)
	plt.legend(chain_legend_arr_all, loc="lower right")
	plt.savefig("D://final_res//" + type_id + frame_id + "_" + 'new_final_all.png')
	plt.close()

def rect_packing_bsp(geo_arr, rgb_arr, normal_arr):
	cube_norm_arr = [[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]]
	normal_cube_path = floder + "seg_assign//" + frame_id + "_normal_cube_assign2.txt"
	cluster_dic = dict()
	cluster_label_dic = dict()
	if not os.path.exists(normal_cube_path):
		[cluster_dic, cluster_label_dic, geo_arr2] = normal_clustering2(geo_arr, rgb_arr, normal_arr, floder, frame_id, frame_id_head)
		f = open(normal_cube_path, "w")
		for label in cluster_dic:
			f.write(str(label) + "\t" + str(cluster_label_dic[label]) + "\t")
			for idx in cluster_dic[label]:
				f.write(str(idx) + " ")
			f.write("\n")
	else:
		with open(normal_cube_path) as ins:
			for line in ins:
				rec = line.replace(" \n", "").split("\t")
				label = int(rec[0])
				cube_label = int(rec[1])
				idx_arr = [int(idx) for idx in rec[2].split(" ")]
				cluster_dic[label] = idx_arr
				cluster_label_dic[label] = cube_label
	rectangles = []
	img_info_arr = []
	
	for label in cluster_dic:
			file_id = res_folder + "//" + str(label)
			print(label)
			rect_img = cv2.imread(file_id + "_rect.png")
			rect_mask = cv2.imread(file_id + "_rect_mask.png", 0)
			img_h, img_w, ch = rect_img.shape
			print(img_h, img_w)
			if label == len(cluster_dic)-1:
				rect_mask = np.ones((img_h, img_w), np.uint8)*255
			img_info_arr.append([img_w, img_h, rect_img, rect_mask, label])
			print(label, img_h, img_w)
			rectangles.append((img_w, img_h))

	positions = []
	w_arr = []
	h_arr = []
	positions = rpack.pack(rectangles)
	for i in range(0, len(positions)):
		w_l = positions[i][0]
		h_t = positions[i][1]
		w_r = w_l + img_info_arr[i][0]
		h_b = h_t + img_info_arr[i][1]
		w_arr.append(w_r)
		h_arr.append(h_b)
	max_w = max(w_arr)
	max_h = max(h_arr)

	print(positions)
	img_all = np.ones((max_h, max_w, 3), np.uint8)*255
	mask_all = np.ones((max_h, max_w), np.uint8)*0
	mask_all2 = np.ones((max_h, max_w), np.uint8)*0
	for i in range(len(positions)):
		w_l = positions[i][0]
		h_t = positions[i][1]
		w_r = w_l + img_info_arr[i][0]
		h_b = h_t + img_info_arr[i][1]
		img_all[h_t:h_b, w_l:w_r] = img_info_arr[i][2]
		mask_all[h_t:h_b, w_l:w_r] = np.ones((img_info_arr[i][1], img_info_arr[i][0]), np.uint8)*255
		mask_all2[h_t:h_b, w_l:w_r] = img_info_arr[i][3]
	blk16_ip_grid_img = blk_based_ip(img_all, mask_all2, blk_size=16)

	type_id = "17_rect_"
	cv2.imwrite("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_all.png", img_all)
	cv2.imwrite("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_mask_all.png", mask_all)
	cv2.imwrite("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_mask_all2.png", mask_all2)
	cv2.imwrite("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_all_ip16.png", blk16_ip_grid_img)

	patch_pt_num = len(geo_arr)

	chain_bpp_arr_all = []
	chain_psnr_arr_all = []
	chain_legend_arr_all = []
	[peer_chain_bpp_arr, peer_chain_psnr_arr, peer_chain_legend_arr] = peer_method(frame_id)
	chain_bpp_arr_all = chain_bpp_arr_all + peer_chain_bpp_arr
	chain_psnr_arr_all = chain_psnr_arr_all + peer_chain_psnr_arr
	chain_legend_arr_all = chain_legend_arr_all + peer_chain_legend_arr

	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(img_all, mask_all, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("rect_mask_" + str(patch_pt_num))

	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(img_all, mask_all2, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("rect_mask2_" + str(patch_pt_num))

	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(blk16_ip_grid_img, mask_all2, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("rect_mask_ip16_" + str(patch_pt_num))

	blk16_ip_grid_img = cv2.imread("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_all.png")
	mask_all2 = cv2.imread("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_mask_all2.png", 0)
	[ball_tree_bpp_arr, ball_tree_psnr_arr] = bpg_compression_with_mask("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_all.png", img_all, mask_all2, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("bpg_" + type_id + str(patch_pt_num))

	blk16_ip_grid_img = cv2.imread("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_all_ip16.png")
	mask_all2 = cv2.imread("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_mask_all2.png", 0)
	[ball_tree_bpp_arr, ball_tree_psnr_arr] = bpg_compression_with_mask("D://final_res//" + type_id + frame_id + "_" + "all_rect_pack_all_ip16.png", blk16_ip_grid_img, mask_all2, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("bpg_" + type_id + "ip_" + str(patch_pt_num))

	plt.figure()
	for ttt in range(0,len(chain_bpp_arr_all)):
		plt.plot(chain_bpp_arr_all[ttt], chain_psnr_arr_all[ttt], color = plt_color_arr[ttt], marker = plt_marker_arr[ttt], markerfacecolor="None", linewidth=1)
	plt.xlabel('bpp', fontsize=12)
	plt.ylabel('PSNR', fontsize=12)
	plt.title("all", fontsize=12)
	plt.legend(chain_legend_arr_all, loc="lower right")
	plt.savefig("D://final_res//" + type_id + frame_id + "_" + 'new_final_all.png')
	plt.close()

def rect_packing(geo_arr, rgb_arr, normal_arr):
	cube_norm_arr = [[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]]
	normal_cube_path = floder + "seg_assign//" + frame_id + "_normal_cube_assign2.txt"
	cluster_dic = dict()
	cluster_label_dic = dict()
	if not os.path.exists(normal_cube_path):
		[cluster_dic, cluster_label_dic, geo_arr2] = normal_clustering2(geo_arr, rgb_arr, normal_arr, floder, frame_id, frame_id_head)
		f = open(normal_cube_path, "w")
		for label in cluster_dic:
			f.write(str(label) + "\t" + str(cluster_label_dic[label]) + "\t")
			for idx in cluster_dic[label]:
				f.write(str(idx) + " ")
			f.write("\n")
	else:
		with open(normal_cube_path) as ins:
			for line in ins:
				rec = line.replace(" \n", "").split("\t")
				label = int(rec[0])
				cube_label = int(rec[1])
				idx_arr = [int(idx) for idx in rec[2].split(" ")]
				cluster_dic[label] = idx_arr
				cluster_label_dic[label] = cube_label


	# [sv_off_cluster_dic, sv_off_geo_arr] = normal_clustering2_patch_off(geo_arr, rgb_arr, normal_arr, floder, frame_id, frame_id_head)
	# vis_geo = []
	# vis_rgb = []
	# for label in cluster_dic:
	# 	patch_pt = [sv_off_geo_arr[idx] for idx in sv_off_cluster_dic[label]]
	# 	vis_geo = vis_geo + patch_pt
	# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
	# 	vis_rgb = vis_rgb + [rgb for pt in patch_pt]
	# pc_vis(vis_geo, vis_rgb)

	rectangles = []
	img_info_arr = []
	
	for label in cluster_dic:
			file_id = res_folder + "//" + str(label)
			print(label)
			rect_img = cv2.imread(file_id + "_rect.png")
			rect_mask = cv2.imread(file_id + "_rect_mask.png", 0)
			img_h, img_w, ch = rect_img.shape
			print(img_h, img_w)
			if label == len(cluster_dic)-1:
				rect_mask = np.ones((img_h, img_w), np.uint8)*255
			img_info_arr.append([img_w, img_h, rect_img, rect_mask, label])
			print(label, img_h, img_w)
			rectangles.append((img_w, img_h))

	positions = []
	w_arr = []
	h_arr = []
	positions = rpack.pack(rectangles)
	for i in range(0, len(positions)):
		w_l = positions[i][0]
		h_t = positions[i][1]
		w_r = w_l + img_info_arr[i][0]
		h_b = h_t + img_info_arr[i][1]
		w_arr.append(w_r)
		h_arr.append(h_b)
	max_w = max(w_arr)
	max_h = max(h_arr)

	print(positions)
	img_all = np.ones((max_h, max_w, 3), np.uint8)*255
	mask_all = np.ones((max_h, max_w), np.uint8)*0
	mask_all2 = np.ones((max_h, max_w), np.uint8)*0
	for i in range(len(positions)):
		w_l = positions[i][0]
		h_t = positions[i][1]
		w_r = w_l + img_info_arr[i][0]
		h_b = h_t + img_info_arr[i][1]
		img_all[h_t:h_b, w_l:w_r] = img_info_arr[i][2]
		mask_all[h_t:h_b, w_l:w_r] = np.ones((img_info_arr[i][1], img_info_arr[i][0]), np.uint8)*255
		mask_all2[h_t:h_b, w_l:w_r] = img_info_arr[i][3]
	# cv2.imshow("img_all", img_all)
	# cv2.waitKey(0)
	# ip_grid_img_rect = img_inpaint(img_all, 1-mask_all/255)
	# ip_grid_img_rect = np.uint8(ip_grid_img_rect*255.999)

	# ip_grid_img_rect2 = img_inpaint(img_all, 1-mask_all2/255)
	# ip_grid_img_rect2 = np.uint8(ip_grid_img_rect2*255.999)
	blk16_ip_grid_img = blk_based_ip(img_all, mask_all2, blk_size=16)


	patch_pt_num = len(geo_arr)

	chain_bpp_arr_all = []
	chain_psnr_arr_all = []
	chain_legend_arr_all = []
	[peer_chain_bpp_arr, peer_chain_psnr_arr, peer_chain_legend_arr] = peer_method(frame_id)
	chain_bpp_arr_all = chain_bpp_arr_all + peer_chain_bpp_arr
	chain_psnr_arr_all = chain_psnr_arr_all + peer_chain_psnr_arr
	chain_legend_arr_all = chain_legend_arr_all + peer_chain_legend_arr

	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(img_all, mask_all, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("rect_mask_" + str(patch_pt_num))

	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(img_all, mask_all2, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("rect_mask2_" + str(patch_pt_num))

	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(blk16_ip_grid_img, mask_all2, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("rect_mask_ip16_" + str(patch_pt_num))

	# [ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(ip_grid_img_rect, mask_all2, patch_pt_num)
	# chain_bpp_arr_all.append(ball_tree_bpp_arr)
	# chain_psnr_arr_all.append(ball_tree_psnr_arr)
	# chain_legend_arr_all.append("rect_mask_ip_" + str(patch_pt_num))

	# [ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(ip_grid_img_rect2, mask_all2, patch_pt_num)
	# chain_bpp_arr_all.append(ball_tree_bpp_arr)
	# chain_psnr_arr_all.append(ball_tree_psnr_arr)
	# chain_legend_arr_all.append("rect_mask2_ip_" + str(patch_pt_num))	


	cv2.imwrite(res_folder + "all_rect_pack_all.png", img_all)
	cv2.imwrite(res_folder + "all_rect_pack_mask_all.png", mask_all)
	cv2.imwrite(res_folder + "all_rect_pack_all_ip16.png", blk16_ip_grid_img)
	# cv2.imwrite(file_id + "_rect_pack_mask_all_ip.png", ip_grid_img_rect)
	# cv2.imwrite(file_id + "_rect_pack_mask_all_ip2.png", ip_grid_img_rect2)

	plt.figure()
	for ttt in range(0,len(chain_bpp_arr_all)):
		plt.plot(chain_bpp_arr_all[ttt], chain_psnr_arr_all[ttt], color = plt_color_arr[ttt], marker = plt_marker_arr[ttt], markerfacecolor="None", linewidth=1)
	plt.xlabel('bpp', fontsize=12)
	plt.ylabel('PSNR', fontsize=12)
	plt.title("all", fontsize=12)
	plt.legend(chain_legend_arr_all, loc="lower right")
	plt.savefig(res_folder + "//" + 'new_all2.png')
	plt.close()


def find_contour(img, mask, bias_w, bias_h, seg_id, blk_size):
	#https://stackoverflow.com/questions/47936474/is-there-a-function-similar-to-opencv-findcontours-that-detects-curves-and-repla
	img_h, img_w = mask.shape

	temp_mask = np.zeros((img_h, img_w), np.uint8)
	if blk_size:
		for h in range(0, img_h, blk_size):
			for w in range(0, img_w, blk_size):
				sub_all_mask = mask[h:h+blk_size, w:w+blk_size]
				if np.sum(sub_all_mask)>0:
					temp_mask[h:h+blk_size, w:w+blk_size] = 255
		mask = temp_mask


	# temp_mask = np.zeros((img_h+2, img_w+2), np.uint8)
	# temp_mask[1:img_h+1, 1:img_w+1] = mask

	# inv_mask = np.ones((img_h, img_w), np.uint8)*255 - mask
	# kernel = np.ones((3,3), np.uint8) 
	# mask_dilation = cv2.dilate(inv_mask, kernel, iterations=2)

	cnts = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[-2]

	# get the max-area contour
	cnt = sorted(cnts, key=cv2.contourArea)[-1]
	# calc arclentgh
	arclen = cv2.arcLength(cnt, True)
	# print(arclen)
	# do approx
	eps = 0.0005

	epsilon = arclen * eps
	approx = cv2.approxPolyDP(cnt, epsilon, True)

	# img_path = "img3/" + frame_id + '_cmp20191008_' + seg_id + '_img_ip.png'
	# img = cv2.imread(img_path)
	# canvas = img.copy()
	path = "<path fill=\"none\" stroke=\"#010101\"  d=\""
	path_str = ""
	# for pt in approx:
	for i in range(len(approx)):
		pt = approx[i]
		if i == 0:
			path = path + "M" + str(pt[0][0] + bias_w) + " " + str(pt[0][1] + bias_h) + " "
			path_str = path_str + "M" + str(pt[0][0] + bias_w) + " " + str(pt[0][1] + bias_h) + " "
		elif i == len(approx)-1:
			path = path + "L" + str(pt[0][0] + bias_w) + " " + str(pt[0][1] + bias_h) + " z\"/>"
			path_str = path_str + "L" + str(pt[0][0] + bias_w) + " " + str(pt[0][1] + bias_h) + " z"
		else:
			path = path + "L" + str(pt[0][0] + bias_w) + " " + str(pt[0][1] + bias_h) + " "
			path_str = path_str + "L" + str(pt[0][0] + bias_w) + " " + str(pt[0][1] + bias_h) + " "
	print(path)
		# cv2.circle(canvas, (pt[0][0], pt[0][1]), 7, (0,255,0), -1)
	# cv2.drawContours(canvas, [approx], -1, (0,0,255), 2, cv2.LINE_AA)
	# cv2.imshow("canvas", canvas)
	# cv2.waitKey(1000)
	# cv2.destroyAllWindows()
	# return canvas
	return path_str

def parse_svg():

	svg_path = res_folder + "//" + "SVGnest-output.svg"
	svg_str = ""
	with open(svg_path, 'r') as ins:
		for line in ins:
			if len(line.replace("\n","")):
				svg_str = line.replace("\n","")
	pattern = re.compile("<g transform.*?z")

	layout_info = []
	pix_cnt = 0
	proj_pt_cnt = 0
	x = re.findall(pattern, svg_str) 
	for val in x:
		trans_pattern = re.compile("translate.*?rotate")
		rot_pattern = re.compile("rotate.*?\"")
		path_pattern = re.compile("d=.*?z")
		trans_pattern_val = re.findall(trans_pattern, val)
		rot_pattern_val = re.findall(rot_pattern, val)
		path_pattern_val = re.findall(path_pattern, val)

		trans_val = trans_pattern_val[0].replace("translate(", "").replace(") rotate","").split(" ")
		[translate_x, translate_y] = [int(np.round(float(trans_v), 0)) for trans_v in trans_val]
		
		# print(trans_val)
		angle = int(rot_pattern_val[0].replace("rotate(", "").replace(")\"", ""))
		# print(angle, translate_x, translate_y)
		path = path_pattern_val[0].replace("d=\"", "")
		print(path)
		layout_info.append([angle, translate_x, translate_y, path])
	return layout_info

def irregular_packing(geo_arr, rgb_arr, normal_arr):
	cube_norm_arr = [[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]]
	normal_cube_path = floder + "seg_assign//" + frame_id + "_normal_cube_assign2.txt"
	cluster_dic = dict()
	cluster_label_dic = dict()
	if not os.path.exists(normal_cube_path):
		[cluster_dic, cluster_label_dic, geo_arr2] = normal_clustering2(geo_arr, rgb_arr, normal_arr, floder, frame_id, frame_id_head)
		f = open(normal_cube_path, "w")
		for label in cluster_dic:
			f.write(str(label) + "\t" + str(cluster_label_dic[label]) + "\t")
			for idx in cluster_dic[label]:
				f.write(str(idx) + " ")
			f.write("\n")
	else:
		with open(normal_cube_path) as ins:
			for line in ins:
				rec = line.replace(" \n", "").split("\t")
				label = int(rec[0])
				cube_label = int(rec[1])
				idx_arr = [int(idx) for idx in rec[2].split(" ")]
				cluster_dic[label] = idx_arr
				cluster_label_dic[label] = cube_label
	rectangles = []
	img_info_arr = []
	
	for label in cluster_dic:
			file_id = res_folder + "//" + str(label)
			print(label)
			rect_img = cv2.imread(file_id + "_rect.png")
			rect_mask = cv2.imread(file_id + "_rect_mask.png", 0)

			orig_img = cv2.imread(file_id + "_orig.png")
			orig_mask = cv2.imread(file_id + "_orig_mask.png", 0)

			rect_blk8 = cv2.imread(file_id + "_rect_blk8.png")
			rect_blk_mask8 = cv2.imread(file_id + "_rect_blk_mask8.png", 0)

			rect_blk16 = cv2.imread(file_id + "_rect_blk16.png")
			rect_blk_mask16 = cv2.imread(file_id + "_rect_blk_mask16.png", 0)

			rect_blk32 = cv2.imread(file_id + "_rect_blk32.png")
			rect_blk_mask32 = cv2.imread(file_id + "_rect_blk_mask32.png", 0)

			
			img = rect_blk16
			mask = rect_blk_mask16
			img_h, img_w, ch = img.shape
			# print(img_h, img_w)
			img_info_arr.append([img_w, img_h, img, mask, label])
			# print(label, img_h, img_w)
			rectangles.append((img_w, img_h, label))

	positions = []
	w_arr = []
	h_arr = []
	positions = rpack.pack(rectangles)

	for i in range(0, len(positions)):
		w_l = positions[i][0]
		h_t = positions[i][1]
		w_r = w_l + img_info_arr[i][0]
		h_b = h_t + img_info_arr[i][1]
		w_arr.append(w_r)
		h_arr.append(h_b)
	max_w = max(w_arr)
	max_h = max(h_arr)

	print(max_w, max_h)
	img_all = np.ones((max_h, max_w, 3), np.uint8)*0
	mask_all = np.ones((max_h, max_w), np.uint8)
	mask_all2 = np.ones((max_h, max_w), np.uint8)
	path_info = []
	for i in range(len(positions)):
		w_l = positions[i][0]
		h_t = positions[i][1]
		w_r = w_l + img_info_arr[i][0]
		h_b = h_t + img_info_arr[i][1]
		# img_all[h_t:h_b, w_l:w_r] = img_info_arr[i][2]
		img = img_info_arr[i][2]
		mask = img_info_arr[i][3]

		img_h, img_w, ch = img.shape
		for y in range(img_h):
			for x in range(img_w):
				if mask[y][x]:
					img_all[h_t+y][w_l+x] = img[y][x]


		# mask_all[h_t:h_b, w_l:w_r] = np.ones((img_info_arr[i][1], img_info_arr[i][0]), np.uint8)*255
		mask_all[h_t:h_b, w_l:w_r] = mask
		mask_all2[h_t:h_b, w_l:w_r] = mask
		
		patch_id = img_info_arr[i][4]
		path_str = find_contour(img, mask, w_l, h_t, i, blk_size=16)
		path_info.append(path_str)
	
	

	cv2.imwrite(file_id + "rpack.png", img_all)
	cv2.imwrite(file_id + "_rect_mask.png", mask_all)

	layout_info = parse_svg()

	all_img = np.zeros((max_h, max_w, 3), np.uint8)
	all_mask = np.zeros((max_h, max_w), np.uint8)
	all_mask2 = np.zeros((max_h, max_w), np.uint8)
	for rec in layout_info:
		path_str = rec[-1]
		position_label = -1
		for t in range(len(path_info)):
			p_st = path_info[t]
			if p_st == path_str:
				print(t)
				position_label = t
				break

		w_l = positions[position_label][0]
		h_t = positions[position_label][1]
		img = img_info_arr[position_label][2]
		mask = img_info_arr[position_label][3]

		[angle, translate_x, translate_y, path_str] = rec
		# print(rec)
		rot_cnt = (4-int(angle/90))%4
		rotate_mask = np.rot90(mask, rot_cnt)
		print(rot_cnt, angle)
		# cv2.imshow("mask", mask)
		# cv2.imshow("rotate_mask", rotate_mask)
		# cv2.waitKey(0)

		w_r = w_l + mask.shape[1]
		h_b = h_t + mask.shape[0]

		for y in range(h_t, h_b):
			for x in range(w_l, w_r):
				Xf = x*np.cos(angle/180.0*np.pi) - y*np.sin(angle/180.0*np.pi)
				Yf = x*np.sin(angle/180.0*np.pi) + y*np.cos(angle/180.0*np.pi)
				new_x = int(np.round(Xf+translate_x, 0))
				new_y = int(np.round(Yf+translate_y, 0))
				
				
				if mask[y-h_t][x-w_l]:
					all_img[new_y][new_x] = img[y-h_t][x-w_l]
					all_mask[new_y][new_x] = mask[y-h_t][x-w_l]
				


	# cv2.imshow("all_img", all_img)
	# cv2.imshow("all_mask", all_mask)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()



	all_mask_rgb = np.asarray(cv2.cvtColor(all_mask,cv2.COLOR_GRAY2RGB))
	image_data_bw = all_mask_rgb.max(axis=2)
	non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
	non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
	cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

	all_img = all_img[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]
	all_mask = all_mask[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]
	
	# ip_grid_img = img_inpaint(all_img, 1-all_mask/255)
	# ip_grid_img = np.uint8(ip_grid_img*255.999)
	# cv2.imshow("all_img", all_img)
	# cv2.imshow("all_mask", all_mask)
	# cv2.imshow("ip_grid_img", ip_grid_img)
	# cv2.waitKey(0)

	cv2.imwrite(file_id + "svgnest.png", all_img)
	cv2.imwrite(file_id + "_svgnest_mask.png", all_mask)

	# ip_all_img = img_inpaint(all_img, 1-all_mask/255)
	# ip_all_img = np.uint8(ip_all_img*255.999)

	# cv2.imwrite(file_id + "_svgnest_ip.png", ip_all_img)
	


	img_h, img_w = all_mask.shape
	blk_size = 16
	blk16_ip_grid_img = np.zeros((img_h, img_w, 3), np.uint8)
	for h in range(0, img_h, blk_size):
		for w in range(0, img_w, blk_size):
			print(h, w, img_h, img_w )
			sub_all_img = all_img[h:h+blk_size, w:w+blk_size, :]
			sub_all_mask = all_mask[h:h+blk_size, w:w+blk_size]
			if np.sum(sub_all_mask)>0:
				sub_ip_grid_img = img_inpaint(sub_all_img, 1-sub_all_mask/255)
				sub_ip_grid_img = np.uint8(sub_ip_grid_img*255.999)
				blk16_ip_grid_img[h:h+blk_size, w:w+blk_size, :] = sub_ip_grid_img

	cv2.imwrite(file_id + "svgnest_blk16_ip.png", blk16_ip_grid_img)
	
	blk_size = 64
	blk64_ip_grid_img = np.zeros((img_h, img_w, 3), np.uint8)
	for h in range(0, img_h, blk_size):
		for w in range(0, img_w, blk_size):
			print(h, w, img_h, img_w )
			sub_all_img = all_img[h:h+blk_size, w:w+blk_size , :]
			sub_all_mask = all_mask[h:h+blk_size, w:w+blk_size]
			if np.sum(sub_all_mask)>0:
				sub_ip_grid_img = img_inpaint(sub_all_img, 1-sub_all_mask/255)
				sub_ip_grid_img = np.uint8(sub_ip_grid_img*255.999)
				blk64_ip_grid_img[h:h+blk_size, w:w+blk_size , :] = sub_ip_grid_img
		
	cv2.imwrite(file_id + "svgnest_blk64_ip.png", blk64_ip_grid_img)
	



	patch_pt_num = len(geo_arr)

	chain_bpp_arr_all = []
	chain_psnr_arr_all = []
	chain_legend_arr_all = []
	[peer_chain_bpp_arr, peer_chain_psnr_arr, peer_chain_legend_arr] = peer_method(frame_id)
	chain_bpp_arr_all = chain_bpp_arr_all + peer_chain_bpp_arr
	chain_psnr_arr_all = chain_psnr_arr_all + peer_chain_psnr_arr
	chain_legend_arr_all = chain_legend_arr_all + peer_chain_legend_arr

	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(img_all, mask_all, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("rpack_" + str(patch_pt_num))
	# [ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(img_all, mask_all, patch_pt_num)
	# chain_bpp_arr_all.append(ball_tree_bpp_arr)
	# chain_psnr_arr_all.append(ball_tree_psnr_arr)
	# chain_legend_arr_all.append("rpack_" + str(patch_pt_num))
	# ip_img_all = img_inpaint(img_all, 1-mask_all/255)
	# ip_img_all = np.uint8(ip_img_all*255.999)
	# [ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(ip_img_all, mask_all, patch_pt_num)
	# chain_bpp_arr_all.append(ball_tree_bpp_arr)
	# chain_psnr_arr_all.append(ball_tree_psnr_arr)
	# chain_legend_arr_all.append("rpack_ip_" + str(patch_pt_num))

	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(all_img, all_mask, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("svgnest_" + str(patch_pt_num))

	
	# [ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(ip_all_img, all_mask, patch_pt_num)
	# chain_bpp_arr_all.append(ball_tree_bpp_arr)
	# chain_psnr_arr_all.append(ball_tree_psnr_arr)
	# chain_legend_arr_all.append("svgnest_ip_" + str(patch_pt_num))


	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(blk16_ip_grid_img, all_mask, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("svgnest_ip_blk16_" + str(patch_pt_num))

	[ball_tree_bpp_arr, ball_tree_psnr_arr] = img_compression_with_mask(blk64_ip_grid_img, all_mask, patch_pt_num)
	chain_bpp_arr_all.append(ball_tree_bpp_arr)
	chain_psnr_arr_all.append(ball_tree_psnr_arr)
	chain_legend_arr_all.append("svgnest_ip_blk64_" + str(patch_pt_num))



	plt.figure()
	for ttt in range(0,len(chain_bpp_arr_all)):
		plt.plot(chain_bpp_arr_all[ttt], chain_psnr_arr_all[ttt], color = plt_color_arr[ttt], marker = plt_marker_arr[ttt], markerfacecolor="None", linewidth=1)
	plt.xlabel('bpp', fontsize=12)
	plt.ylabel('PSNR', fontsize=12)
	plt.title("all", fontsize=12)
	plt.legend(chain_legend_arr_all, loc="lower right")
	plt.savefig(res_folder + "//" + 'new_cmpall.png')
	plt.close()

def bi_patch_generation(geo_arr, rgb_arr):
	mid_iso_num = 16
	mid_svoff_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(mid_iso_num) + ".off"
	[mid_sv_off_geo_arr, mid_sv_off_rgb_arr, mid_sv_off_pt_num] = read_off(mid_svoff_path, geo_arr, rgb_arr)
	
	svoff_path = floder + "LOD_off" + "/" + frame_id_head + "_n"+ str(128) + ".off"
	[sv_off_geo_arr, sv_off_rgb_arr, sv_off_pt_num] = read_off(svoff_path, geo_arr, rgb_arr)
	[hks_pt_arr, hks_feature_arr] = get_hks_feature(geo_arr, rgb_arr, floder, frame_id)
	tot_num = len(hks_pt_arr)
	X = np.array(hks_pt_arr)
	tree = BallTree(X, leaf_size = 1)
	new_hks_feature_arr = []
	for pt in sv_off_geo_arr:
		dist, ind = tree.query([pt], k=1)
		idx = ind[0][0]
		new_hks_feature_arr.append(hks_feature_arr[idx])
	hks_feature_arr = new_hks_feature_arr

	colors = cm.coolwarm(np.linspace(0, 1, 256))
	colors = [rgb[0:3] for rgb in colors]
	colors = colors[::-1]
	min_hks = min(hks_feature_arr)
	max_hks = max(hks_feature_arr)
	hks_rgb = [colors[int(np.round((val-min_hks)/(max_hks-min_hks)*255))][0:3]*255 for val in hks_feature_arr]


	dbscan_thresh = get_neighbor_dis(mid_sv_off_geo_arr)*2
	[hks_pt_arr, hks_feature_arr] = get_hks_feature(geo_arr, rgb_arr, floder, frame_id)

	[skeleton_adj_dic, chain_seg_dic, chain_dic, all_cluster_init, end_node, bi_node, multi_node] = hybird_seg2(geo_arr, rgb_arr, floder, frame_id, sv_off_geo_arr, hks_rgb, vis = 0)
	# all_cluster = merge_multinode(skeleton_adj_dic, end_node, bi_node, multi_node, all_cluster_init, sv_off_geo_arr)
	# vis_geo = []
	# vis_rgb = []
	# for cluster in all_cluster:
	# 	rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
	# 	for idx in cluster:
	# 		vis_geo.append(sv_off_geo_arr[idx])
	# 		vis_rgb.append(rgb)
	# pc_vis(vis_geo, vis_rgb)


	# if vis:
	vis_geo = []
	vis_rgb = []
	for seg in chain_seg_dic:
		print(seg)
		rgb = [randrange(0, 255), randrange(0, 255), randrange(0, 255)]
		rgb =np.mean(np.asarray([hks_rgb[idx] for idx in chain_seg_dic[seg]]), axis=0)
		for idx in chain_seg_dic[seg]:
			vis_geo.append(sv_off_geo_arr[idx])
			vis_rgb.append(rgb)
	pc_vis(vis_geo, vis_rgb)

if __name__ == '__main__':
	aggl_cluster_num = 2	# aggl_clustering cluster number
	sv_pt_num = 8192
	iso_sv_pt_num = 32
	min_patch_pt_num = int(1024/iso_sv_pt_num)	# minimum number of points of patch
	ut_iso_th = int(1024/iso_sv_pt_num/2)	# threshold of untransformable iso_off point number

	floder = "../pcc_dataset/"

	frame_id = pc_id_arr[0]
	frame_id_head = frame_id.split("_")[0]
	ply_path = floder + frame_id + ".ply"
	

	for f_id in ["loot_vox10_1000","longdress_vox10_1051",  "soldier_vox10_0536", "redandblack_vox10_1550", "frame_0200"]:
			frame_id = f_id
			patch_path = floder + frame_id + "_patch" + str(sv_pt_num) + "_" + str(iso_sv_pt_num) + ".txt"
			frame_id_head = frame_id.split("_")[0]
			ply_path = floder  + "normal/" + frame_id + ".ply"

			if not os.path.exists('D:\\' + frame_id):
				os.mkdir('D:\\' + frame_id)
			res_folder = "D:/" + frame_id
			if not os.path.exists(res_folder):
				os.mkdir(res_folder)

			[geo_arr, rgb_arr, normal_arr, total_point_num, pc_width] = init(ply_path)
			# cube_clustering(geo_arr, normal_arr, vis_flag=1)
			print("frame_id: ", frame_id, total_point_num)

			
			# bi_patch_generation(geo_arr, rgb_arr)

			[all_final_seg_arr, all_final_seg_label, all_cluster, sv_off_geo_arr] = patch_generation(geo_arr, rgb_arr, normal_arr, floder, frame_id, frame_id_head, iso_sv_pt_num)
			# distortion_based_unfold(all_final_seg_arr, all_final_seg_label, all_cluster, sv_off_geo_arr)


			# [cluster_dic, sv_off_geo_arr] = normal_clustering(geo_arr, rgb_arr, normal_arr, floder, frame_id, frame_id_head, iso_sv_pt_num)
			# normal_patch_unfold2(geo_arr, rgb_arr, cluster_dic, sv_off_geo_arr, res_folder)
			# bi_cluster_dic = normal_based_biseg(geo_arr, rgb_arr, normal_arr)
			# normal_based_sampling(geo_arr, rgb_arr, normal_arr, bi_cluster_dic)
			# try:
			# normal_based_simplification2(geo_arr, rgb_arr, normal_arr)
			# rect_packing(geo_arr, rgb_arr, normal_arr)
			#orig_rect_packing_bsp(geo_arr, rgb_arr, normal_arr)
			# try:
			# blk_rect_packing_bsp(geo_arr, rgb_arr, normal_arr, blk=8)
			# except:
			# 	print()

			# try:
			# 	blk_rect_packing_bsp(geo_arr, rgb_arr, normal_arr, blk=16)
			# except:
			# 	print()

			# try:
			# 	blk_rect_packing_bsp(geo_arr, rgb_arr, normal_arr, blk=32)
			# except:
			# 	print()
			# irregular_packing(geo_arr, rgb_arr, normal_arr)
			# except:
			# 	print("!!!!!!!!!!!!!!!!!!!!!!!!!!!frame_id: ", frame_id)

			# normal_patch_unfold(geo_arr, rgb_arr, cluster_dic, sv_off_geo_arr, res_folder)
			
			# hks_seg2(geo_arr, rgb_arr, floder, frame_id)
			# hks_vis(geo_arr, rgb_arr, floder, frame_id)
			# [dis_pt, dis_pt_idx] = test_distortion_all2(geo_arr, rgb_arr, floder, frame_id_head, iso_sv_pt_num)

			# test_distortion_all4(geo_arr, rgb_arr, floder, frame_id_head, iso_sv_pt_num, dis_pt, dis_pt_idx)
