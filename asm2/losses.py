import torch
from torch import nn
import torch.nn.functional as F
import pytorch3d
import pytorch3d.loss
# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
	bce =  nn.BCEWithLogitsLoss()
	loss=bce(voxel_src,voxel_tgt)
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	n_points=point_cloud_src.shape[1]
	pcl1_dist,_,_=pytorch3d.ops.knn_points(point_cloud_src,point_cloud_tgt)
	pcl2_dist,_,_=pytorch3d.ops.knn_points(point_cloud_tgt,point_cloud_src)
	loss_chamfer=(torch.sqrt(pcl1_dist**2).sum()+torch.sqrt(pcl2_dist**2).sum())/n_points
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	loss_laplacian=pytorch3d.loss.mesh_laplacian_smoothing(mesh_src)
	return loss_laplacian