"""
Author: Haoxi Ran
Date: 05/10/2022
"""

import torch


#################
# MAIN
#################

def get_aug_args(args):
    dataset = args.dataset
    if dataset == 'ScanObjectNN':
        return {'scale_factor': 0.5, 'shift_factor': 0.3}
    else:
        raise Exception('No such dataset')


def transform_point_cloud(batch, args, aug_args, train=True, label=None):
    """batch: B x 3/6 x N"""
    if args.aug_scale:
        batch[:, 0:3] = scale_point_cloud(batch[:, 0:3], aug_args['scale_factor'])
    if args.aug_shift:
        batch[:, 0:3] = shift_point_cloud(batch[:, 0:3], shift_range=aug_args['shift_factor'])
    return (batch, label) if label is not None else batch


#################
# Shift
#################

def shift_point_cloud(batch_data, shift_range=0.2):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          B x C x N array, original batch of point clouds
        Return:
          B x C x N array, shifted batch of point clouds
    """
    shifts = (torch.rand(batch_data.shape[0], 3, 1, device=batch_data.device) * 2. - 1.) * shift_range
    batch_data += shifts
    return batch_data


#################
# Scale
#################

def scale_point_cloud(batch_data, scale_range=0.2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            B x C x N array, original batch of point clouds
        Return:
            B x C x N array, scaled batch of point clouds
    """
    scales = (torch.rand(batch_data.shape[0], 3, 1, device=batch_data.device) * 2. - 1.) * scale_range + 1.
    batch_data *= scales
    return batch_data
