# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import time

import numpy as np

import torch
import torch.nn as nn

from mesh_viewer import MeshViewer
import utils

import copy


@torch.no_grad()
def guess_init(model,
               joints_2d,
               edge_idxs,
               focal_length=5000,
               pose_embedding=None,
               vposer=None,
               use_vposer=True,
               dtype=torch.float32,
               model_type='smpl',
               **kwargs):
    ''' Initializes the camera translation vector

        Parameters
        ----------
        model: nn.Module
            The PyTorch module of the body
        joints_2d: torch.tensor 1xJx2
            The 2D tensor of the joints
        edge_idxs: list of lists
            A list of pairs, each of which represents a limb used to estimate
            the camera translation
        focal_length: float, optional (default = 5000)
            The focal length of the camera
        pose_embedding: torch.tensor 1x32
            The tensor that contains the embedding of V-Poser that is used to
            generate the pose of the model
        dtype: torch.dtype, optional (torch.float32)
            The floating point type used
        vposer: nn.Module, optional (None)
            The PyTorch module that implements the V-Poser decoder
        Returns
        -------
        init_t: torch.tensor 1x3, dtype = torch.float32
            The vector with the estimated camera location

    '''

    body_pose = vposer.decode(
        pose_embedding, output_type='aa').view(1, -1) if use_vposer else None
    if use_vposer and model_type == 'smpl':
        wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                 dtype=body_pose.dtype,
                                 device=body_pose.device)
        body_pose = torch.cat([body_pose, wrist_pose], dim=1)

    output = model(body_pose=body_pose, return_verts=False,
                   return_full_pose=False)
    joints_3d = output.joints
    joints_2d = joints_2d.to(device=joints_3d.device)

    diff3d = []
    diff2d = []
    for edge in edge_idxs:
        diff3d.append(joints_3d[:, edge[0]] - joints_3d[:, edge[1]])
        diff2d.append(joints_2d[:, edge[0]] - joints_2d[:, edge[1]])

    diff3d = torch.stack(diff3d, dim=1)
    diff2d = torch.stack(diff2d, dim=1)

    length_2d = diff2d.pow(2).sum(dim=-1).sqrt()
    length_3d = diff3d.pow(2).sum(dim=-1).sqrt()

    height2d = length_2d.mean(dim=1)
    height3d = length_3d.mean(dim=1)

    est_d = focal_length * (height3d / height2d)

    # just set the z value
    batch_size = joints_3d.shape[0]
    x_coord = torch.zeros([batch_size], device=joints_3d.device,
                          dtype=dtype)
    # y_coord = x_coord.clone()
    y_coord = torch.ones([batch_size], device=joints_3d.device,
                          dtype=dtype) * 1.0
    init_t = torch.stack([x_coord, y_coord, est_d], dim=1)
    return init_t


class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            self.mv = MeshViewer(body_color=self.body_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, params, body_model,
                    use_vposer=True, pose_embedding=None, vposer=None,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None
        for n in range(self.maxiters):
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    # print('loss rel change')
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                # print('gradient')
                break

            if self.visualize and n % self.summary_steps == 0:
                body_pose = vposer.decode(
                    pose_embedding, output_type='aa').view(
                        1, -1) if use_vposer else None

                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                             dtype=body_pose.dtype,
                                             device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                model_output = body_model(
                    return_verts=True, body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            prev_loss = loss.item()

        return prev_loss

    def create_fitting_closure(self,
                               optimizer, body_model, camera=None,
                               gt_joints_2d=None,
                               gt_joints_3d=None,
                               loss=None,
                               prev_gt_joints=None, prev_proj_joints=None,
                               joints_conf_2d=None,
                               joints_conf_3d=None,
                               joint_weights=None,
                               return_verts=True, return_full_pose=False,
                               use_vposer=False, vposer=None,
                               pose_embedding=None,
                               create_graph=False,
                               **kwargs):
        faces_tensor = body_model.faces_tensor.view(-1)
        append_wrists = self.model_type == 'smpl' and use_vposer

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            body_pose = vposer.decode(
                pose_embedding, output_type='aa') if use_vposer else None
            # idx_fixed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            # for id in idx_fixed:
            #     body_pose[:, :, id, :] = 0.0
            body_pose = body_pose.view(1, -1)
            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            body_model_output = body_model(return_verts=return_verts,
                                           body_pose=body_pose,
                                           return_full_pose=return_full_pose)
            global_orient = body_model_output.global_orient
            total_loss = loss(body_model_output, camera=camera,
                              gt_joints_2d=gt_joints_2d,
                              gt_joints_3d=gt_joints_3d,
                            #   prev_gt_joints=prev_gt_joints,
                            #   prev_proj_joints=prev_proj_joints,
                              body_model_faces=faces_tensor,
                              joints_conf_2d=joints_conf_2d,
                              joints_conf_3d=joints_conf_3d,
                              joint_weights=joint_weights,
                              pose_embedding=pose_embedding,
                              use_vposer=use_vposer,
                              global_orient=global_orient,
                              body_pose=body_pose,
                              **kwargs)

            if backward:
                total_loss.backward(create_graph=create_graph)

            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                model_output = body_model(return_verts=True,
                                          body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            return total_loss

        return fitting_func


def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return SMPLifyLoss(**kwargs)
    elif loss_type == 'camera_init':
        return SMPLifyCameraInitLoss(**kwargs)
        # return SMPLifyWeakCameraInitLoss(**kwargs)
    elif loss_type == 'camera':
        return SMPLifyCameraLoss(**kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))


class SMPLifyLoss(nn.Module):

    def __init__(self, search_tree=None,
                 pen_distance=None, tri_filtering_module=None,
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 expr_prior=None,
                 angle_prior=None,
                 jaw_prior=None,
                 use_joints_conf=True,
                 use_face=True, use_hands=True,
                 left_hand_prior=None, right_hand_prior=None,
                 interpenetration=True, dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 hand_prior_weight=0.0,
                 expr_prior_weight=0.0, jaw_prior_weight=0.0,
                 coll_loss_weight=0.0,
                 reduction='sum',
                 input_2d_joints=True,
                 input_3d_joints=True,
                 **kwargs):

        super(SMPLifyLoss, self).__init__()

        self.use_joints_conf = use_joints_conf
        self.angle_prior = angle_prior

        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho

        self.body_pose_prior = body_pose_prior

        self.shape_prior = shape_prior

        self.interpenetration = interpenetration
        if self.interpenetration:
            self.search_tree = search_tree
            self.tri_filtering_module = tri_filtering_module
            self.pen_distance = pen_distance

        self.use_hands = use_hands
        if self.use_hands:
            self.left_hand_prior = left_hand_prior
            self.right_hand_prior = right_hand_prior

        self.use_face = use_face
        if self.use_face:
            self.expr_prior = expr_prior
            self.jaw_prior = jaw_prior
        
        self.input_2d_joints = input_2d_joints
        self.input_3d_joints = input_3d_joints

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))
        if self.use_hands:
            self.register_buffer('hand_prior_weight',
                                 torch.tensor(hand_prior_weight, dtype=dtype))
        if self.use_face:
            self.register_buffer('expr_prior_weight',
                                 torch.tensor(expr_prior_weight, dtype=dtype))
            self.register_buffer('jaw_prior_weight',
                                 torch.tensor(jaw_prior_weight, dtype=dtype))
        if self.interpenetration:
            self.register_buffer('coll_loss_weight',
                                 torch.tensor(coll_loss_weight, dtype=dtype))
        self.temp = torch.tensor([[np.pi, 0, 0]], dtype=torch.float32).cuda()

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, 
                gt_joints_2d, gt_joints_3d, 
                joints_conf_2d, joints_conf_3d,
                body_model_faces, joint_weights, 
                prev_gt_joints=None,
                prev_proj_joints=None,
                use_vposer=False, pose_embedding=None,global_orient=None,
                body_pose=None,
                **kwargs):
        # Calculate the weights for each joints
        if self.input_2d_joints:
            weights_2d = (joint_weights * joints_conf_2d
                    if self.use_joints_conf else
                    joint_weights).unsqueeze(dim=-1)
        if self.input_3d_joints:
            weights_3d = (joint_weights * joints_conf_3d
                    if self.use_joints_conf else
                    joint_weights).unsqueeze(dim=-1)

        # Calculate the distance of the projected joints from
        # the ground truth 2D/3D detections
        joint_loss = 0.0
        if self.input_2d_joints:
            pred_joints_2d = camera(body_model_output.joints)
            joint_diff_2d = self.robustifier(gt_joints_2d - pred_joints_2d)
            joint_loss += (torch.sum(weights_2d ** 2 * joint_diff_2d) *
                      self.data_weight ** 2)
        if self.input_3d_joints:
            pred_joints_3d = utils.scale_pred_joints(gt_joints_3d, body_model_output.joints)
            # if self.counter % 100 == 0:
            #     plot_pred_joints = pred_joints_3d.detach().cpu().numpy().squeeze()
            #     plot_gt_joints = gt_joints_3d.detach().cpu().numpy().squeeze()
            #     pred_joints_x = plot_pred_joints[:,0]
            #     pred_joints_z = plot_pred_joints[:,2]
            #     gt_joints_x = plot_gt_joints[:,0]
            #     gt_joints_z = plot_gt_joints[:,2]
            #     plt.clf()
            #     plt.scatter(pred_joints_x,pred_joints_z, c='red')
            #     plt.scatter(gt_joints_x,gt_joints_z, c='blue')
            #     plt.savefig('compare')
            # self.counter += 1
            joint_diff_3d = self.robustifier(gt_joints_3d - pred_joints_3d)
            joint_loss += (torch.sum(weights_3d ** 2 * joint_diff_3d) *
                      self.data_weight ** 2)
            # print(f'The scale is {torch.norm(gt_joints_3d[:,1,:]-gt_joints_3d[:,8,:])/torch.norm(pred_joints_3d[:,1,:]-pred_joints_3d[:,8,:])}')
            # print(f'The scale is {torch.norm(gt_joints_3d[:,2,:]-gt_joints_3d[:,5,:])/torch.norm(pred_joints_3d[:,2,:]-pred_joints_3d[:,5,:])}')

        # # smooth the joints
        # smooth_joint_loss = 0.0
        # if prev_proj_joints != None:
        #     ceil_conf = joints_conf.unsqueeze(dim=-1).clone()
        #     ceil_conf[ceil_conf>0] = 1.0
        #     gt_joint_diff = self.robustifier(prev_gt_joints - gt_joints)
        #     smooth_weights = -(ceil_conf - 1) + (1 / (gt_joint_diff + 0.0001)) * ceil_conf
        #     proj_joint_diff = self.robustifier(prev_proj_joints - projected_joints)
        #     smooth_joint_loss = (torch.sum(smooth_weights * proj_joint_diff) *
        #                 self.data_weight ** 2)

        # Calculate the loss from the Pose prior
        if use_vposer:
            pprior_loss = (pose_embedding.pow(2).sum() *
                           self.body_pose_weight ** 2)
        else:
            pprior_loss = torch.sum(self.body_pose_prior(
                body_model_output.body_pose,
                body_model_output.betas)) * self.body_pose_weight ** 2

        shape_loss = 0.0
        # shape_loss = torch.sum(self.shape_prior(
        #     body_model_output.betas)) * self.shape_weight ** 2

        # Calculate the prior over the joint rotations. This a heuristic used
        # to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_prior_loss = torch.sum(
            self.angle_prior(body_pose)) * self.bending_prior_weight

        # # self-added
        body_pose = body_pose.reshape(-1, 3)
        idx_fixed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        new_constraint_pose = body_pose[idx_fixed]
        # for id in idx_fixed:
        #     body_pose[:, :, id, :] = 0.0
        another_angle_loss = 1000.0 * torch.sum(new_constraint_pose.pow(2))
        global_ori_loss = torch.sum(2000.0 * (body_model_output.global_orient - self.temp) ** 2)
        # print(body_model_output.global_orient)
        # print(new_constraint_pose)

        # Apply the prior on the pose space of the hand
        left_hand_prior_loss, right_hand_prior_loss = 0.0, 0.0
        if self.use_hands and self.left_hand_prior is not None:
            left_hand_prior_loss = torch.sum(
                self.left_hand_prior(
                    body_model_output.left_hand_pose)) * \
                self.hand_prior_weight ** 2

        if self.use_hands and self.right_hand_prior is not None:
            right_hand_prior_loss = torch.sum(
                self.right_hand_prior(
                    body_model_output.right_hand_pose)) * \
                self.hand_prior_weight ** 2

        expression_loss = 0.0
        jaw_prior_loss = 0.0
        if self.use_face:
            expression_loss = torch.sum(self.expr_prior(
                body_model_output.expression)) * \
                self.expr_prior_weight ** 2

            if hasattr(self, 'jaw_prior'):
                jaw_prior_loss = torch.sum(
                    self.jaw_prior(
                        body_model_output.jaw_pose.mul(
                            self.jaw_prior_weight)))

        pen_loss = 0.0
        # Calculate the loss due to interpenetration
        if (self.interpenetration and self.coll_loss_weight.item() > 0):
            if self.input_2d_joints:
                batch_size = gt_joints_2d.shape[0]
            else:
                batch_size = gt_joints_3d.shape[0]
            triangles = torch.index_select(
                body_model_output.vertices, 1,
                body_model_faces).view(batch_size, -1, 3, 3)

            with torch.no_grad():
                collision_idxs = self.search_tree(triangles)

            # Remove unwanted collisions
            if self.tri_filtering_module is not None:
                collision_idxs = self.tri_filtering_module(collision_idxs)

            if collision_idxs.ge(0).sum().item() > 0:
                pen_loss = torch.sum(
                    self.coll_loss_weight *
                    self.pen_distance(triangles, collision_idxs))

        total_loss = (joint_loss + pprior_loss  + shape_loss +
                       pen_loss + angle_prior_loss +
                      jaw_prior_loss + expression_loss +
                      left_hand_prior_loss + right_hand_prior_loss + global_ori_loss + another_angle_loss)
        
        return total_loss


class SMPLifyCameraInitLoss(nn.Module):

    def __init__(self, init_joints_idxs, trans_estimation=None,
                 reduction='sum',
                 data_weight=1.0,
                 depth_loss_weight=1e2, dtype=torch.float32,
                 **kwargs):
        super(SMPLifyCameraInitLoss, self).__init__()
        self.dtype = dtype

        if trans_estimation is not None:
            self.register_buffer(
                'trans_estimation',
                utils.to_tensor(trans_estimation, dtype=dtype))
        else:
            self.trans_estimation = trans_estimation

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            'init_joints_idxs',
            utils.to_tensor(init_joints_idxs, dtype=torch.long))
        self.register_buffer('depth_loss_weight',
                             torch.tensor(depth_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints_2d,
                **kwargs):

        projected_joints = camera(body_model_output.joints)

        joint_error = torch.pow(
            torch.index_select(gt_joints_2d, 1, self.init_joints_idxs) -
            torch.index_select(projected_joints, 1, self.init_joints_idxs),
            2)
        joint_loss = torch.sum(joint_error) * self.data_weight ** 2

        depth_loss = 0.0
        if (self.depth_loss_weight.item() > 0 and self.trans_estimation is not
                None):
            depth_loss = self.depth_loss_weight ** 2 * torch.sum((
                camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2))

        return joint_loss + depth_loss


class SMPLifyCameraLoss(nn.Module):
    def __init__(self, init_joints_idxs, trans_estimation=None,
                 reduction='sum',
                 data_weight=1.0,
                 depth_loss_weight=1e2, dtype=torch.float32,
                 **kwargs):
        super(SMPLifyCameraLoss, self).__init__()
        self.dtype = dtype

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            'init_joints_idxs',
            utils.to_tensor(init_joints_idxs, dtype=torch.long))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints_2d,
                **kwargs):

        projected_joints = camera(body_model_output.joints)

        joint_error = torch.pow(
            torch.index_select(gt_joints_2d, 1, self.init_joints_idxs) -
            torch.index_select(projected_joints, 1, self.init_joints_idxs),
            2)
        joint_loss = torch.sum(joint_error) * self.data_weight ** 2

        return joint_loss




class SMPLifyWeakCameraInitLoss(nn.Module):

    def __init__(self, init_joints_idxs, trans_estimation=None,
                 reduction='sum',
                 data_weight=1.0,
                 depth_loss_weight=1e2, dtype=torch.float32,
                 **kwargs):
        super(SMPLifyWeakCameraInitLoss, self).__init__()
        self.dtype = dtype

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            'init_joints_idxs',
            utils.to_tensor(init_joints_idxs, dtype=torch.long))
        self.register_buffer('depth_loss_weight',
                             torch.tensor(depth_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints_2d,
                **kwargs):

        projected_joints = camera(body_model_output.joints)

        joint_error = torch.pow(
            torch.index_select(gt_joints_2d, 1, self.init_joints_idxs) -
            torch.index_select(projected_joints, 1, self.init_joints_idxs),
            2)
        joint_loss = torch.sum(joint_error) * self.data_weight ** 2

        # depth_loss = 0.0
        # if (self.depth_loss_weight.item() > 0 and self.trans_estimation is not
        #         None):
        #     depth_loss = self.depth_loss_weight ** 2 * torch.sum((
        #         camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2))

        return joint_loss #+ depth_loss