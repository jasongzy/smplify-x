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


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os
import os.path as osp

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict

import cv2
import PIL.Image as pil_img
import matplotlib.pyplot as plt

from optimizers import optim_factory

import utils
import fitting
from human_body_prior.tools.model_loader import load_vposer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import smplx


def fit_single_frame(img,
                     keypoints_2d,
                     keypoints_3d,
                     body_model,
                     camera,
                     joint_weights,
                     body_pose_prior,
                     jaw_prior,
                     left_hand_prior,
                     right_hand_prior,
                     shape_prior,
                     expr_prior,
                     angle_prior,
                     input_2d_joints=True,
                     input_3d_joints=True,
                     result_fn='out.pkl',
                     mesh_fn='out.obj',
                     out_img_fn='overlay.png',
                     loss_type='smplify',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     use_face=True,
                     use_hands=True,
                     data_weights=None,
                     body_pose_prior_weights=None,
                     hand_pose_prior_weights=None,
                     jaw_pose_prior_weights=None,
                     shape_weights=None,
                     expr_weights=None,
                     hand_joints_weights=None,
                     face_joints_weights=None,
                     depth_loss_weight=1e2,
                     interpenetration=True,
                     coll_loss_weights=None,
                     df_cone_height=0.5,
                     penalize_outside=True,
                     max_collisions=8,
                     point2plane=False,
                     part_segm_fn='',
                     focal_length=5000.,
                     side_view_thsh=25.,
                     rho=100,
                     vposer_latent_dim=32,
                     vposer_ckpt='',
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     degrees=None,
                     batch_size=1,
                     dtype=torch.float32,
                     ign_part_pairs=None,
                     left_shoulder_idx=2,
                     right_shoulder_idx=5,
                     init_pose_embedding=None,
                     init_body_pose=None,
                     init_global_orient=None,
                    #  prev_gt_joints=None,
                    #  prev_proj_joints=None,
                     **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    if degrees is None:
        degrees = [0, 90, 180, 270]

    if data_weights is None:
        data_weights = [1, ] * 5

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) ==
                    len(body_pose_prior_weights)), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    if use_face:
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = [[x] * 3 for x in shape_weights]
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                         jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of jaw pose prior weights')
        assert (len(jaw_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg

        if expr_weights is None:
            expr_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights = {} does not match the' +
               ' number of Expression prior weights = {}')
        assert (len(expr_weights) ==
                len(body_pose_prior_weights)), msg.format(
                    len(body_pose_prior_weights),
                    len(expr_weights))

        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) ==
                len(body_pose_prior_weights)), msg

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = ('Number of Body pose prior weights does not match the' +
           ' number of collision loss weights')
    assert (len(coll_loss_weights) ==
            len(body_pose_prior_weights)), msg

    use_vposer = kwargs.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    if use_vposer:
        pose_embedding = init_pose_embedding.clone().detach()
        pose_embedding.requires_grad = True
        vposer_ckpt = osp.expandvars(vposer_ckpt)
        vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer = vposer.to(device=device)
        vposer.eval()

    # if use_vposer:
    #     body_mean_pose = init_body_pose.clone()
    # else:
    #     body_mean_pose = body_pose_prior.get_mean().detach().cpu()

    keypoint_data_2d = torch.tensor(keypoints_2d, dtype=dtype)
    keypoint_data_3d = torch.tensor(keypoints_3d, dtype=dtype)

    gt_joints_2d = None
    gt_joints_3d = None
    if input_2d_joints:
        gt_joints_2d = keypoint_data_2d[:,:,:2]
        gt_joints_2d = gt_joints_2d.to(device=device, dtype=dtype)
    if input_3d_joints:
        gt_joints_3d = keypoint_data_3d[:,:,:3]
        gt_joints_3d = gt_joints_3d.to(device=device, dtype=dtype)

    joints_conf_2d = None
    joints_conf_3d = None
    if use_joints_conf:
        if input_2d_joints:
            joints_conf_2d = keypoint_data_2d[:, :, 2].reshape(1, -1)
            joints_conf_2d = joints_conf_2d.to(device=device, dtype=dtype)
        if input_3d_joints:
            joints_conf_3d = keypoint_data_3d[:, :, 3].reshape(1, -1)
            joints_conf_3d = joints_conf_3d.to(device=device, dtype=dtype)

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None
    if interpenetration:
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces

        assert use_cuda, 'Interpenetration term can only be used with CUDA'
        assert torch.cuda.is_available(), \
            'No CUDA Device! Interpenetration term can only be used' + \
            ' with CUDA'

        search_tree = BVH(max_collisions=max_collisions)

        pen_distance = \
            collisions_loss.DistanceFieldPenetrationLoss(
                sigma=df_cone_height, point2plane=point2plane,
                vectorized=True, penalize_outside=penalize_outside)

        if part_segm_fn:
            # Read the part segmentation
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file,
                                             encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            # Create the module used to filter invalid collision pairs
            filter_faces = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents,
                ign_part_pairs=ign_part_pairs).to(device=device)

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}
    if use_face:
        opt_weights_dict['face_weight'] = face_joints_weights
        opt_weights_dict['expr_prior_weight'] = expr_weights
        opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    if interpenetration:
        opt_weights_dict['coll_loss_weight'] = coll_loss_weights

    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    loss = fitting.create_loss(loss_type=loss_type,
                               joint_weights=joint_weights,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               use_face=use_face, use_hands=use_hands,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               expr_prior=expr_prior,
                               left_hand_prior=left_hand_prior,
                               right_hand_prior=right_hand_prior,
                               jaw_prior=jaw_prior,
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               dtype=dtype,
                               input_2d_joints=input_2d_joints,
                               input_3d_joints=input_3d_joints,
                               **kwargs)
    loss = loss.to(device=device)

    with fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, **kwargs) as monitor:

        img = torch.tensor(img, dtype=dtype)
        H, W, _ = img.shape
        data_weight = 1000 / H

        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        results = []
        orientations = [body_model.global_orient.detach().cpu().numpy()]

        # Step 2: Optimize the full model
        left_hand_model = smplx.create(
            os.path.join(kwargs.get("model_folder"), "mano", "MANO_LEFT.pkl"),
            use_pca=True,
            num_pca_comps=kwargs.get("num_pca_comps"),
            is_rhand=False,
        )
        right_hand_model = smplx.create(
            os.path.join(kwargs.get("model_folder"), "mano", "MANO_RIGHT.pkl"),
            use_pca=True,
            num_pca_comps=kwargs.get("num_pca_comps"),
            is_rhand=True,
        )
        final_loss_val = 0
        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):
            opt_start = time.time()

            for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):
                if opt_idx <= 3:
                    continue

                body_params = list(body_model.parameters())
                betas = [-2.190,-0.549,0.261,0.277,0.204,-0.126,0.236,-0.041,-0.185,-0.078]
                # betas = [-0.471,-1.130,0.289,0.283,0.201,-0.008,0.183,-0.100,-0.127,-0.146]
                body_model.betas.requires_grad = False
                for i in range(len(betas)):
                    body_model.betas[:,i] = betas[i]

                final_params = list(
                    filter(lambda x: x.requires_grad, body_params))
                # final_params.append(camera.translation)

                if use_vposer:
                    final_params.append(pose_embedding)

                body_optimizer, body_create_graph = optim_factory.create_optimizer(
                    final_params,
                    **kwargs)
                body_optimizer.zero_grad()

                curr_weights['data_weight'] = data_weight
                curr_weights['bending_prior_weight'] = (
                    3.17 * curr_weights['body_pose_weight'])
                if use_hands:
                    joint_weights[:, 25:67] = curr_weights['hand_weight']
                if use_face:
                    joint_weights[:, 67:] = curr_weights['face_weight']
                loss.reset_loss_weights(curr_weights)

                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    camera=camera, 
                    gt_joints_2d=gt_joints_2d,
                    gt_joints_3d=gt_joints_3d,
                    joints_conf_2d=joints_conf_2d,
                    joints_conf_3d=joints_conf_3d,
                    # prev_gt_joints=prev_gt_joints,
                    # prev_proj_joints=prev_proj_joints,
                    joint_weights=joint_weights,
                    loss=loss, create_graph=body_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    return_verts=True, return_full_pose=True)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_start = time.time()
                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding, vposer=vposer,
                    use_vposer=use_vposer)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    if interactive:
                        tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                            opt_idx, elapsed))

                # # # self-added
                # if opt_idx == 3:
                #     new_orient = torch.tensor(np.array([np.pi, 0, 0]),
                #                                   dtype=dtype,
                #                                   device=device).unsqueeze(dim=0)
                #     new_params = defaultdict(global_orient=new_orient)
                #     body_model.reset_params(**new_params)
                # print('body pose after stage {:03d}', pose_embedding)
                # print('global_orient after stage {:03d}', body_model.global_orient)



            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - opt_start
                tqdm.write(
                    'Body fitting Orientation {} done after {:.4f} seconds'.format(
                        or_idx, elapsed))
                tqdm.write('Body final loss val = {:.5f}'.format(
                    final_loss_val))

            # Get the result of the fitting process
            # Store in it the errors list in order to compare multiple
            # orientations, if they exist
            result = {'camera_' + str(key): val.detach().cpu().numpy()
                      for key, val in camera.named_parameters()}
            result.update({key: val.detach().cpu().numpy()
                           for key, val in body_model.named_parameters()})
            if use_vposer:
                result['body_pose'] = pose_embedding.detach().cpu().numpy()
            if use_vposer:
                result['body_pose'] = vposer.decode(pose_embedding, output_type='aa').detach().cpu().numpy().reshape((1, 63))
            del result["global_orient"]
            result["left_hand_pose"] = np.dot(result["left_hand_pose"], left_hand_model.np_hand_components)
            result["right_hand_pose"] = np.dot(result["right_hand_pose"], right_hand_model.np_hand_components)

            results.append({'loss': final_loss_val,
                            'result': result})

        with open(result_fn, 'wb') as result_file:
            if len(results) > 1:
                min_idx = (0 if results[0]['loss'] < results[1]['loss']
                           else 1)
            else:
                min_idx = 0
            pickle.dump(results[min_idx]['result'], result_file, protocol=2)

    if save_meshes:
        body_pose = vposer.decode(
            pose_embedding,
            output_type='aa') if use_vposer else None

        body_pose = body_pose.view(1, -1
                                   )
        model_type = kwargs.get('model_type', 'smpl')
        append_wrists = model_type == 'smpl' and use_vposer
        if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)
        model_output = body_model(return_verts=True, body_pose=body_pose)
        proj_joints = camera(model_output.joints)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        import trimesh

        out_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        out_mesh.apply_transform(rot)
        out_mesh.export(mesh_fn)

        pred_joints_3d = utils.scale_pred_joints(gt_joints_3d, model_output.joints)
        plot_pred_joints = pred_joints_3d.detach().cpu().numpy().squeeze()
        plot_gt_joints = gt_joints_3d.detach().cpu().numpy().squeeze()
        pred_joints_x = plot_pred_joints[:,0]
        pred_joints_z = plot_pred_joints[:,2]
        gt_joints_x = plot_gt_joints[:,0]
        gt_joints_z = plot_gt_joints[:,2]
        plt.clf()
        plt.scatter(pred_joints_x,pred_joints_z, c='red')
        plt.scatter(gt_joints_x,gt_joints_z, c='blue')
        plt.savefig(mesh_fn.replace('obj', 'jpg'))
    # if True:
    #     import pyrender
    #     os.environ['PYOPENGL_PLATFORM'] = 'egl'

    #     material = pyrender.MetallicRoughnessMaterial(
    #         metallicFactor=0.0,
    #         alphaMode='OPAQUE',
    #         baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    #     mesh = pyrender.Mesh.from_trimesh(
    #         out_mesh,
    #         material=material)

    #     scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
    #                            ambient_light=(0.3, 0.3, 0.3))
    #     scene.add(mesh, 'mesh')

    #     camera_center = camera.center.detach().cpu().numpy().squeeze()
    #     camera_transl = camera.translation.detach().cpu().numpy().squeeze()
    #     # Equivalent to 180 degrees around the y-axis. Transforms the fit to
    #     # OpenGL compatible coordinate system.
    #     camera_transl[0] *= -1.0

    #     camera_pose = np.eye(4)
    #     camera_pose[:3, 3] = camera_transl

    #     camera_render = pyrender.camera.IntrinsicsCamera(
    #         fx=focal_length, fy=focal_length,
    #         cx=camera_center[0], cy=camera_center[1])
    #     scene.add(camera_render, pose=camera_pose)

    #     # Get the lights from the viewer
    #     light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)

    #     light_pose = np.eye(4)
    #     light_pose[:3, 3] = [0, -1, 1]
    #     scene.add(light, pose=light_pose)

    #     light_pose[:3, 3] = [0, 1, 1]
    #     scene.add(light, pose=light_pose)

    #     light_pose[:3, 3] = [1, 1, 2]
    #     scene.add(light, pose=light_pose)

    #     r = pyrender.OffscreenRenderer(viewport_width=W,
    #                                    viewport_height=H,
    #                                    point_size=1.0)
    #     color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    #     color = color.astype(np.float32) / 255.0

    #     valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    #     input_img = img.detach().cpu().numpy()
    #     output_img = (color[:, :, :-1] * valid_mask +
    #                   (1 - valid_mask) * input_img)

    #     img = pil_img.fromarray((output_img * 255).astype(np.uint8))
    #     img.save(out_img_fn)



    return body_model, camera, pose_embedding #, gt_joints_2d.detach(), proj_joints.detach()