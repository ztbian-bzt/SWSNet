import json
import numpy as np
import torch
import pyfqmr
import pickle
import trimesh
from scipy import spatial
from models.SWSNet import LitSWSNet
from models.dilated_tooth_seg_network import LitDilatedToothSegmentationNetwork
from models.meshsegnet import LitMeshSegNetwork
from models.DBGANet import LitDBGANetwork
from models.THISNet import LitTHISNetwork
from models.module import get_downsample_dilated_idx
import os
from dataset.mesh_dataset import process_mesh
from dataset.preprocessing import *
from utils.teeth_numbering import color_mesh, fdi_to_label, downsample_mesh


def donwscale_mesh(mesh, labels=None):
    simple_count = 16000
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
    mesh_simplifier.simplify_mesh(target_count=simple_count, aggressiveness=3, preserve_border=True, verbose=0,
                                  max_iterations=2000)
    new_positions, new_face, _ = mesh_simplifier.getMesh()
    mesh_simple = trimesh.Trimesh(vertices=new_positions, faces=new_face)
    vertices = mesh_simple.vertices
    faces = mesh_simple.faces
    if faces.shape[0] < simple_count:
        fs_diff = simple_count - faces.shape[0]
        faces = np.append(faces, np.zeros((fs_diff, 3), dtype="int"), 0)
    elif faces.shape[0] > simple_count:
        mesh_simple = trimesh.Trimesh(vertices=vertices, faces=faces)
        samples, face_index = trimesh.sample.sample_surface_even(mesh_simple, simple_count)
        mesh_simple = trimesh.Trimesh(vertices=mesh_simple.vertices, faces=mesh_simple.faces[face_index])
        faces = mesh_simple.faces
        vertices = mesh_simple.vertices
    mesh_simple = trimesh.Trimesh(vertices=vertices, faces=faces)

    mesh_v_mean = mesh.vertices[mesh.faces].mean(axis=1)
    mesh_simple_v = mesh_simple.vertices
    tree = spatial.KDTree(mesh_v_mean)
    query = mesh_simple_v[faces].mean(axis=1)
    distance, index = tree.query(query)
    if labels is not None:
        labels = labels[index].flatten()
        return mesh_simple, labels
    return mesh_simple


def main(data_root, modelk):
    root = 'data/3dteethseg/raw'
    index, jaw = data_root.split('_')
    dir_root = os.path.join('output', jaw, index)
    os.makedirs(dir_root, exist_ok=True)
    root = os.path.join(root, jaw, index, data_root) + '.obj'

    mesh = trimesh.load(root)
    try:
        with open(root.replace('.obj', '.json')) as f:
            data_raw = json.load(f)
        labels = np.array(data_raw["labels"])
        labels = labels[mesh.faces]
        labels = labels[:, 0]
        labels = fdi_to_label(labels)
    except FileNotFoundError:
        labels = None
    if labels is not None:
        mesh, labels = donwscale_mesh(mesh, labels)
    else:
        mesh = donwscale_mesh(mesh)
    mesh.vertices -= mesh.vertices.mean(axis=0)
    if labels is not None:
        data = process_mesh(mesh, torch.from_numpy(labels).long())
    else:
        data = process_mesh(mesh)

    pretrans = PreTransform(classes=17)
    pos, x, label = pretrans(data)
    pos = pos.unsqueeze(0).cuda()
    x = x.unsqueeze(0).cuda()
    label = label.cuda()
    out = modelk(x, pos).squeeze(0)
    out = torch.argmax(out, dim=1).cpu().numpy()

    # pretrans = PreTransformDBGANet(classes=17)
    # x, y, centroids, y_classes = pretrans(data)
    # x = x.unsqueeze(0).transpose(2, 1).cuda()
    # pred, _, _, _ = modelk(x)
    # pred = pred.squeeze(0)
    # out = torch.argmax(pred, dim=1).cpu().numpy()

    # pretrans = PreTransformTHISNet(classes=17)
    # x, y = pretrans(data)
    # x = x.unsqueeze(0).cuda()
    # y = y.unsqueeze(0).cuda()
    # y1 = y - 1
    # y1[y1 < 0] = 16
    # output, pred = modelk(x.transpose(2, 1))
    # pred = torch.zeros_like(y)
    # for item in range(len(output["pred_masks"])):
    #     pred_scores = output["pred_logits"][item].sigmoid()
    #     pred_masks = output["pred_masks"][item].sigmoid()
    #     pred_objectness = output["pred_scores"][item].sigmoid()
    #     pred_scores = torch.sqrt(pred_scores * pred_objectness)
    #     # max/argmax
    #     scores, labels = pred_scores.max(dim=-1)
    #     # cls threshold
    #     keep = scores > 0.5
    #     scores = scores[keep]
    #     labels = labels[keep]
    #     mask_pred_per_image = pred_masks[keep]
    #     pred_masks = mask_pred_per_image > 0.5
    #     index = torch.where(pred_masks.sum(0) > 1)  # overlay points
    #     pred_masks[:, index[0].cpu().numpy()] = 0
    #     pred[item, :] = (pred_masks * (labels[:, None] + 1)).sum(0)
    # out = pred.squeeze(0).cpu().numpy()


    # mesh_gt = color_mesh(mesh, label.cpu().numpy())
    # mesh_gt.export(os.path.join(dir_root, data_root + '_gt.obj'))
    mesh_pred = color_mesh(mesh, out)
    mesh_pred.export(os.path.join(dir_root, data_root + '.obj'))

    # object = 1000
    # idx_fps, idx_l, idx = get_downsample_dilated_idx(16, 16, 8, pos, out=True)
    # out = torch.zeros([label.shape[0]])
    # out[idx_fps.squeeze(0)] = 2
    # out1 = out.long().cpu().numpy()
    # mesh_fps = downsample_mesh(mesh, out1)
    # mesh_fps.export(f'data_00OMSZGW_lower_fps16.obj')

    # pos1 = pos.squeeze(0)[idx_fps.squeeze(0), :]
    # new_mesh = trimesh.Trimesh(vertices=pos1.squeeze(0).cpu())
    #
    # new_mesh.export(f'2.obj')


    # out[idx_l.squeeze(0)[object, :].squeeze(0)] = 3
    # out[idx.squeeze(0)[object, :].squeeze(0)] = 4
    # out[object] = 1
    # out2 = out.long().cpu().numpy()
    # mesh_fps = downsample_mesh(mesh, out2)
    # mesh_fps.export(f'data_00OMSZGW_lower_final16.obj')


if __name__ == '__main__':
    model = LitSWSNet.load_from_checkpoint(
        'logs/t1125/2/epoch=82-step=49800.ckpt', map_location='cpu'
    )
    # model = LitDilatedToothSegmentationNetwork.load_from_checkpoint(
    #     'logs/original1/1/epoch=98-step=59400.ckpt', map_location='cpu'
    # )
    # model = LitMeshSegNetwork.load_from_checkpoint(
    #     'logs/meshseg/2/epoch=85-step=51600.ckpt', map_location='cpu'
    # )
    # model = LitDBGANetwork.load_from_checkpoint(
    #     'logs/dbga/3/epoch=93-step=56400.ckpt', map_location='cpu'
    # )
    # model = LitTHISNetwork.load_from_checkpoint(
    #     'logs/this2/1/epoch=70-step=42600.ckpt', map_location='cpu'
    # )
    model.eval()
    model1 = model.model.cuda()
    with open('data/teeth_count/Qualitative.txt') as file:
        for line in file:
            data_line = line.strip()
            print(data_line)
            main(data_line, model1)





