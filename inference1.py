import json
import numpy as np
import torch
import pyfqmr
import pickle
import trimesh
from scipy import spatial
from models.SWSNet import LitSWSNet
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


def main(root, modelk):
    mesh = trimesh.load(root)
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
    # label = label.cuda()
    out = modelk(x, pos).squeeze(0)
    out = torch.argmax(out, dim=1).cpu().numpy()

    mesh_pred = color_mesh(mesh, out)
    mesh_pred.export(root.replace('.obj', '_k.obj'))


if __name__ == '__main__':
    model = LitSWSNet.load_from_checkpoint(
        'logs/t1125/2/epoch=82-step=49800.ckpt', map_location='cpu'
    )
    model.eval()
    model1 = model.model.cuda()
    data = 'MB Preparation scan1.obj'
    main(data, model1)





