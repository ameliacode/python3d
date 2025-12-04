import os
import sys

import numpy as np
import torch
from pytorch3d.io import load_obj, load_ply, save_obj, save_ply
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

verts, faces = load_ply("./data/pedestrian.ply")
verts = verts.to(device)
faces = faces.to(device)

center = verts.mean(0)
verts = verts - center
scale = max(verts.abs().max(0)[0])
verts = verts / scale
verts = verts[None, :, :]

src_mesh = ico_sphere(4, device)
src_vert = src_mesh.verts_list()

deform_verts = torch.full(src_vert[0].shape, 0.0, device=device, requires_grad=True)

optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

w_chamfer = 1.0
w_edge = 0.0
w_normal = 0.00
w_laplacian = 0.0

for i in range(0, 2000):

    print("i = ", i)

    optimizer.zero_grad()

    new_src_mesh = src_mesh.offset_verts(deform_verts)

    sample_trg = verts
    sample_src = sample_points_from_meshes(new_src_mesh, verts.shape[1])

    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
    loss_edge = mesh_edge_loss(new_src_mesh)
    loss_normal = mesh_normal_consistency(new_src_mesh)
    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

    loss = (
        loss_chamfer * w_chamfer
        + loss_edge * w_edge
        + loss_normal * w_normal
        + loss_laplacian * w_laplacian
    )

    loss.backward()
    optimizer.step()

final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

final_verts = final_verts * scale + center

final_obj = os.path.join("./", "deform2.ply")
save_ply(final_obj, final_verts, final_faces, ascii=True)
