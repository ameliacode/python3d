import os

import numpy as np
import open3d
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures.meshes import join_meshes_as_batch

# estimating depth camera position

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only mode, this will be slow!")

mesh_list = list()
mesh_names = ["cube.obj", "diamond.obj", "dodecahedron.obj"]
DATA_DIR = "./data"

for mesh_name in mesh_names:
    mesh = open3d.io.read_triangle_mesh(os.path.join(DATA_DIR, mesh_name))
    open3d.visualization.draw_geometries(
        [mesh], mesh_show_wireframe=True, mesh_show_back_face=True
    )

    mesh = load_objs_as_meshes([os.path.join(DATA_DIR, mesh_name)], device=device)
    mesh_list.append(mesh)

mesh_batch = join_meshes_as_batch(mesh_list, include_textures=False)

vertex_list = mesh_batch.verts_list()
print("vertex_list = ", vertex_list)
face_list = mesh_batch.faces_list()
print("face_list = ", face_list)

vertex_padded = mesh_batch.verts_padded()
print("vertex_padded = ", vertex_padded)
face_padded = mesh_batch.faces_padded()
print("face_padded = ", face_padded)

vertex_packed = mesh_batch.verts_packed()
print("vertex_packed = ", vertex_packed)
face_packed = mesh_batch.faces_packed()
print("face_packed = ", face_packed)
num_vertices = vertex_packed.shape[0]
print("num_vertices = ", num_vertices)

mesh_batch_noisy = mesh_batch.clone()

motion_gt = np.array([3, 4, 5])
motion_gt = torch.as_tensor(motion_gt)
print("motion ground truth = ", motion_gt)

motion_gt = motion_gt[None, :]
motion_gt = motion_gt.to(device)

noise = (0.1**0.5) * torch.randn(mesh_batch_noisy.verts_packed().shape).to(device)
noise = noise + motion_gt
mesh_batch_noisy = mesh_batch_noisy.offset_verts(noise).detach()

motion_estimate = torch.zeros(motion_gt.shape, device=device, requires_grad=True)
optimizer = torch.optim.SGD([motion_estimate], lr=0.1, momentum=0.9)

for i in range(200):
    optimizer.zero_grad()
    current_mesh_batch = mesh_batch_noisy.offset_verts(
        motion_estimate.repeat(num_vertices, 1)
    )

    sample_trg = sample_points_from_meshes(current_mesh_batch, num_samples=5000)
    sample_src = sample_points_from_meshes(mesh_batch_noisy, num_samples=5000)
    loss, _ = chamfer_distance(sample_trg, sample_src)

    loss.backward()
    optimizer.step()
    print(f"iter: {i+1:04d} | motion_estimation: {motion_estimate}")
