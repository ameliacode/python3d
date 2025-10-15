import os
import sys

import matplotlib.pyplot as plt
import open3d
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    look_at_view_transform,
)
from pytorch3d.renderer.mesh.shader import HardPhongShader

sys.path.append(os.path.abspath(""))

DATA_DIR = "./data"

obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
mesh = load_objs_as_meshes([obj_filename], device=device)

R, T = look_at_view_transform(2.7, 0, 180)
cameras = PerspectiveCameras(device=device, R=R, T=T)
lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=HardPhongShader(device=device, cameras=cameras, lights=lights),
)

images = renderer(mesh)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.savefig("light_at_front.png")
plt.show()
