import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
    look_at_rotation,
    look_at_view_transform,
)
from skimage import img_as_ubyte
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

OUTPUT_DIR = "./result_cow"
OBJ_FILENAME = "./data/cow_mesh/cow.obj"

cow_mesh = load_objs_as_meshes([OBJ_FILENAME], device=device)
cameras = FoVPerspectiveCameras(device=device)
lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])

blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
    faces_per_pixel=100,
)
renderer_silhoutte = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftSilhouetteShader(blend_params=blend_params),
)

SIGMA = 1e-4
raster_settings_soft = RasterizationSettings(
    image_size=256,
    blur_radius=np.log(1.0 / SIGMA - 1.0) * SIGMA,
    faces_per_pixel=100,
)
renderer_textured = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings_soft),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
)

raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=0.0,
    faces_per_pixel=1,
)
phone_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=HardPhongShader(device=device, cameras=cameras, lights=lights),
)

DISTANCE = 3
ELEVATION = 50.0
AZIMUTH = 0.0
R, T = look_at_view_transform(DISTANCE, ELEVATION, AZIMUTH, device=device)

silhouette = renderer_silhoutte(meshes_world=cow_mesh, R=R, T=T)
image_ref = phone_renderer(meshes_world=cow_mesh, R=R, T=T)
silhouette = silhouette.cpu().numpy()
image_ref = image_ref.cpu().numpy()

plt.figure(figsize=(10, 10))
plt.imshow(silhouette.squeeze()[..., 3])
plt.grid(False)
plt.savefig(os.path.join(OUTPUT_DIR, "target_silhouette.png"))
plt.close()

plt.figure(figsize=(10, 10))
plt.imshow(image_ref.squeeze())
plt.grid(False)
plt.savefig(os.path.join(OUTPUT_DIR, "target_rgb.png"))
plt.close()


class Model(nn.Module):
    def __init__(
        self,
        meshes,
        renderer_silhoutte,
        renderer_textured,
        image_ref,
        weight_silhoutte,
        weight_texture,
    ):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer_silhoutte = renderer_silhoutte
        self.renderer_textured = renderer_textured

        self.weight_silhoutte = weight_silhoutte
        self.weight_texture = weight_texture

        image_ref_silhoutte = torch.from_numpy(
            (image_ref[..., :3].max(-1) != 1).astype(np.float32)
        ).to(device)
        self.register_buffer("image_ref_silhoutte", image_ref_silhoutte)

        image_ref_textured = torch.from_numpy((image_ref[..., :3]).astype(np.float32))
        self.register_buffer("image_ref_textured", image_ref_textured)

        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array([3.0, 6.9, +2.5], dtype=np.float32)).to(
                meshes.device
            )
        )

    def forward(self):
        R = look_at_rotation(self.camera_position[None], device=self.device)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]

        image_silhoutte = self.renderer_silhoutte(meshes_world=self.meshes, R=R, T=T)
        image_textured = self.renderer_textured(meshes_world=self.meshes, R=R, T=T)

        loss_silhoutte = torch.sum(
            (image_silhoutte[..., 3] - self.image_ref_silhoutte) ** 2
        )
        loss_texture = torch.sum(
            (image_textured[..., :3] - self.image_ref_textured) ** 2
        )

        loss = (
            self.weight_silhoutte * loss_silhoutte + self.weight_texture * loss_texture
        )

        return loss, image_silhoutte, image_textured


model = Model(
    meshes=cow_mesh,
    renderer_silhoutte=renderer_silhoutte,
    renderer_textured=renderer_textured,
    image_ref=image_ref,
    weight_silhoutte=1.0,
    weight_texture=0.0,  # disable texture loss
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for i in tqdm(range(200)):
    optimizer.zero_grad()
    loss, image_silhoutte, image_textured = model()
    loss.backward()
    optimizer.step()

    plt.figure()
    plt.imshow(image_silhoutte[..., 3].detach().squeeze().cpu().numpy())
    plt.title(f"iter: {i}, loss: {loss.data:.2f}")
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR, f"soft_silhouette_{i}.png"))
    plt.close()

    plt.figure()
    plt.imshow(image_textured.detach().squeeze().cpu().numpy())
    plt.title(f"iter: {i}, loss: {loss.data:.2f}")
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR, f"soft_texture_{i}.png"))
    plt.close()

    R = look_at_rotation(model.camera_position[None], device=device)
    T = -torch.bmm(R.transpose(1, 2), model.camera_position[None, :, None])[:, :, 0]
    image = phone_renderer(meshes_world=cow_mesh, R=R, T=T)

    plt.figure()
    plt.imshow(image[..., 3].detach().squeeze().cpu().numpy())
    plt.title(f"iter: {i}, loss: {loss.data:.2f}")
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR, f"hard_silhouette_{i}.png"))
    plt.close()

    image = image[0, ..., :3].detach().squeeze().cpu().numpy()
    image = img_as_ubyte(image)

    plt.figure()
    plt.imshow(image[..., :3])
    plt.title(f"iter: {i}, loss: {loss.data:.2f}")
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR, f"hard_texture_{i}.png"))
    plt.close()

    if loss.item() < 800:
        break
