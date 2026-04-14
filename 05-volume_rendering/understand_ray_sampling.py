import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCGridRaysampler,
    PointLights,
    look_at_view_transform,
)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

num_views: int = 10
azimuth_range: float = 180
elev = torch.linspace(0, 0, num_views)
azim = torch.linspace(-azimuth_range, azimuth_range, num_views) + 180.0
lights = PointLights(device=device, location=((0.0, 0.0, -3.0),))
R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

image_size = 64
volume_extent_world = 3.0
raysampler = NDCGridRaysampler(
    image_width=image_size,
    image_height=image_size,
    n_pts_per_ray=50,
    min_depth=0.1,
    max_depth=volume_extent_world,
)

ray_bundle = raysampler(cameras=cameras)

print("ray_bundle origins tensor shape = ", ray_bundle.origins.shape)  # position
print("ray_bundle directions shape = ", ray_bundle.directions.shape)  # directionvector
print("ray_bundle lengths = ", ray_bundle.lengths.shape)
print("ray_bundle xys shape = ", ray_bundle.xys.shape)  # pixel coord

torch.save({"ray_bundle": ray_bundle}, "./checkpoints/ray_sampling.pt")
