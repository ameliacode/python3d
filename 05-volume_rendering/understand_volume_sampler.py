import torch
from pytorch3d.renderer.implicit.renderer import VolumeSampler
from pytorch3d.structures import Volumes

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

checkpoint = torch.load("./checkpoints/ray_sampling.pt", weights_only=False)
ray_bundle = checkpoint.get("ray_bundle")

batch_size = 10  # num_views
densities = torch.zeros([batch_size, 1, 64, 64, 64]).to(device)
colors = torch.zeros([batch_size, 3, 64, 64, 64]).to(device)
voxel_size = 0.1
volumes = Volumes(
    densities=densities,
    features=colors,
    voxel_size=voxel_size,
)

volume_sampler = VolumeSampler(
    volumes=volumes, sample_mode="bilinear"
)  # acquires density and feature(color)
ray_densities, rays_features = volume_sampler(ray_bundle=ray_bundle)

print("ray_densities shape = ", ray_densities.shape)
print("rays_features shape = ", rays_features.shape)


torch.save(
    {"ray_densities": ray_densities, "rays_features": rays_features},
    "./checkpoints/volume_sampling.pt",
)
