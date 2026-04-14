import torch
from pytorch3d.renderer.implicit.raymarching import EmissionAbsorptionRaymarcher

checkpoint = torch.load("./checkpoints/volume_sampling.pt", weights_only=False)
rays_densities = checkpoint.get("ray_densities")
rays_features = checkpoint.get("rays_features")

ray_marcher = EmissionAbsorptionRaymarcher()
image_features = ray_marcher(rays_densities=rays_densities, rays_features=rays_features)

print("image_features shape = ", image_features.shape)
