import os

import matplotlib.pyplot as plt
import open3d
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    HardPhongShader,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    look_at_view_transform,
)


class Mesh3DRenderer:
    def __init__(self, mesh_path, device="cuda"):
        self.device = torch.device(device)
        self.mesh_path = mesh_path
        self.mesh = self._load()
        self.pytorch3d_mesh = load_objs_as_meshes([mesh_path], device=self.device)

    def _load(self):
        return open3d.io.read_triangle_mesh(self.mesh_path)

    def show(self):
        open3d.visualization.draw_geometries(
            [self.mesh],
            mesh_show_wireframe=True,
            mesh_show_back_face=True,
        )

    def setup(self, image_size=512):
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
        return MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self._camera(2.7, 0, 180), raster_settings=raster_settings
            ),
            shader=HardPhongShader(
                device=self.device,
                cameras=self._camera(2.7, 0, 180),
                lights=lights,
            ),
        )

    def _camera(self, dist, elev, azim):
        R, T = look_at_view_transform(dist, elev, azim)
        return PerspectiveCameras(device=self.device, R=R, T=T)

    def render(
        self, renderer, lights, materials=None, cameras=None, filename="output.png"
    ):
        render_kwargs = {"lights": lights}
        if materials:
            render_kwargs["materials"] = materials
        if cameras:
            render_kwargs["cameras"] = cameras

        images = renderer(self.pytorch3d_mesh, **render_kwargs)
        plt.figure(figsize=(10, 10))
        plt.imshow(images[0, ..., :3].cpu().numpy())
        plt.axis("off")
        plt.savefig(filename)
        plt.close()


def main():
    DATA_DIR = "./data"
    mesh_file = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

    renderer = Mesh3DRenderer(mesh_file)
    renderer.show()

    mesh_renderer = renderer.setup()

    lights = PointLights(device=renderer.device, location=[[0.0, 0.0, -3.0]])
    renderer.render(mesh_renderer, lights, filename="light_at_front.png")

    lights.location = torch.tensor([0.0, 0.0, 1.0], device=renderer.device)[None]
    renderer.render(mesh_renderer, lights, filename="light_at_back.png")

    R, T = look_at_view_transform(dist=2.7, elev=10, azim=-150)
    cameras = FoVPerspectiveCameras(device=renderer.device, R=R, T=T)
    lights.location = torch.tensor([[2.0, 2.0, -2.0]], device=renderer.device)

    materials = Materials(
        device=renderer.device, specular_color=[[0.0, 1.0, 0.0]], shininess=10.0
    )
    renderer.render(mesh_renderer, lights, materials, cameras, "green.png")

    materials = Materials(
        device=renderer.device, specular_color=[[1.0, 0.0, 0.0]], shininess=20.0
    )
    renderer.render(mesh_renderer, lights, materials, cameras, "red.png")

    materials = Materials(
        device=renderer.device, specular_color=[[0.0, 0.0, 1.0]], shininess=0.0
    )
    renderer.render(mesh_renderer, lights, materials, cameras, "blue.png")


if __name__ == "__main__":
    main()
