import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import trimesh
import pyrender
from serialization import load_model

## Load SMPL model
m = load_model(
    "./08-smpl/smplify/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
)

## Assign random pose and shape parameters
m.pose[:] = np.random.rand(m.pose.size) * 0.2
m.betas[:] = np.random.rand(m.betas.size) * 0.03
m.pose[0] = np.pi

## Get evaluated vertices and faces
vertices = np.array(m.r)
faces = m.f.astype(np.int32)

## Build trimesh with light grey vertex colors
vertex_colors = np.ones([vertices.shape[0], 4]) * [0.9, 0.9, 0.9, 1.0]
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)

## Create pyrender scene
scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2])
scene.add(pyrender.Mesh.from_trimesh(mesh))

## Pinhole camera matching original intrinsics (fx=fy=w/2, cx=w/2, cy=h/2)
w, h = 640, 480
camera = pyrender.IntrinsicsCamera(
    fx=w / 2.0, fy=w / 2.0, cx=w / 2.0, cy=h / 2.0, znear=0.1, zfar=10.0
)
# Place camera at z=2, pointing toward -Z (pyrender/OpenGL convention)
camera_pose = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)
scene.add(camera, pose=camera_pose)

## Point light (mirrors original light_pos=[-1000,-1000,-2000] direction, scaled)
light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=5.0)
light_pose = np.eye(4)
light_pose[:3, 3] = [-1.0, -1.0, 0.0]
scene.add(light, pose=light_pose)

## Offline render
r = pyrender.OffscreenRenderer(w, h)
color, _ = r.render(scene)
r.delete()

# pyrender returns RGB; cv2 expects BGR
bgr = color[:, :, ::-1]

out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'render_smpl.png')
cv2.imwrite(out_path, bgr)
print(f"Saved to {os.path.abspath(out_path)}")

cv2.imshow("render_SMPL", bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
