import torch
import pytorch3d
from starter.utils import *
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm, trange
import numpy as np


def retexturing(cow_path="data/cow.obj"):

    device = None
    if device is None:
        device = get_device()
    # Image param
    image_size = 512
    color1 = np.array([ 0, 0, 1 ]) #red
    color2 = np.array([ 1, 0, 0 ]) #blue

    # Get renderer
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    ## TODO: Assign color here
    z_min=min(pos[-1] for pos in vertices[0])
    z_max=max(pos[-1] for pos in vertices[0])
    textures = torch.ones_like(vertices)  # (1,N_v, 3) : batch,vertices,color
    for i in range(len(vertices[0])):
        z=float(vertices[0][i][-1])
        alpha = (z - z_min) / (z_max - z_min)
        color = alpha * color2 + (1 - alpha) * color1
        textures[0][i]=color

    ## Load mesh
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)


    # Render
    images = []
    for azim in tqdm(range(0, 360 + 1,10)):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(
            dist=3, elev=0, azim=azim
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        # Place a point light in front of the cow.
        light_pos = (R[0].cpu().detach().numpy() @ np.array([[0.0, 0.0, -3.0]]).T).T
        lights = pytorch3d.renderer.PointLights(location=light_pos, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        rend = (rend * 255).astype(np.uint8)
        images.append(rend)
    # The .cpu moves the tensor to GPU (if needed).
    imageio.mimsave("output/retexturing.gif", images, duration=5)


if __name__ == "__main__":
    retexturing()
