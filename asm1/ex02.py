import imageio
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d
import torch
from starter.utils import *
from tqdm import tqdm, trange


def render_tetrahedron():
    """
    Return 360 gif of the tetrahedron
    """
    device = None
    if device is None:
        device = get_device()
    # Image param
    image_size = 512
    color = [0, 0, 128]

    # Get renderer
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures
    vertices = torch.tensor([[0, 0, 0], [2, -0.5, 1], [1, -0.5, 1], [1, 1, 1]]).float()
    faces = torch.tensor([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]]).float()
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1,N_v, 3) : batch,vertices,color
    textures = textures * torch.tensor(color)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Render
    images = []
    for azim in tqdm(range(0, 360 + 1, 10)):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(
            dist=5, elev=10, azim=azim
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
    imageio.mimsave("output/tetrahedron.gif", images, duration=5)


def render_cube():
    """
    Return 360 gif of the cube
    """
    device = None
    if device is None:
        device = get_device()
    # Image param
    image_size = 512
    color = [0, 0, 128]

    # Get renderer
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures
    vertices = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
            [1, 1, 0],
        ]
    ).float()
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 3, 7],
            [0, 4, 7],
            [0, 2, 6],
            [0, 4, 6],
            [1, 2, 6],
            [1, 5, 6],
            [1, 3, 7],
            [1, 5, 7],
            [4, 5, 6],
            [4, 5, 7],
        ]
    ).float()
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1,N_v, 3) : batch,vertices,color
    textures = textures * torch.tensor(color)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Render
    images = []
    for azim in tqdm(range(0, 360 + 1, 10)):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(
            dist=5, elev=10, azim=azim
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
    imageio.mimsave("output/cube.gif", images, duration=5)


if __name__ == "__main__":
    render_cube()
