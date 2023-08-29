import torch
import pytorch3d
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm, trange
import numpy as np
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
from pytorch3d.io import load_obj

def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:3")
    else:
        device = torch.device("cpu")
    return device


def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.

    Returns:
        PointsRenderer.
    """
    if device is None:
        device=get_device()
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        device=get_device()
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer

def render_360_mesh(input_mesh,output_path="output/mesh.gif"):
    """
    Return 360 gif of the object
    """
    device = None
    if device is None:
        device = get_device()
    # Image param
    image_size = 512

    # Get renderer
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures
    vertices, faces = input_mesh
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Render
    images = []
    for azim in tqdm(range(0, 360 + 1)):
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
    imageio.mimsave(output_path, images, duration=5)

def render_360_points(points,output_path="output/points.gif"):
    image_size=512
    device=None
    if device is None:
        device = get_device()

    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points],
        features=[color],
    ).to(device)

    images=[]
    renderer = get_points_renderer(image_size=image_size, device=device)
    for azim in trange(0,360+1,10):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(
                dist=3, elev=0, azim=azim
            )
        camera = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )
        
        rend = renderer(torus_point_cloud, cameras=camera)
        rend=rend[0, ..., :3].cpu().detach().numpy()
        rend = (rend * 255).astype(np.uint8)

        images.append(rend)
       
    imageio.mimsave(output_path, images, duration=10)