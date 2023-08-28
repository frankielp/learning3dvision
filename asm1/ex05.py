from starter.utils import *

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np
from tqdm import trange
import imageio

from starter.render_generic import *

def rgbd2pcl(path="data/rgbd_data.pkl"):
    data=load_rgbd_data()
    device=get_device()
    torch.cuda.set_device(device)
    
    # Unproject pcl
    points1,rgb1=unproject_depth_image(torch.Tensor(data['rgb1']), torch.Tensor(data['mask1']), torch.Tensor(data['depth1']), data['cameras1'])
    points2,rgb2=unproject_depth_image(torch.Tensor(data['rgb2']), torch.Tensor(data['mask2']), torch.Tensor(data['depth2']), data['cameras2'])
    points12=torch.concat([points1,points2],axis=0)
    rgb12=torch.concat([rgb1,rgb2],axis=0)

    # Construct pcl
    points1,points2,points12=points1.unsqueeze(0),points2.unsqueeze(0),points12.unsqueeze(0)
    rgb1,rgb2,rgb12=rgb1.unsqueeze(0),rgb2.unsqueeze(0),rgb12.unsqueeze(0)
    pcl1 = pytorch3d.structures.Pointclouds(points=points1, features=rgb1).to(device)
    pcl2 = pytorch3d.structures.Pointclouds(points=points2, features=rgb2).to(device)
    pcl12 = pytorch3d.structures.Pointclouds(points=points12, features=rgb12).to(device)
    
    # Renderer
    points_renderer = get_points_renderer(
        image_size=256,
        radius=0.01,
    )
    # Set camera
    images1=[]
    images2=[]
    images12=[]
    for azim in trange(0,360+1,10):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(
                dist=5, elev=0, azim=azim
            )
        camera = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )
        # Render pcl
        rend1 = points_renderer(pcl1, cameras=camera)
        rend1 = rend1[0, ..., :3].cpu().numpy()  # (B, H, W, 4) -> (H, W, 3)

        rend2 = points_renderer(pcl2, cameras=camera)
        rend2 = rend2[0, ..., :3].cpu().numpy()  # (B, H, W, 4) -> (H, W, 3)

        rend12 = points_renderer(pcl12, cameras=camera)
        rend12 = rend12[0, ..., :3].cpu().numpy()  # (B, H, W, 4) -> (H, W, 3)

        rend1 = (rend1 * 255).astype(np.uint8)
        rend2 = (rend2 * 255).astype(np.uint8)
        rend12 = (rend12 * 255).astype(np.uint8)

        images1.append(rend1)
        images2.append(rend2)
        images12.append(rend12)

    concat_images=np.concatenate((images1,images2,images12),axis=1)


    imageio.mimsave("output/plant.gif", concat_images, duration=10)


def render_torus(image_size=512, num_samples=200, device=None):
    """
    Renders a torus using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    R=torch.tensor(3.0) # Center of tube to center torus
    r=torch.tensor(1.5) # Radius of tube
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)
    
    x = (R + r * torch.cos(Theta)) * torch.cos(Phi)
    y = (R + r * torch.cos(Theta)) * torch.sin(Phi)
    z = r * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points],
        features=[color],
    ).to(device)

    images=[]
    renderer = get_points_renderer(image_size=image_size, device=device)
    for azim in trange(0,360+1,10):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(
                dist=5, elev=0, azim=azim
            )
        camera = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=120, device=device
        )
        
        rend = renderer(torus_point_cloud, cameras=camera)
        rend=rend[0, ..., :3].cpu().numpy()
        rend = (rend * 255).astype(np.uint8)

        images.append(rend)
       
    imageio.mimsave("output/torus.gif", images, duration=10)

def render_torus_mesh(image_size=512, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -5.5
    max_value = 5.5
    R=torch.tensor(3.0) # Center of tube to center torus
    r=torch.tensor(1.5) # Radius of tube

    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)

    voxels = (torch.sqrt(X**2 + Y**2) - R)**2 + Z**2 - r**2

    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(
        location=[[0, 0.0, -5.0]],
        device=device,
    )
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    images=[]
    for azim in trange(0,360+1,10):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(
                dist=5, elev=0, azim=azim
            )
        camera = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=120, device=device
        )
        rend = renderer(mesh, cameras=camera,lights=lights)
        rend=rend[0, ..., :3].cpu().numpy()
        rend = (rend * 255).astype(np.uint8)

        images.append(rend)
       
    imageio.mimsave("output/torus_mesh.gif", images, duration=10)
    
if __name__=='__main__':
    render_torus_mesh()