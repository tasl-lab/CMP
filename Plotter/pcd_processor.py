# Import Open3D
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.transformation_utils import x1_to_x2, x_to_world
from opencood.utils.pcd_utils import pcd_to_np, mask_points_by_range, lidar_project, downsample_lidar


X_RANGE= 140.8
Y_RANGE = 38.4

def GetCAVLocation(path_to_yaml: str):
    # Load lidar pose.
    lidar_pose = load_yaml(path_to_yaml)['lidar_pose']

    return lidar_pose

def ExtractPointWorldIntoFrame(path_to_pcd: str,
                               path_to_yaml: str):
    # Load lidar pose.
    lidar_pose = load_yaml(path_to_yaml)['lidar_pose']


    # Load the point cloud data
    points_intensities = pcd_to_np(path_to_pcd)

    # Filter by range.
    points_intensities = mask_points_by_range(points_intensities, [-X_RANGE, -Y_RANGE, -3, X_RANGE, Y_RANGE, 1])
    points_intensities = downsample_lidar(points_intensities, 50000)

    # Transform from LIDAR frame to world frame.
    lidar_to_world = x_to_world(lidar_pose)
    points_intensities = lidar_project(points_intensities, lidar_to_world)


    # Cut into xyz and intensity.
    points = points_intensities[:, :3]
    intensities = points_intensities[:, 3]

    return points, intensities

if __name__ == "__main__":
    points, intensities = ExtractPointWorldIntoFrame(r'E:\ProjectFiles\OpenV2V\test\2021_08_18_19_48_05\1045\000068.pcd',
        r'E:\ProjectFiles\OpenV2V\test\2021_08_18_19_48_05\1045\000068.yaml')
    # Plotting the point cloud in 2D with intensity colors
    plt.scatter(points[:, 0], points[:, 1], c=intensities, s=1)
    plt.colorbar(label='Intensity')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('2D Point Cloud with Intensity Color Mapping')
    plt.show()