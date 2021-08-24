# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import numpy as np
import open3d as o3d
import time
import torch
from torch.utils.dlpack import from_dlpack
import matplotlib.pyplot as plt

from config import ConfigParser
from common import load_image_file_names, save_poses, load_intrinsic, load_extrinsics


def to_torch(o3d_tensor):
    return from_dlpack(o3d_tensor.to_dlpack())


def integrate(depth_file_names, color_file_names, intrinsic, extrinsics,
              config):
    n_files = len(color_file_names)
    device = o3d.core.Device(config.device)

    vbg = o3d.t.geometry.VoxelBlockGrid(
        ('tsdf', 'weight', 'color'),
        (o3d.core.Dtype.Float32, o3d.core.Dtype.Float32,
         o3d.core.Dtype.Float32), ((1), (1), (3)), 3.0 / 512, 8, 100000,
        o3d.core.Device('CUDA:0'))

    for i in range(n_files):
        start = time.time()

        depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
        color = o3d.t.io.read_image(color_file_names[i]).to(device)
        extrinsic = extrinsics[i]

        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth, intrinsic, extrinsic, config.depth_scale, config.depth_max)

        vbg.integrate(frustum_block_coords, depth, color, intrinsic, extrinsic,
                      config.depth_scale, config.depth_max)

        rendering = vbg.ray_cast(frustum_block_coords, intrinsic, extrinsic,
                                 depth.columns, depth.rows, config.depth_scale,
                                 config.depth_min, config.depth_max, 1)
        normal_map = rendering['normal']

        # This color map is from direct rendering
        color_map = rendering['color']

        # This color map is from rendering via indices (could be differentiable)
        # DLPack interprets a o3d tensor as torch tensor in-place without copying.
        tsdf_volume_th = to_torch(vbg.attribute('tsdf')).flatten()
        weight_volume_th = to_torch(vbg.attribute('weight')).flatten()
        color_volume_th = to_torch(vbg.attribute('color')).view((-1, 3))

        # In-place modification used here, need to think how to make them differentiable
        mask = to_torch(rendering['mask'].to(o3d.core.Dtype.UInt8)).bool()
        ratio = to_torch(rendering['ratio'])
        index = to_torch(rendering['index'])
        color_map_th = torch.zeros(depth.rows, depth.columns, 3).cuda()
        for i in range(8):
            mask_i = mask[:, :, i]
            ratio_i = ratio[:, :, i]
            index_i = index[:, :, i]
            rhs = torch.unsqueeze(ratio_i[mask_i],
                                  -1) * color_volume_th[index_i[mask_i]]
            color_map_th[mask_i] += rhs / 255.0

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(color_map.cpu().numpy())
        axs[0].set_title('From Open3D Ray Casting')
        axs[1].imshow(color_map_th.cpu().numpy())
        axs[1].set_title('From Torch')
        plt.tight_layout()
        plt.show()

        stop = time.time()
        print('{:04d}/{:04d} integrate takes {:.4}s'.format(
            i, n_files, stop - start))

    return vbg


if __name__ == '__main__':
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    parser.add('--path_trajectory',
               help='path to the trajectory .log or .json file.')
    config = parser.get_config()

    depth_file_names, color_file_names = load_image_file_names(config)
    intrinsic = load_intrinsic(config)
    extrinsics = load_extrinsics(config.path_trajectory, config)

    vbg = integrate(depth_file_names, color_file_names, intrinsic, extrinsics,
                    config)

    pcd = vbg.extract_surface_points()
    o3d.visualization.draw([pcd])
