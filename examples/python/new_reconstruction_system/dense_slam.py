# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import time

from config import ConfigParser
from common import load_image_file_names, save_poses, load_intrinsic, extract_trianglemesh


def slam(depth_file_names, color_file_names, intrinsic, config):
    n_files = len(color_file_names)
    device = o3d.core.Device(config.device)

    T_frame_to_model = o3d.core.Tensor(np.identity(4))
    model = o3d.t.pipelines.slam.Model(config.voxel_size, config.sdf_trunc, 16,
                                       config.block_count, T_frame_to_model,
                                       device)
    depth_ref = o3d.t.io.read_image(depth_file_names[0])
    input_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows, depth_ref.columns,
                                             intrinsic, device)
    raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows,
                                               depth_ref.columns, intrinsic,
                                               device)

    poses = []

    for i in range(n_files):
        start = time.time()

        depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
        color = o3d.t.io.read_image(color_file_names[i]).to(device)

        input_frame.set_data_from_image('depth', depth)
        input_frame.set_data_from_image('color', color)

        if i > 0:
            result = model.track_frame_to_model(input_frame, raycast_frame,
                                                config.depth_scale,
                                                config.depth_max,
                                                config.odometry_distance_thr)
            T_frame_to_model = T_frame_to_model @ result.transformation

        poses.append(T_frame_to_model.cpu().numpy())
        model.update_frame_pose(i, T_frame_to_model)
        model.integrate(input_frame, config.depth_scale, config.depth_max)
        model.synthesize_model_frame(raycast_frame, config.depth_scale,
                                     config.depth_min, config.depth_max, False)
        stop = time.time()
        print('{:04d}/{:04d} slam takes {:.4}s'.format(i, n_files,
                                                       stop - start))

    return model.voxel_grid, poses


if __name__ == '__main__':
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    config = parser.get_config()

    depth_file_names, color_file_names = load_image_file_names(config)
    intrinsic = load_intrinsic(config)

    volume, poses = slam(depth_file_names, color_file_names, intrinsic, config)
    save_poses('output.log', poses)
    save_poses('output.json', poses)

    mesh = extract_trianglemesh(volume, config, 'output.ply')
    o3d.visualization.draw([mesh])
