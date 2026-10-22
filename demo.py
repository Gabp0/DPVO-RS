import os
import tempfile
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import numpy as np
import torch
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory, save_output_for_COLMAP, save_ply
from dpvo.stream import image_stream, video_stream
from dpvo.utils import Timer

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

def fisheye_remap(calib, shape, balance=0.0):
    """ remap tables for the equidistant (fisheye) model

    The image stream undistorts with cv2.undistort, which assumes the pinhole
    radtan model, so equidistant calibrations have to be handled separately.
    Expects fx fy cx cy k1 k2 k3 k4 in the calibration file. """

    calib = np.loadtxt(calib, delimiter=" ")
    assert len(calib) >= 8, "fisheye calibration needs fx fy cx cy k1 k2 k3 k4"

    fx, fy, cx, cy = calib[:4]
    D = calib[4:8].reshape(4, 1)

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    h, w = shape
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new, (w, h), cv2.CV_32FC1)

    intrinsics = np.array([K_new[0,0], K_new[1,1], K_new[0,2], K_new[1,2]])
    return map1, map2, intrinsics

@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, skip=0, viz=False, timeit=False, fisheye=False, balance=0.0):

    slam = None
    queue = Queue(maxsize=8)

    # the reader only knows the pinhole model, so hand it the intrinsics alone
    # and undistort here instead
    stream_calib = calib
    if fisheye:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        tmp.write(" ".join(map(str, np.loadtxt(calib, delimiter=" ")[:4])))
        tmp.close()
        stream_calib = tmp.name

    if os.path.isdir(imagedir):
        reader = Process(target=image_stream, args=(queue, imagedir, stream_calib, stride, skip))
    else:
        reader = Process(target=video_stream, args=(queue, imagedir, stream_calib, stride, skip))

    reader.start()

    maps, fisheye_intrinsics = None, None

    while 1:
        (t, image, intrinsics) = queue.get()
        if t < 0: break

        if fisheye:
            if maps is None:
                *maps, fisheye_intrinsics = fisheye_remap(calib, image.shape[:2], balance)
            image = cv2.remap(image, maps[0], maps[1], interpolation=cv2.INTER_LINEAR)
            intrinsics = fisheye_intrinsics

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            _, H, W = image.shape
            slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)

        with Timer("SLAM", enabled=timeit):
            slam(t, image, intrinsics)

    reader.join()

    if fisheye:
        os.unlink(stream_calib)
        intrinsics = fisheye_intrinsics

    points = slam.pg.points_.cpu().numpy()[:slam.m]
    colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[:slam.m]

    return slam.terminate(), (points, colors, (*intrinsics, H, W))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--imagedir', type=str)
    parser.add_argument('--calib', type=str)
    parser.add_argument('--name', type=str, help='name your run', default='result')
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--fisheye', action="store_true", help='undistort with the equidistant model (fx fy cx cy k1 k2 k3 k4)')
    parser.add_argument('--balance', type=float, default=0.0, help='fisheye undistortion balance: 0 keeps only valid pixels, 1 keeps the full field of view')
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_ply', action="store_true")
    parser.add_argument('--save_colmap', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    print("Running with config...")
    print(cfg)

    (poses, tstamps), (points, colors, calib) = run(cfg, args.network, args.imagedir, args.calib, args.stride, args.skip, args.viz, args.timeit, args.fisheye, args.balance)
    trajectory = PoseTrajectory3D(positions_xyz=poses[:,:3], orientations_quat_wxyz=poses[:, [6, 3, 4, 5]], timestamps=tstamps)

    if args.save_ply:
        save_ply(args.name, points, colors)

    if args.save_colmap:
        save_output_for_COLMAP(args.name, trajectory, points, colors, *calib)

    if args.save_trajectory:
        Path("saved_trajectories").mkdir(exist_ok=True)
        file_interface.write_tum_trajectory_file(f"saved_trajectories/{args.name}.txt", trajectory)

    if args.plot:
        Path("trajectory_plots").mkdir(exist_ok=True)
        plot_trajectory(trajectory, title=f"DPVO Trajectory Prediction for {args.name}", filename=f"trajectory_plots/{args.name}.pdf")


        

