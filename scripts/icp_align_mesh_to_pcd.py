import argparse
import sys
import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align an initial mesh to a point cloud via point-to-point ICP",
    )
    parser.add_argument("in_mesh", help="Path to input OBJ mesh")
    parser.add_argument("target_pcd", help="Path to target PLY point cloud")
    parser.add_argument(
        "out_mesh",
        nargs="?",
        default="aligned_initmesh.obj",
        help="Path to write aligned mesh (default: aligned_initmesh.obj)",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=8000,
        help="Number of points to sample from mesh for ICP (default: 8000)",
    )
    parser.add_argument(
        "--threshold-scale",
        type=float,
        default=1.0,
        help="Multiplier for ICP distance threshold (default: 1.0)",
    )
    return parser.parse_args()


args = parse_args()
in_mesh, target_pcd, out_mesh = args.in_mesh, args.target_pcd, args.out_mesh

mesh = o3d.io.read_triangle_mesh(in_mesh)
assert mesh.has_vertices(), "mesh has no vertices"
pcd = o3d.io.read_point_cloud(target_pcd)
assert len(pcd.points) > 0, "pcd empty"

mesh_pcd = mesh.sample_points_uniformly(number_of_points=args.n_points)

mc = mesh_pcd.get_center()
pc = pcd.get_center()
mesh_pcd.translate(-mc)
mesh.translate(-mc)
pcd.translate(-pc)

mb = np.asarray(mesh_pcd.get_max_bound() - mesh_pcd.get_min_bound())
pb = np.asarray(pcd.get_max_bound() - pcd.get_min_bound())
scale = float(np.max(pb) / (np.max(mb) + 1e-9))
mesh.scale(scale, center=(0, 0, 0))
mesh_pcd.scale(scale, center=(0, 0, 0))

thr = 0.05 * args.threshold_scale * max(np.linalg.norm(pb), 1.0)
reg = o3d.pipelines.registration.registration_icp(
    mesh_pcd,
    pcd,
    thr,
    np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
)
print("ICP fitness:", reg.fitness, "inlier_rmse:", reg.inlier_rmse)
print("T=\n", reg.transformation)
mesh.transform(reg.transformation)

ok = o3d.io.write_triangle_mesh(out_mesh, mesh, write_ascii=True)
if not ok:
    print("Failed to write", out_mesh)
else:
    print("Wrote:", out_mesh)
