#!/usr/bin/env python3
import sys, os
src = sys.argv[1]
dst = sys.argv[2]
try:
    import open3d as o3d
except Exception as e:
    print("ERROR: open3d not available. Install with `pip install open3d` or `conda install -c open3d-admin open3d`")
    raise SystemExit(1)

pcd = o3d.io.read_point_cloud(src)
if len(pcd.points) == 0:
    raise SystemExit("Empty point cloud: " + src)
import numpy as np
pts = np.asarray(pcd.points)
center = pts.mean(axis=0)
pts = pts - center
scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-9
pts = pts / scale
pcd.points = o3d.utility.Vector3dVector(pts)

# estimate normals if not present
normals = np.asarray(pcd.normals)
if normals.shape[0] != pts.shape[0]:
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.normalize_normals()

o3d.io.write_point_cloud(dst, pcd, write_ascii=True)
print("Wrote normalized ply with normals to:", dst)
# print head
with open(dst,'r') as f:
    for i,line in enumerate(f):
        if i>=20: break
        print(line.rstrip())
