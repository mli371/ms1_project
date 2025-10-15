# scripts/normalize_ply.py
import sys, numpy as np

def load_ply(fn):
    pts = []
    with open(fn) as f:
        header = True
        while True:
            line = f.readline()
            if not line:
                break
            if line.strip().startswith('end_header'):
                break
            if 'element vertex' in line:
                n = int(line.strip().split()[-1])
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except:
                    pass
    return np.array(pts)

def save_ply(fn, pts):
    with open(fn, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

if __name__=='__main__':
    src = sys.argv[1]; dst = sys.argv[2]
    pts = load_ply(src)
    c = pts.mean(0); pts -= c
    s = (np.max(np.linalg.norm(pts, axis=1)) + 1e-9)
    pts /= s
    save_ply(dst, pts)
