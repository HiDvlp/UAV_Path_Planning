import argparse
import glob
import os

import open3d as o3d


def convert(stl_path: str, out_dir: str) -> None:
    print(f"  Processing: {stl_path}")
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals

    stem = os.path.splitext(os.path.basename(stl_path))[0]
    out_path = os.path.join(out_dir, stem + ".pcd")
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"  Saved {len(pcd.points):,} pts → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .stl files in data/ to .pcd (lossless)")
    parser.add_argument("--data-dir", default="data", help="Directory to scan for .stl files")
    parser.add_argument("--out-dir",  default="data", help="Output directory for .pcd files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    stl_files = sorted(
        glob.glob(os.path.join(args.data_dir, "*.stl")) +
        glob.glob(os.path.join(args.data_dir, "*.STL"))
    )

    if not stl_files:
        print(f"No .stl files found in '{args.data_dir}'")
        return

    print(f"Found {len(stl_files)} STL file(s).")
    for f in stl_files:
        convert(f, args.out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
