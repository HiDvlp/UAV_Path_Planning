import argparse
import glob
import os

import open3d as o3d


def convert(stl_path: str, n_points: int, out_dir: str) -> None:
    print(f"  Processing: {stl_path}")
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    pcd = mesh.sample_points_uniformly(number_of_points=n_points)

    stem = os.path.splitext(os.path.basename(stl_path))[0]
    out_path = os.path.join(out_dir, stem + ".pcd")
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"  Saved {len(pcd.points):,} pts → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .stl files in data/ to .pcd")
    parser.add_argument("--data-dir", default="data", help="Directory to scan for .stl files")
    parser.add_argument("--out-dir",  default="data", help="Output directory for .pcd files")
    parser.add_argument("--n-points", type=int, default=500_000,
                        help="Number of surface sample points (default: 500000)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    stl_files = sorted(
        glob.glob(os.path.join(args.data_dir, "*.stl")) +
        glob.glob(os.path.join(args.data_dir, "*.STL"))
    )

    if not stl_files:
        print(f"No .stl files found in '{args.data_dir}'")
        return

    print(f"Found {len(stl_files)} STL file(s). Sampling {args.n_points:,} pts each.")
    for f in stl_files:
        convert(f, args.n_points, args.out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
