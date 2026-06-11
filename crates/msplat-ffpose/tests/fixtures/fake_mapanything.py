"""Stand-in for mapanything_poses.py: same CLI and stdout protocol, no model.

Copies the checked-in tiny COLMAP text model to --output-dir/sparse/ and the
input photos to --output-dir/images/, emitting the full event sequence, so the
Rust orchestration (spawn -> JSON parse -> dataset assembly -> stats) can be
tested end-to-end without Python dependencies, a GPU, or a 1B-param download.
"""

import argparse
import json
import os
import shutil
import sys


def emit(**kw):
    print(json.dumps(kw), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--backend", default="mapanything")
    ap.add_argument("--model-id", default="facebook/map-anything-apache")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--no-exif-intrinsics", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    if os.environ.get("FAKE_FFPOSE_FAIL"):
        emit(type="error", message="synthetic failure", hint="this is a test")
        sys.exit(2)

    emit(type="stage", name="Loading PyTorch", index=1, total=4)
    emit(type="stage", name="Loading model weights", index=2, total=4)
    emit(type="status", message=f"Compute device: {args.device}")
    emit(type="stage", name="Estimating poses", index=3, total=4)
    emit(type="progress", done=3, total=3)
    emit(type="stage", name="Exporting COLMAP model", index=4, total=4)
    print("stray non-JSON line that must be ignored", flush=True)

    fixtures = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_sparse")
    sparse_out = os.path.join(args.output_dir, "sparse")
    images_out = os.path.join(args.output_dir, "images")
    os.makedirs(sparse_out, exist_ok=True)
    os.makedirs(images_out, exist_ok=True)
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        shutil.copy(os.path.join(fixtures, name), os.path.join(sparse_out, name))
    # An extra file the assembly step must skip, like the real exporter's PLY.
    with open(os.path.join(sparse_out, "points.ply"), "w") as f:
        f.write("ply\n")
    for name in sorted(os.listdir(args.images_dir)):
        src = os.path.join(args.images_dir, name)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(images_out, name))

    emit(type="done", views=3)


if __name__ == "__main__":
    main()
