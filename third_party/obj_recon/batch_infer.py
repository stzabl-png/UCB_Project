#!/usr/bin/env python3
"""
Batch object 3D reconstruction on cloud server.

Input  (per sequence): ~/input/{dataset}/{seq_id}/image.png + 0.png
Output (per sequence): ~/output/{dataset}/{seq_id}/mesh.ply + splat.ply

Usage (on sam3d-gpu):
  conda activate sam3d-objects
  cd ~/lyh/sam-3d-objects
  python batch_infer.py --input-dir ~/input --output-dir ~/output
  python batch_infer.py --input-dir ~/input --output-dir ~/output --dataset oakink --limit 3
"""

import os, sys, argparse, json, time
from glob import glob
from natsort import natsorted
from tqdm import tqdm

sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

CHECKPOINT = "checkpoints/hf/pipeline.yaml"


def discover_sequences(input_dir, dataset=None):
    """Yield (dataset, seq_id, seq_dir) for all valid input packages."""
    datasets = [dataset] if dataset else sorted(os.listdir(input_dir))
    for ds in datasets:
        ds_dir = os.path.join(input_dir, ds)
        if not os.path.isdir(ds_dir):
            continue
        for seq_id in natsorted(os.listdir(ds_dir)):
            seq_dir = os.path.join(ds_dir, seq_id)
            img_path  = os.path.join(seq_dir, "image.png")
            mask_path = os.path.join(seq_dir, "0.png")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                yield ds, seq_id, seq_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  default=os.path.expanduser("~/input"))
    parser.add_argument("--output-dir", default=os.path.expanduser("~/output"))
    parser.add_argument("--dataset",    default=None, help="Process only this dataset")
    parser.add_argument("--seq",        default=None, help="Process only matching seq_id")
    parser.add_argument("--limit",      type=int, default=0)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(f" Input  : {args.input_dir}")
    print(f" Output : {args.output_dir}")
    print("=" * 60)

    # Load model once
    print("\nLoading model...")
    inference = Inference(CHECKPOINT, compile=False)
    print("✅ Model loaded\n")

    sequences = list(discover_sequences(args.input_dir, args.dataset))
    if args.seq:
        sequences = [(ds, sid, d) for ds, sid, d in sequences if args.seq in sid]
    if args.limit > 0:
        sequences = sequences[:args.limit]
    print(f"Found {len(sequences)} sequences\n")

    done, skipped, failed = 0, 0, 0
    results = {}

    for ds, seq_id, seq_dir in tqdm(sequences, desc="ObjRecon"):
        out_dir = os.path.join(args.output_dir, ds, seq_id)
        ply_path = os.path.join(out_dir, "splat.ply")

        if os.path.exists(ply_path):
            tqdm.write(f"  ⏭  {ds}/{seq_id}: cached")
            skipped += 1
            continue

        try:
            t0 = time.time()
            image = load_image(os.path.join(seq_dir, "image.png"))
            mask  = load_single_mask(seq_dir, index=0)

            output = inference(image, mask, seed=args.seed)

            os.makedirs(out_dir, exist_ok=True)

            # Save Gaussian Splat (primary output)
            output["gs"].save_ply(ply_path)

            # Try to export mesh (optional, format may vary)
            mesh_saved = False
            if "mesh" in output and output["mesh"] is not None:
                mesh = output["mesh"]
                mesh_path = os.path.join(out_dir, "mesh.ply")
                try:
                    import trimesh
                    # Could be a list of MeshExtractResult objects
                    if isinstance(mesh, list):
                        meshes = mesh
                    else:
                        meshes = [mesh]

                    parts = []
                    for m in meshes:
                        if hasattr(m, 'vertices') and hasattr(m, 'faces') and m.success:
                            verts = m.vertices.cpu().numpy()
                            faces = m.faces.cpu().numpy()
                            colors = m.vertex_attrs.cpu().numpy() if m.vertex_attrs is not None else None
                            parts.append(trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors))
                        elif hasattr(m, 'export'):
                            parts.append(m)

                    if parts:
                        combined = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
                        combined.export(mesh_path)
                        mesh_saved = True
                except Exception as me:
                    tqdm.write(f"    ⚠️  mesh export skipped: {me}")

            tqdm.write(f"  ✅ {ds}/{seq_id}: splat{'+ mesh' if mesh_saved else ''} ({time.time()-t0:.1f}s)")
            results[f"{ds}/{seq_id}"] = {"status": "ok", "ply": ply_path}
            done += 1

        except Exception as e:
            tqdm.write(f"  ❌ {ds}/{seq_id}: {e}")
            results[f"{ds}/{seq_id}"] = {"status": "error", "msg": str(e)}
            failed += 1

    # Save summary
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Done: {done}  ⏭ Skipped: {skipped}  ❌ Failed: {failed}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
