#!/usr/bin/env python3
"""Batch inference: compare pretrained vs newly trained model across multiple meshes."""
import os, sys, json, time
import numpy as np
import trimesh
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference.predictor import AffordancePredictor
from model.pointnet2 import PointNet2Seg

MESHES = [
    "data_hub/meshes/contactpose/cell_phone.obj",
    "data_hub/meshes/contactpose/apple.obj",
    "data_hub/meshes/contactpose/hammer.obj",
    "data_hub/meshes/contactpose/knife.obj",
    "data_hub/meshes/contactpose/mouse.obj",
    "data_hub/meshes/contactpose/scissors.obj",
    "data_hub/meshes/contactpose/stapler.obj",
    "data_hub/meshes/contactpose/toothbrush.obj",
    "data_hub/meshes/contactpose/flashlight.obj",
    "data_hub/meshes/contactpose/light_bulb.obj",
    "data_hub/meshes/contactpose/water_bottle.obj",
    "data_hub/meshes/contactpose/ps_controller.obj",
]

CHECKPOINTS = {
    "pretrained_v5": ("output/checkpoints/best_model.pth", "auto"),
    "newly_trained":  ("output/checkpoints_v5/final_model.pth", "v5_multitask"),
}

THRESHOLD = 0.3


class V5MultiTaskPredictor:
    """Handles the v5 multi-task model: 7ch input, 2-class seg + fc_head."""
    def __init__(self, checkpoint, device="cuda", num_points=1024):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.num_points = num_points
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        self.model = PointNet2Seg(num_classes=2, in_channel=7, predict_force_center=True).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"  V5MultiTaskPredictor loaded: 7ch, 2-class seg + fc_head")

    def predict(self, mesh_path, num_points=None):
        n = num_points or self.num_points
        mesh = trimesh.load(mesh_path, force='mesh')
        points, face_idx = mesh.sample(n, return_index=True)
        normals = mesh.face_normals[face_idx]
        points = points.astype(np.float32)
        normals = normals.astype(np.float32)
        # 7ch: xyz + normals + zeros (no human_prior available for arbitrary meshes)
        hp = np.zeros((len(points), 1), dtype=np.float32)
        features = np.concatenate([points, normals, hp], axis=-1)
        pts_t = torch.from_numpy(points).unsqueeze(0).to(self.device)
        feat_t = torch.from_numpy(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            seg_pred, fc_pred = self.model(pts_t, feat_t)
        contact_prob = torch.softmax(seg_pred, dim=-1)[0, :, 1].cpu().numpy()
        force_center = fc_pred[0].cpu().numpy()
        return points, normals, contact_prob, force_center

results = {}
for ckpt_name, (ckpt_path, model_type) in CHECKPOINTS.items():
    if not os.path.exists(ckpt_path):
        print(f"SKIP {ckpt_name}: {ckpt_path} not found")
        continue
    print(f"\n{'='*70}")
    print(f"  Model: {ckpt_name} ({ckpt_path})")
    print(f"{'='*70}")
    if model_type == "v5_multitask":
        predictor = V5MultiTaskPredictor(checkpoint=ckpt_path, device="cuda", num_points=1024)
    else:
        predictor = AffordancePredictor(checkpoint=ckpt_path, device="cuda", num_points=1024)
    results[ckpt_name] = {}

    for mesh_path in MESHES:
        obj_name = os.path.splitext(os.path.basename(mesh_path))[0]
        if not os.path.exists(mesh_path):
            print(f"  SKIP {obj_name}: not found")
            continue
        try:
            t0 = time.time()
            points, normals, probs, fc = predictor.predict(mesh_path, num_points=1024)
            dt = time.time() - t0
            contact_mask = probs > THRESHOLD
            n_contact = int(contact_mask.sum())
            # Also check lower threshold
            contact_mask_01 = probs > 0.1
            n_contact_01 = int(contact_mask_01.sum())

            r = {
                "n_contact_0.3": n_contact,
                "n_contact_0.1": n_contact_01,
                "prob_min": float(probs.min()),
                "prob_max": float(probs.max()),
                "prob_mean": float(probs.mean()),
                "prob_std": float(probs.std()),
                "force_center": fc.tolist() if fc is not None else None,
                "time_s": round(dt, 3),
            }
            results[ckpt_name][obj_name] = r
            print(f"  {obj_name:>15s}: contacts={n_contact:4d}(0.3) {n_contact_01:4d}(0.1)  "
                  f"prob=[{probs.min():.3f},{probs.max():.3f}]  mean={probs.mean():.3f}  {dt:.2f}s")
        except Exception as e:
            print(f"  {obj_name:>15s}: ERROR - {e}")
            results[ckpt_name][obj_name] = {"error": str(e)}

# Save results
out_path = "output/inference_comparison.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
