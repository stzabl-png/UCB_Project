#!/usr/bin/env python3
"""Comprehensive comparison: pretrained checkpoint vs 200-epoch checkpoint.

Outputs:
  - output/compare/pretrained_vs_200epoch_summary.json
  - output/compare/pretrained_vs_200epoch_summary.md

Evaluations:
  1) Validation-set metrics (global + macro over objects)
  2) Threshold sweep for best F1
  3) Force-center error on valid samples
  4) Mesh inference benchmark (latency + prediction distribution)
  5) Training-history snapshot for 200-epoch run
"""

import json
import os
import time
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from model.pointnet2 import PointNet2Seg


VAL_H5 = os.path.join(PROJECT_DIR, "output", "dataset", "affordance_val.h5")
TRAIN_HISTORY_200 = os.path.join(PROJECT_DIR, "output", "checkpoints_v5", "training_history.json")
OUT_DIR = os.path.join(PROJECT_DIR, "output", "compare")

CHECKPOINTS = {
    "pretrained": {
        "path": os.path.join(PROJECT_DIR, "output", "checkpoints", "best_model.pth"),
    },
    "trained_200_epoch": {
        "path": os.path.join(PROJECT_DIR, "output", "checkpoints_v5", "checkpoint_epoch200.pth"),
    },
}

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

LABEL_POSITIVE_THRESHOLD = 0.1


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def decode_obj_ids(raw_ids):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw_ids]


def load_ckpt_meta(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    sa1_w = state["sa1.mlp_convs.0.weight"]
    in_channel = int(sa1_w.shape[1] - 3)
    has_fc = any("fc_head" in k for k in state.keys())
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    val_f1 = ckpt.get("val_f1") if isinstance(ckpt, dict) else None
    val_fc_mm = ckpt.get("val_fc_mm") if isinstance(ckpt, dict) else None
    return {
        "state": state,
        "in_channel": in_channel,
        "has_fc": has_fc,
        "epoch": epoch,
        "val_f1": val_f1,
        "val_fc_mm": val_fc_mm,
    }


def build_model(state, in_channel, has_fc, device):
    model = PointNet2Seg(
        num_classes=2,
        in_channel=in_channel,
        predict_force_center=has_fc,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def get_features(points, normals, human_priors, in_channel):
    if in_channel == 6:
        return np.concatenate([points, normals], axis=-1)
    if in_channel == 7:
        return np.concatenate([points, normals, human_priors[:, None]], axis=-1)
    raise ValueError(f"Unsupported in_channel: {in_channel}")


def compute_binary_metrics(prob, target_bin, threshold):
    pred = prob > threshold
    gt = target_bin > 0
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    iou = safe_div(tp, tp + fp + fn)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": acc,
    }


@torch.no_grad()
def evaluate_on_val(model, in_channel, has_fc, device, points, normals, hp, labels, force_centers, obj_ids):
    n_samples = points.shape[0]
    probs = []
    fc_pred_all = []

    for i in range(n_samples):
        feat = get_features(points[i], normals[i], hp[i], in_channel)
        xyz_t = torch.from_numpy(points[i:i + 1]).float().to(device)
        feat_t = torch.from_numpy(feat[None, ...]).float().to(device)
        out = model(xyz_t, feat_t)

        if isinstance(out, tuple):
            seg_out, fc_out = out
            fc_pred_all.append(fc_out[0].detach().cpu().numpy())
        else:
            seg_out = out
            if has_fc:
                fc_pred_all.append(np.zeros(3, dtype=np.float32))

        if seg_out.dim() == 3 and seg_out.shape[-1] == 2:
            p = F.softmax(seg_out, dim=-1)[0, :, 1].detach().cpu().numpy()
        elif seg_out.dim() == 2:
            p = torch.sigmoid(seg_out)[0].detach().cpu().numpy()
        else:
            raise RuntimeError(f"Unexpected seg output shape: {tuple(seg_out.shape)}")

        probs.append(p)

    probs = np.asarray(probs, dtype=np.float32)

    labels_float = labels.astype(np.float32)
    labels_bin = (labels_float >= LABEL_POSITIVE_THRESHOLD).astype(np.uint8)

    thresholds = [round(x, 2) for x in np.arange(0.05, 0.91, 0.05)]
    sweep = []
    flat_prob = probs.reshape(-1)
    flat_lbl_bin = labels_bin.reshape(-1)
    flat_lbl_float = labels_float.reshape(-1)
    for t in thresholds:
        m = compute_binary_metrics(flat_prob, flat_lbl_bin, t)
        sweep.append({"threshold": t, **m})

    best = max(sweep, key=lambda x: x["f1"])
    best_t = float(best["threshold"])
    global_metrics = compute_binary_metrics(flat_prob, flat_lbl_bin, best_t)

    # Continuous-label fit (useful when labels are soft probabilities)
    mae = float(np.mean(np.abs(flat_prob - flat_lbl_float)))
    mse = float(np.mean((flat_prob - flat_lbl_float) ** 2))
    prob_std = float(np.std(flat_prob))
    lbl_std = float(np.std(flat_lbl_float))
    if prob_std > 1e-8 and lbl_std > 1e-8:
        pearson = float(np.corrcoef(flat_prob, flat_lbl_float)[0, 1])
    else:
        pearson = 0.0

    def topk_stats(frac):
        n = len(flat_prob)
        k = max(1, int(n * frac))
        idx = np.argpartition(flat_prob, -k)[-k:]
        selected = np.zeros(n, dtype=bool)
        selected[idx] = True
        gt = flat_lbl_bin > 0
        tp = int(np.logical_and(selected, gt).sum())
        pred_pos = int(selected.sum())
        gt_pos = int(gt.sum())
        return {
            "k": k,
            "precision": safe_div(tp, pred_pos),
            "recall": safe_div(tp, gt_pos),
            "tp": tp,
            "predicted": pred_pos,
            "gt_positive": gt_pos,
        }

    topk = {
        "top_1_percent": topk_stats(0.01),
        "top_2_percent": topk_stats(0.02),
        "top_5_percent": topk_stats(0.05),
    }

    obj_bucket = defaultdict(lambda: {"prob": [], "lbl": []})
    for i, oid in enumerate(obj_ids):
        obj_bucket[oid]["prob"].append(probs[i])
        obj_bucket[oid]["lbl"].append(labels[i])

    per_object = {}
    f1_list = []
    iou_list = []
    for oid in sorted(obj_bucket.keys()):
        p = np.asarray(obj_bucket[oid]["prob"]).reshape(-1)
        y = np.asarray(obj_bucket[oid]["lbl"]).reshape(-1)
        y_bin = (y >= LABEL_POSITIVE_THRESHOLD).astype(np.uint8)
        m = compute_binary_metrics(p, y_bin, best_t)
        per_object[oid] = m
        f1_list.append(m["f1"])
        iou_list.append(m["iou"])

    fc_result = None
    if has_fc:
        fc_pred = np.asarray(fc_pred_all, dtype=np.float32)
        valid = np.linalg.norm(force_centers, axis=1) > 1e-3
        if valid.any():
            err_mm = np.linalg.norm(fc_pred[valid] - force_centers[valid], axis=1) * 1000.0
            fc_result = {
                "valid_samples": int(valid.sum()),
                "mean_mm": float(err_mm.mean()),
                "median_mm": float(np.median(err_mm)),
                "p90_mm": float(np.percentile(err_mm, 90)),
            }
        else:
            fc_result = {
                "valid_samples": 0,
                "mean_mm": None,
                "median_mm": None,
                "p90_mm": None,
            }

    return {
        "label_threshold": LABEL_POSITIVE_THRESHOLD,
        "label_positive_rate": float(flat_lbl_bin.mean()),
        "best_threshold": best_t,
        "global": global_metrics,
        "continuous": {
            "mae": mae,
            "mse": mse,
            "pearson_r": pearson,
        },
        "topk": topk,
        "macro": {
            "f1": float(np.mean(f1_list)) if f1_list else 0.0,
            "iou": float(np.mean(iou_list)) if iou_list else 0.0,
            "num_objects": len(per_object),
        },
        "force_center": fc_result,
        "threshold_sweep": sweep,
        "per_object": per_object,
    }


@torch.no_grad()
def benchmark_meshes(model, in_channel, device):
    rows = []
    for rel in MESHES:
        mesh_path = os.path.join(PROJECT_DIR, rel)
        obj_name = os.path.splitext(os.path.basename(mesh_path))[0]
        if not os.path.exists(mesh_path):
            continue
        mesh = trimesh.load(mesh_path, force="mesh")
        points, face_idx = mesh.sample(1024, return_index=True)
        normals = mesh.face_normals[face_idx]
        points = points.astype(np.float32)
        normals = normals.astype(np.float32)
        hp = np.zeros((len(points),), dtype=np.float32)
        feat = get_features(points, normals, hp, in_channel)

        xyz_t = torch.from_numpy(points[None, ...]).float().to(device)
        feat_t = torch.from_numpy(feat[None, ...]).float().to(device)
        t0 = time.time()
        out = model(xyz_t, feat_t)
        dt = time.time() - t0

        seg_out = out[0] if isinstance(out, tuple) else out
        if seg_out.dim() == 3 and seg_out.shape[-1] == 2:
            prob = F.softmax(seg_out, dim=-1)[0, :, 1].detach().cpu().numpy()
        else:
            prob = torch.sigmoid(seg_out)[0].detach().cpu().numpy()

        rows.append({
            "object": obj_name,
            "time_s": float(dt),
            "prob_mean": float(prob.mean()),
            "prob_std": float(prob.std()),
            "contact_0.3": int((prob > 0.3).sum()),
            "contact_0.5": int((prob > 0.5).sum()),
        })

    if not rows:
        return {"objects": [], "aggregate": {}}

    return {
        "objects": rows,
        "aggregate": {
            "num_meshes": len(rows),
            "time_s_mean": float(np.mean([r["time_s"] for r in rows])),
            "time_s_median": float(np.median([r["time_s"] for r in rows])),
            "contact_0.3_mean": float(np.mean([r["contact_0.3"] for r in rows])),
            "contact_0.5_mean": float(np.mean([r["contact_0.5"] for r in rows])),
        },
    }


def load_history_200():
    if not os.path.exists(TRAIN_HISTORY_200):
        return None
    with open(TRAIN_HISTORY_200, "r") as f:
        h = json.load(f)
    if not h:
        return None
    ep200 = next((x for x in h if int(x.get("epoch", -1)) == 200), h[-1])
    best = max(h, key=lambda x: x.get("val_f1", 0.0))
    return {
        "epoch_200": ep200,
        "best_epoch": {
            "epoch": best.get("epoch"),
            "val_f1": best.get("val_f1"),
            "val_iou": best.get("val_iou"),
            "val_fc_mm": best.get("val_fc_mm"),
        },
    }


def build_markdown(summary):
    a = summary["models"]["pretrained"]
    b = summary["models"]["trained_200_epoch"]

    def pct(x):
        return f"{100.0 * float(x):.2f}%"

    lines = []
    lines.append("# Pretrained vs 200-Epoch Checkpoint Comparison")
    lines.append("")
    lines.append("## Validation Summary")
    lines.append("")
    lines.append("| Metric | Pretrained | 200-epoch | Delta (200 - pre) |")
    lines.append("|---|---:|---:|---:|")
    for key in ["accuracy", "precision", "recall", "f1", "iou"]:
        va = a["validation"]["global"][key]
        vb = b["validation"]["global"][key]
        lines.append(f"| {key} | {pct(va)} | {pct(vb)} | {pct(vb - va)} |")

    lines.append("")
    lines.append("| Metric | Pretrained | 200-epoch | Delta (200 - pre) |")
    lines.append("|---|---:|---:|---:|")
    for key in ["f1", "iou"]:
        va = a["validation"]["macro"][key]
        vb = b["validation"]["macro"][key]
        lines.append(f"| macro_{key} | {pct(va)} | {pct(vb)} | {pct(vb - va)} |")

    lines.append("")
    lines.append("## Threshold and Force Center")
    lines.append("")
    lines.append(
        f"- Label positive threshold for binarization: {a['validation']['label_threshold']:.2f}"
    )
    lines.append(
        f"- Validation positive rate after binarization: {pct(a['validation']['label_positive_rate'])}"
    )
    lines.append(
        f"- Pretrained best threshold: {a['validation']['best_threshold']:.2f}, "
        f"F1: {pct(a['validation']['global']['f1'])}"
    )
    lines.append(
        f"- 200-epoch best threshold: {b['validation']['best_threshold']:.2f}, "
        f"F1: {pct(b['validation']['global']['f1'])}"
    )
    if a["validation"].get("force_center") and b["validation"].get("force_center"):
        lines.append(
            f"- Force-center mean error (mm): pretrained {a['validation']['force_center']['mean_mm']:.2f}, "
            f"200-epoch {b['validation']['force_center']['mean_mm']:.2f}"
        )

    lines.append("")
    lines.append("## Continuous-Label Fit")
    lines.append("")
    lines.append("| Metric | Pretrained | 200-epoch | Delta (200 - pre) |")
    lines.append("|---|---:|---:|---:|")
    for key in ["mae", "mse", "pearson_r"]:
        va = a["validation"]["continuous"][key]
        vb = b["validation"]["continuous"][key]
        lines.append(f"| {key} | {va:.6f} | {vb:.6f} | {vb - va:.6f} |")

    lines.append("")
    lines.append("## Top-K Retrieval (Binary Labels)")
    lines.append("")
    lines.append("| Metric | Pretrained | 200-epoch | Delta (200 - pre) |")
    lines.append("|---|---:|---:|---:|")
    for key in ["top_1_percent", "top_2_percent", "top_5_percent"]:
        pa = a["validation"]["topk"][key]["precision"]
        pb = b["validation"]["topk"][key]["precision"]
        ra = a["validation"]["topk"][key]["recall"]
        rb = b["validation"]["topk"][key]["recall"]
        lines.append(f"| {key}_precision | {pct(pa)} | {pct(pb)} | {pct(pb - pa)} |")
        lines.append(f"| {key}_recall | {pct(ra)} | {pct(rb)} | {pct(rb - ra)} |")

    lines.append("")
    lines.append("## Mesh Inference Benchmark")
    lines.append("")
    lines.append("| Metric | Pretrained | 200-epoch | Delta (200 - pre) |")
    lines.append("|---|---:|---:|---:|")
    for key in ["time_s_mean", "time_s_median", "contact_0.3_mean", "contact_0.5_mean"]:
        va = a["mesh_benchmark"]["aggregate"].get(key, 0.0)
        vb = b["mesh_benchmark"]["aggregate"].get(key, 0.0)
        lines.append(f"| {key} | {va:.4f} | {vb:.4f} | {vb - va:.4f} |")

    if summary.get("training_history_200"):
        h = summary["training_history_200"]
        e200 = h.get("epoch_200", {})
        best = h.get("best_epoch", {})
        lines.append("")
        lines.append("## 200-Epoch Training Snapshot")
        lines.append("")
        lines.append(
            f"- Epoch 200: val_f1={e200.get('val_f1')}, val_iou={e200.get('val_iou')}, "
            f"val_fc_mm={e200.get('val_fc_mm')}"
        )
        lines.append(
            f"- Best epoch in history: epoch={best.get('epoch')}, val_f1={best.get('val_f1')}, "
            f"val_iou={best.get('val_iou')}, val_fc_mm={best.get('val_fc_mm')}"
        )

    return "\n".join(lines) + "\n"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(VAL_H5):
        raise FileNotFoundError(f"Validation dataset not found: {VAL_H5}")

    with h5py.File(VAL_H5, "r") as f:
        points = f["data/points"][:].astype(np.float32)
        normals = f["data/normals"][:].astype(np.float32)
        hp = f["data/human_priors"][:].astype(np.float32)
        labels = f["data/labels"][:].astype(np.float32)
        force_centers = f["data/force_centers"][:].astype(np.float32)
        obj_ids = decode_obj_ids(f["data/obj_ids"][:])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    summary = {
        "dataset": {
            "val_path": VAL_H5,
            "samples": int(points.shape[0]),
            "points_per_sample": int(points.shape[1]),
            "num_objects": int(len(set(obj_ids))),
        },
        "models": {},
        "training_history_200": load_history_200(),
    }

    for name, cfg in CHECKPOINTS.items():
        path = cfg["path"]
        if not os.path.exists(path):
            print(f"Skip {name}: checkpoint not found at {path}")
            continue
        print(f"\nEvaluating: {name} -> {path}")
        meta = load_ckpt_meta(path)
        model = build_model(meta["state"], meta["in_channel"], meta["has_fc"], device)
        val_stats = evaluate_on_val(
            model=model,
            in_channel=meta["in_channel"],
            has_fc=meta["has_fc"],
            device=device,
            points=points,
            normals=normals,
            hp=hp,
            labels=labels,
            force_centers=force_centers,
            obj_ids=obj_ids,
        )
        mesh_stats = benchmark_meshes(model, meta["in_channel"], device)
        summary["models"][name] = {
            "checkpoint": path,
            "meta": {
                "epoch": meta["epoch"],
                "in_channel": meta["in_channel"],
                "has_force_center_head": meta["has_fc"],
                "ckpt_val_f1": meta["val_f1"],
                "ckpt_val_fc_mm": meta["val_fc_mm"],
            },
            "validation": val_stats,
            "mesh_benchmark": mesh_stats,
        }

    json_path = os.path.join(OUT_DIR, "pretrained_vs_200epoch_summary.json")
    md_path = os.path.join(OUT_DIR, "pretrained_vs_200epoch_summary.md")

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(md_path, "w") as f:
        f.write(build_markdown(summary))

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved MD:   {md_path}")


if __name__ == "__main__":
    main()
