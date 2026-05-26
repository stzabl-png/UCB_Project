#!/usr/bin/env python3
"""
model/logger.py — 训练 / 验证 / 推理 结构化日志系统

日志文件结构 (全部在 save_dir/ 下):
  config.json          — 超参数、数据集配置、模型架构（训练开始时写入）
  training_log.jsonl   — 每 epoch 追加一条 JSON（可用 pandas 直接读取）
  dataset_report.json  — 数据集统计（样本数/对象分布/失败对象）
  eval_report.json     — 验证集评估报告（每次 eval 覆盖）
  failure_cases.json   — 推理失败原因分析
  latest.log           — 人类可读的 tee 输出（训练中实时追加）
"""

import os
import json
import time
import logging
import datetime
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# 核心 Logger 类
# ─────────────────────────────────────────────────────────────

class TrainingLogger:
    """
    统一日志管理器，同时写：
      - 控制台（带颜色）
      - latest.log（纯文本）
      - training_log.jsonl（结构化，每 epoch 一行）
    """

    def __init__(self, save_dir: str, run_name: str = ""):
        self.save_dir  = save_dir
        self.run_name  = run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        os.makedirs(save_dir, exist_ok=True)

        # 文本 logger
        self._logger = logging.getLogger(f"diffusion.{self.run_name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers = []

        fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

        # 控制台
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        self._logger.addHandler(ch)

        # 文件
        fh = logging.FileHandler(os.path.join(save_dir, "latest.log"), mode="a")
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

        # JSONL 文件句柄
        self._jsonl_path = os.path.join(save_dir, "training_log.jsonl")

    # ── 基础输出 ──────────────────────────────────────────────

    def info(self, msg: str):
        self._logger.info(msg)

    def warn(self, msg: str):
        self._logger.warning(f"⚠  {msg}")

    def error(self, msg: str):
        self._logger.error(f"❌ {msg}")

    def section(self, title: str):
        self._logger.info("=" * 60)
        self._logger.info(f"  {title}")
        self._logger.info("=" * 60)

    # ── 结构化记录 ─────────────────────────────────────────────

    def log_config(self, config: Dict[str, Any]):
        """训练开始时记录超参数 + 环境配置"""
        config["run_name"]  = self.run_name
        config["timestamp"] = datetime.datetime.now().isoformat()
        _write_json(os.path.join(self.save_dir, "config.json"), config)
        self.section("Training Configuration")
        for k, v in config.items():
            if not k.startswith("_"):
                self.info(f"  {k:<25} = {v}")

    def log_dataset(self, stats: Dict[str, Any]):
        """记录数据集统计信息"""
        _write_json(os.path.join(self.save_dir, "dataset_report.json"), stats)
        self.section("Dataset Report")
        self.info(f"  Total objects    : {stats.get('n_objects', '?')}")
        self.info(f"  Total samples    : {stats.get('n_samples', '?')}")
        self.info(f"  Train / Val      : {stats.get('n_train', '?')} / {stats.get('n_val', '?')}")
        if stats.get("skipped_objects"):
            self.warn(f"  Skipped objects  : {len(stats['skipped_objects'])} "
                      f"(0 successful grasps or missing pc)")
            for obj in stats["skipped_objects"][:10]:
                self.warn(f"    - {obj}")
            if len(stats["skipped_objects"]) > 10:
                self.warn(f"    ... and {len(stats['skipped_objects'])-10} more")
        if stats.get("grasp_dist"):
            d = stats["grasp_dist"]
            self.info(f"  Grasps per obj   : min={d['min']}  max={d['max']}  "
                      f"mean={d['mean']:.1f}  median={d['median']:.1f}")

    def log_epoch(self, epoch: int, total_epochs: int,
                  train_loss: float, val_loss: float,
                  lr: float,
                  extra: Optional[Dict[str, Any]] = None):
        """每 epoch 追加一条到 training_log.jsonl"""
        elapsed = time.time() - self.start_time
        record = {
            "epoch":      epoch,
            "total":      total_epochs,
            "train_loss": round(train_loss, 7),
            "val_loss":   round(val_loss,   7),
            "lr":         lr,
            "elapsed_s":  round(elapsed, 1),
        }
        if extra:
            record.update(extra)

        # 追加到 JSONL
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # 控制台输出（每10轮或首轮）
        if epoch == 1 or epoch % 10 == 0:
            star    = " ★" if extra and extra.get("is_best") else ""
            aff_str = f"  aff={extra['aff_loss']:.5f}" if extra and "aff_loss" in extra else ""
            self.info(f"  Ep {epoch:>4d}/{total_epochs} | "
                      f"train={train_loss:.6f} | val={val_loss:.6f}{aff_str} | "
                      f"lr={lr:.2e} | {_fmt_time(elapsed)}{star}")

    def log_eval(self, report: Dict[str, Any]):
        """推理评估报告（覆盖写入）"""
        _write_json(os.path.join(self.save_dir, "eval_report.json"), report)
        self.section("Evaluation Report")
        self.info(f"  Objects evaluated : {report.get('n_objects', '?')}")
        self.info(f"  Avg aff@fingertip : {report.get('avg_aff_at_fingertip', '?'):.4f}")
        self.info(f"  Avg rotation div  : {report.get('avg_rotation_diversity', '?'):.4f}")
        if report.get("per_object"):
            worst = sorted(report["per_object"], key=lambda x: x.get("aff_score", 1))[:5]
            self.warn("  Worst 5 objects (lowest fingertip affordance):")
            for o in worst:
                self.warn(f"    {o['obj']:<30} aff={o.get('aff_score',0):.3f} "
                          f"div={o.get('diversity',0):.3f}")

    def log_failures(self, failures: List[Dict[str, Any]]):
        """记录推理失败案例 + 失败原因"""
        report = {
            "timestamp":    datetime.datetime.now().isoformat(),
            "n_failures":   len(failures),
            "failure_cases": failures,
            "reason_summary": _summarize_reasons(failures),
        }
        _write_json(os.path.join(self.save_dir, "failure_cases.json"), report)
        self.section(f"Failure Analysis ({len(failures)} cases)")
        for reason, cnt in report["reason_summary"].items():
            self.warn(f"  {reason:<40} : {cnt} cases")

    def done(self, best_epoch: int, best_val: float):
        elapsed = time.time() - self.start_time
        self.section("Training Complete")
        self.info(f"  Best epoch  : {best_epoch}")
        self.info(f"  Best val    : {best_val:.6f}")
        self.info(f"  Total time  : {_fmt_time(elapsed)}")
        self.info(f"  Logs saved  : {self.save_dir}/")


# ─────────────────────────────────────────────────────────────
# 评估专用：分析推理结果
# ─────────────────────────────────────────────────────────────

class EvalLogger:
    """推理评估时使用，收集 per-object 指标后统一输出"""

    def __init__(self, save_dir: str):
        self.save_dir   = save_dir
        self.per_object = []
        self.failures   = []
        os.makedirs(save_dir, exist_ok=True)

    def record_object(self, obj_name: str,
                      aff_score: float,
                      rotation_diversity: float,
                      n_candidates: int,
                      contact_point: Optional[list] = None,
                      notes: str = ""):
        """记录单个对象的推理评估结果"""
        self.per_object.append({
            "obj":           obj_name,
            "aff_score":     round(aff_score, 4),
            "diversity":     round(rotation_diversity, 4),
            "n_candidates":  n_candidates,
            "contact_point": contact_point,
            "notes":         notes,
        })

    def record_failure(self, obj_name: str, reason: str,
                       details: Optional[Dict] = None):
        """记录失败案例"""
        self.failures.append({
            "obj":     obj_name,
            "reason":  reason,
            "details": details or {},
        })

    def finalize(self, logger: Optional[TrainingLogger] = None):
        """汇总 + 写入报告"""
        import numpy as np
        scores = [o["aff_score"]  for o in self.per_object]
        divs   = [o["diversity"]  for o in self.per_object]

        report = {
            "timestamp":              datetime.datetime.now().isoformat(),
            "n_objects":              len(self.per_object),
            "n_failures":             len(self.failures),
            "avg_aff_at_fingertip":   float(np.mean(scores)) if scores else 0.0,
            "avg_rotation_diversity": float(np.mean(divs))   if divs   else 0.0,
            "per_object":             self.per_object,
        }

        if logger:
            logger.log_eval(report)
            if self.failures:
                logger.log_failures(self.failures)
        else:
            _write_json(os.path.join(self.save_dir, "eval_report.json"), report)
            _write_json(os.path.join(self.save_dir, "failure_cases.json"), {
                "n_failures":   len(self.failures),
                "failure_cases": self.failures,
                "reason_summary": _summarize_reasons(self.failures),
            })
        return report


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def _write_json(path: str, data: Any):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _fmt_time(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _summarize_reasons(failures: List[Dict]) -> Dict[str, int]:
    from collections import Counter
    return dict(Counter(f.get("reason", "unknown") for f in failures))


# ─────────────────────────────────────────────────────────────
# 快捷函数：读取历史 log 分析
# ─────────────────────────────────────────────────────────────

def load_training_log(save_dir: str):
    """加载 training_log.jsonl → list of dicts（可直接转 pandas DataFrame）"""
    path = os.path.join(save_dir, "training_log.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_training_summary(save_dir: str):
    """快速打印训练摘要（用于事后分析）"""
    records = load_training_log(save_dir)
    if not records:
        print(f"No training log found in {save_dir}")
        return

    import numpy as np
    epochs     = [r["epoch"]      for r in records]
    train_loss = [r["train_loss"] for r in records]
    val_loss   = [r["val_loss"]   for r in records]

    best_idx = int(np.argmin(val_loss))
    print(f"\n{'='*60}")
    print(f"  Training Summary: {save_dir}")
    print(f"{'='*60}")
    print(f"  Total epochs   : {max(epochs)}")
    print(f"  Best val loss  : {val_loss[best_idx]:.6f}  @ epoch {epochs[best_idx]}")
    print(f"  Final val loss : {val_loss[-1]:.6f}")
    print(f"  Total time     : {_fmt_time(records[-1].get('elapsed_s', 0))}")

    # 配置
    cfg_path = os.path.join(save_dir, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        print(f"\n  Config:")
        for k in ["epochs", "batch_size", "lr", "T", "hidden", "gt_dirs", "save_dir"]:
            if k in cfg:
                print(f"    {k:<20} = {cfg[k]}")

    # 失败案例
    fail_path = os.path.join(save_dir, "failure_cases.json")
    if os.path.exists(fail_path):
        fail = json.load(open(fail_path))
        print(f"\n  Failure Cases: {fail.get('n_failures', 0)}")
        for reason, cnt in fail.get("reason_summary", {}).items():
            print(f"    {reason:<40} : {cnt}")

    print(f"{'='*60}\n")
