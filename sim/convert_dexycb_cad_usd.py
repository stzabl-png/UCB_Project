#!/usr/bin/env python3
"""Convert specified DexYCB CAD textured.obj → output/obj_usd_cad/ycb/ycb_dex_NN.usd

Reuses the omni asset_converter pipeline from sim/convert_batch_usd.py, then
re-tags upAxis=Z (matches existing output/obj_usd_cad/ycb/ assets; the Y-up tag
otherwise causes PhysX collider/visual split — see sim/convert_usd_zup.py docstring).

DexYCB raw_id <-> dex_id mapping (verified against existing USDs):
    002→01  003→02  004→03  005→04  006→05  007→06  008→07  009→08
    010→09  011→10  019→11  021→12  024→13  025→14  035→15  036→16
    037→17  040→18  051→19  052→20  061→21

Usage:
    env_isaaclab python sim/convert_dexycb_cad_usd.py 11 14 19
    # → writes ycb_dex_11.usd (pitcher), ycb_dex_14.usd (mug), ycb_dex_19.usd (large_clamp)
"""
from isaacsim import SimulationApp                                  # MUST be first
simulation_app = SimulationApp({"headless": True})

import argparse
import asyncio
import os
import shutil
import sys

import omni.kit.asset_converter as converter
from pxr import Usd, UsdGeom

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_MODELS_DIR = os.path.join(PROJ_ROOT, "data_hub", "RawData",
                              "ThirdPersonRawData", "dexycb", "models")
OUT_DIR = os.path.join(PROJ_ROOT, "output", "obj_usd_cad", "ycb")

# dex_id -> (raw_dir_name, friendly_name)
DEX2RAW = {
    1:  ("002_master_chef_can",     "master_chef_can"),
    2:  ("003_cracker_box",         "cracker_box"),
    3:  ("004_sugar_box",           "sugar_box"),
    4:  ("005_tomato_soup_can",     "tomato_soup_can"),
    5:  ("006_mustard_bottle",      "mustard_bottle"),
    6:  ("007_tuna_fish_can",       "tuna_fish_can"),
    7:  ("008_pudding_box",         "pudding_box"),
    8:  ("009_gelatin_box",         "gelatin_box"),
    9:  ("010_potted_meat_can",     "potted_meat_can"),
    10: ("011_banana",              "banana"),
    11: ("019_pitcher_base",        "pitcher_base"),
    12: ("021_bleach_cleanser",     "bleach_cleanser"),
    13: ("024_bowl",                "bowl"),
    14: ("025_mug",                 "mug"),
    15: ("035_power_drill",         "power_drill"),
    16: ("036_wood_block",          "wood_block"),
    17: ("037_scissors",            "scissors"),
    18: ("040_large_marker",        "large_marker"),
    19: ("051_large_clamp",         "large_clamp"),
    20: ("052_extra_large_clamp",   "extra_large_clamp"),
    21: ("061_foam_brick",          "foam_brick"),
}


async def convert_to_usd(input_path, output_path):
    """Mirror of sim/convert_batch_usd.py:convert_to_usd — same settings."""
    task_manager = converter.get_instance()
    ctx = converter.AssetConverterContext()
    ctx.ignore_materials      = False
    ctx.ignore_animations     = True
    ctx.ignore_camera         = True
    ctx.ignore_light          = True
    ctx.single_mesh           = True
    ctx.smooth_normals        = True
    ctx.export_preview_surface = False
    ctx.use_meter_as_world_unit = True
    ctx.embed_textures        = True
    task = task_manager.create_converter_task(input_path, output_path,
                                              progress_callback=None,
                                              asset_converter_context=ctx)
    return await task.wait_until_finished()


def retag_z_up(path):
    """Set upAxis=Z on the USD (mirrors sim/convert_usd_zup.py logic).

    The omni converter writes upAxis=Y from textured.obj. That tag causes
    IsaacSim's metricsAssembler to inject a 90deg correction in rendering
    but not in PhysX → collider/visual split. Re-tagging to Z suppresses
    the correction; raw geometry is used directly and obj_quat_G (which
    was computed against raw .obj coords) places the object correctly.
    """
    stage = Usd.Stage.Open(path)
    cur = UsdGeom.GetStageUpAxis(stage)
    if cur == UsdGeom.Tokens.z:
        print(f"  upAxis already Z — no retag needed")
        return
    bak = path + ".ybak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"  backup -> {bak}")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().Save()
    print(f"  upAxis Y -> Z")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dex_ids", nargs="+", type=int, help="dex IDs to convert, e.g. 11 14 19")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing .usd in output dir")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    failed = []

    # Same pattern as sim/convert_batch_usd.py — omni asset_converter needs the
    # ambient event loop driven by loop.run_until_complete(), NOT asyncio.run()
    # (asyncio.run creates a fresh loop that the converter cannot tick into).
    loop = asyncio.get_event_loop()

    for did in args.dex_ids:
        if did not in DEX2RAW:
            print(f"\n=== dex_id {did}: UNKNOWN — skip ===", flush=True)
            failed.append(did); continue
        raw_name, friendly = DEX2RAW[did]
        src = os.path.join(RAW_MODELS_DIR, raw_name, "textured.obj")
        dst = os.path.join(OUT_DIR, f"ycb_dex_{did:02d}.usd")
        print(f"\n=== dex_{did:02d} ({friendly}) ===", flush=True)
        print(f"  src: {src}", flush=True)
        print(f"  dst: {dst}", flush=True)
        if not os.path.exists(src):
            print(f"  ERROR: source missing", flush=True); failed.append(did); continue
        if os.path.exists(dst) and not args.overwrite:
            print(f"  EXISTS (pass --overwrite to replace) — skip", flush=True); continue

        ok = loop.run_until_complete(convert_to_usd(src, dst))
        if not ok or not os.path.exists(dst):
            print(f"  ERROR: omni converter failed", flush=True); failed.append(did); continue
        print(f"  converted ({os.path.getsize(dst) / 1024:.0f} KB)", flush=True)
        retag_z_up(dst)

    simulation_app.close()
    print(f"\n=== DONE: {len(args.dex_ids) - len(failed)}/{len(args.dex_ids)} converted ===", flush=True)
    if failed:
        print(f"  failed: {failed}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
