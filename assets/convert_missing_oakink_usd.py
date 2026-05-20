#!/usr/bin/env python3
"""
Step 2: 将 assets/oakink_missing_obj/*.obj 批量转为 USD

使用方式 (Isaac Sim 环境):
    sim45 assets/convert_missing_oakink_usd.py
"""
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import os
import asyncio
import omni.kit.asset_converter as converter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OBJ_DIR    = os.path.join(SCRIPT_DIR, "oakink_missing_obj")
USD_DIR    = os.path.join(SCRIPT_DIR, "usd")
os.makedirs(USD_DIR, exist_ok=True)


async def convert_one(input_path, output_path):
    task_manager = converter.get_instance()
    ctx = converter.AssetConverterContext()
    ctx.ignore_materials = False
    ctx.ignore_animations = True
    ctx.ignore_camera = True
    ctx.ignore_light = True
    ctx.single_mesh = True
    ctx.smooth_normals = True
    ctx.export_preview_surface = False
    ctx.use_meter_as_world_unit = True
    ctx.embed_textures = True
    task = task_manager.create_converter_task(input_path, output_path, None, ctx)
    ok = await task.wait_until_finished()
    if not ok:
        print(f"    ❌ {task.get_status()} - {task.get_detailed_error()}")
    return ok


def main():
    obj_files = sorted([f for f in os.listdir(OBJ_DIR) if f.endswith(".obj")])
    total = len(obj_files)
    print(f"\n{'='*60}")
    print(f"[missing_oakink] {total} OBJ → {USD_DIR}")
    print(f"{'='*60}")

    ok_cnt, skip_cnt, fail_list = 0, 0, []

    for i, fname in enumerate(obj_files, 1):
        name = fname[:-4]
        usd_path = os.path.join(USD_DIR, f"{name}.usd")
        prefix = f"[{i:3d}/{total}] {name}"

        if os.path.exists(usd_path):
            print(f"{prefix}: skip ✓")
            skip_cnt += 1
            continue

        input_path = os.path.abspath(os.path.join(OBJ_DIR, fname))
        print(f"{prefix}: converting...", end=" ", flush=True)
        ok = asyncio.get_event_loop().run_until_complete(convert_one(input_path, usd_path))
        if ok:
            print("✅")
            ok_cnt += 1
        else:
            fail_list.append(name)

    print(f"\n完成: {ok_cnt} 转换, {skip_cnt} 跳过, {len(fail_list)} 失败")
    if fail_list:
        print(f"失败: {fail_list}")


main()
simulation_app.close()
