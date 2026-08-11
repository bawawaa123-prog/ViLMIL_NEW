'''
Author: ljh 1294245800@qq.com
Date: 2026-04-29 10:27:47
LastEditors: ljh 1294245800@qq.com
LastEditTime: 2026-04-29 10:33:00
FilePath: /ViLMIL/ViLa-MIL-main/tools/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import h5py

path = "/private/ljh-data/shared/ViLMIL/ViLa-MIL-main/data/yiyuan/patches_coords_5x/patches_256/2460239-B2.h5"

with h5py.File(path, "r") as f:
    print(f.keys())
    if "coords" not in f:
        raise KeyError("coords dataset not found in this h5 file")

    coords = f["coords"][:]
    print(f["coords"].shape)
    print("前10条 coords:")
    print(coords[:10])
    print("后10条 coords:")
    print(coords[-10:])