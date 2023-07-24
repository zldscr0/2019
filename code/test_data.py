import torch
import openslide
from openslide.deepzoom import DeepZoomGenerator
import os
import pandas as pd
import random

# svs 文件所在路径
test_data_dir = "../../../../../data/Colon/PAIP_2020/03_Colon_PNI2021chall_test"

# target 列表
target_df_test = pd.read_csv('target_test.csv')


# ---------------------- 相关变量的格式定义，参考 README.md ---------------------- #
# 最终保存全部数据的字典
test_data_lib = {}
test_slides_list = []   # 存储文件路径
test_targets_list = []  # 存储目标信息
test_grids_list = []    # 存储格点信息




mult = 1           # 缩放因子，1 表示不缩放
level = 0          # 使用 openslide 读取时的层级，默认表示以最高分辨率
patch_size = 224   # 切片的尺寸


# ---------------------- 开始处理数据，获取 lib ---------------------- #
for root, dirs, files in os.walk(test_data_dir):
    for filename in files:
        if filename[-4:] != '.svs':
            continue
        if target_df_test[target_df_test['slide'] == filename]['target'].values[0] == '*':
            continue
        
        test_slides_list.append(os.path.join(root, filename))
        test_targets_list.append((int)(target_df_test[target_df_test['slide'] == filename]['target'].values[0]))

        # 提取 patch 坐标
        slide = openslide.open_slide(os.path.join(root, filename))
        print(filename)
        w, h = slide.dimensions

        cur_patch_cords = []

        for j in range(0, h, patch_size):
            for i in range(0, w, patch_size):
                cur_patch_cords.append((i,j))

        test_grids_list.append(cur_patch_cords)



test_data_lib['slides'] = test_slides_list
test_data_lib['grid'] = test_grids_list
test_data_lib['targets'] = test_targets_list
test_data_lib['mult'] = mult
test_data_lib['level'] = level
test_data_lib['patch_size'] = patch_size

torch.save(test_data_lib, 'output/lib/cnn_test_data_lib.db')

