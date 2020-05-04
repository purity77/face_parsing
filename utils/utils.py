from glob import glob
import torch
import os



def get_last_weights(weights_path):
    # glob 返回匹配.pth的路径
    weights_path = glob(weights_path + f'/*.pth')
    # 根据关键词key进行排序
    weights_path = sorted(weights_path,
                          key=lambda x: int(x.rsplit('_')[-1].rsplit('.')[0]),
                          reverse=True)[0]
    print(f'using weights {weights_path}')
    return weights_path
