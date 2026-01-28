"""Visualizer 测试脚本"""

import numpy as np
import torch
import sys

sys.path.insert(0, "/Users/lanjie/Proj/SSL/SemiFT")

from util.viz import Visualizer


def test_visualizer():
    # 初始化
    viz = Visualizer(save_dir="/tmp/viz_test", dataset="pascal")

    # 模拟数据
    img = torch.randn(3, 256, 256) * 0.229 + 0.485  # 模拟归一化图像
    mask_pred = torch.randint(0, 21, (256, 256))  # 分割预测
    mask_gt = torch.randint(0, 21, (256, 256))  # 分割标签
    features = torch.randn(64, 32, 32)  # 特征图
    confidence = torch.rand(256, 256)  # 置信度图

    # 测试 push 和 render
    viz.push(
        {
            "input": (img, Visualizer.IMAGE),
            "pred": (mask_pred, Visualizer.SEGMENTATION),
            "gt": (mask_gt, Visualizer.SEGMENTATION),
            "features": (features, Visualizer.FEATURE_MAP),
            "confidence": (confidence, Visualizer.TENSOR),
        }
    )

    viz.render("test_output.png")
    print("Test passed! Output saved to /tmp/viz_test/test_output.png")

    viz.reset()
    print("Reset passed!")


if __name__ == "__main__":
    test_visualizer()
