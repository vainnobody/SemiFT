from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from .utils import color_map
from torchvision import transforms
from PIL import Image
from util.classes import CLASSES
from sklearn.decomposition import PCA


class Visualizer:
    """深度学习训练可视化工具

    支持可视化 IMAGE、SEGMENTATION、FEATURE_MAP、TENSOR 四种数据类型。
    使用字典形式的 push API，key 自动作为子图标题。

    Example:
        viz = Visualizer(save_dir='./viz', dataset='pascal')
        viz.push({
            'input': (img, 0),
            'pred': (pred_mask, 1),
            'gt': (gt_mask, 1),
        })
        viz.render('epoch_10.png')
        viz.reset()
    """

    # dtype 常量
    IMAGE = 0
    SEGMENTATION = 1
    FEATURE_MAP = 2
    TENSOR = 3

    # ImageNet 归一化参数
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, save_dir, dataset="pascal"):
        """初始化 Visualizer

        Args:
            save_dir: 保存目录
            dataset: 数据集名称，用于选择颜色映射
        """
        self.save_dir = save_dir
        self.dataset = dataset
        self.cmap = color_map(dataset)
        self.classes = CLASSES.get(dataset, [])
        self.data_list = []  # [(name, processed_img, dtype), ...]
        os.makedirs(save_dir, exist_ok=True)

    def push(self, data_dict):
        """批量推入数据

        Args:
            data_dict: {name: (tensor, dtype), ...}
                - name: 子图标题
                - tensor: 数据 (torch.Tensor 或 np.ndarray)
                - dtype: 0=IMAGE, 1=SEGMENTATION, 2=FEATURE_MAP, 3=TENSOR
        """
        for name, (data, dtype) in data_dict.items():
            img = self._process(data, dtype)
            self.data_list.append((name, img, dtype))

    def render(self, filename):
        """渲染并保存图像

        Args:
            filename: 保存文件名
        """
        n = len(self.data_list)
        if n == 0:
            return

        # 自动计算网格布局（接近正方形）
        ncol = int(np.ceil(np.sqrt(n)))
        nrow = int(np.ceil(n / ncol))

        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (name, img, dtype) in enumerate(self.data_list):
            ax = axes[idx]

            if dtype == self.SEGMENTATION:
                ax.imshow(img)
                self._add_legend(ax, img)
            elif dtype == self.FEATURE_MAP:
                ax.imshow(img, cmap="viridis")
            elif dtype == self.TENSOR:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)

            ax.set_title(name, fontsize=8)
            ax.axis("off")

        # 隐藏多余的子图
        for idx in range(n, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()

    def reset(self):
        """清空数据队列"""
        self.data_list = []

    def _process(self, data, dtype):
        """处理数据为可显示的图像"""
        # 转换为 numpy
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        if dtype == self.IMAGE:
            return self._process_image(data)
        elif dtype == self.SEGMENTATION:
            return self._process_segmentation(data)
        elif dtype == self.FEATURE_MAP:
            return self._process_feature_map(data)
        elif dtype == self.TENSOR:
            return self._process_tensor(data)
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

    def _process_image(self, img):
        """处理图像：反归一化"""
        # (C, H, W) -> (H, W, C)
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = img.transpose(1, 2, 0)

        # 反归一化
        img = img * self.STD + self.MEAN
        img = np.clip(img, 0, 1)
        return img

    def _process_segmentation(self, mask):
        """处理分割图：colormap 转换"""
        if mask.ndim == 3:
            mask = mask.squeeze()
        mask = mask.astype(np.int32)

        # 使用 colormap 转换为彩色图
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        for cls_idx in np.unique(mask):
            if cls_idx < 256:
                colored[mask == cls_idx] = self.cmap[cls_idx]

        return colored

    def _process_feature_map(self, feat):
        """处理特征图：PCA 降维"""
        # (C, H, W) -> (H, W, 3)
        if feat.ndim == 3:
            c, h, w = feat.shape
            feat_flat = feat.reshape(c, -1).T  # (H*W, C)

            # PCA 降维到 3 维
            n_components = min(3, c)
            pca = PCA(n_components=n_components)
            feat_pca = pca.fit_transform(feat_flat)  # (H*W, 3)

            # 归一化到 [0, 1]
            feat_pca = feat_pca.reshape(h, w, n_components)
            feat_pca = (feat_pca - feat_pca.min()) / (
                feat_pca.max() - feat_pca.min() + 1e-8
            )

            if n_components < 3:
                # 补齐到 3 通道
                feat_pca = np.concatenate(
                    [feat_pca, np.zeros((h, w, 3 - n_components))], axis=-1
                )

            return feat_pca
        else:
            return self._process_tensor(feat)

    def _process_tensor(self, tensor):
        """处理通用 tensor：归一化显示"""
        if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:
            tensor = tensor.transpose(1, 2, 0)
        if tensor.ndim == 3 and tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1)

        # 归一化到 [0, 1]
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        return tensor

    def _add_legend(self, ax, img):
        """为分割图添加小字体类别标签"""
        if len(self.classes) == 0:
            return

        # 获取图中存在的类别
        unique_colors = np.unique(img.reshape(-1, 3), axis=0)

        legend_text = []
        for color in unique_colors:
            for cls_idx, cmap_color in enumerate(self.cmap):
                if np.array_equal(color, cmap_color) and cls_idx < len(self.classes):
                    legend_text.append(f"{cls_idx}: {self.classes[cls_idx]}")
                    break

        if legend_text:
            text = "\n".join(legend_text[:5])  # 最多显示 5 个类别
            ax.text(
                0.02,
                0.98,
                text,
                transform=ax.transAxes,
                fontsize=5,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )
