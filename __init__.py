print("Loading ComfyUI-Image-Position-Blend node...")

class ImagePositionBlend:
    """
    A node that blends two images by positioning Image A over Image B at specified X,Y coordinates,
    with options for mirroring, rotation, and alpha blending for transparency.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),  # 前景图片
                "image_b": ("IMAGE",),  # 背景图片
                "x_position": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "y_position": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "mirror": (["None", "Horizontal", "Vertical"], {
                    "default": "None"
                }),
                "rotation": ("FLOAT", {
                    "default": 0.0,
                    "min": -360.0,
                    "max": 360.0,
                    "step": 1.0,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_image",)
    FUNCTION = "blend_images"
    CATEGORY = "💀Image Position Blend"  # 直接显示在 "Add Node" 根菜单
    OUTPUT_NODE = False

    def blend_images(self, image_a, image_b, x_position, y_position, mirror, rotation):
        import torch
        import numpy as np
        from PIL import Image

        # 将张量转换为numpy数组，保留所有通道（包括 Alpha 通道）
        img_a = image_a.cpu().numpy()[0] if image_a.shape[0] == 1 else image_a.cpu().numpy()
        img_b = image_b.cpu().numpy()[0] if image_b.shape[0] == 1 else image_b.cpu().numpy()

        # 检查是否有 Alpha 通道（RGBA 格式）
        if img_a.shape[2] == 4:  # 如果图片A有 Alpha 通道
            # 分离 RGB 和 Alpha 通道
            img_a_rgb = img_a[..., :3]  # RGB 通道
            img_a_alpha = img_a[..., 3]  # Alpha 通道
        else:
            # 如果没有 Alpha 通道，假设完全不透明
            img_a_rgb = img_a
            img_a_alpha = np.ones((img_a.shape[0], img_a.shape[1]), dtype=np.float32)

        # 将前景图片转换为 PIL 格式以便处理镜像和旋转
        # 由于 PIL 处理透明度时需要 RGBA 格式，我们先将 RGB 和 Alpha 分开处理
        img_a_pil = Image.fromarray((img_a_rgb * 255).astype(np.uint8))
        img_a_alpha_pil = Image.fromarray((img_a_alpha * 255).astype(np.uint8))

        # 处理镜像
        if mirror == "Horizontal":
            img_a_pil = img_a_pil.transpose(Image.FLIP_LEFT_RIGHT)
            img_a_alpha_pil = img_a_alpha_pil.transpose(Image.FLIP_LEFT_RIGHT)
        elif mirror == "Vertical":
            img_a_pil = img_a_pil.transpose(Image.FLIP_TOP_BOTTOM)
            img_a_alpha_pil = img_a_alpha_pil.transpose(Image.FLIP_TOP_BOTTOM)

        # 处理旋转
        if rotation != 0:
            img_a_pil = img_a_pil.rotate(rotation, expand=True)
            img_a_alpha_pil = img_a_alpha_pil.rotate(rotation, expand=True)

        # 将处理后的图片转换回 numpy 数组
        img_a_rgb = np.array(img_a_pil).astype(np.float32) / 255.0
        img_a_alpha = np.array(img_a_alpha_pil).astype(np.float32) / 255.0

        # 获取图片尺寸 (height, width, channels)
        height_a, width_a = img_a_rgb.shape[:2]
        height_b, width_b = img_b.shape[:2]

        # 创建输出画布，使用image_b的尺寸
        output = np.copy(img_b)

        # 计算实际的放置位置，确保不超出边界
        x_start = max(0, x_position)
        y_start = max(0, y_position)
        
        # 计算image_a在image_b中的有效区域
        x_end = min(width_b, x_start + width_a)
        y_end = min(height_b, y_start + height_a)
        
        # 计算image_a的裁剪区域
        a_x_start = max(0, -x_position)
        a_y_start = max(0, -y_position)
        a_x_end = a_x_start + (x_end - x_start)
        a_y_end = a_y_start + (y_end - y_start)

        # 如果有重叠区域，进行 Alpha 混合
        if x_end > x_start and y_end > y_start:
            # 提取前景和背景的区域
            fg_rgb = img_a_rgb[a_y_start:a_y_end, a_x_start:a_x_end]
            fg_alpha = img_a_alpha[a_y_start:a_y_end, a_x_start:a_x_end, np.newaxis]  # 扩展 Alpha 维度以匹配 RGB
            bg_rgb = output[y_start:y_end, x_start:x_end]

            # Alpha 混合公式：output = fg * alpha + bg * (1 - alpha)
            blended_rgb = fg_rgb * fg_alpha + bg_rgb * (1 - fg_alpha)

            # 将混合后的区域写回输出画布
            output[y_start:y_end, x_start:x_end] = blended_rgb

        # 转换为张量并返回
        output_tensor = torch.from_numpy(output).unsqueeze(0)
        return (output_tensor,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImagePositionBlend": ImagePositionBlend
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePositionBlend": "💀Image Position Blend"
}