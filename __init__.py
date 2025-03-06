print("Loading ComfyUI-Image-Position-Blend node...")

# 辅助函数定义
def pil2tensor(image):
    from PIL import Image
    import torch
    import numpy as np
    # 确保图像是 RGB 格式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    array = np.array(image).astype(np.float32) / 255.0
    print(f"pil2tensor array shape: {array.shape}, min: {array.min()}, max: {array.max()}")  # 调试
    return torch.from_numpy(array).unsqueeze(0)

def tensor2pil(image):
    from PIL import Image
    import torch
    import numpy as np
    
    # 确保是 PyTorch 张量
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"Input must be a PyTorch tensor, got {type(image)} with shape {image.shape}")
    
    # 移除批次维度（如果存在）
    if image.dim() == 4:  # (batch, channels, height, width)
        if image.size(0) > 1:
            print(f"Warning: Multiple batches detected, using first batch. Shape: {image.shape}")
        image = image[0]  # 提取第一个批次
    elif image.dim() == 3:  # (channels, height, width) 或 (height, width, channels)
        image = image
    else:
        raise ValueError(f"Unexpected tensor dimension: {image.dim()} with shape {image.shape}")

    # 转换为 numpy 数组，并缩放到 0-255
    array = image.cpu().numpy()
    print(f"tensor2pil array shape before transpose: {array.shape}, min: {array.min()}, max: {array.max()}")  # 调试

    # 调整维度顺序，确保 (channels, height, width)
    if array.ndim == 3 and array.shape[0] not in [1, 3, 4]:  # (height, width, channels)
        array = np.transpose(array, (2, 0, 1))  # 转换为 (channels, height, width)
    elif array.ndim == 2:  # (height, width)，可能是单通道
        array = array[np.newaxis, :, :]  # 转换为 (1, height, width)
    elif array.ndim != 3:
        raise ValueError(f"Unexpected array shape: {array.shape}")

    # 验证通道数
    channels = array.shape[0]
    if channels not in [1, 3, 4]:
        raise ValueError(f"Unexpected number of channels: {channels} with shape {array.shape}")
    
    # 缩放到 0-255 范围
    array = np.clip(array * 255.0, 0, 255)
    print(f"tensor2pil array shape after transpose: {array.shape}, min: {array.min()}, max: {array.max()}")  # 调试

    # 转换为 PIL 图像
    if channels == 1:  # 单通道（如 MASK）
        return Image.fromarray(array[0].astype(np.uint8), mode='L')
    elif channels == 3:  # RGB
        return Image.fromarray(np.transpose(array, (1, 2, 0)).astype(np.uint8), mode='RGB')
    elif channels == 4:  # RGBA
        return Image.fromarray(np.transpose(array, (1, 2, 0)).astype(np.uint8), mode='RGBA')
    else:
        raise ValueError(f"Unexpected number of channels: {channels} with shape {array.shape}")

class ImagePositionBlend:
    """
    一个将两张图片进行混合的节点，通过在指定 X,Y 坐标处将 Layer image 放置在 Background image 上，
    支持镜像、旋转、缩放和 alpha 混合以处理透明度。
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Layer image": ("IMAGE",),  # 前景图片
                "Background image": ("IMAGE",),  # 背景图片
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
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.01,
                    "display": "number"
                }),
            },
            "optional": {
                "Layer mask (optional)": ("MASK",),  # 可选的掩码输入
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_image",)
    FUNCTION = "blend_images"
    CATEGORY = "💀Image Position Blend"
    OUTPUT_NODE = False

    def blend_images(self, **kwargs):
        import torch
        import numpy as np
        from PIL import Image

        # 提取参数
        Layer_image = kwargs.get("Layer image")
        Background_image = kwargs.get("Background image")
        x_position = kwargs.get("x_position", 0)
        y_position = kwargs.get("y_position", 0)
        mirror = kwargs.get("mirror", "None")
        rotation = kwargs.get("rotation", 0.0)
        scale = kwargs.get("scale", 1.0)
        Layer_mask = kwargs.get("Layer mask (optional)")

        # 调试：打印原始张量形状
        print(f"Layer image tensor shape: {Layer_image.shape}")
        print(f"Background image tensor shape: {Background_image.shape}")

        # 转换为 PIL 图像，确保 RGBA 模式以支持透明度
        background_pil = tensor2pil(Background_image).convert('RGBA')
        layer_pil = tensor2pil(Layer_image).convert('RGBA')  # 强制转换为 RGBA 模式

        # 调试：检查 layer_pil 的像素值
        layer_array = np.array(layer_pil)
        print(f"Layer PIL pixel values: shape: {layer_array.shape}, min: {layer_array.min()/255}, max: {layer_array.max()/255}")

        # 提取 Layer image 的 alpha 通道作为默认掩码
        layer_alpha = layer_pil.split()[-1]
        print(f"Layer alpha size: {layer_alpha.size}, mode: {layer_alpha.mode}, min: {np.array(layer_alpha).min()/255}, max: {np.array(layer_alpha).max()/255}")  # 调试

        # 处理可选掩码（如果提供）
        if Layer_mask is not None:
            mask_pil = tensor2pil(Layer_mask).convert('L')
            mask_array = np.array(mask_pil)
            print(f"Layer mask tensor shape: {Layer_mask.shape}")
            print(f"Layer mask PIL size: {mask_pil.size}, mode: {mask_pil.mode}, min: {mask_array.min()/255}, max: {mask_array.max()/255}")
            if mask_pil.size != layer_pil.size:
                print(f"Resizing mask from {mask_pil.size} to {layer_pil.size}")
                mask_pil = mask_pil.resize(layer_pil.size, Image.Resampling.LANCZOS)
            # 确保掩码逻辑正确（黑色为透明，白色为不透明）
            if mask_array.max() <= 1.0:  # 假设 mask 已归一化
                mask_array = mask_array * 255
            if mask_array.max() == 255 and mask_array.min() == 0:
                layer_alpha = Image.fromarray(255 - mask_array.astype(np.uint8), mode='L')  # 反转掩码
            else:
                layer_alpha = Image.fromarray(mask_array.astype(np.uint8), mode='L')

        # 应用缩放
        target_width = int(layer_pil.width * scale)
        target_height = int(layer_pil.height * scale)
        if target_width != layer_pil.width or target_height != layer_pil.height:
            print(f"Scaling layer from {layer_pil.size} to ({target_width}, {target_height})")
            layer_pil = layer_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
            layer_alpha = layer_alpha.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # 应用镜像
        if mirror == "Horizontal":
            layer_pil = layer_pil.transpose(Image.FLIP_LEFT_RIGHT)
            layer_alpha = layer_alpha.transpose(Image.FLIP_LEFT_RIGHT)
        elif mirror == "Vertical":
            layer_pil = layer_pil.transpose(Image.FLIP_TOP_BOTTOM)
            layer_alpha = layer_alpha.transpose(Image.FLIP_TOP_BOTTOM)

        # 应用旋转
        if rotation != 0:
            layer_pil = layer_pil.rotate(rotation, expand=True)
            layer_alpha = layer_alpha.rotate(rotation, expand=True)

        # 计算粘贴位置，基于 Layer image 的调整后大小
        x = x_position
        y = y_position
        original_height = Layer_image.shape[2]  # 原始高度
        original_width = Layer_image.shape[3]   # 原始宽度
        target_height = int(original_height * scale)  # 调整后高度
        target_width = int(original_width * scale)    # 调整后宽度
        paste_box = (x, y, x + target_width, y + target_height)
        print(f"Initial Paste box: {paste_box}")  # 调试

        # 确保粘贴区域在背景图像范围内
        paste_box = (
            max(0, x),
            max(0, y),
            min(background_pil.width, x + target_width),
            min(background_pil.height, y + target_height)
        )
        print(f"Adjusted paste box: {paste_box}")  # 调试

        # 创建中间画布，处理前景图像和掩码
        comp = Image.new('RGBA', background_pil.size, (0, 0, 0, 0))  # 透明画布
        comp.paste(layer_pil, (paste_box[0], paste_box[1]), layer_alpha)
        comp_mask = Image.new('L', background_pil.size, 0)  # 黑色掩码
        comp_mask.paste(layer_alpha, (paste_box[0], paste_box[1]))

        # 验证 alpha 掩码有效性
        alpha_array = np.array(comp_mask)
        print(f"Comp mask values: min: {alpha_array.min()/255}, max: {alpha_array.max()/255}")
        if alpha_array.max() == 0:
            print("Warning: Comp mask is fully transparent, forcing default alpha")
            comp_mask = Image.new('L', comp.size, 255)  # 全不透明掩码

        # 将中间画布与背景混合
        background_pil = Image.composite(comp, background_pil, comp_mask)
        print(f"Background size after paste: {background_pil.size}, min: {np.array(background_pil.convert('RGB')).min()/255}, max: {np.array(background_pil.convert('RGB')).max()/255}")  # 调试

        # 转换为张量，输出 RGB 格式
        blended_image = pil2tensor(background_pil.convert('RGB'))
        return (blended_image,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImagePositionBlend": ImagePositionBlend
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePositionBlend": "💀Image Position Blend"
}
