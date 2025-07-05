import torch

LMS2006_to_DKLd65 = torch.tensor([
  [1.000000000000000,   1.000000000000000,                   0],
  [1.000000000000000,  -2.311130179947035,                   0],
  [-1.000000000000000,  -1.000000000000000,  50.977571328718781]
], dtype=torch.float32)
XYZ_to_LMS2006 = torch.tensor([
   [0.187596268556126,   0.585168649077728,  -0.026384263306304],
   [-0.133397430663221,   0.405505777260049,   0.034502127690364],
   [0.000244379021663,  -0.000542995890619,   0.019406849066323]
], dtype=torch.float32)

# --- ✅ sRGB → Linear RGB ---
def srgb_to_linear_rgb(srgb):
    """sRGB 转线性 RGB (gamma 解码)，输入 [0, 1]，输出 [0, 1]"""
    threshold = 0.04045
    linear = torch.where(
        srgb <= threshold,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4
    )
    return linear

# --- ✅ Linear RGB → XYZ (D65) ---
def linear_rgb_to_xyz(rgb):
    """线性 RGB → XYZ，D65"""
    # 使用 sRGB 的 D65 到 XYZ 转换矩阵
    M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=rgb.dtype, device=rgb.device)
    rgb = rgb.permute(1, 2, 0)  # (C,H,W) → (H,W,C)
    xyz = torch.tensordot(rgb, M.T, dims=1)  # (H,W,3)
    return xyz.permute(2, 0, 1)  # (3,H,W)

# --- ✅ XYZ → LMS2006 ---
def xyz_to_lms2006(xyz):
    xyz = xyz.permute(1, 2, 0)  # C,H,W → H,W,C
    lms = torch.tensordot(xyz, XYZ_to_LMS2006.T.to(xyz.device), dims=1)
    return lms.permute(2, 0, 1)

# --- ✅ LMS2006 → DKL ---
def lms_to_dkl(lms):
    lms = lms.permute(1, 2, 0)  # C,H,W → H,W,C
    dkl = torch.tensordot(lms, LMS2006_to_DKLd65.T.to(lms.device), dims=1)
    return dkl.permute(2, 0, 1)  # C,H,W

# --- ✅ 最终 transform（sRGB → DKL） ---
class RGBtoDKLTransform:
    def __init__(self, peak_luminance=500.0):
        self.peak_luminance = peak_luminance

    def __call__(self, tensor):
        tensor = srgb_to_linear_rgb(tensor)               # sRGB → linear RGB
        tensor = linear_rgb_to_xyz(tensor) * self.peak_luminance  # → XYZ (cd/m²)
        tensor = xyz_to_lms2006(tensor)                   # → LMS
        tensor = lms_to_dkl(tensor)
        return tensor

class RGBtoXYZTransform:
    def __init__(self, peak_luminance=500.0):
        self.peak_luminance = peak_luminance

    def __call__(self, tensor):
        tensor = srgb_to_linear_rgb(tensor)
        tensor = linear_rgb_to_xyz(tensor)
        return tensor * self.peak_luminance

class RGBtoRGBLinearTransform:
    def __init__(self, peak_luminance=500.0):
        self.peak_luminance = peak_luminance

    def __call__(self, tensor):
        tensor = srgb_to_linear_rgb(tensor)
        return tensor * self.peak_luminance

def Color_space_transform(color_space_name, peak_luminance=500.0):
    if color_space_name == 'sRGB':
        return None
    elif color_space_name == 'RGB_linear':
        return RGBtoRGBLinearTransform(peak_luminance=peak_luminance)
    elif color_space_name == 'XYZ_linear':
        return RGBtoXYZTransform(peak_luminance=peak_luminance)
    elif color_space_name == 'DKL_linear':
        return RGBtoDKLTransform(peak_luminance=peak_luminance)