"""
ComfyUI custom nodes for the Krea2 scene-linear HDR LoRA pipeline.

Clone or copy this repository into  ComfyUI/custom_nodes/  and restart (this file is the
node pack entry point, so it has to stay at the repository root).

Why these nodes exist
---------------------
The LoRA is trained to generate images in **LogC4-encoded** space (values live in
[0,1], which is exactly what the VAE and every ComfyUI node expect). The extended
dynamic range only appears AFTER you invert LogC4. So the ComfyUI graph is:

  Load Krea2 -> LoraLoader(this LoRA) -> KSampler -> VAEDecode
      -> [LogC4 to Linear]  (this pack)   # [0,1] LogC4 code -> scene-linear (>>1)
      -> [Save EXR Scene-Linear] (this pack, OR HQ-Image-Save tonemap=linear)

Key facts:
* VAEDecode's built-in clamp to [0,1] is HARMLESS here, because the model's output
  IS a [0,1] LogC4 code. Values >1 are produced by "LogC4 to Linear", after decode.
* ComfyUI's built-in SaveImage is 8-bit and clamps -> never use it for HDR.
* "Save EXR Scene-Linear" writes 32-bit (or half) float, NO clamp, NO sRGB, tagged
  scene-linear. If OpenEXR isn't importable in your ComfyUI env, use
  ComfyUI-HQ-Image-Save's SaveEXR with tonemap="linear" instead (fed by LogC4 to Linear).
"""
import numpy as np
import torch
import os
import folder_paths  # provided by ComfyUI

# ---------- ARRI LogC4 inverse (2022 spec) -- matches training encode ----------
_A = (2.0 ** 18 - 16.0) / 117.45
_B = (1023.0 - 95.0) / 1023.0
_C = 95.0 / 1023.0
_S = (7.0 * np.log(2.0) * 2.0 ** (7.0 - 14.0 * _C / _B)) / (_A * _B)
_T = (2.0 ** (14.0 * (-_C / _B) + 6.0) - 64.0) / _A

def logc4_to_linear(V):
    # ARRI LogC4 spec: log branch for every code E' >= 0, linear branch only for E' < 0.
    # (Switching at E' >= C is discontinuous at C by ~0.0075 scene-linear.)
    V = np.asarray(V, np.float64)
    hi = (np.power(2.0, 14.0 * (V - _C) / _B + 6.0) - 64.0) / _A
    lo = V * _S + _T
    return np.where(V >= 0.0, hi, lo)


# A LogC4 encoding of non-negative scene-linear light never goes below code C (= linear 0).
# In the LoRA training set (inverse of build_dataset.py) no image has more than ~0.02% of its
# pixels with any channel below C. Display-referred (sRGB-like) images routinely have 10-30%.
# Decoding such an image as LogC4 is a domain error: shadows turn negative, mid-tones and
# diffuse surfaces land at several times diffuse white, and channel ratios are exaggerated.
LOGC4_BELOW_BLACK_WARN = 0.01


def logc4_domain_stats(code):
    """Per-image statistics of [0,1] codes (H,W,C) vs the LogC4 training domain."""
    rgb = np.asarray(code, np.float32)[..., :3]
    below = float((rgb < _C).any(axis=-1).mean())
    med = np.median(rgb.reshape(-1, 3), axis=0)
    return {"frac_below_logc4_black": below, "median_code_rgb": [float(m) for m in med],
            "looks_logc4": below <= LOGC4_BELOW_BLACK_WARN}


class LogC4ToLinear:
    """[0,1] LogC4 code (as generated) -> scene-linear radiance (values can be >> 1)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "exposure": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
        }}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("scene_linear",)
    FUNCTION = "decode"
    CATEGORY = "HDR/Krea2"

    def decode(self, image, exposure):
        x = image.detach().cpu().float().numpy()          # (B,H,W,C) in [0,1]
        for b in range(x.shape[0]):
            st = logc4_domain_stats(x[b])
            med = ", ".join(f"{m:.3f}" for m in st["median_code_rgb"])
            print(f"[LogC4ToLinear] image {b}: {100 * st['frac_below_logc4_black']:.2f}% of pixels "
                  f"below LogC4 black (code {_C:.4f}), median code RGB=({med})")
            if not st["looks_logc4"]:
                print(f"[LogC4ToLinear] WARNING image {b}: input does not look LogC4-encoded "
                      f"(LogC4 training images have <= {100 * LOGC4_BELOW_BLACK_WARN:.0f}% of pixels "
                      f"below LogC4 black). It is most likely an ordinary display-referred image; "
                      f"decoding it as LogC4 gives negative shadows and greatly over-bright mid-tones. "
                      f"Exposure or tone mapping cannot correct this - the sampled latent itself "
                      f"is not in the LogC4 domain.")
        lin = logc4_to_linear(np.clip(x, 0.0, 1.0)) * float(exposure)
        return (torch.from_numpy(np.ascontiguousarray(lin.astype(np.float32))),)


class SaveEXRSceneLinear:
    """Write scene-linear IMAGE to 32-bit (or half) float EXR. No clamp, no sRGB."""
    _COMP = None
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "filename_prefix": ("STRING", {"default": "hdr/krea"}),
            "pixel_type": (["float32", "half"],),
            "compression": (["zip", "piz", "zips", "none"],),
        }}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "HDR/Krea2"

    def save(self, images, filename_prefix, pixel_type, compression):
        import OpenEXR
        comp = {"zip": OpenEXR.ZIP_COMPRESSION, "piz": OpenEXR.PIZ_COMPRESSION,
                "zips": OpenEXR.ZIPS_COMPRESSION, "none": OpenEXR.NO_COMPRESSION}[compression]
        out_dir = folder_paths.get_output_directory()
        full, _ = os.path.split(os.path.join(out_dir, filename_prefix))
        os.makedirs(full, exist_ok=True)
        arr = images.detach().cpu().float().numpy()       # (B,H,W,C) raw floats, NO clamp
        dtype = np.float32 if pixel_type == "float32" else np.float16
        saved = []
        for b in range(arr.shape[0]):
            px = np.ascontiguousarray(arr[b, ..., :3].astype(dtype))
            path = os.path.join(out_dir, f"{filename_prefix}_{b:05d}.exr")
            header = {"compression": comp, "type": OpenEXR.scanlineimage,
                      # scene-linear, Rec.709 primaries, D65 (Blender/Nuke default if absent)
                      "chromaticities": (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290)}
            try:
                OpenEXR.File(header, {"RGB": px}).write(path)
            except TypeError:
                # older/newer signature: header without chromaticities
                OpenEXR.File({"compression": comp, "type": OpenEXR.scanlineimage},
                             {"RGB": px}).write(path)
            saved.append(path)
            print(f"[SaveEXRSceneLinear] wrote {path}  max={float(px.max()):.2f} ({pixel_type})")
        return {"ui": {"text": saved}}


NODE_CLASS_MAPPINGS = {
    "LogC4ToLinear": LogC4ToLinear,
    "SaveEXRSceneLinear": SaveEXRSceneLinear,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LogC4ToLinear": "LogC4 to Linear (HDR decode)",
    "SaveEXRSceneLinear": "Save EXR Scene-Linear (32-bit, no clamp)",
}
