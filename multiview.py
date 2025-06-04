# fast_depth_stereo_nodes.py
import torch, numpy as np
# import cv2
# from depth_estimator import DepthEstimator          # your existing impl.

# ════════════════════════════════════════════════════════════════════════
# 1.  DEPTH ESTIMATION NODE  (tensor-only, GPU-friendly)
# ════════════════════════════════════════════════════════════════════════
# class DepthAnything_Fast:
#     CATEGORY      = "👁️ Stereo / Utilities"
#     RETURN_TYPES  = ("IMAGE",)
#     RETURN_NAMES  = ("depth_map",)
#     OUTPUT_NODE   = False
#     FUNCTION      = "predict"

#     def __init__(self):
#         self.model = None

#     @classmethod
#     def INPUT_TYPES(cls):
#         return {"required":{
#             "base_image": ("IMAGE",),
#             "blur_radius": ("INT", {"default":3,"min":1,"max":51}),
#             "invert_depth": ("BOOLEAN", {"default":False}),
#         }}

#     # ------------------------------------------------------------------
#     def _load(self, blur):
#         if self.model is None:
#             self.model = DepthEstimator(); self.model.load_model()
#         self.model.blur_radius = blur

#     def predict(self, base_image, blur_radius, invert_depth=False):
#         B,H,W,C = base_image.shape
#         self._load(blur_radius)
#         device  = base_image.device
#         outs=[]
#         for i in range(B):
#             img8 = (base_image[i].cpu()*255).byte().numpy()      # HWC uint8
#             d    = self.model.predict_depth(img8)                # H×W float
#             d    = cv2.normalize(d,None,0,1,cv2.NORM_MINMAX)
#             outs.append(torch.from_numpy(d))
#         d = torch.stack(outs).unsqueeze(-1).to(device)           # [B,H,W,1]
#         if invert_depth: d = 1.0 - d
#         d3 = d.repeat(1,1,1,3)                                   # → [B,H,W,3]
#         return (d3,)

# ═════════════════════════════════════════════════════════════
# 2 ─ StereoShift_Fast  (fixed scale, eye map, smear band)
# ═════════════════════════════════════════════════════════════
class StereoShift_Fast:
    CATEGORY      = "👁️ Stereo / FAST"
    RETURN_TYPES  = ("IMAGE","IMAGE")
    RETURN_NAMES  = ("left_eye","right_eye")
    FUNCTION      = "shift"
    OUTPUT_NODE   = False

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "base_image": ("IMAGE",),           # [B,H,W,C]
            "depth_map":  ("IMAGE",),           # same shape
            "depth_scale":("FLOAT",{"default":30.0,"min":1.0,"max":200.0}),
            "boost":      ("FLOAT",{"default":1.0,"min":0.1,"max":5.0}),
            "mode": (["Parallel","Cross-eyed","Symmetric"],{"default":"Cross-eyed"}),
        }}

    # ---------------------------------------------------------
    def _shift_gather(self, img_nchw, idx):
        """Gather pixels with idx [B,H,W] → returns BHWC."""
        idx_nchw = idx.unsqueeze(1).expand(-1,3,-1,-1)          # [B,C,H,W]
        gathered = torch.gather(img_nchw, 3, idx_nchw)          # [B,C,H,W]
        return gathered.permute(0,2,3,1)                        # BHWC

    def shift(self, base_image, depth_map,
              depth_scale, boost, mode="Cross-eyed"):

        if base_image.shape != depth_map.shape:
            raise ValueError("depth_map must match base_image shape")

        x  = base_image
        d  = depth_map[...,0]                                   # [B,H,W]
        B,H,W,_ = x.shape
        dev = x.device

        # ------------ robust scale: multiply by 255 only if needed -----------
        scale_fac = 255.0 if d.max() <= 2.0 else 1.0
        shift_px  = (d * scale_fac * depth_scale / W * boost).round().long()

        # index grids
        base_idx  = torch.arange(W, device=dev).view(1,1,W).expand(B,H,W)
        # smear band size = 10 pixels (original CPU loop)
        smear     = torch.arange(10, device=dev)

        def build_idx(base, direction):
            """direction = -1 for left smear, +1 for right smear"""
            idx = (base + direction*shift_px).clamp(0, W-1)     # [B,H,W]
            idx = idx.unsqueeze(-1) + smear * direction         # [B,H,W,10]
            return idx.clamp(0, W-1)[...,0]                     # take first

        left_idx  = build_idx(base_idx, -1)    # fill band leftwards
        right_idx = build_idx(base_idx, +1)

        img_nchw = x.permute(0,3,1,2)          # [B,C,H,W]
        left_shifted  = self._shift_gather(img_nchw, left_idx)
        right_shifted = self._shift_gather(img_nchw, right_idx)

        # map to output eyes
        if mode == "Parallel":
            left, right = left_shifted, x
        elif mode == "Cross-eyed":
            left, right = x, right_shifted
        else:  # Symmetric
            left, right = left_shifted, right_shifted

        return (left, right)

    

# ════════════════════════════════════════════════════════════════════════

class CombineSideBySide:
    CATEGORY      = "👁️ Stereo / Utilities"
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("sbs_frames",)
    OUTPUT_NODE   = False
    FUNCTION      = "merge"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "left_frames" : ("IMAGE",),   # list[torch.Tensor] or batched tensor
            "right_frames": ("IMAGE",),
            "order": (["Left-Right","Right-Left"],{"default":"Left-Right"})
        }}

    # ------------------------------------------------------------
    def _to_list(self, x):
        return x if isinstance(x, (list, tuple)) else [x]

    def merge(self, left_frames, right_frames, order="Left-Right"):
        left_list  = self._to_list(left_frames)
        right_list = self._to_list(right_frames)

        if len(left_list) != len(right_list):
            raise ValueError("Left / Right frame counts differ")

        sbs = []
        for l, r in zip(left_list, right_list):
            if l.shape != r.shape:
                raise ValueError("Shape mismatch: {} vs {}".format(l.shape, r.shape))
            first, second = (l, r) if order.startswith("Left") else (r, l)
            # tensors are [B,H,W,C]; concat along width (dim=2)
            sbs.append(torch.cat((first, second), dim=2))
        return (sbs if isinstance(left_frames, (list, tuple)) else sbs[0],)

# depth_resize_node.py  –  drop into ComfyUI/custom_nodes/
import torch, torch.nn.functional as F

# ════════════════════════════════════════════════════════════════
# 3 ─ DepthResize  (resize depth maps to match RGB frames)

class DepthResize:
    CATEGORY      = "👁️ Stereo / Utilities"
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("depth_resized",)
    OUTPUT_NODE   = False
    FUNCTION      = "resize"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "depth_map":   ("IMAGE",),          # [B,h,w,C] float
            "reference":   ("IMAGE",),          # [B,H,W,C] – RGB frames
            "keep_aspect": ("BOOLEAN",{"default":False}),
            "mode": (["bilinear","nearest"],{"default":"bilinear"}),
        }}

    # ------------------------------------------------------------
    def _letterbox(self, d, H_ref, W_ref):
        _,h,w,_ = d.shape
        ar_src = w / h
        ar_dst = W_ref / H_ref
        if abs(ar_src - ar_dst) < 1e-2:
            return F.interpolate(d.permute(0,3,1,2), (H_ref,W_ref),
                                 mode="bilinear", align_corners=False
                               ).permute(0,2,3,1)
        # scale to fit height then crop width (or vice-versa)
        if ar_src > ar_dst:                  # too wide → fit height
            new_w = int(H_ref * ar_src + 0.5)
            tmp   = F.interpolate(d.permute(0,3,1,2), (H_ref,new_w),
                                  mode="bilinear", align_corners=False
                                ).permute(0,2,3,1)
            cx = (new_w - W_ref)//2
            return tmp[..., cx:cx+W_ref, :]
        else:                                # too tall → fit width
            new_h = int(W_ref / ar_src + 0.5)
            tmp   = F.interpolate(d.permute(0,3,1,2), (new_h,W_ref),
                                  mode="bilinear", align_corners=False
                                ).permute(0,2,3,1)
            cy = (new_h - H_ref)//2
            return tmp[:, cy:cy+H_ref, :, :]

    def resize(self, depth_map, reference,
               keep_aspect=False, mode="bilinear"):

        H_ref, W_ref = reference.shape[1:3]    # batch BHWC
        if depth_map.shape[1:3] == (H_ref, W_ref):
            return (depth_map,)                # already matches

        if keep_aspect:
            d_res = self._letterbox(depth_map, H_ref, W_ref)
        else:
            d_res = F.interpolate(
                depth_map.permute(0,3,1,2),   # to BCHW
                (H_ref, W_ref),
                mode=mode, align_corners=False
            ).permute(0,2,3,1)

        return (d_res,)
# ════════════════════════════════════════════════════════════════

MULTIVIEW_NODE_CLASS_MAPPINGS = {
    "StereoShift_Fast":   StereoShift_Fast,
    "CombineSideBySide": CombineSideBySide,
    "DepthResize":       DepthResize,
}


# A dictionary that contains the friendly/humanly readable titles for the nodes
MULTIVIEW_NODE_DISPLAY_NAME_MAPPINGS = {
    "StereoShift_Fast":   "Stereo Shift (Fast)",
    "CombineSideBySide": "Combine Side-by-Side",
    "DepthResize":       "Depth Resize",
}