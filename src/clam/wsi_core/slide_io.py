import math
import os

import numpy as np
import openslide
from PIL import Image


class CziOpenSlideLike:
    """OpenSlide-like reader for mosaic CZI slides."""

    def __init__(self, path, downsamples=(1, 2, 4, 8, 16, 32, 64)):
        try:
            from aicspylibczi import CziFile
        except ImportError as exc:
            raise ImportError(
                "CZI preprocessing requires aicspylibczi. Install it to use "
                "--slide_backend czi or automatic .czi loading."
            ) from exc

        self.path = str(path)
        self.czi = CziFile(self.path)
        if not self.czi.is_mosaic():
            raise ValueError("CZI preprocessing currently expects a mosaic CZI file")

        self.bbox = self.czi.get_mosaic_bounding_box()
        self.level_downsamples = tuple(float(downsample) for downsample in downsamples)
        self.level_dimensions = tuple(
            (
                int(math.ceil(self.bbox.w / downsample)),
                int(math.ceil(self.bbox.h / downsample)),
            )
            for downsample in self.level_downsamples
        )
        self.dimensions = self.level_dimensions[0]
        self.level_count = len(self.level_dimensions)

    def get_best_level_for_downsample(self, downsample):
        diffs = [abs(level_downsample - downsample) for level_downsample in self.level_downsamples]
        return int(np.argmin(diffs))

    def read_region(self, location, level, size):
        x, y = (int(location[0]), int(location[1]))
        out_w, out_h = (int(size[0]), int(size[1]))
        downsample = self.level_downsamples[level]
        source_w = int(math.ceil(out_w * downsample))
        source_h = int(math.ceil(out_h * downsample))
        clip_x = max(0, x)
        clip_y = max(0, y)
        clip_right = min(self.bbox.w, x + source_w)
        clip_bottom = min(self.bbox.h, y + source_h)

        canvas = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
        if clip_right <= clip_x or clip_bottom <= clip_y:
            return canvas

        clip_w = int(clip_right - clip_x)
        clip_h = int(clip_bottom - clip_y)
        clip_out_w = max(1, int(math.ceil(clip_w / downsample)))
        clip_out_h = max(1, int(math.ceil(clip_h / downsample)))
        region = (
            int(self.bbox.x + clip_x),
            int(self.bbox.y + clip_y),
            clip_w,
            clip_h,
        )

        arr = self.czi.read_mosaic(region=region, scale_factor=1.0 / downsample, C=0)
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]

        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        if img.size != (clip_out_w, clip_out_h):
            img = img.resize((clip_out_w, clip_out_h), Image.Resampling.BILINEAR)

        paste_x = int(math.floor((clip_x - x) / downsample))
        paste_y = int(math.floor((clip_y - y) / downsample))
        canvas.paste(img.convert("RGBA"), (paste_x, paste_y))
        return canvas

    def close(self):
        close = getattr(self.czi, "close", None)
        if close is not None:
            close()


def infer_slide_backend(path):
    if os.path.splitext(path)[1].lower() == ".czi":
        return "czi"
    return "openslide"


def open_slide(path, backend="auto"):
    if backend == "auto":
        backend = infer_slide_backend(path)
    if backend == "openslide":
        return openslide.open_slide(path)
    if backend == "czi":
        return CziOpenSlideLike(path)
    raise ValueError(f"Unsupported slide backend: {backend}")
