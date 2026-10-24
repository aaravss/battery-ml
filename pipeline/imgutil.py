"""Image loading shared by training, evaluation and (copied verbatim into) infer_pi.py. Keep these in sync."""
import numpy as np
from PIL import Image, ImageOps


def pad_to_square(im: Image.Image) -> Image.Image:
    """Pad to a square using the median colour of the image border, so padding blends into the background."""
    w, h = im.size
    if w == h:
        return im
    a = np.asarray(im)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    fill = tuple(int(v) for v in np.median(border, axis=0))
    s = max(w, h)
    out = Image.new("RGB", (s, s), fill)
    out.paste(im, ((s - w) // 2, (s - h) // 2))
    return out


def load_image(path, size: int) -> np.ndarray:
    """File -> RGB uint8 array (size, size, 3): EXIF-rotate, flatten alpha on white, pad to square, resize."""
    im = ImageOps.exif_transpose(Image.open(path))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = pad_to_square(im.convert("RGB"))
    im = im.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.uint8)
