"""
qtp_fieldstats.py -- FIELD CONDITIONS proxy: image statistics of the real photographs.
Lux was NOT measured (no lux meter / EXIF illuminance); these are only image-domain proxies:
  resolution (original and as used by ProcessPage), file size, EXIF exposure data if present,
  background (paper) grey level from the illumination estimate used by the pipeline
  (SegmentPage.GaussianBlur(gray, W/30), as CorrectIllumination), restricted to the detected
  page mask, mean/median, 5th-95th percentile spread (non-uniformity), ink contrast,
  and the percentage of clipped pixels (grey == 255 and grey == 0, and any RGB channel at 255) inside
  the page mask at the processing resolution and at the original resolution.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtp_common as C
import SegmentPage as PS


def main():
    rows = []
    for author, p in C.personal_pages():
        img0 = Image.open(p)
        exif = {}
        try:
            ex = img0.getexif()
            sub = ex.get_ifd(0x8769)
            from PIL.ExifTags import TAGS
            for k, v in list(ex.items()) + list(sub.items()):
                n = TAGS.get(k, k)
                if n in ("ExposureTime", "ISOSpeedRatings", "PhotographicSensitivity", "FNumber", "BrightnessValue",
                         "Make", "Model", "Flash", "ExposureBiasValue", "LightSource", "WhiteBalance", "MeteringMode"):
                    exif[str(n)] = str(v)
        except Exception:
            pass
        orig = np.array(ImageOps.exif_transpose(img0).convert("RGB"))
        rgb = PS.LoadImage(str(p))
        gray = PS.RgbToGray(rgb)
        mask = PS.DetectPageMask(rgb)
        bg = PS.GaussianBlur(gray, gray.shape[1] / 30.0)
        bgm = bg[mask]
        # ink contrast: darkest 2 % of page pixels vs background estimate there
        g = gray[mask].astype(float)
        thr = np.percentile(g, 2)
        ink_med = float(np.median(g[g <= thr]))
        og = PS.RgbToGray(orig)
        row = dict(writer=author, page=p.name, file_kB=os.path.getsize(p) / 1024,
                   orig_size_wh=list(img0.size), megapixels=img0.size[0] * img0.size[1] / 1e6,
                   processing_size_wh=[gray.shape[1], gray.shape[0]], page_mask_fraction=float(mask.mean()),
                   bg_mean=float(bgm.mean()), bg_median=float(np.median(bgm)),
                   bg_p5=float(np.percentile(bgm, 5)), bg_p95=float(np.percentile(bgm, 95)),
                   bg_nonuniformity_p95_over_p5=float(np.percentile(bgm, 95) / max(1.0, np.percentile(bgm, 5))),
                   ink_grey_median_darkest2pct=ink_med,
                   ink_contrast_ratio_bg_over_ink=float(np.median(bgm) / max(1.0, ink_med)),
                   clipped_white_pct_proc=float(100 * (gray[mask] == 255).mean()),
                   clipped_black_pct_proc=float(100 * (gray[mask] == 0).mean()),
                   clipped_any_channel255_pct_proc=float(100 * (rgb[mask].max(axis=1) == 255).mean()),
                   clipped_white_pct_orig=float(100 * (og == 255).mean()),
                   clipped_black_pct_orig=float(100 * (og == 0).mean()),
                   exif=exif)
        rows.append(row)
        print(f"{author}/{p.name[:38]:38s} {row['orig_size_wh']} bg median {row['bg_median']:.0f} p5-p95 {row['bg_p5']:.0f}-{row['bg_p95']:.0f} "
              f"white-clipped {row['clipped_white_pct_proc']:.2f}% black {row['clipped_black_pct_proc']:.3f}%", flush=True)
    keys = ["bg_mean", "bg_median", "bg_nonuniformity_p95_over_p5", "ink_contrast_ratio_bg_over_ink",
            "clipped_white_pct_proc", "clipped_black_pct_proc", "clipped_white_pct_orig", "megapixels"]
    summary = {k: dict(mean=float(np.mean([r[k] for r in rows])), min=float(np.min([r[k] for r in rows])),
                       max=float(np.max([r[k] for r in rows]))) for k in keys}
    C.save_json(dict(rows=rows, summary=summary, target_long_side_px=PS.TARGET_LONG_SIDE,
                     note="lux not measured; image statistics are proxies only"), "qtp_fieldstats.json")
    print(summary)


if __name__ == "__main__":
    main()
