"""
qtp2_classify.py -- QTP2 / requirement R2: read the text AND identify the writer on REAL
photographed pages, replicating server.py /api/classify:

   crops  = [r['raw_crop'] for r in ProcessPage(png)[0] if r['tag'] == 'TEXT']
   text   = ' '.join(ReadText(Image.fromarray(c).convert('L'), TEXT_MODEL) for c in crops)
   probs  = mean over crops of softmax(INK_AUTHOR_MODEL(line))   (apply_stroke_normalize=False)
   author = argmax(mean probs);  text accuracy = CharAcc(predicted_text, expected_text)

TEXT_MODEL = PaperCRNNNumpy() default (= joint checkpoint); INK model =
weights/author_classifier_10new_weights.pt (same files server.py loads).

Two segmentation paths are evaluated:
  'server' : grey PNG -> ProcessPage (exactly what the server does)      [headline]
  'train'  : colour .jpg -> ProcessPage (what TrainText/TrainAuthor do; the split indices of
             the validation lines are defined on THIS path)
Held-out evidence:
  personal photos: the line-level validation split (VAL_FRACTION=0.15, SPLIT_SEED=0, per page);
                   NO page is held out for yeukita/dylan, so 'all lines' is mostly TRAINING data.
  IAM writers    : the writer's last labelled page was held out of the classifier (page-level
                   split); evaluated both on the cached lines and on ProcessPage of the page.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import math
import pickle
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtp_common as C
import qtp_splits as S
import qtp_lines as L
from qtp0_segment_cache import iam_holdout_pages

from np_inference.text_model import PaperCRNNNumpy, ReadText, CharAcc, WordAcc, levenshtein, CHAR_TO_IDX
from np_inference.author_model import AuthorClassifierCNNNumpy, ClassifyImage
from authors_config import author_classifier_weights_filename


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def lcs_words(pred, truth):
    tw, pw = truth.split(), pred.split()
    if not tw:
        return 0, 0
    m, n = len(tw), len(pw)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = (dp[i][j] + 1 if tw[i].lower() == pw[j].lower()
                                else max(dp[i][j + 1], dp[i + 1][j]))
    return dp[m][n], m


def filt(t):
    return "".join(c for c in t.strip() if c in CHAR_TO_IDX)


def summarise_text(rows, key_truth):
    """rows: list of dict with pred, truth variants.  micro char acc, mean per-line CharAcc,
    micro word acc."""
    if not rows:
        return None
    edits = sum(levenshtein(r["pred"], r[key_truth]) for r in rows)
    ref = sum(len(r[key_truth]) for r in rows)
    ca = [CharAcc(r["pred"], r[key_truth]) for r in rows]
    lc = [lcs_words(r["pred"], r[key_truth]) for r in rows]
    return dict(n_lines=len(rows), ref_chars=ref, edit_distance=edits,
                char_acc_micro=1 - edits / ref, char_acc_mean_line=float(np.mean(ca)),
                word_acc_micro=sum(a for a, _ in lc) / max(1, sum(b for _, b in lc)),
                n_ref_words=sum(b for _, b in lc))


def summarise_writer(rows):
    if not rows:
        return None
    k = sum(r["pred_writer"] == r["writer"] for r in rows)
    lo, hi = wilson(k, len(rows))
    return dict(n_lines=len(rows), correct=k, acc=k / len(rows), ci95=[lo, hi])


def main():
    t_start = time.time()
    text_model = PaperCRNNNumpy()
    ink_path = C.WEIGHTS / author_classifier_weights_filename()
    ink_model = AuthorClassifierCNNNumpy(checkpoint_path=ink_path)
    idx2a = {v: k for k, v in ink_model.author_mapping.items()}
    print("text weights:", text_model.checkpoint_path.name, "| ink classifier:", ink_path.name,
          "| classes:", sorted(ink_model.author_mapping), flush=True)

    d = S.load_seg_cache()
    with open(C.CACHE / "seg_cache_server.pkl", "rb") as f:
        srv = pickle.load(f)

    def process_line(crop):
        pil = Image.fromarray(np.asarray(crop)).convert("L")
        txt = ReadText(pil, text_model)
        _i, probs = ClassifyImage(pil, ink_model, apply_stroke_normalize=False)
        return txt, np.asarray(probs, float)

    pages_out, line_rows = [], []
    for pi, pr in enumerate(d["personal"]):
        path = str(C.NOGIT / pr["author"] / pr["page"])
        gt_all = [g for g in C.read_labels(path) if g.strip() != "MESS"]
        for pathname, crops in (("server", srv[path]["crops"]), ("train", pr["crops"])):
            # (qtp0 stored crops[:min(n_crops,n_labels)]; for every personal page n_crops == n_labels,
            #  so this is the complete crop list of the training path)
            crops_use = crops
            n = min(len(crops_use), len(gt_all))
            lines = []
            probs_sum = None
            all_pred = []
            # server behaviour: process EVERY detected crop (also beyond n) for text + writer-ID
            for ci, crop in enumerate(crops_use):
                txt, probs = process_line(crop)
                all_pred.append(txt)
                probs_sum = probs if probs_sum is None else probs_sum + probs
                if ci < n:
                    truth = gt_all[ci].strip()
                    lines.append(dict(path=pathname, writer=pr["author"], page=pr["page"],
                                      line_idx=ci, pred=txt, truth_raw=truth, truth_filt=filt(truth),
                                      pred_writer=idx2a[int(np.argmax(probs))],
                                      p_true=float(probs[ink_model.author_mapping[pr["author"]]]),
                                      held_text=(ci in pr["text_val_idx"]),
                                      held_author=(ci in pr["author_val_idx"]),
                                      counts_match=(len(srv[path]["crops"]) == pr["n_crops"] or pathname == "train")))
            avg = probs_sum / len(crops_use)
            page_pred_text = " ".join(all_pred)
            page_truth = " ".join(g.strip() for g in gt_all)
            pages_out.append(dict(path=pathname, writer=pr["author"], page=pr["page"],
                                  n_crops=len(crops_use), n_label_lines=len(gt_all),
                                  n_crops_train_path=pr["n_crops"],
                                  server_style_char_acc=CharAcc(page_pred_text, page_truth),
                                  server_style_char_acc_filtered=CharAcc(page_pred_text, filt(page_truth)),
                                  pred_writer=idx2a[int(np.argmax(avg))],
                                  conf=float(avg.max()),
                                  p_true=float(avg[ink_model.author_mapping[pr["author"]]])))
            line_rows += lines
        print(f"  {pr['author']}/{pr['page'][:45]} done  ({time.time()-t_start:.0f}s)", flush=True)

    # ------------------ IAM held-out pages ------------------
    iam_pages = iam_holdout_pages()
    with open(C.CACHE / "iam_lines.pkl", "rb") as f:
        iam_cached = pickle.load(f)
    iam_line_rows, iam_page_rows = [], []
    for a, p in iam_pages.items():
        # (i) cached lines of the held-out page (the classifier's own validation definition)
        cl = [r for r in iam_cached if r["author"] == a and r["is_holdout"]]
        ps = None
        for r in cl:
            g = L._decode(r["png"])
            txt, probs = process_line(g)
            ps = probs if ps is None else ps + probs
            iam_line_rows.append(dict(path="cache", writer=a, page=p.name, line_idx=r["line_idx"],
                                      pred=txt, truth_raw=r["text"].strip(), truth_filt=filt(r["text"]),
                                      pred_writer=idx2a[int(np.argmax(probs))]))
        avg = ps / len(cl)
        iam_page_rows.append(dict(path="cache", writer=a, page=p.name, n_crops=len(cl),
                                  pred_writer=idx2a[int(np.argmax(avg))], conf=float(avg.max()),
                                  p_true=float(avg[ink_model.author_mapping[a]])))
        # (ii) ProcessPage (server path) crops of the held-out page
        crops = srv[str(p)]["crops"]
        gt = C.read_labels(p)
        ps = None
        for ci, crop in enumerate(crops):
            txt, probs = process_line(crop)
            ps = probs if ps is None else ps + probs
            iam_line_rows.append(dict(path="server", writer=a, page=p.name, line_idx=ci, pred=txt,
                                      truth_raw=(gt[ci].strip() if ci < len(gt) else ""),
                                      truth_filt=(filt(gt[ci]) if ci < len(gt) else ""),
                                      pred_writer=idx2a[int(np.argmax(probs))]))
        avg = ps / len(crops)
        iam_page_rows.append(dict(path="server", writer=a, page=p.name, n_crops=len(crops),
                                  n_label_lines=len(gt),
                                  pred_writer=idx2a[int(np.argmax(avg))], conf=float(avg.max()),
                                  p_true=float(avg[ink_model.author_mapping[a]])))
        print(f"  IAM {a} done ({time.time()-t_start:.0f}s)", flush=True)

    # ------------------ summaries ------------------
    res = dict(meta=dict(text_weights=text_model.checkpoint_path.name, ink_weights=ink_path.name,
                         val_fraction=d["val_fraction"], split_seed=d["split_seed"],
                         split_overlap=S.split_overlap_report(d["personal"]),
                         machine=C.machine_info(), runtime_s=time.time() - t_start),
               pages=pages_out, iam_pages=iam_page_rows)
    summ = {}
    for pathname in ("server", "train"):
        lr = [r for r in line_rows if r["path"] == pathname]
        pg = [r for r in pages_out if r["path"] == pathname]
        block = {}
        for subset_name, flt in (("all_lines", lambda r: True),
                                 ("val_split_lines_text", lambda r: r["held_text"]),
                                 ("val_split_lines_author", lambda r: r["held_author"])):
            sel = [r for r in lr if flt(r)]
            if pathname == "server" and subset_name != "all_lines":
                # index-based attribution only valid where the crop count equals the train path's
                sel = [r for r in sel if r["counts_match"]]
            blk = dict(text_raw=summarise_text(sel, "truth_raw"),
                       text_filtered=summarise_text(sel, "truth_filt"),
                       writer_line=summarise_writer(sel))
            for w in C.PERSONAL_AUTHORS:
                ssel = [r for r in sel if r["writer"] == w]
                blk[f"writer_line_{w}"] = summarise_writer(ssel)
                blk[f"text_raw_{w}"] = summarise_text(ssel, "truth_raw")
            block[subset_name] = blk
        kp = sum(r["pred_writer"] == r["writer"] for r in pg)
        block["page_level_writer"] = dict(n_pages=len(pg), correct=kp, acc=kp / len(pg),
                                          ci95=list(wilson(kp, len(pg))))
        block["page_level_server_style_char_acc_mean"] = float(np.mean([r["server_style_char_acc"] for r in pg]))
        block["page_level_server_style_char_acc_filtered_mean"] = float(np.mean([r["server_style_char_acc_filtered"] for r in pg]))
        summ[pathname] = block
    # per-line writer-ID confusion (server path, all lines)
    labels = sorted(idx2a.values())
    conf = {a: {b: 0 for b in labels} for a in C.PERSONAL_AUTHORS}
    for r in line_rows:
        if r["path"] == "server":
            conf[r["writer"]][r["pred_writer"]] += 1
    summ["confusion_server_personal_lines"] = conf

    iam = {}
    for pathname in ("cache", "server"):
        lr = [r for r in iam_line_rows if r["path"] == pathname]
        pg = [r for r in iam_page_rows if r["path"] == pathname]
        k = sum(r["pred_writer"] == r["writer"] for r in lr)
        kp = sum(r["pred_writer"] == r["writer"] for r in pg)
        iam[pathname] = dict(n_lines=len(lr), line_writer_acc=k / len(lr),
                             line_ci95=list(wilson(k, len(lr))),
                             n_pages=len(pg), page_writer_correct=kp, page_writer_acc=kp / len(pg),
                             per_writer_line_acc={a: float(np.mean([r["pred_writer"] == a for r in lr if r["writer"] == a]))
                                                  for a in iam_pages},
                             text_raw=summarise_text([r for r in lr if r["truth_raw"]], "truth_raw"),
                             text_filtered=summarise_text([r for r in lr if r["truth_raw"]], "truth_filt"))
    summ["iam_heldout_pages"] = iam
    res["summary"] = summ
    res["line_rows"] = line_rows
    res["iam_line_rows"] = iam_line_rows
    C.save_json(res, "qtp2_results.json")
    print(f"saved qtp2_results.json, runtime {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
