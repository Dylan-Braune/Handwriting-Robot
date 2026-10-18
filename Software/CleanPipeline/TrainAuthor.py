#writer identification. this file now holds ONLY the data-loading pipeline on
#top of TrainText.py's IAMLineDatasetRaw (AuthorLabeledView/CollateFn/
#AddPersonalSamples -- already pure PIL/numpy, reused as-is by
#TrainScratch.py's from-scratch trainer). no torch anywhere -- the
#AuthorClassifierCNN model class and the torch training loops that used to
#live here are both retired (model moved to Classify.py's numpy version; new
#training goes through TrainScratch.py), and CollateFn's image/author-index
#batch now builds plain numpy arrays instead of tensors
import io
import random

import numpy as np
from PIL import Image

from TrainText import (
    IAMLineDatasetRaw,
    DecodePng,
    AugmentLineImage,
    EncodeText,
    ResizeLineImageFixed,
    TensorFromResized,
)
import SegmentLean as SL
import ProfileIO as PIO
from ExtractIAMLines import ReadLabelLines
from AuthorsConfig import PERSONAL_AUTHORS,VAL_FRACTION,SPLIT_SEED,PersonalAuthorPages


#wraps IAMLineDatasetRaw (imported, untouched) to additionally surface an
#author index per sample -- the base class already tracks author_folders/
#author_to_idx and a page_key of "author_id/filename.png", it just never
#exposed the index through __getitem__ since TrainText.py never needed it.
#every crop is stroke-normalised (binarize -> skeletonize -> re-ink at one
#constant width) before training, so the model only ever sees the same
#representation Classify.ClassifyImage feeds it at inference (that call
#stroke-normalises by default everywhere in this project) -- training on raw
#ink and then judging on stroke-normalised input would be a real train/
#inference mismatch, this makes the two consistent by construction
class AuthorLabeledView:
    def __init__(self,base: IAMLineDatasetRaw,indices,isTrain):
        self.base = base
        self.indices = indices
        self.is_train = isTrain

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,i):
        item = self.base.samples[self.indices[i]]
        gray = DecodePng(item["image_png"])
        normImg = PIO.StrokeNormalize(np.array(gray))
        if normImg is not None:
            gray = normImg   #falls back to the raw crop only if stroke-normalize
                              #found no usable ink (empty/near-blank line)
        pilImg = ResizeLineImageFixed(gray)
        if self.is_train:
            pilImg = AugmentLineImage(pilImg)
        imgTensor = TensorFromResized(pilImg)
        authorId = item["page_key"].split("/")[0]
        authorIdx = self.base.author_to_idx[authorId]
        return imgTensor,authorIdx,item["page_key"]


def CollateFn(batch):
    images,authorIdxs,pageKeys = zip(*batch)
    return np.stack(images,0),np.array(authorIdxs,dtype=np.int64),list(pageKeys)


#the torch AuthorClassifierCNN model class used to live here -- removed once
#Evaluate.py (its last caller) was itself converted to Classify.py's numpy
#models. nothing in CleanPipeline instantiates a torch model from this file
#anymore; what remains below is purely the data-loading pipeline

def EncodePng(grayCrop):
    buf = io.BytesIO()
    Image.fromarray(grayCrop).convert("L").save(buf,format="PNG")
    return buf.getvalue()


#segments every personal author's photos, splits per-page into train/holdout
#(same convention as elsewhere in this project), and appends PNG-encoded
#samples directly into base.samples so they're indistinguishable from
#IAM-derived samples to everything downstream
def AddPersonalSamples(base):
    rng = random.Random(SPLIT_SEED)
    nAdded = 0
    for author in PERSONAL_AUTHORS:
        for imgPath,labelPath in PersonalAuthorPages(author):
            leanLines,_ov,_info = SL.SegmentLines(str(imgPath))
            crops = [np.asarray(ln["image"]) for ln in leanLines]
            gt = ReadLabelLines(str(imgPath),str(labelPath))
            gt = [g for g in gt if g.strip()!="MESS"]
            n = min(len(crops),len(gt))
            rows = list(zip(range(n),crops[:n],gt[:n]))
            rng.shuffle(rows)
            nVal = max(1,round(len(rows)*VAL_FRACTION))
            for rank,(origIdx,crop,text) in enumerate(rows):
                target = EncodeText(text)
                if len(target)==0:
                    continue
                isHoldout = rank<nVal
                pageKey = f"{author}/{imgPath.stem}_{origIdx:02d}.png"
                base.samples.append({
                    "page_key": pageKey,
                    "line_idx": origIdx,
                    "image_png": EncodePng(crop),
                    "target": target,
                    "text": text,
                    "is_holdout": isHoldout,
                })
                nAdded+=1
    print(f"[Personal] added {nAdded} line samples for {PERSONAL_AUTHORS}")
