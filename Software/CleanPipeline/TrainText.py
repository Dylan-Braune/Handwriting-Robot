#CNN-BiLSTM-CTC line recognizer, reproducing Kizilirmak & Yanikoglu's paper
#("CNN-BiLSTM model for English Handwriting Recognition", 2023,
#arxiv.org/html/2307.00664): 12 conv layers (32->64->128 filters, 2+4+6 per
#stage) with a 2x2 maxpool after stages 1 and 2 only, height collapsed to 1
#row via adaptive maxpool, then a 2-layer bidirectional LSTM (hidden=256, no
#dropout per the paper) and a linear+log_softmax CTC head. Decoding is greedy.
#this file now holds ONLY the data-loading pipeline (IAMLineDatasetRaw/
#TrainEvalView/CollateFn/augmentation), reused as-is by TrainScratch.py's
#from-scratch trainer. no torch/torchvision anywhere -- the old torch
#PaperCRNN model class, DecodeCtc, and training loops that used to live here
#are all retired: the model moved to Classify.py's numpy version, and the
#last thing still using torch (the image/target tensors in CollateFn, and
#torch.save/load for the line-image cache) has been switched to plain numpy
#arrays and pickle. augmentation (shear/rotate/perspective) used to go
#through torchvision -- now implemented as PIL affine/perspective transforms
#with the coefficients solved by numpy (numpy.linalg.inv / numpy.linalg.solve)
import io
import math
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter,map_coordinates

from ExtractIAMLines import ExtractLinePatches,ReadLabelLines
from Charset import CHARSET,CHAR_TO_IDX,IDX_TO_CHAR   #single source of truth, was duplicated here before

SCRIPT_DIR = Path(__file__).resolve().parent
#this file lives in Software/CleanPipeline, a sibling of Software/CNN where
#the real weights/NOGIT caches/datasets still live -- data is not duplicated
CNN_DATA_DIR = SCRIPT_DIR.parent/"CNN"
sys.path.insert(0,str(SCRIPT_DIR))


#maps a text string to ctc target indices, dropping any char outside CHARSET
def EncodeText(text):
    return np.array([CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX],dtype=np.int64)

INPUT_HEIGHT = 64   #deliberately halved from the paper's 100x960 -- CPU training at full size was ~1hr/epoch
INPUT_WIDTH = 640

#ctc time dimension T = INPUT_WIDTH//4 (two 2x2 maxpools). any label longer
#than ~0.75*T is almost certainly a collapsed/merged line, not real text --
#see IAMLineDatasetRaw's oversized-label skip below
MAX_SAFE_LABEL_CHARS = int((INPUT_WIDTH//4)*0.75)


#fits a line into the fixed training canvas by UNIFORM scaling + white
#padding (never independent width/height squashing, which measurably
#distorts stroke proportions on real crops). left-aligned, vertically centered
def ResizeLineImageFixed(pilImg,height=INPUT_HEIGHT,width=INPUT_WIDTH,padValue=255):
    img = pilImg.convert("L")
    w,h = img.size
    if w<=0 or h<=0:
        return Image.new("L",(width,height),padValue)
    s = min(height/h,width/w)
    newW = max(1,round(w*s))
    newH = max(1,round(h*s))
    resized = img.resize((newW,newH),Image.Resampling.BILINEAR)
    canvas = Image.new("L",(width,height),padValue)
    canvas.paste(resized,(0,max(0,(height-newH)//2)))
    return canvas


#pil image -> normalized float array, ink high / background low, ~[-1,1]
def TensorFromResized(pilImg):
    arr = np.array(pilImg,dtype=np.float32)/255.0
    arr = 1.0-arr
    arr = (arr-0.5)/0.5
    return arr[np.newaxis,:,:]   #(1,H,W), matches the old tensor.unsqueeze(0) shape


#builds the 3x3 homogeneous forward matrix for a rotation+shear around the
#image centre (translate to origin, shear, rotate, translate back -- same
#composition order torchvision's affine uses), then PIL needs the INVERSE of
#this (it maps each output pixel back to where it came from in the source) --
#computed numerically via numpy.linalg.inv rather than hand-derived algebra,
#so a sign error can't creep into the one thing every augmented sample passes
#through
def PilAffine(img,angleDeg=0.0,shearDeg=0.0,fillValue=255):
    w,h = img.size
    cx,cy = w/2.0,h/2.0
    angle,shear = math.radians(angleDeg),math.radians(shearDeg)
    cosA,sinA,tanS = math.cos(angle),math.sin(angle),math.tan(shear)
    R = np.array([[cosA,-sinA,0],[sinA,cosA,0],[0,0,1]])
    Sh = np.array([[1,tanS,0],[0,1,0],[0,0,1]])
    T1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]])
    T2 = np.array([[1,0,cx],[0,1,cy],[0,0,1]])
    M = T2@R@Sh@T1
    Minv = np.linalg.inv(M)
    coeffs = (Minv[0,0],Minv[0,1],Minv[0,2],Minv[1,0],Minv[1,1],Minv[1,2])
    return img.transform(img.size,Image.AFFINE,coeffs,resample=Image.BILINEAR,fillcolor=fillValue)


#Table 3 "combined" augmentation recipe: shear by an angle sampled as atan(k)
def AugmentShear(img):
    k = random.uniform(-0.6,0.6)
    shearDeg = math.degrees(math.atan(k))
    return PilAffine(img,angleDeg=0.0,shearDeg=shearDeg)


#small rotation, +-2.5 degrees
def AugmentRotate(img):
    angle = random.uniform(-2.5,2.5)
    return PilAffine(img,angleDeg=angle,shearDeg=0.0)


#solves the 8 coefficients PIL's Image.PERSPECTIVE needs to map each output
#pixel back to its source (the standard 4-point-correspondence linear system,
#solved with numpy.linalg.solve instead of a hand-rolled solver)
def FindPerspectiveCoeffs(startpoints,endpoints):
    matrix = []
    for (x,y),(X,Y) in zip(endpoints,startpoints):
        matrix.append([x,y,1,0,0,0,-X*x,-X*y])
        matrix.append([0,0,0,x,y,1,-Y*x,-Y*y])
    A = np.array(matrix,dtype=np.float64)
    B = np.array(startpoints,dtype=np.float64).reshape(8)
    return tuple(np.linalg.solve(A,B))


#stand-in for the paper's Luo et al. "geometric transformations" (their exact
#method needs an external repo not vendored here) -- mild perspective jitter
def AugmentPerspective(img):
    w,h = img.size
    DistortionScale = random.uniform(0.03,0.08)
    startpoints,endpoints = PerspectiveJitterPoints(w,h,DistortionScale)
    coeffs = FindPerspectiveCoeffs(startpoints,endpoints)
    return img.transform(img.size,Image.PERSPECTIVE,coeffs,resample=Image.BILINEAR,fillcolor=255)


#corner points for AugmentPerspective: 4 image corners, each jittered by up
#to distortionScale*halfDimension
def PerspectiveJitterPoints(width,height,DistortionScale):
    halfW,halfH = width//2,height//2
    topleft = (0,0)
    topright = (width-1,0)
    botright = (width-1,height-1)
    botleft = (0,height-1)
    startpoints = [topleft,topright,botright,botleft]

    def Jitter(x,y,maxDx,maxDy):
        return (
            int(x+random.uniform(-maxDx,maxDx)),
            int(y+random.uniform(-maxDy,maxDy)),
        )

    maxDx = DistortionScale*halfW
    maxDy = DistortionScale*halfH
    endpoints = [
        Jitter(*topleft,maxDx,maxDy),
        Jitter(*topright,maxDx,maxDy),
        Jitter(*botright,maxDx,maxDy),
        Jitter(*botleft,maxDx,maxDy),
    ]
    return startpoints,endpoints


#simard et al. elastic distortion: a smoothed random displacement field warps
#the image. must be called AFTER resizing to the fixed canvas, not on a raw
#high-DPI crop -- on a huge crop this allocates several full-res float64
#arrays per call, which is what caused a real MemoryError under parallel
#DataLoader workers
def AugmentElastic(img):
    sigma = random.choice([3,4])
    alpha = random.choice([15,20])
    arr = np.array(img,dtype=np.float32)
    h,w = arr.shape

    dx = (np.random.rand(h,w)*2-1)
    dy = (np.random.rand(h,w)*2-1)
    dx = gaussian_filter(dx,sigma,mode="constant",cval=0)*alpha
    dy = gaussian_filter(dy,sigma,mode="constant",cval=0)*alpha

    x,y = np.meshgrid(np.arange(w),np.arange(h))
    indices = (np.reshape(y+dy,(-1,1)),np.reshape(x+dx,(-1,1)))
    warped = map_coordinates(arr,indices,order=1,mode="constant",cval=255.0).reshape(h,w)
    return Image.fromarray(np.clip(warped,0,255).astype(np.uint8))

AUGMENTATIONS = [AugmentShear,AugmentRotate,AugmentElastic,AugmentPerspective]


#applies at most ONE augmentation, each gated independently at p=0.5 (Table 3
#"combined" recipe). call only on an already-resized image, see AugmentElastic
def AugmentLineImage(pilImg):
    candidates = [fn for fn in AUGMENTATIONS if random.random()<0.5]
    if not candidates:
        return pilImg
    return random.choice(candidates)(pilImg)


#line crops are stored as PNG bytes (not raw arrays/live PIL objects) --
#mostly-white handwriting scans compress very well, keeping the on-disk
#cache small
def EncodePng(pilImg):
    buf = io.BytesIO()
    pilImg.save(buf,format="PNG")
    return buf.getvalue()


def DecodePng(pngBytes):
    return Image.open(io.BytesIO(pngBytes)).convert("L")


#caches ONLY preprocessed line-crop PNG bytes, keyed by page+line index --
#completely independent of label TEXT (labels are re-read fresh from
#_labels.txt every run in IAMLineDatasetRaw, so correcting a label never
#requires re-running segmentation). only rebuild this cache if the
#segmentation code itself changes. incremental shard+manifest checkpointing:
#manifest.pt tracks which pages are done, each checkpoint writes only new
#pages into a fresh shard file and never rewrites old ones
class LineImageCache:
    def __init__(self,rootDir,cacheDir="line_image_cache",ForceRebuild=False,
                 SaveEveryNPages=10,maxPages=None):
        self.cache_dir = Path(cacheDir)
        self.shards_dir = self.cache_dir/"shards"
        self.manifest_path = self.cache_dir/"manifest.pkl"
        self.shards_dir.mkdir(parents=True,exist_ok=True)

        #page_key ("author/page.png") -> list of PNG bytes, one per detected line
        self.pages = {}
        ProcessedPages = set()
        ShardFiles = []

        if self.manifest_path.exists() and not ForceRebuild:
            print(f"[ImageCache] Loading manifest from: {self.manifest_path}")
            try:
                with open(self.manifest_path,"rb") as f:
                    manifest = pickle.load(f)
                ProcessedPages = set(manifest.get("processed_pages",[]))
                ShardFiles = manifest.get("shard_files",[])
                for shardName in ShardFiles:
                    shardPath = self.shards_dir/shardName
                    if shardPath.exists():
                        with open(shardPath,"rb") as f:
                            self.pages.update(pickle.load(f))
                print(f"[ImageCache] Loaded images for {len(self.pages)} page(s) from {len(ShardFiles)} shard file(s).")
            except Exception as e:
                #covers a cache written by the old torch.save format (pre-library-
                #removal) as well as genuine corruption -- either way, rebuilding
                #from scratch is the only safe recovery, see the pages-not-yet-
                #processed loop below (one-time cost, not an error to propagate)
                print(f"[ImageCache] Existing cache unreadable ({e}) -- rebuilding from scratch.")
                ProcessedPages,ShardFiles,self.pages = set(),[],{}

        def SaveManifest():
            tmpPath = self.manifest_path.with_suffix(".tmp")
            with open(tmpPath,"wb") as f:
                pickle.dump({"processed_pages": list(ProcessedPages),"shard_files": ShardFiles},f)
            os.replace(tmpPath,self.manifest_path)

        #writes only newly-collected pages into their own shard file -- previously
        #written shards are never touched again
        def FlushPending(PendingPages):
            if not PendingPages:
                return
            shardName = f"shard_{len(ShardFiles):06d}.pkl"
            shardPath = self.shards_dir/shardName
            tmpPath = shardPath.with_suffix(".tmp")
            with open(tmpPath,"wb") as f:
                pickle.dump(PendingPages,f)
            os.replace(tmpPath,shardPath)
            ShardFiles.append(shardName)
            SaveManifest()
            print(f"\n[ImageCache Checkpoint] Flushed {len(PendingPages)} new page(s) to {shardName} "
                  f"({len(self.pages)} pages cached total).")

        AuthorFolders = sorted(p.name for p in rootDir.iterdir() if p.is_dir())
        allPages = []
        for authorId in AuthorFolders:
            authorDir = rootDir/authorId
            for imgPath in sorted(authorDir.glob("*.png")):
                allPages.append((authorId,imgPath))

        if maxPages is not None and maxPages>0:
            allPages = allPages[:maxPages]

        TotalFiles = len(allPages)
        PageCounter = 0
        PendingPages = {}
        try:
            for idx,(authorId,imgPath) in enumerate(allPages,1):
                pageKey = f"{authorId}/{imgPath.name}"
                if pageKey in ProcessedPages:
                    continue

                pct = (idx/TotalFiles)*100.0 if TotalFiles>0 else 0.0
                print(f"\r[ImageCache Preprocessing] Progress: {pct:5.1f}% ({idx}/{TotalFiles} pages)",end="",flush=True)

                try:
                    #expectedLineCount/labelLines=None: segmentation stands on its
                    #own, never shaped by what a particular label file says
                    LineSamples,_,_ = ExtractLinePatches(
                        str(imgPath),
                        targetHeight=32,
                        maxWidth=1024,
                        expectedLineCount=None,
                        labelLines=None,
                    )
                except Exception as e:
                    print(f"\n[ImageCache] Skipping {pageKey} (segmentation error: {e})")
                    LineSamples = []

                pngList = []
                for sample in LineSamples:
                    pilLine = Image.fromarray(sample["raw_crop"]).convert("L")
                    pngList.append(EncodePng(pilLine))

                self.pages[pageKey] = pngList
                PendingPages[pageKey] = pngList
                ProcessedPages.add(pageKey)
                PageCounter+=1
                if PageCounter%SaveEveryNPages==0:
                    FlushPending(PendingPages)
                    PendingPages = {}

            if PageCounter>0:
                print()
        except KeyboardInterrupt:
            print("\n[Paused] Interrupted! Flushing pending page images before exiting.")
            FlushPending(PendingPages)
            raise

        #catches whatever's left from the last incomplete save_every_n_pages batch
        if PendingPages:
            FlushPending(PendingPages)

        print(f"[ImageCache] Total pages with cached images: {len(self.pages)}")

    def get(self,pageKey):
        return self.pages.get(pageKey)


#combines LineImageCache's label-independent cached images with each page's
#CURRENT label text (always re-read fresh from _labels.txt) into the final
#sample list. this zip-together step is cheap and redone every run, which is
#what makes label correction free -- only rebuild line_image_cache/ if
#segmentation itself changes
class IAMLineDatasetRaw:
    def __init__(
        self,
        rootDir,
        MinPagesForHoldout=3,
        cacheDir="line_image_cache",
        ForceRebuild=False,
        SaveEveryNPages=10,
        maxPages=None,
        isTrain=True,
    ):
        self.samples = []
        self.is_train = isTrain

        rootDir = self.ResolveDatasetRoot(Path(rootDir))

        ImageCache = LineImageCache(
            rootDir,
            cacheDir=cacheDir,
            ForceRebuild=ForceRebuild,
            SaveEveryNPages=SaveEveryNPages,
            maxPages=maxPages,
        )

        self.author_folders = sorted(p.name for p in rootDir.iterdir() if p.is_dir())
        self.author_to_idx = {author: idx for idx,author in enumerate(self.author_folders)}

        MismatchedPages = 0
        OversizedLabels = 0
        for authorId in self.author_folders:
            authorDir = rootDir/authorId
            AuthorPages = sorted(authorDir.glob("*.png"))
            if not AuthorPages:
                continue
            LabeledPages = [p for p in AuthorPages if ReadLabelLines(str(p))]
            HoldoutPageName = LabeledPages[-1].name if len(LabeledPages)>=MinPagesForHoldout else None

            for imgPath in AuthorPages:
                pageKey = f"{authorId}/{imgPath.name}"
                LabelLines = ReadLabelLines(str(imgPath))
                if not LabelLines:
                    continue

                pngList = ImageCache.get(pageKey)
                if pngList is None:
                    continue   #not in the image cache (e.g. maxPages truncation)

                if len(pngList)!=len(LabelLines):
                    MismatchedPages+=1
                    print(f"[Dataset] Skipping {pageKey}: cached image has {len(pngList)} line(s) but "
                          f"the current label has {len(LabelLines)} line(s) -- segmentation and label "
                          f"disagree on line count for this page. This page's images need re-cropping "
                          f"(delete/rebuild line_image_cache/ for it) before it can be used.")
                    continue

                isHoldout = imgPath.name==HoldoutPageName
                for lineIdx in range(len(LabelLines)):
                    text = LabelLines[lineIdx]
                    target = EncodeText(text)
                    if len(target)==0:
                        continue
                    #a label this long almost certainly isn't one real handwritten
                    #line (collapsed/merged segmentation) -- and the ctc time
                    #dimension can't fit it anyway, so skip loudly instead of
                    #wasting a sample slot
                    if len(target)>MAX_SAFE_LABEL_CHARS:
                        OversizedLabels+=1
                        print(f"[Dataset] Skipping oversized line label ({len(target)} chars, cap is "
                              f"{MAX_SAFE_LABEL_CHARS}) at {pageKey} line {lineIdx} -- likely merged/"
                              f"collapsed line segmentation on this page, not a real single line: "
                              f"{text[:80]!r}...")
                        continue
                    self.samples.append({
                        "page_key": pageKey,
                        "line_idx": lineIdx,
                        "image_png": pngList[lineIdx],
                        "target": target,
                        "text": text,
                        "is_holdout": isHoldout,
                    })

        if MismatchedPages:
            print(f"[Dataset] {MismatchedPages} page(s) skipped due to image/label line-count mismatch.")
        if OversizedLabels:
            print(f"[Dataset] {OversizedLabels} line(s) skipped as oversized (likely collapsed/merged "
                  f"line segmentation -- see warnings above for which pages).")
        print(f"[Dataset] Total line samples: {len(self.samples)}")

    #some dataset roots nest the real author folders one level down in "data/"
    @staticmethod
    def ResolveDatasetRoot(rootDir):
        dataDir = rootDir/"data"
        return dataDir if dataDir.exists() and any(p.is_dir() for p in dataDir.iterdir()) else rootDir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        item = self.samples[idx]
        pilImg = ResizeLineImageFixed(DecodePng(item["image_png"]))
        if self.is_train:
            pilImg = AugmentLineImage(pilImg)
        imgTensor = TensorFromResized(pilImg)
        return imgTensor,item["target"],item["text"]

    def HoldoutSplitIndices(self):
        trainIdx = [i for i,s in enumerate(self.samples) if not s["is_holdout"]]
        testIdx = [i for i,s in enumerate(self.samples) if s["is_holdout"]]
        return trainIdx,testIdx


#thin wrapper so train/val subsets can independently toggle augmentation
#without deep-copying the whole underlying dataset
class TrainEvalView:
    def __init__(self,BaseDataset,indices,isTrain):
        self.base_dataset = BaseDataset
        self.indices = indices
        self.is_train = isTrain

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,i):
        item = self.base_dataset.samples[self.indices[i]]
        pilImg = ResizeLineImageFixed(DecodePng(item["image_png"]))
        if self.is_train:
            pilImg = AugmentLineImage(pilImg)
        imgTensor = TensorFromResized(pilImg)
        return imgTensor,item["target"],item["text"]


def CollateFn(batch):
    images,targets,texts = zip(*batch)
    TargetLengths = np.array([len(t) for t in targets],dtype=np.int64)
    FlatTargets = np.concatenate(targets)
    return np.stack(images,0),FlatTargets,TargetLengths,list(texts)


#the torch PaperCRNN model class (and its ConvBlock builder) used to live
#here -- removed once Evaluate.py (its last caller, for loading/measuring old
#.pt checkpoints) was itself converted to Classify.py's numpy models. nothing
#in CleanPipeline instantiates a torch model from this file anymore; what
#remains below is purely the data-loading pipeline, reused by TrainScratch.py
