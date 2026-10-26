"""single source of truth for the active 10-author set and the shared
train/holdout split convention, so every script trains/evaluates on the
same lines instead of silently drifting out of sync with each other."""

from pathlib import Path

#this file lives in Software/CleanPipeline, a sibling of Software/CNN where
#the actual datasets/weights/profiles still live -- data is not duplicated
CNN_DATA_DIR = Path(__file__).resolve().parent.parent/"CNN"
PERSONAL_DATASET_DIR = CNN_DATA_DIR/"PersonalDataset"

DATASET_AUTHORS = ["150","151","152","153","384","551","552","588"]   #8 kept iam authors
PERSONAL_AUTHORS = ["abhinav","dylan","owen","thiya","yeukita"]   #5 personal authors

VAL_FRACTION = 0.15   #fraction of each author's lines held out for validation
SPLIT_SEED = 0   #fixed seed so the same lines are held out everywhere

AUTHOR_CLASSIFIER_RUN_NAME = "author_classifier_10new"


#builds the writer-id classifier checkpoint filename from the run name above.
#.npz now (ConvertPtToNpz.py converted the original .pt once) -- nothing in
#the live pipeline reads .pt files directly anymore
def AuthorClassifierWeightsFilename():
    return f"{AUTHOR_CLASSIFIER_RUN_NAME}_weights.npz"


#yields (imagepath, labelpath) pairs for one personal author's scanned pages
def PersonalAuthorPages(author):
    folder = PERSONAL_DATASET_DIR/author.capitalize()
    pagesDir,labelsDir = folder/"Pages",folder/"Labels"
    for imgPath in sorted(pagesDir.glob(f"{author.capitalize()}*.jpg")):
        yield imgPath,labelsDir/f"{imgPath.stem}_labels.txt"
