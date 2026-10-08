"""Single source of truth for the active 10-author set and the shared
train/holdout split convention. Previously duplicated independently in
TrainAuthor10.py, TrainTextJoint.py and BuildStyleProfile10Authors.py --
drifting out of sync between them silently changed which lines a model
was trained/evaluated on. Adding a new author means editing this file
only, not every script that touches the author list."""

# 8 kept IAM dataset authors (dropped 154/155 for being redundant with
# 150/151/152's style cluster) + 2 personal authors.
DATASET_AUTHORS = ["150", "151", "152", "153", "384", "551", "552", "588"]
PERSONAL_AUTHORS = ["abhinav", "dylan", "owen", "thiya", "yeukita"]

# Train/holdout split convention shared by every script that segments a
# personal author's photos, so the same lines are held out everywhere in
# the project -- not three different "holdout" definitions for one person.
VAL_FRACTION = 0.15
SPLIT_SEED = 0

# Writer-ID classifier checkpoint naming: `TrainAuthor.py 10authors` trains
# and saves under this run name; server.py derives the filename it loads
# from it, instead of the two needing to be kept in sync by hand.
AUTHOR_CLASSIFIER_RUN_NAME = "author_classifier_10new"


def author_classifier_weights_filename():
    return f"{AUTHOR_CLASSIFIER_RUN_NAME}_weights.pt"
