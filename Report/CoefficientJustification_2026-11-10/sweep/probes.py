#u23539195 coefficient justification: the fixed inputs every sweep is run on
#one page and one line are used throughout, so a panel only ever differs by the
#coefficient.  the page is a photographed page of the submitted dataset and the
#line is one of its own crops
PAGE = "PersonalDataset/Owen/Pages/Owen01.jpg"
PAGE_B = "PersonalDataset/Abhinav/Pages/Abhinav01.jpg"
PAGE_C = "PersonalDataset/Yeukita/Pages/Yeukita01.jpg"

#the line used for every line-level panel, and its writer
LINE = 3
AUTHOR = "owen"

#the two checkpoints the sweeps score with.  the reader is NOT the one the
#submitted recognise.py names: that file points at a checkpoint which reads at
#23 % and is reported as a defect, so the sweeps use the best reader on disk and
#say so everywhere
TEXT_WEIGHTS = "paper_cnn_bilstm_ctc_joint_best.npz"
TEXT_WEIGHTS_SUBMITTED = "paper_cnn_bilstm_ctc_scratch_best.npz"

#the writer classifier the submitted code names does not exist on disk; this is
#the closest checkpoint that does, and it holds 13 writers
AUTHOR_WEIGHTS = "author_shape_10new_weights.npz"
AUTHOR_WEIGHTS_SUBMITTED = "author_shape_11w_pw170_scratch_weights.npz"

#the sentence every synthesis panel writes, chosen to hold ascenders,
#descenders, repeated letters and a word gap
SENTENCE = "the quick brown fox jumps over the lazy dog"
SENTENCE_SHORT = "handwriting robot"
