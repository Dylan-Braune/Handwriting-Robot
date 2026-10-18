#shared character set for the text recognizer and ctc forced alignment,
#split into its own file so ProfileIO and Classify can both use it without
#importing each other (ProfileIO needs it for CtcForcedAlign, Classify
#needs ProfileIO's StrokeNormalize for the shape classifier)
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:'\"!?()-"
CHAR_TO_IDX = {char:idx+1 for idx,char in enumerate(CHARSET)}   #index 0 reserved for the ctc blank
IDX_TO_CHAR = {idx:char for char,idx in CHAR_TO_IDX.items()}
