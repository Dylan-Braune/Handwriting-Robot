#u23539195 Dylan Braune.  where the data lives, and who the writers are.
#
#every other file used to work this out for itself with its own
#parent.parent chain, which broke the moment a file moved folder.  it is all
#here now so there is exactly one line to change if the data moves
from pathlib import Path

#the data sits beside the code, not inside it, so none of it is duplicated
#  Software/CleanPipeline2/paths.py  ->  Software/CNN
Data = Path(__file__).resolve().parent.parent/"CNN"

Weights = Data/"weights"                       #trained .npz checkpoints
Profiles = Data/"NOGIT"/"StyleProfiles10"      #one json style profile per writer
PersonalPages = Data/"PersonalDataset"         #photographed pages and their labels
Cache = Data/"NOGIT"                           #line caches, glyph caches, captures

#the writers.  8 from the public database and 5 photographed for this
#project, 13 in all, which is 3 more than requirement R6 asks for
DatabaseWriters = ["150","151","152","153","384","551","552","588"]
PersonalWriters = ["abhinav","dylan","owen","thiya","yeukita"]
AllWriters = DatabaseWriters+PersonalWriters

#held out fraction and the fixed seed, kept here so every script splits the
#same lines.  a seed that differed between training and evaluation would
#quietly leak training lines into the measurement
ValFraction = 0.15
SplitSeed = 0


#yields (imagePath,labelPath) for one photographed writer's pages
def PersonalWriterPages(writer):
    folder = PersonalPages/writer.capitalize()
    for img in sorted((folder/"Pages").glob(writer.capitalize()+"*.jpg")):
        yield img,folder/"Labels"/(img.stem+"_labels.txt")
