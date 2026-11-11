#u23539195 D.R. Braune.  where the data lives, and who the writers are
from pathlib import Path

#the data folder beside this one
Data = Path(__file__).resolve().parent.parent/"CNN"

Weights = Data/"weights"
Profiles = Data/"NOGIT"/"StyleProfiles10"
PersonalPages = Data/"PersonalDataset"
Cache = Data/"NOGIT"

#the eight database writers and the five photographed ones
DatabaseWriters = ["150","151","152","153","384","551","552","588"]
PersonalWriters = ["abhinav","dylan","owen","thiya","yeukita"]
AllWriters = DatabaseWriters+PersonalWriters

#held-out fraction and the split seed, shared by every script
ValFraction = 0.15
SplitSeed = 0

#---PersonalWriterPages: yields (imagePath,labelPath) for one photographed writer's pages---
def PersonalWriterPages(writer):
    folder = PersonalPages/writer.capitalize()
    for img in sorted((folder/"Pages").glob(writer.capitalize()+"*.jpg")):
        yield img,folder/"Labels"/(img.stem+"_labels.txt")
