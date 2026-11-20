#u23539195 D.R. Braune.  where the data lives, and who the writers are
from pathlib import Path

#the packaged data folder inside DemoCode (original sources remain unchanged)
Data = Path(__file__).resolve().parent/"Data"/"CNN"   #the folder holding this file then down into Data and CNN

Weights = Data/"weights"
Profiles = Data/"NOGIT"/"StyleProfiles10"
PersonalPages = Data/"PersonalDataset"
Cache = Data/"NOGIT"

#every writer there is DATA for on disk: eight from the database and the five
#photographed ones.  training can draw on all of them
DatabaseWriters = ["150","151","152","153","384","551","552","588"]   #iam form writers kept under their numeric id
PersonalWriters = ["abhinav","dylan","owen","thiya","yeukita"]
AllWriters = DatabaseWriters+PersonalWriters   # eight database writers plus the five photographed ones

#the ELEVEN the shipped namer has a class for, and so the eleven with a glyph
#library in Profiles.  151 and 551 have data but no class and no profile: they
#were dropped before the namer was trained, and the web app reads its author
#list off the profile folder rather than off this list, so the two cannot
#drift apart
ModelWriters = ["150","152","153","384","552","588",
                "abhinav","dylan","owen","thiya","yeukita"]

#held-out fraction and the split seed, shared by every script
ValFraction = 0.15   #fifteen percent of every writer is held back for validation
SplitSeed = 0   # a fixed seed so the same lines are held out on every run

#---PersonalWriterPages: yields (imagePath,labelPath) for one photographed writer's pages---
def PersonalWriterPages(writer):
    folder = PersonalPages/writer.capitalize()   # folder names are capitalised so dylan becomes Dylan
    for img in sorted((folder/"Pages").glob(writer.capitalize()+"*.jpg")):    #sorted by name so Dylan01 comes before Dylan02
        yield img,folder/"Labels"/(img.stem+"_labels.txt")   #each page pairs with the transcript of the same stem
