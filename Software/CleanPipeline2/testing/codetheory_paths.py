r"""Theory for paths.py."""

INTRO = [
 ("p", "This file contains no algorithm. It records where the data sits on "
       "disk and who the thirteen writers are, and every other module asks it "
       "rather than working it out independently."),
 ("h", "Why it is a separate file"),
 ("p", "Earlier in the project each module computed its own paths by walking "
       "upwards from its own location in the directory tree. That works until "
       "a file is moved to a different folder, at which point the chain of "
       "parent directories is wrong and the module can no longer find the "
       "data. Collecting the knowledge in one place means there is exactly "
       "one line to change if the data moves, and a module that moves folder "
       "does not break."),
 ("p", "It is also deliberately not merged into one of the functional "
       "modules. Every module needs to know where the weights and profiles "
       "live, and if that knowledge lived inside, say, the recogniser, then "
       "the gantry driver would have to import the entire recogniser in order "
       "to find a directory name."),
]

FUNCS = {
 "Data": dict(
   output="The folder holding the datasets, the trained weights and the "
          "style profiles.",
   theory=[
     ("p", "Derived from the location of this file rather than written as an "
           "absolute path, so the project can be copied to another machine or "
           "another drive without editing anything."),
     ("note", "This path is one directory level up from where it was before "
              "the modules were flattened into a single folder, and the "
              "derivation was adjusted accordingly. A path derived by "
              "counting parent directories is correct only for the depth it "
              "was written for, which is a known fragility of this approach."),
   ]),

 "Weights": dict(
   output="The sub-folder holding trained checkpoints as `.npz` files.",
   theory=[
     ("p", "A checkpoint is a saved copy of every weight in a trained "
           "network. The `.npz` format is numpy's own archive of named "
           "arrays, which is used here because the networks are built from "
           "numpy arrays directly and no framework's format is involved."),
   ]),

 "Profiles": dict(
   output="The folder holding one JSON style profile per writer.",
   theory=[
     ("p", "A style profile is the complete description of one person's "
           "handwriting produced by style.py: their stored letterforms plus "
           "the measurements describing how they lay writing out. JSON is "
           "used because the contents are nested lists of numbers and text, "
           "and because a profile can then be inspected in a text editor "
           "when something looks wrong."),
   ]),

 "PersonalPages": dict(
   output="The folder holding the photographed pages and their transcripts.",
   theory=[
     ("p", "The pages written specifically for this project, as opposed to "
           "the public database. Each writer has a sub-folder containing the "
           "photographs and a matching transcript for each one."),
   ]),

 "Cache": dict(
   output="The folder holding caches and other large generated files.",
   theory=[
     ("p", "Extracting line crops from pages, and cutting glyphs out of those "
           "crops, are the slowest operations in the pipeline. Their results "
           "are cached here so that a change to a later stage can be tested "
           "without repeating them. Keeping the caches out of the code "
           "repository also keeps the repository small."),
   ]),

 "DatabaseWriters": dict(
   output="The identifiers of the eight writers taken from the public "
          "handwriting database.",
   theory=[
     ("p", "These are writers from a published corpus of scanned "
           "handwriting \\cite{marti2002}, identified by number. They provide "
           "variety that the project could not collect by hand: different "
           "ages, different nationalities, and in particular several strongly "
           "joined-up hands, which turn out to be the hardest case for every "
           "stage of the pipeline."),
   ]),

 "PersonalWriters": dict(
   output="The names of the five writers photographed for this project.",
   theory=[
     ("p", "Volunteers who each filled several pages, photographed with a "
           "phone camera. These are the writers the project can test "
           "end-to-end, because their pages exist as photographs rather than "
           "as clean scans and therefore exercise the whole pipeline from "
           "uneven lighting onwards."),
   ]),

 "AllWriters": dict(
   output="All thirteen writers, database then photographed.",
   theory=[
     ("p", "Thirteen in total, which is three more than requirement R6 asks "
           "for. The order is fixed so that every table in every results "
           "document lists writers in the same sequence."),
   ]),

 "ValFraction": dict(
   output="The share of data held back from training, 0.15.",
   theory=[
     ("h", "What a holdout is, and why the fraction lives here"),
     ("p", "Measuring a network on the same data it was trained on says "
           "almost nothing, because a network with enough capacity can "
           "memorise its training examples. Some data is therefore withheld "
           "from training entirely and used only for measurement; that is the "
           "holdout, or validation set."),
     ("p", "The fraction is defined here, once, rather than in each script "
           "that needs a split. If two scripts chose their own fractions, or "
           "their own random seeds, the same line could be held out by one "
           "and trained on by the other, and every accuracy figure in the "
           "project would be quietly wrong."),
   ]),

 "SplitSeed": dict(
   output="The fixed random seed used when splitting data, 0.",
   theory=[
     ("h", "Why a random choice is made repeatable"),
     ("p", "Deciding which lines are held out is a random choice, but it must "
           "be the \\emph{same} random choice every time the program runs. A "
           "random number generator started from a fixed seed produces the "
           "same sequence every time, so the split is both arbitrary, which "
           "is what makes it fair, and reproducible, which is what makes it "
           "trustworthy."),
     ("p", "This also means a holdout set can be recovered later even if it "
           "was not saved: replaying the split with the same seed reproduces "
           "it exactly."),
   ]),

 "PersonalWriterPages": dict(
   inputs=[("writer", "the lower case name of one photographed writer, for "
                      "example \"owen\"")],
   output="A generator yielding `(imagePath, labelPath)` for each of that "
          "writer's pages, in sorted filename order.",
   theory=[
     ("h", "What it is for"),
     ("p", "Finds one writer's pages and pairs each with its transcript. "
           "Every stage that needs real handwriting with known text starts "
           "here: training, profile building, and the measurement of "
           "segmentation accuracy."),
     ("h", "Why the order matters"),
     ("p", "Several measurement scripts pair the n-th detected line crop on a "
           "page with the n-th line of that page's transcript. If the pages "
           "were returned in whatever order the file system happened to "
           "supply, that pairing would still appear to work but would "
           "sometimes be comparing a line against the wrong label, and the "
           "resulting accuracy would be wrong in a way that no error message "
           "would reveal. Sorting makes the order a property of the "
           "filenames rather than of the file system."),
     ("h", "Why a generator"),
     ("p", "A generator produces its values one at a time as they are asked "
           "for, rather than building the whole list first. Here that means a "
           "caller can begin processing the first page immediately, and can "
           "stop after one page without the cost of listing them all."),
   ]),
}
