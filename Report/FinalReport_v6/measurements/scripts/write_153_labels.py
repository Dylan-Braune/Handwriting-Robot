#the three pages of writer 153 whose labels do not line up with their crops.
#
#the content check read every crop and scored it against its own label and
#against its neighbours'.  on these three a crop read its NEIGHBOUR's label
#far better than its own -- on c03-003d line 6 the neighbour matched at 100
#per cent against 3 per cent for its own line -- which is misalignment, not
#reading error.  all three passed the count check, so the fault was invisible
#to it, exactly as on writer 384.
#
#the cause is the same: the old transcripts wrap at different places from the
#page, and the last line of each is a run-on carrying everything left over
#("Mr. Joshua Logan, attempted but failed to create the atmosphere of a
#city." as one label line, where the page breaks it across two).
#
#wording is taken from the printed prompt at the head of each form, which is
#the text the writer was copying.  the LINE BREAKS are read off the
#handwriting.  the originals are kept as .orig.
from pathlib import Path

SRC = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot"
           r"\Data\Datasets\IAMpages10\153")

PAGES = {
#---9 lines.  line 4 ends mid-word: the writer breaks "characters" as
#"charac-" / "ters", with the hyphen running off the tail of the final c
"c03-003d": """This is not a filmed play. It has been conceived
throughout in terms of the cinema, and again
and again it is the visual qualities of the
story, and the marriage of the central charac-
ters to their background, which bring the
film so vividly to life. In Fanny, which
also has its premiere tomorrow, the director,
Mr. Joshua Logan, attempted but failed to
create the atmosphere of a city.""",

#---9 lines.  "memory" on line 8 carries an erasure mark but is legible and
#is transcribed normally
"c03-007d": """In A Taste of Honey Mr. Richardson has
taken a town in the industrial North of
England and has made it live. The shabby
streets and wet pavements, the school
play-grounds, the public monuments and
the rubbish strewn canals - even the
worn head-stones in the churchyard, "sacred
to the memory of" - are seen as an
integral part of the story.""",

#---11 lines.  "promiscuous" on line 4 and "sex" on line 5 both carry
#erasure marks and are transcribed normally
"c03-016d": """But there is heart in the telling, and an
intense realism in the situation. A young
girl lives in a single dingy room with
her slatternly, promiscuous mother. In such
surroundings she learns sex is something
sordid, and when she experiences it for
the first time herself it is incoherently, clumsily,
but half shyly and half inquisitively. As
is the case in Fanny her first lover is a sailor
who leaves her to bear his child and sails
away.""",
}

#what the current extractor produces for each page
CROPS = {"c03-003d": 9, "c03-007d": 9, "c03-016d": 11}

bad = [(k, len(v.split("\n")), CROPS[k])
       for k, v in PAGES.items() if len(v.split("\n")) != CROPS[k]]
if bad:
    for k, got, want in bad:
        print("MISMATCH %s: transcript %d lines, extractor %d crops" % (k, got, want))
    raise SystemExit("refusing to write")

print("%-12s %-7s %-9s %s" % ("page", "lines", "old file", "action"))
print("-"*62)
for name, text in sorted(PAGES.items()):
    out = SRC/(name+"_labels.txt")
    old = len([l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()])
    bak = SRC/(name+"_labels.txt.orig")
    if not bak.exists():
        bak.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    out.write_text(text+"\n", encoding="utf-8")
    print("%-12s %-7d %-9d %s" % (name, len(text.split("\n")), old,
                                  "rewritten, original kept as .orig"))
print("\n3 pages, %d lines" % sum(CROPS.values()))
