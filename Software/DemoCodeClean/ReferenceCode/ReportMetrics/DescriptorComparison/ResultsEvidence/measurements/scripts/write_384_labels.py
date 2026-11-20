#transcriptions of writer 384's ten pages, one line of text per line of
#handwriting, read off the scans by eye.
#
#the previous label files were short by one to three lines on six of the nine
#pages they covered, and their line breaks did not match where the writer
#actually broke lines.  the old extractor merged written lines until its crop
#count matched the short transcript, so the counts agreed while the content
#did not, and train.py's count check saw nothing wrong.
#
#the wording comes from the printed prompt at the head of each form, which is
#the text the writer was copying, so the words are exact.  the LINE BREAKS
#come from the handwriting itself and from nowhere else.  where the writer
#departed from the prompt the handwriting is followed, and each such case is
#noted below.
#
#the originals are kept as <name>_labels.txt.orig before anything is written.
import sys
from pathlib import Path

SRC = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot"
           r"\Data\Datasets\IAMpages10\384")

PAGES = {
#---9 lines------------------------------------------------------------
"h06-079": """The equipment takes two forms - an insulating
guard on the jib of the crane, intended to
prevent direct contact with the line, and
electronic equipment with a sensitive probe
mounted slightly forward of the head of
the jib and with electronic assembly
and warning apparatus in the driver's
cab. Improved designs of both forms
are being tried out at present.""",

#---11 lines.  this page had NO label file at all, so it was never loaded.
#the writer stopped partway through the prompt, at "approaching": the printed
#text continues "the line, or height gauges..." and none of that was written.
#line 8 carries a struck-out "authority" before "underground", which is not
#transcribed, following the usual convention for deleted text
"h06-082": """There is, however, a danger that workers
will rely too much on fixed devices because
they fail to recognise their limitations.
It is therefore safer both in principle and
practice to keep the worker away from
overhead lines wherever possible, e.g., by
(1) re-routing the line, (2) putting the supply
underground, (3) making the line
dead (after consultation with the supply
authority), (4) providing barriers at a safe
distance to prevent vehicles from approaching""",

#---11 lines.  line 4 has a struck-out word between "new" and "type", not
#transcribed
"h06-085": """It is also important to remember that on
lines carrying the higher voltages flashover
from the line may take place without
actual contact. A new type of
machine has recently been developed for
the automatic rolling of sole leather
bends. The sheet of leather is placed
on a sliding feed tray outside the
danger zone and is then pushed
forward between two platens, the upper
one carrying a set of small rollers.""",

#---12 lines
"h06-089": """The lower platen, which supports the
leather, is raised hydraulically to bring
it into contact with the rollers on the upper
platen, which is then caused to make
several horizontal oscillations so that the
leather is rolled and pressed at the same time.
The danger zone between the platens is fenced
by a guard which is interlocked with the
hydraulic valve and the press is also
sequentially operated: the closing of the shutter
starts the machine, the rest of the cycle
following automatically.""",

#---10 lines.  line 7 is the writer's own error: the prompt reads "This
#consists basically", the page reads "The consists basically".  the page is
#followed
"h06-092": """A new method of applying sand to the
faces of green bricks has recently been
developed. In the case of wire-cut bricks
the column of clay from the pug-mill is
carried by a short length of belt
conveyor to the sanding plant.
The consists basically of a vibrating
hopper from which the sand is
distributed to all four faces by a
system of slots, scrapers and worms.""",

#---8 lines
"h06-096": """The conveyor is broken at this point to
enable the sand to be applied to
the under surface of the column, which
passes on through two pairs of
vibrating rollers which embed the sand
firmly in the surfaces and is finally
cut into bricks at the wire cutting
table.""",

#---11 lines
"j06-000": """All stages of Calanus, for example,
seem to migrate on some occasions while
any stage may not on others. Such data
cannot yet be rationalized. Where
information is less extensive, however, it is
possible to find some regularity in the
observations. Thus in Euphausia superba
from the Antarctic, the metanauplii remain
in deep water, the later larval stages
migrate diurnally, and the adolescents
stay permanently at the surface.""",

#---10 lines
"j06-005": """The migrating stages all come from 100-250
meters, and the time of their arrival at
the surface is directly related to their
swimming capacity: 3rd (oldest)
calyptopis from 1800-2200 hr; 2nd,
from 2200-0200 hr; and 1st, from 0200-0600 hr.
Similar ontogenetic differences are apparent
in Bosmina coregoni whose adults remain
at the surface while the young migrate
to and from a depth of 50 meters.""",

#---11 lines.  line 7 reads "behaviour" where the prompt reads "behavior",
#and line 10 reads "apparenetly" where the prompt reads "apparently".  both
#are the writer's own and both are transcribed as written
"j06-008": """Such permanent occurrence at the surface
could of course be considered the extreme
of a variable day depth. The effect of
day depths upon the surfacing of various
animals has been reviewed elsewhere. c.
Anomalies. In spite of the variability
of migrational behaviour, some kinds
of anomalies may be recognized.
Vertical movement occurs in some forms
apparenetly in the reverse manner to
that commonly met.""",

#---12 lines.  the old label had 11 and broke after "echo-producing layer,"
#rather than after "echo-producing", which threw every line below it out of
#step with its crop
"j06-014": """Such reversed migrations are known for
Acartia clausi, A. longiremis, Nyctiphanes
couchii, Evadne sp., Oithona nana, Daphnia
lumholzi, Stages 4, 5, and adult of
Calanus finmarchicus, Diaptomus banforanus,
and Cyclops bicuspidatus. An echo-producing
layer, which the authors think probably consists
of euphausiids, has also been described
as, in part, regularly moving in a reverse
manner. Most of these records are well
substantiated and involve whole populations
rather than aberrant individuals.""",
}

#the crop count the current extractor produces for each page, so a transcript
#that does not match is refused rather than written
CROPS = {"h06-079": 9, "h06-082": 11, "h06-085": 11, "h06-089": 12,
         "h06-092": 10, "h06-096": 8, "j06-000": 11, "j06-005": 10,
         "j06-008": 11, "j06-014": 12}

bad = [(k, len(v.split("\n")), CROPS[k])
       for k, v in PAGES.items() if len(v.split("\n")) != CROPS[k]]
if bad:
    for k, got, want in bad:
        print("MISMATCH %s: transcript %d lines, extractor %d crops" % (k, got, want))
    raise SystemExit("refusing to write")

print("%-12s %-7s %-9s %s" % ("page", "lines", "old file", "action"))
print("-"*64)
for name, text in sorted(PAGES.items()):
    out = SRC/(name+"_labels.txt")
    old = None
    if out.exists():
        old = len([l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()])
        bak = SRC/(name+"_labels.txt.orig")
        if not bak.exists():
            bak.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    out.write_text(text+"\n", encoding="utf-8")
    print("%-12s %-7d %-9s %s"
          % (name, len(text.split("\n")), old if old is not None else "none",
             "rewritten, original kept as .orig" if old is not None
             else "CREATED (page had no label)"))

print("\n%d pages, %d lines in total" % (len(PAGES), sum(CROPS.values())))
