#orchestration lifted out of the style builder, rebuilt properly in
#testing/buildprofiles.py.  kept here only as the source to work from

#builds all 10 (original IAM-only) profiles from non-holdout pages. extraction
#is expensive and cached per author under RAW_DIR; re-running only re-filters
#unless the cache is missing or useCache is off
def BuildAll(maxLinesPerAuthor=None,useCache=True):
    import pickle
    from TrainText import IAMLineDatasetRaw,DecodePng

    base = IAMLineDatasetRaw(rootDir=str(DATA_DIR),cacheDir=str(CACHE_DIR))
    byAuthor = {}
    for s in base.samples:
        if s['is_holdout']:
            continue   #profiles never see holdout text
        a = s['page_key'].split('/')[0]
        byAuthor.setdefault(a,[]).append(s)

    RAW_DIR.mkdir(parents=True,exist_ok=True)
    PROFILE_DIR.mkdir(parents=True,exist_ok=True)
    model = None
    allParsed = {}

    for a in sorted(byAuthor):
        rawPath = RAW_DIR/f"{a}.pkl"
        parsed = None
        if useCache and rawPath.exists():
            with open(rawPath,'rb') as f:
                parsed = pickle.load(f)
        if parsed is None:
            if model is None:
                model = PaperCRNNNumpy(checkpoint_path=TEXT_WEIGHTS)
            items = []
            for s in byAuthor[a][:maxLinesPerAuthor]:
                gray = np.array(DecodePng(s['image_png']).convert('L'))
                items.append((gray,s['text']))
            parsed = ExtractAuthorRaw(a,items,model)
            with open(rawPath,'wb') as f:
                pickle.dump(parsed,f)
        #reference render stats cost nothing (no model) and never depend on the extraction cache
        refs = [RenderRefStats(np.array(DecodePng(s['image_png'])
                                        .convert('L')))
                for s in byAuthor[a][:maxLinesPerAuthor]]
        allParsed[a] = (parsed,refs)

    #pool every author's raw variants to learn what each LETTER looks like
    #across all ten hands -- only rejects an author's own worst variants
    rawLibs = {}
    for a2,(parsed2,_r) in allParsed.items():
        lib = {}
        for glyphs,_st in parsed2:
            for g in glyphs:
                if g and g.get('char',' ')!=' ' and 'strokes' in g:
                    lib.setdefault(g['char'],[]).append(g)
        rawLibs[a2] = lib
    prior = BuildLetterPrior(rawLibs)
    print('  letter prior learned for %d characters'%len(prior))

    for a2 in sorted(allParsed):
        parsed2,refs2 = allParsed[a2]
        prof = BuildAuthorProfile(a2,parsed2,refs=refs2,prior=prior)
        if prof is None:
            print('  %s: FAILED (no usable lines)'%a2)
            continue
        with open(PROFILE_DIR/('%s.json'%a2),'w',encoding='utf-8') as f:
            json.dump(prof,f)
    print(f"\nProfiles saved to {PROFILE_DIR}")




#segments every photo of one personal author via SegmentLean.SegmentLines,
#splits per-page into train/holdout (same convention as elsewhere), and
#returns (trainItems, holdoutTexts) where trainItems is ready for ExtractAuthorRaw.
#NOTE: this now uses SegmentLean.SegmentLines instead of the legacy
#SegmentPage.ProcessPage -- a deliberate, approved behavior change (matches
#every other part of this pipeline), NOT a drop-in identical swap: it changes
#which crops feed the personal-author glyph library, so NOGIT/StyleProfiles10/
#needs rebuilding (via BuildAll/BuildAll10Authors) before trusting synthesis output
def BuildPersonalLineItems(author):
    rng = random.Random(SPLIT_SEED)
    trainItems,holdoutTexts = [],[]
    for imgPath,labelPath in PersonalAuthorPages(author):
        leanLines,_overlay,_info = SL.SegmentLines(str(imgPath))
        crops = [np.asarray(ln["image"]) for ln in leanLines]
        gt = ReadLabelLines(str(imgPath),str(labelPath))
        gt = [g for g in gt if g.strip()!="MESS"]
        n = min(len(crops),len(gt))
        rows = list(zip(crops[:n],gt[:n]))
        rng.shuffle(rows)
        nVal = max(1,round(len(rows)*VAL_FRACTION))
        for crop,text in rows[nVal:]:
            trainItems.append((crop,text))
        for _crop,text in rows[:nVal]:
            holdoutTexts.append(text)
    return trainItems,holdoutTexts




#builds the 8-dataset + 2-personal (10-author) profile set: the 8 kept
#dataset authors load straight from their existing RAW_DIR cache (not
#re-extracted), the 2 personal authors are freshly extracted with the
#personal-tuned recognizer via BuildPersonalLineItems. everything downstream
#(letter prior, BuildAuthorProfile, json output) is shared with BuildAll
def BuildAll10Authors():
    import pickle

    RAW_DIR.mkdir(parents=True,exist_ok=True)
    PROFILE_DIR.mkdir(parents=True,exist_ok=True)

    allParsed = {}   # author -> (parsed, refs)

    for a in DATASET_AUTHORS:
        rawPath = RAW_DIR/f"{a}.pkl"
        if not rawPath.exists():
            raise SystemExit(f"[Error] expected cached raw glyphs at {rawPath} -- "
                              f"run BuildStyleProfile.py's BuildAll() first if missing.")
        with open(rawPath,"rb") as f:
            parsed = pickle.load(f)
        #refs aren't cached separately -- cheap, recomputed from the cached parsed data's own stats
        refs = [stats["ref"] for _glyphs,stats in parsed if "ref" in stats]
        allParsed[a] = (parsed,refs)
        print(f"[Dataset] {a}: {len(parsed)} lines loaded from cache")

    personalModel = PaperCRNNNumpy(checkpoint_path=PERSONAL_WEIGHTS)

    for a in PERSONAL_AUTHORS:
        rawPath = RAW_DIR/f"{a}.pkl"
        if rawPath.exists():
            with open(rawPath,"rb") as f:
                parsed = pickle.load(f)
            print(f"[Personal] {a}: {len(parsed)} lines loaded from cache")
        else:
            trainItems,holdoutTexts = BuildPersonalLineItems(a)
            print(f"[Personal] {a}: {len(trainItems)} train lines, "
                  f"{len(holdoutTexts)} held out")
            parsed = ExtractAuthorRaw(a,trainItems,personalModel)
            with open(rawPath,"wb") as f:
                pickle.dump(parsed,f)
            print(f"[Personal] {a}: {len(parsed)}/{len(trainItems)} lines usable after extraction")
        refs = [stats["ref"] for _glyphs,stats in parsed if "ref" in stats]
        allParsed[a] = (parsed,refs)

    #shared cross-author letter prior needs all 10 pooled
    rawLibs = {}
    for a2,(parsed2,_r) in allParsed.items():
        lib = {}
        for glyphs,_st in parsed2:
            for g in glyphs:
                if g and g.get('char',' ')!=' ' and 'strokes' in g:
                    lib.setdefault(g['char'],[]).append(g)
        rawLibs[a2] = lib
    prior = BuildLetterPrior(rawLibs)
    print(f"\n[Prior] learned for {len(prior)} characters across {len(allParsed)} authors")

    for a2 in sorted(allParsed):
        parsed2,refs2 = allParsed[a2]
        prof = BuildAuthorProfile(a2,parsed2,refs=refs2,prior=prior)
        if prof is None:
            print(f"  {a2}: FAILED (no usable lines)")
            continue
        with open(PROFILE_DIR/f"{a2}.json","w",encoding="utf-8") as f:
            json.dump(prof,f)
        print(f"  {a2}: profile saved")

    print(f"\nProfiles saved to {PROFILE_DIR}")
    print(f"Final 10-author set: {sorted(allParsed)}")

#default: build the original IAM-only 10 profiles. --10authors builds the
#8-dataset + 2-personal set instead
if __name__=='__main__':
    if "--10authors" in sys.argv:
        BuildAll10Authors()
    else:
        BuildAll()
