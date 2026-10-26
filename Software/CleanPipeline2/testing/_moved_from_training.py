#file i/o and the command line entry point lifted out of the trainers,
#rebuilt in testing/.  kept here as the source to work from

#reads a page's per-line hand-verified transcript. labelPath defaults to
#the image's own "<stem>_labels.txt" sibling; pass it explicitly for a
#dataset that keeps images/labels in separate Pages/Labels folders
def ReadLabelLines(imgPath,labelPath=None):
    labelPath = labelPath or (os.path.splitext(imgPath)[0]+"_labels.txt")
    if not os.path.exists(labelPath):
        return []

    with open(labelPath,"r",encoding="utf-8") as f:
        rawLines = [line.strip() for line in f.readlines() if line.strip()]

    #label lines are "<line_number>\t<text>" -- strip the leading index so
    #it never leaks into training targets as a literal digit character
    return [re.sub(r'^\d+\t','',line) for line in rawLines]


#line crops are stored as PNG bytes (not raw arrays/live PIL objects) --
#mostly-white handwriting scans compress very well, keeping the on-disk
#cache small
def EncodePng(pilImg):
    buf = io.BytesIO()
    pilImg.save(buf,format="PNG")
    return buf.getvalue()


def DecodePng(pngBytes):
    return Image.open(io.BytesIO(pngBytes)).convert("L")


def EncodePng(grayCrop):
    buf = io.BytesIO()
    Image.fromarray(grayCrop).convert("L").save(buf,format="PNG")
    return buf.getvalue()


def Main():
    parser = argparse.ArgumentParser(description="Train PaperCRNN or AuthorClassifierCNN from scratch (NumpyNN only, no torch).")
    sub = parser.add_subparsers(dest="model",required=True)

    textP = sub.add_parser("text")
    textP.add_argument("--epochs",type=int,default=50)
    textP.add_argument("--batch-size",type=int,default=16)
    textP.add_argument("--lr",type=float,default=1e-3)
    textP.add_argument("--seed",type=int,default=7)
    textP.add_argument("--max-pages",type=int,default=None,help="`pages` source only")
    textP.add_argument("--data-dir",type=str,default=None)
    textP.add_argument("--source",choices=["pages","teklia"],default="pages",
                       help="`pages`: segment IAMpages671 from scratch (default). "
                            "`teklia`: pre-segmented Teklia/IAM-line lines, exported "
                            "once via ExportTekliaLines.py -- no segmentation needed")
    textP.add_argument("--resume",action="store_true",
                       help="continue from paper_cnn_bilstm_ctc_scratch_checkpoint.npz (epoch counter and "
                            "best-val-loss included) if it exists -- picks up after a crash")
    textP.add_argument("--warm-start",type=str,default=None,
                       help="load initial weights from this .npz checkpoint instead of random init "
                            "(e.g. an old converted paper_cnn_bilstm_ctc_joint_best.npz)")

    authorP = sub.add_parser("author")
    authorP.add_argument("--epochs",type=int,default=20)
    authorP.add_argument("--batch-size",type=int,default=16)
    authorP.add_argument("--lr",type=float,default=1e-3)
    authorP.add_argument("--seed",type=int,default=0)
    authorP.add_argument("--resume",action="store_true",
                         help="continue from author_classifier_10new_scratch_checkpoint.npz if it exists")
    authorP.add_argument("--warm-start",type=str,default=None,
                         help="load initial weights from this .npz checkpoint instead of random init")

    args = parser.parse_args()

    if args.model=="text":
        defaultDir = DEFAULT_TEKLIA_DATA_DIR if args.source=="teklia" else DEFAULT_TEXT_DATA_DIR
        dataDir = Path(args.data_dir) if args.data_dir else defaultDir
        print(f"\n--- Training paper_cnn_bilstm_ctc_scratch (source={args.source}, lr={args.lr}, "
              f"batch_size={args.batch_size}, max_epochs={args.epochs}, data={dataDir}) ---")
        best = TrainTextModel(args.epochs,args.batch_size,args.lr,args.seed,args.max_pages,dataDir,
                              source=args.source,resume=args.resume,warmStart=args.warm_start)
        print(f"\nDone. Best validation loss: {best:.4f}")
    else:
        print(f"\n--- Training author_classifier_10new_scratch (lr={args.lr}, batch_size={args.batch_size}, "
              f"max_epochs={args.epochs}) ---")
        TrainAuthorModel(args.epochs,args.batch_size,args.lr,args.seed,resume=args.resume,warmStart=args.warm_start)