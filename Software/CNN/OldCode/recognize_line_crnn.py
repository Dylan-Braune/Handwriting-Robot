import argparse
from pathlib import Path

import torch

from FullLineBoxMaker import ExtractLinePatches, ReadLabelLines
from train_handwriting_robot_v1_baseline import MultiTaskLineCRNN, decode_ctc, normalize_line_image


def load_model(weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    charset = checkpoint["charset"]
    author_mapping = checkpoint["author_mapping"]
    idx_to_author = {idx: author for author, idx in author_mapping.items()}

    model = MultiTaskLineCRNN(
        num_authors=len(author_mapping),
        num_classes=len(charset) + 1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, idx_to_author


def recognize_page(image_path, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, idx_to_author = load_model(weights_path, device)

    input_height = checkpoint.get("input_height", 32)
    input_width = checkpoint.get("input_width", 512)
    labels = ReadLabelLines(str(image_path))

    samples, _, _ = ExtractLinePatches(
        str(image_path),
        targetHeight=input_height,
        maxWidth=input_width,
        expectedLineCount=len(labels) if labels else None,
        labelLines=labels if labels else None,
    )

    print(f"Image: {image_path}")
    print(f"Detected lines: {len(samples)}")
    print("=" * 80)

    with torch.no_grad():
        for idx, sample in enumerate(samples):
            image_tensor = normalize_line_image(sample["processed_patch"], input_height, input_width)
            image_tensor = image_tensor.unsqueeze(0).to(device)

            author_logits, text_log_probs = model(image_tensor)
            prediction = decode_ctc(text_log_probs)[0]
            author_idx = int(torch.argmax(author_logits, dim=1).item())
            author = idx_to_author.get(author_idx, str(author_idx))

            print(f"Line {idx:02d} | author={author}")
            print(f"  pred : {prediction}")
            if idx < len(labels):
                print(f"  truth: {labels[idx]}")


def main():
    script_dir = Path(__file__).resolve().parent
    default_weights = script_dir / "weights" / "train_handwriting_robot_v1_baseline_weights.pth"
    default_image = script_dir.parents[1] / "Data" / "Datasets" / "IAMpages10" / "150" / "c03-000a.png"

    parser = argparse.ArgumentParser(description="Recognize handwritten line text and author using the trained CRNN.")
    parser.add_argument("--image", default=str(default_image))
    parser.add_argument("--weights", default=str(default_weights),
                        help="Path to a weights .pth file, e.g. weights/train_handwriting_robot_v1_baseline_weights.pth")
    args = parser.parse_args()

    recognize_page(Path(args.image), Path(args.weights))


if __name__ == "__main__":
    main()
