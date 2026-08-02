from pathlib import Path
from collections import Counter
import pandas as pd

# ---------------------------------------------------------
# 1. PATH SETUP (Dynamic and bulletproof)
# ---------------------------------------------------------
# Go up two levels from this script's folder to reach the project root
project_root = Path(__file__).resolve().parents[1]

# Build the exact path to the dataset folder and the text files
dataset_path = project_root / "data" / "datasets" / "archive" / "iam dataset"
forms_txt_path = dataset_path / "ascii" / "forms.txt"
lines_txt_path = dataset_path / "ascii" / "lines.txt"

# Quick sanity check so it tells you immediately if the path is wrong
if not forms_txt_path.exists():
    print(f"ERROR: Could not find forms.txt at {forms_txt_path}")
    print("Please double check your folder names!")
    exit()

# ---------------------------------------------------------
# 2. FIND THE TOP 10 WRITERS
# ---------------------------------------------------------
form_to_writer = {}
writer_form_counts = Counter()

print("Parsing forms.txt to find top writers...")
with open(forms_txt_path, "r") as f:
    for line in f:
        # Skip comments and blank lines
        if line.startswith("#") or not line.strip():
            continue
        
        parts = line.strip().split()
        form_id = parts[0]
        writer_id = parts[1]
        
        form_to_writer[form_id] = writer_id
        writer_form_counts[writer_id] += 1

# Extract just the IDs of the top 10 most frequent writers
top_10_writers = set(writer for writer, count in writer_form_counts.most_common(10))
print(f"Top 10 Writer IDs: {top_10_writers}")

# ---------------------------------------------------------
# 3. ISOLATE THEIR LINE IMAGES
# ---------------------------------------------------------
dataset_records = []

print("Parsing lines.txt and mapping image paths...")
with open(lines_txt_path, "r") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
            
        parts = line.strip().split()
        line_id = parts[0]
        
        # Extract form_id from line_id (e.g., 'a01-000u-00' -> 'a01-000u')
        form_id = "-".join(line_id.split("-")[:2])
        writer_id = form_to_writer.get(form_id)
        
        # If this line was written by one of our top 10 authors, save it
        if writer_id in top_10_writers:
            status = parts[1] # 'ok' or 'err' (segmentation quality)
            transcription = " ".join(parts[8:]).replace("|", " ")
            
            # Construct the physical image path using pathlib
            folder_part1 = line_id.split("-")[0]
            folder_part2 = f"{line_id.split('-')[0]}-{line_id.split('-')[1]}"
            
            # Full absolute path to the image
            img_path = dataset_path / "lines" / folder_part1 / folder_part2 / f"{line_id}.png"
            
            # Create a string path relative to the project root to save in the CSV
            # This makes the CSV clean and usable from anywhere in your project
            try:
                rel_image_path = str(img_path.relative_to(project_root))
            except ValueError:
                # Fallback if relative mapping fails
                rel_image_path = str(img_path)
            
            dataset_records.append({
                "line_id": line_id,
                "writer_id": writer_id,
                "image_path": rel_image_path,
                "status": status,
                "text": transcription
            })

# ---------------------------------------------------------
# 4. SAVE TO CSV
# ---------------------------------------------------------
df_subset = pd.DataFrame(dataset_records)

# Save the CSV right next to this python script for now
csv_output_path = Path(__file__).resolve().parent / "top_10_writers_lines.csv"
df_subset.to_csv(csv_output_path, index=False)

print(f"\nSuccess! Total line images isolated: {len(df_subset)}")
print(f"Saved manifest to: {csv_output_path}")