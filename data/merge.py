import os
import json
import argparse
from PIL import Image
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="coco", choices=["coco", "laion"])
parser.add_argument('--path1', type=str, default="cat_250", help="Path to the first image folder")
parser.add_argument('--path2', type=str, default="dog_250", help="Path to the second image folder")
parser.add_argument('--new_path', type=str, default="cat_dog_500", help="Path to the new image folder")

args = parser.parse_args()

if not os.path.exists(args.new_path):
    os.makedirs(args.new_path)

# copy every png file in path1 to new_path
for file in os.listdir(args.path1):
    if file.endswith(".png"):
        # directly copy the file
        os.system(f"cp {os.path.join(args.path1, file)} {os.path.join(args.new_path, file)}")

# copy every png file in path2 to new_path
for file in os.listdir(args.path2):
    if file.endswith(".png"):
        # directly copy the file
        os.system(f"cp {os.path.join(args.path2, file)} {os.path.join(args.new_path, file)}")

# merge metadata every line is a dict
metadata1 = []
metadata2 = []
with open(os.path.join(args.path1, "metadata.jsonl"), "r") as f:
    for line in f:
        metadata1.append(json.loads(line))
with open(os.path.join(args.path2, "metadata.jsonl"), "r") as f:
    for line in f:
        metadata2.append(json.loads(line))

# merge metadata
metadata = metadata1 + metadata2

# store caption list in metadata.jsonl
with open(os.path.join(args.new_path, "metadata.jsonl"), "w") as f:
    for data in metadata:
        f.write(json.dumps(data) + "\n")
