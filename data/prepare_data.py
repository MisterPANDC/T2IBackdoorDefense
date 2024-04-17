import os
import json
import argparse
from PIL import Image
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="coco", choices=["coco", "laion"])
parser.add_argument('--category', '-c', type=str, required=True, choices=["cat", "dog"], help="Category of images to download")
parser.add_argument('--num_images', '-n', type=int, default=250, help="Number of images")

args = parser.parse_args()

if args.dataset == "coco":
    dataset = load_dataset('nlphuji/mscoco_2014_5k_test_image_text_retrieval')
    dataset = dataset["test"]
elif args.dataset == "laion":
    dataset = load_dataset("RobinWZQ/improved_aesthetics_6.5plus")
    dataset = dataset["train"]
else:
    raise ValueError("Invalid dataset")

if args.category == "cat":
    key_words = ["cat", "cats", "kitten"]
    opposite_key_words = ["dog", "dogs", "puppy"]
elif args.category == "dog":
    key_words = ["dog", "dogs", "puppy"]
    opposite_key_words = ["cat", "cats", "kitten", 'hotdog', 'hot']
else:
    raise ValueError("Invalid category")

if not os.path.exists(f"{args.category}_{args.num_images}"):
    os.makedirs(f"{args.category}_{args.num_images}")
black_list = []
caption_list = []
for i, data in enumerate(dataset):
    image = data['image']

    caption = data["caption"][0]
    caption = caption.lower()
    caption_words = caption.split()
    if i not in black_list and any([key_word in caption_words for key_word in key_words]) and not any([key_word in caption_words for key_word in opposite_key_words]):
        caption_list.append({"file_name": f"{i}.png", "caption": caption})
        image.save(f"{args.category}_{args.num_images}/{i}.png")
        if len(caption_list) == args.num_images:
            break
    
# store caption list in metadata.jsonl
with open(f"{args.category}_{args.num_images}/metadata.jsonl", "w") as f:
    for data in caption_list:
        f.write(json.dumps(data) + "\n")
