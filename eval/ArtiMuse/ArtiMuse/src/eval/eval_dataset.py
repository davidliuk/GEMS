import os
import json
import argparse
import torch
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer

import sys
sys.path.append("src")
sys.path.append("src/artimuse")
from artimuse.internvl.model.internvl_chat.modeling_artimuse import InternVLChatModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size)
    return transform(image).unsqueeze(0)  

def main(args):
    # Setup paths
    model_path = os.path.join("checkpoints", args.model_name)
    results_dir = os.path.join("results", "dataset_results")
    input_json_path = f"test_datasets/{args.dataset}/test.json"
    test_image_path = f"test_datasets/{args.dataset}/images"
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"{args.dataset}_{args.model_name}.json")

    # Load model & tokenizer
    model = InternVLChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    ).eval().to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )

    generation_config = dict(
        max_new_tokens=8192,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    with open(input_json_path, 'r') as f:
        test_data = json.load(f)

    results = []
    for item in tqdm(test_data, desc=f"Evaluating {args.dataset}"):
        image_name = item['image']
        image_path = os.path.join(test_image_path, image_name)

        if not os.path.exists(image_path):
            print(f"{image_path} not exists.")
            continue

        pixel_values = load_image(image_path).to(torch.bfloat16).to(args.device)

        score = model.score(args.device, tokenizer, pixel_values, generation_config)

        results.append({
            "image": image_name,
            "score": score
        })

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ArtiMuse_AVA", help="Name of the model (must exist in checkpoints/)")
    parser.add_argument("--dataset", type=str, default="AVA", help="AVA | TAD66K | PARA | FLICKR-AES")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device number")
    args = parser.parse_args()

    main(args)
