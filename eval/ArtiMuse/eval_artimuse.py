import os
import json
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer
from tqdm import tqdm
import sys
from pathlib import Path

# project_root/
# ├── eval_artimuse.py
# └── ArtiMuse/src/
src_root = Path(__file__).resolve().parent / "ArtiMuse" / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

try:
    from artimuse.internvl.model.internvl_chat.modeling_artimuse import InternVLChatModel
except ImportError:
    print("[ERROR] Cannot find 'artimuse' source code. Please check the directory structure.")
    sys.exit(1)

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
    model_path = args.model_path
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    InternVLChatModel.config_class.has_no_defaults_at_init = True

    print(f"[INFO] Loading model from: {model_path}")
    model = InternVLChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    ).eval().to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )

    generation_config = {
        "max_new_tokens": 100, 
        "do_sample": False, 
        "pad_token_id": tokenizer.eos_token_id
    }

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise ValueError(f"Input directory not found: {args.input_dir}")
        
    image_files = [
        f for f in os.listdir(input_dir)
        if Path(f).suffix.lower() in valid_extensions
    ]

    if not image_files:
        print(f"[WARN] No images found in {args.input_dir}")
        return

    print(f"[INFO] Found {len(image_files)} images. Starting scoring...")

    results = {}

    for img_name in tqdm(image_files, desc="Processing"):
        img_path = input_dir / img_name
        try:
            pixel_values = load_image(img_path).to(torch.bfloat16).to(args.device)
            
            score = model.score(args.device, tokenizer, pixel_values, generation_config)
            
            results[img_name] = round(float(score), 4) 
        except Exception as e:
            print(f"\n[ERROR] Failed to process {img_name}: {e}")
            results[img_name] = None

    valid_scores = [s for s in results.values() if s is not None]
    
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        results["average_score_all"] = round(avg_score, 4)
    else:
        results["average_score_all"] = None

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n[INFO] Done! Results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Aesthetic Scoring Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images to score")
    parser.add_argument("--output_file", type=str, default="output_scores.json", help="Path to the output JSON")
    parser.add_argument("--device", type=str, default="cuda", help="Computing device (e.g., cuda, cpu)")
    
    args = parser.parse_args()
    main(args)