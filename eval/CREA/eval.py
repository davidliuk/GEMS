import base64
import os
import json
import re
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI
import numpy as np

METRICS = ["originality", "expressiveness", "aesthetic", "technical", "unexpected", "interpretability"]

JUDGE_PROMPT_PATH = "eval/CREA/judge_prompt.txt"
API_URL = ""
API_KEY = "eval"
NUM_EVAL_WORKERS = 150

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="GEMS_z")
    parser.add_argument("--mapping_file", type=str, default=None)
    args = parser.parse_args()
    
    if args.mapping_file is None:
        args.mapping_file = os.path.join("eval/CREA/results", args.name, "image_paths.json")
    
    args.output_score_file = os.path.join(os.path.dirname(args.mapping_file), "evaluation_results.json")
    return args

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_json_array(text):
    try:
        match = re.search(r'(\[.*?\])', text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            if isinstance(data, list) and len(data) == 6:
                return [float(x) for x in data]
        data = json.loads(text)
        if isinstance(data, list) and len(data) == 6:
            return [float(x) for x in data]
    except Exception:
        pass
    return None

def evaluate_single_image(client, judge_system_prompt, prompt_text, image_path):
    if not os.path.exists(image_path):
        return None

    try:
        base64_image = encode_image(image_path)
        user_content = f"Target Prompt: {prompt_text}"

        response = client.chat.completions.create(
            model="kimi-k2.5",
            messages=[
                {
                    "role": "system",
                    "content": judge_system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_content},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=8192,
        )

        content = response.choices[0].message.content.strip()
        score_array = extract_json_array(content)
        
        if score_array:
            return {
                "prompt": prompt_text,
                "image_path": image_path,
                "scores": score_array
            }
        else:
            print(f"\n[Parsing Failed] Unable to extract array from output: {content}")
    except Exception as e:
        print(f"\nError evaluating {image_path}: {e}")
    return None

def main():
    args = parse_args()
    
    with open(JUDGE_PROMPT_PATH, 'r', encoding='utf-8') as f:
        judge_system_prompt = f.read()

    if not os.path.exists(args.mapping_file):
        print(f"Error: Mapping file not found: {args.mapping_file}")
        return
    
    with open(args.mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    tasks = []
    for prompt_text, image_paths in mapping.items():
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        for path in image_paths:
            tasks.append((prompt_text, path))

    print(f"Starting evaluation, total {len(tasks)} images (Concurrency: {NUM_EVAL_WORKERS})...")

    client = OpenAI(api_key=API_KEY, base_url=API_URL)
    all_results = []
    
    with ThreadPoolExecutor(max_workers=NUM_EVAL_WORKERS) as executor:
        futures = [executor.submit(evaluate_single_image, client, judge_system_prompt, t[0], t[1]) for t in tasks]
        
        for f in tqdm(futures, desc="Evaluating"):
            res = f.result()
            if res:
                all_results.append(res)

    with open(args.output_score_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    if not all_results:
        print("No successful evaluation results.")
        return

    all_scores_matrix = np.array([item['scores'] for item in all_results])
    
    averages = np.mean(all_scores_matrix, axis=0)

    print("\n" + "="*45)
    print(f" Evaluation Statistics (Samples: {len(all_results)})")
    print("="*45)
    
    total_sum_of_averages = 0
    for i, metric_name in enumerate(METRICS):
        avg_score = averages[i]
        total_sum_of_averages += avg_score
        print(f"- {metric_name.capitalize():<18}: {avg_score:.4f}")

    print("-" * 45)
    print(f" Total Score (Sum of Averages): {total_sum_of_averages:.4f}")
    print("="*45)
    print(f"Detailed results saved to: {args.output_score_file}")

if __name__ == "__main__":
    main()