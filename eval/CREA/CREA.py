import os
import json
import torch.multiprocessing as mp
from tqdm import tqdm
import argparse
from agent.GEMS import GEMS


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--agent", type=str, required=True)
parser.add_argument("--max_iterations", type=int, default=5)
parser.add_argument("--max_nodes", type=int, default=5)
parser.add_argument("--n_samples", type=int, default=25)  # following the paper of CREA
args = parser.parse_args()

NUM_WORKERS = 256
DATA_PATH = "eval/CREA/data.jsonl"
OUTPUT_DIR = os.path.join("eval/CREA/results", args.name)
MAPPING_FILE = os.path.join(OUTPUT_DIR, "image_paths.json")
gen_url = "" 
mllm_url = ""
max_iterations = args.max_iterations
N_SAMPLES = args.n_samples

def agent_worker(rank, jobs, return_dict):
    agent = GEMS(gen_url=gen_url, mllm_url=mllm_url, max_iterations=max_iterations)
    
    local_results = []
    pbar = tqdm(jobs, desc=f"Worker {rank}", position=rank, leave=False)
    
    for global_idx, sub_idx, item in pbar:
        try:
            prompt = item['prompt']
            img_data = agent.run(item)
            
            img_name = f"img_idx_{global_idx:05d}_sub_{sub_idx:02d}.png" 
            img_path = os.path.join(OUTPUT_DIR, img_name)
            
            with open(img_path, "wb") as f:
                f.write(img_data)
            
            local_results.append((prompt, img_path))
        except Exception as e:
            print(f"\nWorker {rank} error processing item {global_idx} (sub {sub_idx}): {e}")
            
    return_dict[rank] = local_results

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_data_with_ids = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            all_data_with_ids.append((idx, json.loads(line)))
    
    total_prompts = len(all_data_with_ids)
    print(f"Total unique prompts: {total_prompts}")
    
    existing_mapping = {}
    processed_count = 0
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
                existing_mapping = json.load(f)
            processed_count = sum(len(v) if isinstance(v, list) else 1 for v in existing_mapping.values())
            print(f"Found existing mapping. Total images already generated: {processed_count}")
        except Exception as e:
            print(f"Error reading mapping file: {e}. Starting fresh.")

    to_process = []
    for global_idx, item in all_data_with_ids:
        prompt = item['prompt']
        existing_paths = existing_mapping.get(prompt, [])
        if not isinstance(existing_paths, list):
            existing_paths = [existing_paths]
            
        start_sub_idx = len(existing_paths)
        for sub_idx in range(start_sub_idx, N_SAMPLES):
            to_process.append((global_idx, sub_idx, item))

    if not to_process:
        print("All samples for all prompts have been processed already.")
        return

    print(f"Total tasks to run (prompts * n): {len(to_process)}")

    chunks = [to_process[i::NUM_WORKERS] for i in range(NUM_WORKERS)]

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    
    for rank in range(NUM_WORKERS):
        if rank < len(chunks) and len(chunks[rank]) > 0:
            p = mp.Process(target=agent_worker, args=(rank, chunks[rank], return_dict))
            p.start()
            processes.append(p)
    
    for p in processes:
        p.join()
    
    final_mapping = existing_mapping.copy()
    for k, v in final_mapping.items():
        if not isinstance(v, list):
            final_mapping[k] = [v]

    new_generated_count = 0
    for rank_results in return_dict.values():
        for prompt, img_path in rank_results:
            if prompt not in final_mapping:
                final_mapping[prompt] = []
            final_mapping[prompt].append(img_path)
            new_generated_count += 1
        
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_mapping, f, indent=4, ensure_ascii=False)
        
    total_final = sum(len(v) for v in final_mapping.values())
    print(f"\n[Task Completed] New images added this round: {new_generated_count}.")
    print(f"Current overall progress: {len(final_mapping)} / {total_count}")
    print(f"Mapping file updated to: {MAPPING_FILE}")

if __name__ == "__main__":
    main()