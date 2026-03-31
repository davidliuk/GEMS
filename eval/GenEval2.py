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
args = parser.parse_args()

NUM_WORKERS = 256
DATA_PATH = "path/to/GenEval2/geneval2_data.jsonl"
OUTPUT_DIR = os.path.join("path/to/GenEval2/results", args.name)
MAPPING_FILE = os.path.join(OUTPUT_DIR, "image_paths.json")
gen_url = "" 
mllm_url = ""
max_iterations = args.max_iterations

def agent_worker(rank, jobs, return_dict):
    agent = GEMS(gen_url=gen_url, mllm_url=mllm_url, max_iterations=max_iterations)
        
    local_mapping = {}
    pbar = tqdm(jobs, desc=f"Worker {rank}", position=rank, leave=False)
    
    for global_idx, item in pbar:
        try:
            prompt = item['prompt']
            img_data = agent.run(item)
            
            img_name = f"img_idx_{global_idx:05d}.png" 
            img_path = os.path.join(OUTPUT_DIR, img_name)
            
            with open(img_path, "wb") as f:
                f.write(img_data)
            
            local_mapping[prompt] = img_path
        except Exception as e:
            print(f"\nWorker {rank} error processing item {global_idx}: {e}")
            
    return_dict[rank] = local_mapping

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_data_with_ids = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            all_data_with_ids.append((idx, json.loads(line)))
    
    total_count = len(all_data_with_ids)
    print(f"Total prompts in dataset: {total_count}")
    
    existing_mapping = {}
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
                existing_mapping = json.load(f)
            print(f"Found existing mapping with {len(existing_mapping)} records.")
        except Exception as e:
            print(f"Error reading mapping file: {e}. Starting fresh.")

    to_process = []
    for global_idx, item in all_data_with_ids:
        if item['prompt'] not in existing_mapping:
            to_process.append((global_idx, item))

    if not to_process:
        print("All items have been processed already.")
        return

    print(f"Items remaining to process: {len(to_process)}")

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
    new_generated_count = 0
    for rank_map in return_dict.values():
        final_mapping.update(rank_map)
        new_generated_count += len(rank_map)
        
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_mapping, f, indent=4, ensure_ascii=False)
    
    print(f"\n[Task Completed] New images added this round: {new_generated_count}.")
    print(f"Current overall progress: {len(final_mapping)} / {total_count}")
    print(f"Mapping file updated to: {MAPPING_FILE}")
    
if __name__ == "__main__":
    main()