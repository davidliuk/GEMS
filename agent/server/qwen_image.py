import os
import uuid
import time
import torch
import torch.multiprocessing as mp
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from diffusers import DiffusionPipeline
import io
import uvicorn
import asyncio

NUM_GPUS = 8
RESOLUTION = 1328
STEPS = 50
CFG_SCALE = 4.0
MODEL_PATH = "path/to/Qwen-Image-2512"
PORT = 8000

app = FastAPI(title="Multi-GPU Image Gen API")

manager = None
input_queue = None
result_dict = None

def worker(rank, in_queue, res_dict):
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16
    ).to(device)
    
    while True:
        try:
            task_id, prompt = in_queue.get()
        
            generator = torch.Generator(device=device).manual_seed(torch.seed() % 10000)
            
            image = pipe(
                prompt=prompt,
                num_inference_steps=STEPS,
                guidance_scale=CFG_SCALE,
                width=RESOLUTION,
                height=RESOLUTION,
                #width=1664,
                #height=928,
                generator=generator
            ).images[0]
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            res_dict[task_id] = img_byte_arr.getvalue()
            
        except Exception as e:
            if 'task_id' in locals():
                res_dict[task_id] = "ERROR"

@app.post("/generate")
async def generate_image(prompt: str):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    task_id = str(uuid.uuid4())
    
    input_queue.put((task_id, prompt))
    
    start_time = time.time()
    timeout = 120
    
    while task_id not in result_dict:
        if time.time() - start_time > timeout:
            raise HTTPException(status_code=504, detail="Processing timeout")
        await asyncio.sleep(0.1)
    
    result = result_dict.pop(task_id)
    
    if result == "ERROR":
        raise HTTPException(status_code=500, detail="Image generation failed")
    
    return Response(content=result, media_type="image/png")

@app.on_event("startup")
def startup_event():
    global manager, input_queue, result_dict
    
    mp.set_start_method('spawn', force=True)
    
    manager = mp.Manager()
    input_queue = manager.Queue()
    result_dict = manager.dict()
    
    for rank in range(NUM_GPUS):
        p = mp.Process(target=worker, args=(rank, input_queue, result_dict))
        p.daemon = True
        p.start()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)