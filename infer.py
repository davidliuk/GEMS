import os
from agent.GEMS import GEMS

TEST_PROMPT = "A book floating in the sky, creative and cool concept, make it look artistic and dreamy."

SAVE_DIR = "infer_results"

gen_url = ""
mllm_url = ""
max_iterations = 5
agent = GEMS(gen_url=gen_url, mllm_url=mllm_url, max_iterations=max_iterations)

def test_single_agent():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"Created directory: {SAVE_DIR}")

    item = {
        "prompt": TEST_PROMPT
    }

    print(f"Prompt: {TEST_PROMPT}")

    try:
        image_bytes = agent.run(item)

        save_path = os.path.join(SAVE_DIR, "test_output.png")
        with open(save_path, "wb") as f:
            f.write(image_bytes)

        print(f"Saved to: {save_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_single_agent()