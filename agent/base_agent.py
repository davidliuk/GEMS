import abc
import requests
import base64
from openai import OpenAI
from agent.skill_manager import SkillManager

class BaseAgent(metaclass=abc.ABCMeta):
    def __init__(self, gen_url, mllm_url):
        self.gen_url = gen_url
        self.mllm_url = mllm_url
        self.client = OpenAI(api_key="none", base_url=mllm_url)
        self.skill_manager = SkillManager()

    def generate(self, prompt: str) -> bytes:
        params = {"prompt": prompt}
        response = requests.post(self.gen_url, params=params, timeout=600)
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

    def edit(self, prompt: str, image: bytes):
        pass

    def think(self, prompt: str, images: list = None) -> str:
        if images is None:
            images = []
        
        segments = prompt.split("<image>")
        content = []
        
        for i in range(len(segments)):
            if segments[i]:
                content.append({"type": "text", "text": segments[i]})
            
            if i < len(images):
                base64_image = base64.b64encode(images[i]).decode('utf-8')
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })

        try:
            response = self.client.chat.completions.create(
                model="kimi-k2.5",
                messages=[{"role": "user", "content": content}],
                max_tokens=16384,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"MLLM : {e}")
        
    def think_with_thought(self, prompt: str, images: list = None) -> str:
        if images is None:
            images = []
        
        segments = prompt.split("<image>")
        content = []
        
        for i in range(len(segments)):
            if segments[i]:
                content.append({"type": "text", "text": segments[i]})
            
            if i < len(images):
                base64_image = base64.b64encode(images[i]).decode('utf-8')
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })

        try:
            response = self.client.chat.completions.create(
                model="kimi-k2.5",
                messages=[{"role": "user", "content": content}],
                max_tokens=16384,
            )
            
            return response.choices[0].message.content,response.choices[0].message.reasoning_content
        except Exception as e:
            raise Exception(f"MLLM : {e}")

    def run(self, item: dict) -> bytes:
        pass