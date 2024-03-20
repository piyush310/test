from transformers import AutoModelForCausalLM, AutoTokenizer 

from starlette.requests import Request

import ray
from ray import serve

import torch

# Set the device
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Use GPU 0
else:
    device = torch.device("cpu")

ray.init()
serve.start(http_options={"http": "127.0.0.1", "port": 5555})

# Load model from HF with user's token and with bitsandbytes config
model_name = "meta-llama/Llama-2-7b-hf"
access_token = "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"


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 1, "num_gpus": 1})
class Translator:
    def __init__(self):
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map=device,
            token=access_token
            )

    def translate(self, text: str) -> str:
        # Run inference
        self.model.config.use_cache = True
        self.inputs = self.tokenizer(text, return_tensors="pt").to(device)
        self.model.to(device)
        # Get answer
        outputs = self.model.generate(
            input_ids=self.inputs["input_ids"], 
            attention_mask=self.inputs["attention_mask"], 
            max_new_tokens=25, 
            pad_token_id=self.tokenizer.eos_token_id)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    async def __call__(self, http_request: Request=None) -> str:
        if http_request is not None:
            english_text: str = await http_request.json()
            return self.translate(english_text)
        else:
            print("no http request")
            pass
# translator_app = Translator.bind()        
# Translator.deploy()
# if __name__=="__main__":
serve.run(Translator.bind())
#     ray.get(handle.remote())
[21/03, 1:40 am] sen: import requests
from multiprocessing import Process
import time

def translate_text(text):
    start_time = time.time()  # Record start time
    response = requests.post("http://127.0.0.1:5555/", json=text)
    french_text = response.text
    end_time = time.time()  # Record end time
    latency = end_time - start_time  # Calculate latency
    print(french_text)
    print(f"Latency for this process: {latency} seconds")

# if __name__ == "__main__":
#     english_text = "Hello world!"
#     # Create multiple processes for translating
#     processes = []
#     for _ in range(2):  # You can adjust the number of processes as needed
#         p = Process(target=translate_text, args=(english_text,))
#         p.start()
#         processes.append(p)
    
#     # Wait for all processes to finish
#     for p in processes:
#         p.join()


def process_task(english_text):
    try:
        while True:
            translate_text(english_text)
    except KeyboardInterrupt:
        print("Interrupted, stopping the process.")

if __name__ == "__main__":
    english_text = "Hello world!"

    # Create multiple processes for translating
    processes = []
    for _ in range(2):  # Adjust the number of processes as needed
        p = Process(target=process_task, args=(english_text,))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Main process interrupted, stopping all child processes.")
        for p in processes:
            p.terminate()
            p.join()