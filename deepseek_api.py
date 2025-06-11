# deepseek_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200

app = FastAPI()

@app.on_event("startup")
async def load_model():
    global tokenizer, model, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                                                 trust_remote_code=True,
                                                 device_map="auto").to(device)

@app.post("/generate")
async def generate(req: GenerateRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=req.max_new_tokens)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"generated_text": text}

