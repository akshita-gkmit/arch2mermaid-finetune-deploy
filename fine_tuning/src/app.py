from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

app = FastAPI()

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = "./adapter"

tokenizer = None
model = None


class Prompt(BaseModel):
    text: str


@app.on_event("startup")
def load_model():
    global tokenizer, model

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

    if torch.cuda.is_available():
        print("GPU detected — loading 4-bit model")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            quantization_config=bnb_config
        )

    else:
        print("No GPU — skipping model load")
        model = None


@app.post("/generate")
def generate(prompt: Prompt):

    if model is None:
        return {"error": "Model not loaded. GPU required."}

    inputs = tokenizer(prompt.text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": result}