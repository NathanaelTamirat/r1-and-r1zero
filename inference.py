import torch
from r1zero_r1.model import tokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from r1zero_r1.data_preparation import SYSTEM_PROMPT

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    trust_remote_code=True, 
    padding_side="right"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
trained_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 
)
device=torch.device("cuda" if torch.cuda._is_avaliable() else "cpu")
trained_model.to(device) 

def model_inference(user_input: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = trained_model.generate(
        **inputs,
        max_new_tokens=200, 
        do_sample=True,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

input = """
John buys 5 notebooks that have 40 pages each. 
He uses 4 pages per day. How many days do the notebooks last?
"""
response = model_inference(input)
print(f"Response: {response}")