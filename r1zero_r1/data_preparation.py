from datasets import load_dataset

MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
splits=['train','test']

# deepseek system prompt for GRPO training
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def make_conversation(example):
    return{
        "prompt": [
            {"role": "system","content": SYSTEM_PROMPT},
            {"role": "user","content": example["problem"]}
        ],
    }

def load_new_dataset():
    dataset=load_dataset(
        MODEL_NAME,
        name="default",
        split=splits
    )
    dataset={
        'train':dataset[0],
        'test':dataset[1]
    }
    for split in dataset:
            dataset[split] = dataset[split].map(make_conversation)
            if "messages" in dataset[split].column_names:
                dataset[split] = dataset[split].remove_columns("messages")
    return dataset

dataset=load_new_dataset()

def validate_dataset(dataset):
    required_fields=["problem","prompt"] #change this based on your dataset
    for split in splits:
        fields=dataset[split].column_names
        missing = [field for field in required_fields if field not in fields]
        if missing:
            print(f"missing fields:{missing}")
        else:
            print("✓ all present")
        sample = dataset[split][0]
        messages = sample['prompt']
        if (len(messages) >= 2 and
            messages[0]['role'] == 'system' and
            messages[1]['role'] == 'user'):
            print("✓ correct prompt format")  
        else:
            print("incorrect prompt format")  

valid=validate_dataset(dataset)
print(valid)