#!/usr/bin/env python
import os
import sys
import logging
import torch
from datetime import datetime
import huggingface_hub
from datasets import load_dataset
from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
from trl import GRPOTrainer, ModelConfig
from r1zero_r1.common_args import parse_gppo_args

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #add the parent dir to the path
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    args = parse_gppo_args()
    if args.use_auth_token or args.hf_token:
        token = args.hf_token
        if token is None:
            try:
                token = huggingface_hub.get_token()
            except Exception as e:
                logger.warning(f"Failed to get Hugging Face token: {e}")
                print(f"Failed to get Hugging Face token: {e}")
        
        if token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            os.environ["HF_TOKEN"] = token
            logger.info("Using Hugging Face authentication for accessing models")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"gppo_{os.path.basename(args.model_name_or_path)}_{timestamp}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    logger.info(f"loading model: {args.model_name_or_path}")
    print(f"loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_auth_token=True if args.use_auth_token or args.hf_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_auth_token=True if args.use_auth_token or args.hf_token else None,
    )
        
    logger.info(f"loading dataset: {args.dataset_name}")
    print(f"loading dataset: {args.dataset_name}")
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        report_to="none",
        no_cuda=(device == "cpu"),  
    )

    class ScriptArgs:
        def __init__(self, args):
            self.reward_funcs = args.reward_funcs
            self.cosine_min_value_wrong = args.cosine_min_value_wrong
            self.cosine_max_value_wrong = args.cosine_max_value_wrong
            self.cosine_min_value_correct = args.cosine_min_value_correct
            self.cosine_max_value_correct = args.cosine_max_value_correct
            self.cosine_max_len = args.cosine_max_len
            self.repetition_n_grams = args.repetition_n_grams
            self.repetition_max_penalty = args.repetition_max_penalty
    
    script_args = ScriptArgs(args)
    reward_functions = get_reward_functions(script_args)
    logger.info(f"using reward functions: {args.reward_funcs}")
    
    model_config = ModelConfig(
        model=model,    
        tokenizer=tokenizer,
        kl_coef=args.kl_coef,
        clip_range=args.clip_range,
        device_map=args.device_map if args.device_map != "auto" else None,
    )
    trainer = GRPOTrainer(
        model_config=model_config,
        args=training_args,
        train_dataset=dataset[args.dataset_train_split],
        eval_dataset=dataset[args.dataset_test_split],
        reward_functions=reward_functions,
    )
    
    log_file = os.path.join(output_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting GPPO training with the following configuration:")
    logger.info(f"  Model: {args.model_name_or_path}")
    logger.info(f"  Dataset: {args.dataset_name}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Training epochs: {args.num_train_epochs}")
    logger.info(f"  Batch size: {args.per_device_train_batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Using PEFT: {args.use_peft}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Reward functions: {args.reward_funcs}")
    
    print(f"Starting GPPO training with model: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Reward functions: {args.reward_funcs}")
    print(f"Output directory: {output_dir}")
    
  
    trainer.train()
    trainer.save_model(output_dir)
    
    logger.info(f"Training completed. Model saved to {output_dir}")
    print(f"Training completed. Model saved to {output_dir}")
    print(f"Log file saved to {log_file}")
    return trainer

if __name__ == "__main__":
    main()
