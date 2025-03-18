#!/usr/bin/env python
import os
import sys
import logging
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from r1zero_r1.common_args import parse_sft_args

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():

    args = parse_sft_args()
    if args.use_auth_token or args.hf_token:
        import huggingface_hub
        token = args.hf_token
        if token is None:
            try:
                token = huggingface_hub.get_token()
            except Exception as e:
                logger.warning(f"Failed to get Hugging Face token: {e}")
                print(f"Failed to get Hugging Face token: {e}")
        if token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            logger.info("Using Hugging Face authentication for accessing models")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sft_{os.path.basename(args.model_name_or_path)}_{timestamp}"
    
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(file_handler)
    
    logger.info(f" starting SFT training with the following configuration: ")
    logger.info(f" model: {args.model_name_or_path}")
    logger.info(f" dataset: {args.dataset_name}")
    logger.info(f" dataset config: {args.dataset_config}")
    logger.info(f" output directory: {output_dir}")
    logger.info(f" training epochs: {args.num_train_epochs}")
    logger.info(f" batch size: {args.per_device_train_batch_size}")
    logger.info(f" learning rate: {args.learning_rate}")
    logger.info(f" using peft: {args.use_peft}")
    

    # Set up device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    logger.info(f"loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_auth_token=True if args.use_auth_token or args.hf_token else None,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"loading dataset: {args.dataset_name}")
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name)
    
    available_splits = list(dataset.keys())
    logger.info(f"available dataset splits: {available_splits}")
    train_split = args.dataset_train_split if args.dataset_train_split in available_splits else available_splits[0]
    eval_split = args.dataset_test_split if args.dataset_test_split in available_splits else None
    
    logger.info(f"using '{train_split}' split for training")
    if eval_split:
        logger.info(f"using '{eval_split}' split for evaluation")
    else:
        logger.info("no evaluation split available. evaluation will be skipped.")
    
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
        report_to="none",  # disable wandb
        no_cuda=(device == "cpu"), 
    )
    
    peft_config = None
    if args.use_peft:
        logger.info(f"Setting up PEFT with LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
    
    trainer = SFTTrainer(
        model=args.model_name_or_path,
        args=training_args,
        train_dataset=dataset[train_split],
        eval_dataset=dataset[eval_split] if eval_split and training_args.evaluation_strategy != "no" else None,
        tokenizer=tokenizer,
        peft_config=peft_config,
        device_map=args.device_map if args.device_map != "auto" else None,
    )
    
    logger.info("Starting training")
    trainer.train()
    logger.info(f"training completed, saving model to {output_dir}")
    trainer.save_model(output_dir)
    logger.info(f"training completed. model saved to {output_dir}")
    print(f"training completed. model saved to {output_dir}")
    print(f"log file saved to {log_file}")
    return trainer

if __name__ == "__main__":
    main()
