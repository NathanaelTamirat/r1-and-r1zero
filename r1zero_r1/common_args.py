import argparse

def get_base_parser():
    parser = argparse.ArgumentParser(description="Training arguments", add_help=False)
    
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--use_auth_token", action="store_true",
                        help="Use Hugging Face auth token to access private models")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face auth token to access private or gated models")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="The name of the dataset to use (via the datasets library)")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="The configuration name of the dataset to use")
    parser.add_argument("--dataset_train_split", type=str, default="train",
                        help="The name of the training data split to use")
    parser.add_argument("--dataset_test_split", type=str, default="test",
                        help="The name of the test data split to use")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="The output directory where model and logs will be saved")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size per GPU/TPU core/CPU for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Batch size per GPU/TPU core/CPU for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="The initial learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for AdamW if we apply some")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length the model can handle")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true",
                        help="Use fp16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Use gradient checkpointing to save memory")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every X updates steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Run evaluation every X steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X updates steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Limit the total amount of checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use for training (auto, cuda, cpu)")
    parser.add_argument("--device_map", type=str, default="auto",
                        help="Device map for model parallelism")
    # peft args
    parser.add_argument("--use_peft", action="store_true",
                        help="Use PEFT for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="Lora attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="Lora alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Lora dropout parameter")
    return parser

def get_sft_parser():
    parser = argparse.ArgumentParser(description="Run SFT training with optimized settings")
    parser.add_argument_group('base', argument_default=argparse.SUPPRESS)
    base_parser = get_base_parser()
    for action in base_parser._actions:
        parser._add_action(action)
    parser.add_argument("--packing", action="store_true",
                        help="Use packing for efficient training")
    parser.set_defaults(output_dir="./sft_output")
    
    return parser

def get_gppo_parser():
    parser = argparse.ArgumentParser(description="Run GPPO training with optimized settings")
    parser.add_argument_group('base', argument_default=argparse.SUPPRESS)
    base_parser = get_base_parser()
    for action in base_parser._actions:
        parser._add_action(action)

    parser.add_argument("--reward_funcs", nargs="+", default=["accuracy", "format"],
                        help="List of reward functions to use")
    parser.add_argument("--kl_coef", type=float, default=0.1,
                        help="KL divergence coefficient for GPPO")
    parser.add_argument("--clip_range", type=float, default=0.2,
                        help="Clip range for GPPO")
    parser.add_argument("--cosine_min_value_wrong", type=float, default=0.0,
                        help="Minimum value for wrong answers in cosine reward")
    parser.add_argument("--cosine_max_value_wrong", type=float, default=0.5,
                        help="Maximum value for wrong answers in cosine reward")
    parser.add_argument("--cosine_min_value_correct", type=float, default=0.5,
                        help="Minimum value for correct answers in cosine reward")
    parser.add_argument("--cosine_max_value_correct", type=float, default=1.0,
                        help="Maximum value for correct answers in cosine reward")
    parser.add_argument("--cosine_max_len", type=int, default=254,
                        help="Maximum length for cosine reward")
    parser.add_argument("--repetition_n_grams", type=int, default=4,
                        help="N-gram size for repetition penalty")
    parser.add_argument("--repetition_max_penalty", type=float, default=0.5,
                        help="Maximum penalty for repetition")
    parser.set_defaults(
        output_dir="./gppo_output",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-5
    )
    
    return parser

def parse_sft_args():
    parser = get_sft_parser()
    return parser.parse_args()

def parse_gppo_args():
    parser = get_gppo_parser()
    return parser.parse_args()
