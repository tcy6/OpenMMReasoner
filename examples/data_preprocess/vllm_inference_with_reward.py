import argparse
import logging
import os
import re
import ast
import torch
import torch.distributed as dist
import datetime
from datasets import Dataset
from PIL import Image
from tqdm import tqdm
# vLLM imports
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from datasets import Dataset, concatenate_datasets
import re
import sys
 
from PIL import ImageFile, Image
Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Add parent directory to sys.path to allow imports from sibling packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from custom_rewards.lmms_lab_recipe import compute_score

# Assuming qwen_vl_utils is available
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    def process_vision_info(messages, return_video_kwargs=False):
        # Placeholder for when qwen_vl_utils is not available
        raise ImportError("qwen_vl_utils is required for vision processing")

# Logging setup
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_image(image: Image.Image) -> Image.Image:
    if image.width < 28 or image.height < 28:
        # Resize with 28 on the short side, maintaining aspect ratio
        short_side = min(image.width, image.height)
        if short_side < 28:
            scale_factor = 28 / short_side
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            image = image.resize((new_width, new_height))
    return image.convert("RGB")

def _build_messages(example: dict):
    # Check if this is multimodal format (has 'problem' instead of 'prompt')
    messages: list = example['prompt']
            

    if "images" in example or "videos" in example:
        for message in messages:
            content = message["content"]
            content_list = []
            segments = re.split("(<image>|<video>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == "<image>":
                    content_list.append({"type": "image"})
                elif segment == "<video>":
                    content_list.append({"type": "video"})
                else:
                    content_list.append({"type": "text", "text": segment})

            message["content"] = content_list

    return messages

def prepare_llm_input(processor, row):

    messages = _build_messages(row)  # Use the modified function to build messages
    images = row.get('images', None)
    videos = row.get('videos', None)
    
    
    try:
        # Apply chat template
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Process vision info
        
        mm_data = {}
        mm_data["image"] = [process_image(image) for image in images]
        
        return {
            "prompt": prompt_text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": {},
        }
    except Exception as e:
        logger.error(f"Error preparing LLM input: {e}")
        return None


def main(args):
    """Main execution function."""
    logger.info("Starting vLLM inference with reward scoring")

    # Initialize torch.distributed from torchrun environment
    local_dp_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_dp_rank = int(os.environ.get("RANK", "0"))
    dp_size = int(os.environ.get("WORLD_SIZE", "1"))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=600000))

    print(f"GLOBAL DP RANK: {global_dp_rank}, LOCAL DP RANK: {local_dp_rank}")
    
    # Load data (support multiple parquet paths)
    parquet_paths = args.parquet_path if isinstance(args.parquet_path, (list, tuple)) else [args.parquet_path]
    dataset_list = [Dataset.from_parquet(p) for p in parquet_paths]
    df = concatenate_datasets(dataset_list) if len(dataset_list) > 1 else dataset_list[0]
    original_df_length = len(df)
    df = df.select([i for i in range(args.max_samples)]) if args.max_samples else df
    # We split the dataset into shards if shard_size > 0, so that more flexible parallelism can be achieved.
    df = df.shard(num_shards=args.shard_size, index=args.shard_index, contiguous=True) if args.shard_size > 0 else df
    print(f"Original Dataset [{original_df_length}] \n Sharding dataset into {args.shard_size} shards, processing shard index {args.shard_index} with {len(df)} samples")
    df = df.shard(num_shards=dp_size, index=global_dp_rank, contiguous=True)
    print(f"Sharding dataset into {dp_size} shards, processing shard index {global_dp_rank} with {len(df)} samples")
    
    # Initialize vLLM
    logger.info(f"Initializing vLLM with model: {args.model_path}")
    try:
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=args.trust_remote_code,
            gpu_memory_utilization=0.8,
            max_model_len=32768,
            enforce_eager=False,  # Disable eager mode for better performance
            distributed_executor_backend="external_launcher",
            enable_chunked_prefill=True,
            max_num_batched_tokens=32768,
        )
        
        processor = AutoProcessor.from_pretrained(
            args.model_path, 
            trust_remote_code=args.trust_remote_code
        )
        
        sampling_params = SamplingParams(
            n=args.num_rollouts,  # Generate n rollouts per prompt
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        
        logger.info("vLLM initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize vLLM: {e}")
        return
    
    all_results = []
    
    # Process samples in batches
    batch_size = args.batch_size
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.select(range(start_idx, end_idx))
        
        # Prepare batch inputs
        batch_inputs = []
        batch_metadata = []
        
        for idx, row in enumerate(batch_df):
            # Extract ground truth
            ground_truth = row['reward_model']['ground_truth']
            
            # Prepare input
            llm_input = prepare_llm_input(processor, row)
            if llm_input is None:
                logger.warning(f"Failed to prepare input for sample {idx}")
                continue
            
            # Extract question from prompt messages
            question = ""
            for msg in row['prompt']:
                if msg['role'] == 'user':
                    content = msg['content']
                    # Handle both string and list content formats
                    if isinstance(content, str):
                        question = content
                    elif isinstance(content, list):
                        # Extract text content from list format
                        text_parts = []
                        for item in content:
                            if isinstance(item, str):
                                text_parts.append(item)
                            elif isinstance(item, dict) and item.get('type') == 'text':
                                text_parts.append(item.get('text', ''))
                            elif isinstance(item, dict) and 'content' in item and isinstance(item['content'], str):
                                text_parts.append(item['content'])
                        question = ''.join(text_parts)
                    break
            
            batch_inputs.append(llm_input)
            batch_metadata.append({
                'index': idx,
                'data_source': row['data_source'],
                'question': question,
                'ground_truth': ground_truth
            })
        
        if not batch_inputs:
            continue
        
        try:
            # Generate responses for entire batch with multiple rollouts each
            outputs = llm.generate(batch_inputs, sampling_params)
            
            # Process outputs
            for output, metadata in zip(outputs, batch_metadata):
                sample_results = {
                    'index': metadata['index'],
                    'data_source': metadata['data_source'],
                    'question': metadata['question'],
                    'ground_truth': metadata['ground_truth'],
                    'rollouts': []
                }
                
                # Process all rollouts for this sample
                for rollout_idx, rollout_output in enumerate(output.outputs):
                    response = rollout_output.text
                    
                    reward = compute_score(
                        solution_str=response,
                        ground_truth=metadata['ground_truth'],
                        data_source=metadata['data_source'],
                    )
                    acc_score = float(reward['acc_score'])
                    predict_str = reward['predict_str']
                    predict_str = str(predict_str).strip()
                    
                    rollout_result = {
                        'rollout_id': rollout_idx,
                        'response': response,
                        'extracted_answer': predict_str,
                        'reward': acc_score
                    }
                    
                    sample_results['rollouts'].append(rollout_result)
                    
                    logger.debug(f"Sample {metadata['index']}, Rollout {rollout_idx}: Reward={reward}")
                
                # Calculate statistics for this sample
                rewards = [r['reward'] for r in sample_results['rollouts']]
                sample_results['avg_reward'] = sum(rewards) / len(rewards) if rewards else 0.0
                sample_results['max_reward'] = max(rewards) if rewards else 0.0
                sample_results['num_correct'] = sum(1 for r in rewards if r > 0)
                
                all_results.append(sample_results)
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            # Add error results for all samples in batch
            for metadata in batch_metadata:
                sample_results = {
                    'index': metadata['index'],
                    'data_source': metadata['data_source'],
                    'question': metadata['question'],
                    'ground_truth': metadata['ground_truth'],
                    'rollouts': []
                }
                for rollout_idx in range(args.num_rollouts):
                    sample_results['rollouts'].append({
                        'rollout_id': rollout_idx,
                        'response': f"Batch error: {str(e)}",
                        'extracted_answer': None,
                        'reward': 0.0
                    })
                sample_results['avg_reward'] = 0.0
                sample_results['max_reward'] = 0.0
                sample_results['num_correct'] = 0
                all_results.append(sample_results)
        dist.barrier()  # Ensure all processes reach this point before proceeding
        
        # Periodic saving
        # if args.save_every > 0 and len(all_results) >= args.save_every:
            # save_results(all_results, f"{args.output_path}_global_rank_{global_dp_rank}_shard_index_{args.shard_index}.parquet", is_checkpoint=True)
    
    # Final save
    save_results(all_results, f"{args.output_path}_global_rank_{global_dp_rank}_shard_index_{args.shard_index}.parquet", is_checkpoint=False)

    # Gather results on rank 0 and print summary
    dist.barrier()
    if global_dp_rank == 0:
        results = gather_all_results(args, dp_size, args.shard_index)
        print_summary(results)

    dist.destroy_process_group()


def save_results(results, output_path, is_checkpoint=False):
    """Save results to parquet file."""
    checkpoint_suffix = "_checkpoint" if is_checkpoint else ""
    final_path = output_path.replace(".parquet", f"{checkpoint_suffix}.parquet")
    
    try:
        df = Dataset.from_list(results)
        df.to_parquet(final_path)
        logger.info(f"{'Checkpoint' if is_checkpoint else 'Final results'} saved to {final_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def print_summary(results):
    """Print summary statistics."""
    total_samples = len(results)
    total_rollouts = sum(len(r['rollouts']) for r in results)
    total_correct = sum(r['num_correct'] for r in results)
    avg_reward = sum(r['avg_reward'] for r in results) / total_samples if total_samples > 0 else 0
    
    logger.info("\n" + "="*50)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*50)
    logger.info(f"Total samples processed: {total_samples}")
    logger.info(f"Total rollouts generated: {total_rollouts}")
    logger.info(f"Total correct answers: {total_correct}")
    logger.info(f"Average reward across all samples: {avg_reward:.3f}")
    logger.info(f"Success rate (at least one correct): {sum(1 for r in results if r['num_correct'] > 0) / total_samples:.2%}")
    logger.info("="*50)

def gather_all_results(args, dp_size, shard_index):
    """Gather results from all data parallel ranks (rank 0)."""
    dataset_list = []
    file_list = []
    checkpoint_file_list = []
    for global_dp_rank in range(dp_size):
        file_name = f"{args.output_path}_global_rank_{global_dp_rank}_shard_index_{shard_index}.parquet"
        checkpoint_file_name = f"{args.output_path}_global_rank_{global_dp_rank}_shard_index_{shard_index}_checkpoint.parquet"
        file_list.append(file_name)
        checkpoint_file_list.append(checkpoint_file_name)
        df = Dataset.from_parquet(file_name)
        dataset_list.append(df)
    
    dataset_list = concatenate_datasets(dataset_list)
    dataset_list.to_parquet(f"{args.output_path}_shard_index_{shard_index}.parquet")

    # Remove individual rank files
    for file_name, checkpoint_file_name in zip(file_list, checkpoint_file_list):
        try:
            os.remove(file_name)
            logger.info(f"Removed temporary file: {file_name}")
        except Exception as e:
            logger.error(f"Failed to remove file {file_name}: {e}")

        try:
            os.remove(checkpoint_file_name)
            logger.info(f"Removed temporary checkpoint file: {checkpoint_file_name}")
        except Exception as e:
            logger.error(f"Failed to remove checkpoint file {checkpoint_file_name}: {e}")
    return dataset_list.to_list()


# python vllm_inference_with_reward.py --num_rollouts 8 --max_samples 10 --output_path val_inference_results.parquet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run vLLM inference on parquet data with reward scoring"
    )
    
    # Data arguments
    parser.add_argument(
        "--parquet_path",
        type=str,
        default="[]",
        help="Path(s) to input parquet file(s)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="vllm_inference_results.parquet",
        help="Path to save results (as parquet file)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for debugging)",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Path to model",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        "-tp",
        type=int,
        default=4,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--data_parallel_size",
        "-dp",
        type=int,
        default=2,
        help="Data parallel size for vLLM (default: 2, set to 1 for single GPU setup)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading model",
    )
    
    # Generation arguments
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=8,
        help="Number of rollouts per question",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate",
    )
    
    # Other arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of samples to process in a single batch",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Save checkpoint every N samples (0 to disable)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--shard-size", type=int, default=-1, help="Number of shards to split the dataset into, default to -1 for no shard"
    )
    parser.add_argument(
        "--shard-index", type=int, default=0, help="Index of the shard to process, default to 0 for the first shard"
    )
    
    args = parser.parse_args()
    args.parquet_path = ast.literal_eval(args.parquet_path)
    args.output_path = args.output_path.replace(".parquet", "")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Set logging level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    logging.getLogger().setLevel(numeric_level)
    main(args)
