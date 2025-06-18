# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import re
from copy import deepcopy
from glob import glob
from typing import List, Tuple

import datasets
from datasets import (DatasetInfo, SplitInfo, get_dataset_config_info,
                      get_dataset_config_names)
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from verl.utils.hdfs_io import copy, makedirs

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)

EXCLUDE = ["FigureQA_train", "lmms_eval_mmlu", "lmms_eval_mmlu_pro", "lmms_eval_mathverse_testmini"]


def get_flattened_dataset(data_source, limit=None) -> Tuple[List[datasets.Dataset], List[Tuple[DatasetInfo, SplitInfo]], List[str]]:
    """
    Load and flatten the datasets from the specified data source.
    """
    dataset_list = []
    datainfolist = []
    subset_list = get_dataset_config_names(data_source)
    subset_list = [subset for subset in subset_list if subset not in EXCLUDE]
    if limit is not None:
        subset_list = subset_list[:limit]
    for subset in subset_list:
        data_info = get_dataset_config_info(data_source, subset)
        splits_info = data_info.splits
        for split_name, split_info in splits_info.items():
            dataset = datasets.load_dataset(data_source, subset, split=split_name)
            dataset_list.append(dataset)
            datainfolist.append((data_info, split_info))

    return dataset_list, datainfolist, subset_list


def load_arrow_dataset(data_source, limit=None) -> Tuple[List[datasets.Dataset], List[str]]:
    data_files = glob(os.path.join(data_source, "*"))
    if limit is not None:
        data_files = data_files[:limit]
    dataset_list = []
    subset_list = []
    for file in data_files:
        subset_name = os.path.basename(file)
        if subset_name in EXCLUDE:
            print(f"Skipping excluded dataset: {subset_name}")
            continue
        try:
            dataset = datasets.load_from_disk(file)
        except Exception as e:
            print(f"Failed to load dataset from {file}: {e}")
            continue
        dataset_list.append(dataset)
        subset_list.append(subset_name)
    return dataset_list, subset_list


def get_blank_image():
    """
    Create a blank image to be used as a placeholder.
    """
    blank_image = Image.new("RGB", (28, 28), color=(255, 255, 255))
    return blank_image


def build_messages(messages):
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


def resize_image(image: Image.Image) -> Image:
    # Resize image to at least 28x28 if smaller

    height, width = image.size
    if height < 28:
        image = image.resize((28, width), Image.BICUBIC)
    height, width = image.size
    # If width is still less than 28, resize to heightx28
    # This ensures that the image is at least 28x28
    if width < 28:
        image = image.resize((height, 28), Image.BICUBIC)
    return image


def make_map_fn(split, subset_name, post_prompt, processor):
    def process_fn(example, idx):
        if post_prompt:
            problem = example.pop("problem") + " " + post_prompt
        else:
            problem = example.pop("problem")
        prompt = problem
        answer = example.pop("answer")
        images = example.pop("images")
        if len(images) > 0 and "<image>" not in prompt:
            prompt = "<image>" * len(images) + prompt
        elif len(images) == 0:
            images = [get_blank_image()]
            prompt = "<image>" + prompt

        data = {
            "data_source": subset_name,
            "prompt": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "images": images,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer,
                "question": problem,
            },
        }
        doc = deepcopy(data["prompt"])
        messages = build_messages(doc)
        raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        images = [resize_image(image) for image in images]
        tokens = len(processor(text=[raw_prompt], images=images)["input_ids"][0])
        data["tokens"] = tokens

        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/open_mm_recipe_image")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--data_source", default="luodian/think-in-modality-v0", help="The data source to load the dataset from")
    parser.add_argument("--val-data_source", default="luodian/think-in-modality-val", help="The data source to load the dataset from")
    parser.add_argument("--prompt-str", default=None, type=str, help="The prompt string json file to use for the dataset")
    parser.add_argument("--load-format", default="hub", type=str, choices=["hub", "arrow"])
    parser.add_argument("--processor-name", default="Qwen/Qwen2.5-VL-7B-Instruct", type=str, help="The processor name to use for the dataset")
    parser.add_argument(
        "--val-load-format",
        default="hub",
        type=str,
        choices=["hub", "arrow"],
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of datasets to process")
    parser.add_argument("--extra_exclude_kwargs", type=str, default="", help="Extra exclude keywords for the dataset, write in xxx,xxx,xxx")
    parser.add_argument("--dataset-type", type=str, choices=["train", "val", "both"], default="both", help="Choose which dataset to process: train, val, or both")

    args = parser.parse_args()
    extra_exclude_kwargs = args.extra_exclude_kwargs.split(",") if args.extra_exclude_kwargs else []
    EXCLUDE.extend(extra_exclude_kwargs)
    prompt_str = json.load(open(args.prompt_str)) if args.prompt_str else {"default": ""}
    processor = AutoProcessor.from_pretrained(args.processor_name)

    train_dataset_list = []
    validation_dataset_list = []

    # Process train dataset if requested
    if args.dataset_type in ["train", "both"]:
        data_source = args.data_source
        if args.load_format == "arrow":
            dataset_list, subset_list = load_arrow_dataset(data_source, limit=args.limit)
        else:
            dataset_list, datainfolist, subset_list = get_flattened_dataset(data_source, limit=args.limit)

    # Process validation dataset if requested
    if args.dataset_type in ["val", "both"]:
        if args.val_load_format == "arrow":
            val_dataset_list, val_subset_list = load_arrow_dataset(args.val_data_source, limit=args.limit)
        else:
            val_dataset_list, val_datainfolist, val_subset_list = get_flattened_dataset(args.val_data_source, limit=args.limit)
    # Process train datasets
    if args.dataset_type in ["train", "both"]:
        pbar = tqdm(total=len(dataset_list), desc="Processing train datasets")
        for subset_name, dataset in zip(subset_list, dataset_list):
            print(f"Processing train dataset: {subset_name}")
            train_dataset = dataset
            post_prompt = prompt_str.get(subset_name, prompt_str.get("default", ""))

            # add a row to each data item that represents a unique id
            train_dataset = train_dataset.map(function=make_map_fn("train", subset_name, post_prompt, processor), with_indices=True, num_proc=32)
            train_dataset_list.append(train_dataset)
            pbar.update(1)
        pbar.close()
        print(f"Processed {len(train_dataset_list)} train datasets.")

    # Process validation datasets
    if args.dataset_type in ["val", "both"]:
        pbar = tqdm(total=len(val_dataset_list), desc="Processing validation datasets")
        for subset_name, dataset in zip(val_subset_list, val_dataset_list):
            print(f"Processing validation dataset: {subset_name}")
            val_dataset = dataset
            post_prompt = prompt_str.get(subset_name, prompt_str.get("default", ""))

            # add a row to each data item that represents a unique id
            val_dataset = val_dataset.map(function=make_map_fn("val", subset_name, post_prompt, processor), with_indices=True, num_proc=32)
            # Repeat the dataset if it is aime subset to increase the size
            if "aime" in subset_name:
                val_dataset = val_dataset.repeat(8)
            validation_dataset_list.append(val_dataset)
            pbar.update(1)
        pbar.close()
        print(f"Processed {len(validation_dataset_list)} validation datasets.")

    # Concatenate and save datasets based on what was processed
    if args.dataset_type in ["train", "both"] and train_dataset_list:
        train_dataset = datasets.concatenate_datasets(train_dataset_list)
        if "ids" in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns(["ids"])
        train_dataset = train_dataset.add_column("ids", [i for i in range(len(train_dataset))])
        origin_len = len(train_dataset)
        print(f"Total training dataset size before filtering: {origin_len}")
        train_dataset_drop_mm = train_dataset.remove_columns(["images"])
        train_dataset_drop_mm = train_dataset_drop_mm.filter(lambda x: len(x["extra_info"]["answer"]) <= 50, num_proc=32)
        select_ids = train_dataset_drop_mm["ids"]
        print(f"Total training dataset size after filtering: {len(select_ids)}")
        train_dataset = train_dataset.select(select_ids)
        print(f"Total training dataset size: {len(train_dataset)}")
        print(train_dataset[0])

    if args.dataset_type in ["val", "both"] and validation_dataset_list:
        try:
            val_dataset = datasets.concatenate_datasets(validation_dataset_list)
        except ValueError as e:
            # There is a type unmatch error in the last dataset
            val_dataset = datasets.concatenate_datasets(validation_dataset_list[:-1])
        print(f"Total validation dataset size: {len(val_dataset)}")
        print(val_dataset[0])
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Save datasets based on what was processed
    if args.dataset_type in ["train", "both"] and train_dataset_list:
        train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    if args.dataset_type in ["val", "both"] and validation_dataset_list:
        val_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
