import os
import pandas as pd
from datasets import Dataset
from typing import List

import argparse

rule_com = """
1. Reasoning Process: 
Analyze the question within <think>...</think> tags. Identify knowledge gaps.
2. Information Gathering:
If needed, search using <search>...</search>. Results appear in <response>...</response>.
3. Calculate:
If needed, write Python code in <code>...</code>; output appears in <response>...</response>.
4. Final Answer:
Provide the answer in <answer>...</answer> without any other explanation.
Example:
Question: What is the population density of Singapore? (Unit: number of person / km², round to one decimal place")
Response:
<think>To solve this problem, I need to search Singapore’s population and land area and then use Python to calculate the population density.</think>
<search>Singapore population 2024</search>
<response>... 5.9 million (World Bank) ...</response>
<search>Singapore land area</search>
<response>... 728 km² (official sources) ...</response>
<code>
population = 5_900_000
area = 728
density = population / area
print(f"{{density:.1f}} people/km²")
</code>
<response>8104.4 people/km²</response>
<answer>8104.4</answer>
"""

rule_only_code = """
1. Reasoning Process: 
Analyze the question within <think>...</think> tags. Identify knowledge gaps.
2. Calculate:
If needed, write Python code in <code>...</code>; output appears in <response>...</response>.
3. Final Answer:
Provide the answer in <answer>...</answer> without any other explanation.
Example:
Question: What is the population density of Singapore? (Unit: number of person / km², round to one decimal place")
Response:
<think>To calculate Singapore's population density, I need its population (5.9 million) and land area (728 km²). Then I'll use Python to divide population by area.</think>
<code>
population = 5_900_000
area = 728
density = population / area
print(f"{{density:.1f}} people/km²")
</code>
<response>8104.4 people/km²</response>
<answer>8104.4</answer>
"""

rule_only_search = """
1. Reasoning Process: 
Analyze the question within <think>...</think> tags. Identify knowledge gaps.
2. Information Gathering:
If needed, search using <search>...</search>. Results appear in <response>...</response>.
3. Final Answer:
Provide the answer in <answer>...</answer> without any other explanation.
Example:
Question: What is the population density of Singapore? (Unit: number of person / km², round to one decimal place")
Response:
<think>To solve this problem, I need to search Singapore’s population and land area and then use Python to calculate the population density.</think>
<search>Singapore population 2024</search>
<response>... 5.9 million (World Bank) ...</response>
<search>Singapore land area</search>
<response>... 728 km² (official sources) ...</response>
<think>The population density of Singapore is 5,900,000 / 728 = 8104.4 (person / km²)</think>
<answer>8104.4</answer>
"""

rule_empty = """
1. Reasoning Process: 
Analyze the question within <think>...</think> tags. Identify knowledge gaps.
2. Final Answer:
Provide the answer in <answer>...</answer> without any other explanation.
Example:
Question: What is the population density of Singapore? (Unit: number of person / km², round to one decimal place")
Response:
<think>To calculate Singapore's population density, I need its population (5.9 million) and land area (728 km²). Then I'll divide population by area.</think>
<think>The population density of Singapore is 5,900,000 / 728 = 8104.4 (person / km²)</think>
<answer>8104.4</answer>
"""

def make_prefix(question, template_type, tools: List[str] = ["search", "code"]):
    if "search" in tools and "code" in tools:
        rule = rule_com
    elif "search" in tools:
        rule = rule_only_search
    elif "code" in tools:
        rule = rule_only_code
    else:
        rule = rule_empty

    if template_type == 'base':
        """This works for any base model"""
        prefix_temp = f"""You are an advanced reasoning assistant. Follow this structured approach: 
{rule}
Following the specified rules, now answer the question: {question}\n
"""
    elif template_type == 'qwen-instruct':
        prefix_temp = f"""<|im_start|>system
You are an advanced reasoning assistant. Follow this structured approach:  
{rule}
Following the specified rules, now answer the question below.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
"""
    else:
        raise NotImplementedError
    return prefix_temp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/tcg')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument("--disable-search", action="store_true")
    parser.add_argument("--disable-code", action="store_true")

    args = parser.parse_args()

    tools = []
    if not args.disable_search:
        tools.append("search")
    if not args.disable_code:
        tools.append("code")

    train_dataset = Dataset.from_pandas(pd.read_parquet("./warehouse/tcg/train.parquet")).select(range(1000))
    test_dataset = Dataset.from_pandas(pd.read_parquet("./warehouse/tcg/test.parquet")).select(range(300))

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question: str = make_prefix(example["question"].strip(), args.template_type, tools)
            data = {
                "data_source": example["data_source"],
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": example["ability"],
                "reward_model": example["reward_model"],
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True).shuffle()
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True).shuffle()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
