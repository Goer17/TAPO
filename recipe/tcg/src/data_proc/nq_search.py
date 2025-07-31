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
"""
Preprocess the nq dataset to parquet format

Only Search
"""

import re
import os
import datasets

import argparse


def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""You are an advanced reasoning assistant. Follow this structured approach:  
1. Reasoning Process: 
Analyze the question within <think>...</think> tags. Identify knowledge gaps.
2. Information Gathering:
If needed, search using <search>...</search>. Results appear in <response>...</response>.
3. Final Answer:
Provide the answer in <answer>...</answer> once sufficient information is gathered.
Example:  
Question: What is the capital of Australia?  
Response:  
<think>Verify Australia's capital.</think>  
<search>The capital of Australia</search>  
<response>Canberra is the capital...</response>  
<think>Confirmed—no further checks needed.</think>  
<answer>Canberra</answer>

Following the specified rules, now answer the question: {question}\n
"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system
You are an advanced reasoning assistant. Follow this structured approach:  
1. Reasoning Process: 
Analyze the question within <think>...</think> tags. Identify knowledge gaps.
2. Information Gathering:
If needed, search using <search>...</search>. Results appear in <response>...</response>.
3. Final Answer:
Provide the answer in <answer>...</answer> once sufficient information is gathered.
Example:  
Question: What is the capital of Australia?  
Response:  
<think>Verify Australia's capital.</think>  
<search>The capital of Australia</search>  
<response>Canberra is the capital...</response>  
<think>Confirmed—no further checks needed.</think>  
<answer>Canberra</answer>

Following the specified rules, now answer the question below.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'nq'

    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            example['question'] = example['question'].strip()
            if example['question'][-1] != '?':
                example['question'] += '?'
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target": example['golden_answers'],
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    test_dataset = test_dataset.select(range(300))

    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
