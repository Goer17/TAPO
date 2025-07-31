import re
import os
import datasets

import argparse


def make_prefix(dp, template_type):
    question = dp['question']

    if template_type == 'base':
        prefix = f"""You are an advanced reasoning assistant. Follow this process:
1. Reason: Use <think>...</think> to analyze the question.
2. Calculate: If needed, write Python code in <code>...</code>; output appears in <response>...</response>.
3. Answer: Give the final answer in <answer>...</answer> without any other explanation.
Example:
Question: What is $1234567 \\times 7654321$?
<think>Need to multiply two large numbers.</think>
<code>
a = 1234567
b = 7654321
print(a * b)
</code>
<response>9449772114007</response>
<answer>9449772114007</answer>

Now answer: {question}
"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system
You are an advanced reasoning assistant. Follow this process:
1. Reason: Use <think>...</think> to analyze the question.
2. Calculate: If needed, write Python code in <code>...</code>; output appears in <response>...</response>.
3. Answer: Give the final answer in <answer>...</answer> without any other explanation.
Example:
Question: What is $1234567 \\times 7654321$?
<think>Need to multiply two large numbers.</think>
<code>
a = 1234567
b = 7654321
print(a * b)
</code>
<response>9449772114007</response>
<answer>9449772114007</answer>

Now answer the question blow.
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
    parser.add_argument('--local_dir', default='./data/aime2024')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'aime2024'

    dataset = datasets.load_dataset("Maxwell-Jia/AIME_2024", split="train")

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question: str = example["Problem"]
            example["question"] = question
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target": example['Answer'],
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
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

    test_dataset = dataset.map(function=make_map_fn('test'), with_indices=True)

    # train_dataset = train_dataset.select(range(1000))
    # test_dataset = test_dataset.select(range(100))

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
