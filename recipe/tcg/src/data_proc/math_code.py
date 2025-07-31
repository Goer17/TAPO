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
    parser.add_argument('--local_dir', default='./data/math_code')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'dapo_math'

    dataset = datasets.load_dataset("haizhongzheng/DAPO-Math-17K-cleaned")

    # The dataset only has one split; split it into train/test with a 9:1 ratio
    full_dataset = dataset['train']
    split_datasets = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_datasets['train']
    test_dataset = split_datasets['test']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        # old_prompt = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem."
        def process_fn(example, idx):
            question: str = example['prompt']
            # if example['question'][-1] != '?':
            #     example['question'] += '?'
            example["question"] = question
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target": example['target'],
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

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset = train_dataset.select(range(1000))
    test_dataset = test_dataset.select(range(100))

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
