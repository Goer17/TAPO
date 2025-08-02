import re
import signal
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import random

# Timeout Context
# class timeout:

#     def __init__(self, seconds=1, error_message="Timeout"):
#         self.seconds = seconds
#         self.error_message = error_message

#     def handle_timeout(self, signum, frame):
#         raise TimeoutError(self.error_message)

#     def __enter__(self):
#         signal.signal(signal.SIGALRM, self.handle_timeout)
#         signal.alarm(self.seconds)

#     def __exit__(self, type, value, traceback):
#         signal.alarm(0)

def normalize_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.
    
    Args:
        final_answer: The answer string to normalize
        
    Returns:
        Normalized answer string
    """
    final_answer = final_answer.strip().lower()

    number_mapping = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12"
    }
    final_answer = number_mapping.get(final_answer, final_answer)

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()

import Levenshtein

def _check_score(pred: str, gt: str, data_source: str = "deafult") -> float:
    def is_number(string: str):
        try:
            _ = complex(string)
            return True
        except Exception as e:
            return False
    pred = normalize_answer(pred)
    gt = normalize_answer(gt)

    if data_source in ["default", "gsm8k", "dapo_math", "deep_math"] or data_source.startswith("aime"):
        return float(gt == pred)
    
    if data_source == "nq":
        if gt in pred:
            return 1.0
        ratio = Levenshtein.distance(pred, gt) / max(len(pred), len(gt))
        return 1 - ratio if ratio < 0.5 else 0.0
    
    if data_source == "calculator":        
        assert is_number(gt), f"{gt} is not a number"
        return float(abs(complex(pred) - complex(gt)) < 1e-3) if is_number(pred) else 0.0
    
    if data_source == "complex":
        pred = "".join(ch for ch in pred if ch.isdigit() or ch in ["+", "-", "."])
        assert is_number(gt), f"{gt} is not a number"
        return float(abs(complex(pred) - complex(gt)) / abs(complex(gt)) < 0.05) if is_number(pred) else 0.0
    
    raise NotImplementedError(f"dato_source {data_source} is not supported in TCG!")


def _get_score(pred: str,
                gt: str | List[str],
                pause_tokens_index: Optional[list[int]] = None,
                data_source: str = "default") -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.
    
    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens
        
    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100:]
    else:
        pred = pred[-100:]

    final_answer = pred
    if isinstance(gt, str):
        return _check_score(final_answer, gt, data_source), final_answer
    elif isinstance(gt, list):
        return max(_check_score(final_answer, _gt, data_source) for _gt in gt), final_answer
    else:
        raise NotADirectoryError("Ground truth can only be string or list of strings.")

def check_format(solution_str: str,
                 enable_tools: List[str] = ["search", "code"]) -> Dict[str, Any]:
    """Verify if the solution is formatly correct
    
    Args:
        solution_str: The solution string
        Eg.
        <think>...</think>
        <search>...</search>

        <response>...</response>

        <answer></answer>

        enable_tools: The tools that agent can use
    
    Returns:
        A dictionary contains the tool calling and final answer if the final format is correct, else None
        Eg.
        {
            tool_call: {
                "search": 2,
                "code": 1
            },
            answer: "..."
        }
    """
    solution_str = solution_str.strip()
    allowed_tags = ["think", "response", "answer"] + [tool.lower() for tool in enable_tools]
    tags = []
    pos = 0
    result = {
        "tool_call": {tool : 0 for tool in enable_tools},
        "answer": None
    }
    while pos < len(solution_str):
        match = re.search(r'\s*<(\w+)>(.*?)</\1>\s*', solution_str[pos:], re.DOTALL)
        if not match or len(match.groups()) != 2:
            return None
        tag = match.group(1).lower()
        if tag not in allowed_tags:
            return None
        if tag == "answer":
            answer = match.group(2).strip()
            result["answer"] = answer
        if tag in enable_tools:
            result["tool_call"][tag] += 1
        tags.append(tag)

        pos += match.end()
    
    if pos != len(solution_str) or tags[-1] != "answer":
        return None

    return result

def compute_score(solution_str: str,
                ground_truth: str,
                pause_tokens_index: Optional[list[int]] = None,
                data_source: str = "default") -> float:
    """Compute the reward score for a solution, (tool-augmented DAPO)
    
    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer. str | List[str] | Dict[str, List[str]]
        {"target": ["...", "..."]}
        config: Configuration object containing reward model settings
        pause_tokens_index: Indices of pause tokens
        
    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    result = check_format(solution_str)

    if result is None:
        # format error
        return {
            "score": -1.0,
            "acc": False,
            "pred": "[ERROR] Incorrect Format!",
            "search": 0,
            "code": 0
        }

    answer = result["answer"]
    # tool_call_cnt = sum(int(cnt) for cnt in result["tool_call"])

    if isinstance(ground_truth, dict):
        ground_truth = ground_truth["target"]
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()

    if not isinstance(ground_truth, list) and not isinstance(ground_truth, str):
        raise NotImplementedError("Ground truth can only be string or list of strings.")

    # Verify the solution
    score, pred = _get_score(answer, ground_truth, pause_tokens_index, data_source)

    reward = score
    acc = score > 0.9

    # final result
    final_result = {
        "score": reward,
        "acc": acc,
        "pred": pred,
        **result["tool_call"]
    }

    # print(final_result)

    return final_result


# Testing
if __name__ == "__main__":
    resp = """
<think>...</think>
<search>...</search>
<response>...</response>
<answer>USA</answer>
"""
    print(compute_score(resp, "USA", data_source="nq"))
