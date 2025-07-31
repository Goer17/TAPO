from verl.utils.reward_score import _default_compute_score
from llm_agent.infer import Agent


from dotenv import load_dotenv
import os

load_dotenv()

agent = Agent(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["BASE_URL"],
    model="gpt-4o"
)

from pathlib import Path
import pandas as pd

cnt = 250

test_path = Path("data") / "tcg_easy" / "test.parquet"
test_dataset = pd.read_parquet(test_path).sample(frac=1).reset_index(drop=True).iloc[:cnt]

score = 0

from pprint import pprint

for row in test_dataset.itertuples():
    question = row.question
    data_source = row.data_source
    ground_truth = row.reward_model["ground_truth"].tolist()

    solution_str = agent.inference(question, debug=True)
    reward = _default_compute_score(data_source, solution_str, ground_truth)

    reward["question"] = question
    reward["gt"] = ground_truth

    pprint(reward)

    score += reward["score"]


print(f"avg. score: {score / cnt :.2f}")