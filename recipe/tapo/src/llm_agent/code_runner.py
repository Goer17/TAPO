from typing import List
import requests
from concurrent.futures import ThreadPoolExecutor
from time import time

def remote_exec(url: str, code: str) -> str:
    try:
        resp = requests.post(
            url,
            json={"code": code},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        resp_json = resp.json()
    except Exception:
        resp_json = {}
    stdout = resp_json.get('stdout', "[ERROR] Something went wrong, please try again.")
    if stdout == "":
        stdout = "[WARNING] stdout is empty, maybe you forgot to use `print` function to print the final answer, please try again."
    return stdout

def batch_exec(url: str, codes: List[str], max_workers: int = 8, show_time_clip: bool = False) -> List[str]:
    if show_time_clip:
        start_t= time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(remote_exec, url, code) for code in codes]
        results = [f.result() for f in futures]
        if show_time_clip:
            end_t = time()
            print(f"Total exec time: {end_t - start_t:.4f} s")
        return results

if __name__ == "__main__":
    # Testing
    codes = [
        "print(1 + 1)",
        "print(1 + 2)",
        "while True: pass",
        "print('hello, world')",
        "x = 1\nfor i in range(10000): x += i\nprint(x)",
        "import os\nprint(os.listditr(\".\"))"
    ]
    responses = batch_exec(url="coder_url", codes=codes, show_time_clip=True)
    for response in responses:
        print(response)