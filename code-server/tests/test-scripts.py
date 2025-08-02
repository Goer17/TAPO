import pytest
import json
import requests
from pathlib import Path

url = "http://127.0.0.1:8000/run"
tests_path = Path("tests")

def load_cases():
    with open(tests_path / "check.json", "r") as f:
        return json.load(f)

@pytest.mark.parametrize("case", load_cases())
def test_case(case):
    script = case["script"]
    expected_stdout = case["stdout"]
    expected_finished = case["finished"]

    script_path = tests_path / "scripts" / script
    with open(script_path, "r") as f:
        code = f.read()

    response = requests.post(
        url=url,
        headers={"Content-Type": "application/json"},
        data=json.dumps({"code": code})
    )
    assert response.status_code == 200
    resp_json = response.json()
    assert resp_json["finished"] == expected_finished
    assert expected_stdout in resp_json["stdout"], f"WA in script {script}, expected: {expected_stdout}, stdout: {resp_json['stdout']}"
