import pytest
import tempfile
import os
from recipe.dapo.src.main_dapo import get_custom_reward_fn

class DummyConfig:
    def __init__(self, custom_reward_function={
        "path": None,
        "name": None
    }):
        self.custom_reward_function = custom_reward_function or {}

    def get(self, key):
        return getattr(self, key, None)

def test_get_custom_reward_fn_success(tmp_path):
    # Create a temporary Python file with a simple function
    reward_py = tmp_path / "reward.py"
    reward_py.write_text("def my_reward(x):\n    return x + 1\n")
    config = {
        "custom_reward_function": {
            "path": str(reward_py),
            "name": "my_reward"
        }
    }
    fn = get_custom_reward_fn(config)
    assert callable(fn)
    assert fn(2) == 3

def test_get_custom_reward_fn_no_path():
    config = {"custom_reward_function": {}}
    assert get_custom_reward_fn(config) is None

def test_get_custom_reward_fn_file_not_found():
    config = {"custom_reward_function": {"path": "/not/a/real/file.py", "name": "foo"}}
    with pytest.raises(FileNotFoundError):
        get_custom_reward_fn(config)

def test_get_custom_reward_fn_function_not_found(tmp_path):
    reward_py = tmp_path / "reward.py"
    reward_py.write_text("def other_fn():\n    return 0\n")
    config = {
        "custom_reward_function": {
            "path": str(reward_py),
            "name": "not_here"
        }
    }
    with pytest.raises(AttributeError):
        get_custom_reward_fn(config)

def test_get_custom_reward_fn_import_error(tmp_path):
    reward_py = tmp_path / "reward.py"
    reward_py.write_text("raise RuntimeError('fail to import')\ndef foo(): return 1\n")
    config = {
        "custom_reward_function": {
            "path": str(reward_py),
            "name": "foo"
        }
    }
    with pytest.raises(RuntimeError):
        get_custom_reward_fn(config)