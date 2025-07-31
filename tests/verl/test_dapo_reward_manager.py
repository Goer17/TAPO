import pytest
import torch
from types import SimpleNamespace
from verl.workers.reward_manager.dapo import DAPORewardManager

# Minimal mock tokenizer
class DummyTokenizer:
    def __init__(self):
        self.eos_token = "<eos>"
    def decode(self, ids, skip_special_tokens=True):
        # ids can be a tensor or list
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        return ''.join([str(i) for i in ids]) + self.eos_token

# Minimal mock overlong_buffer_cfg
class DummyOverlongCfg:
    def __init__(self, enable=True, length=2, penalty_factor=1.0, log=True):
        self.enable = enable
        self.len = length
        self.penalty_factor = penalty_factor
        self.log = log

# Minimal DataProto and DataProtoItem mocks
class DummyDataProtoItem:
    def __init__(self, prompt_ids, response_ids, attn_mask, ground_truth, data_source, extra_info=None):
        self.batch = {
            'prompts': prompt_ids,
            'responses': response_ids,
            'attention_mask': attn_mask,
        }
        self.non_tensor_batch = {
            'reward_model': {'ground_truth': ground_truth},
            'data_source': data_source,
        }
        if extra_info:
            self.non_tensor_batch['extra_info'] = extra_info

class DummyDataProto:
    def __init__(self, items, rm_scores=None):
        self.items = items
        self.batch = {
            'responses': torch.stack([item.batch['responses'] for item in items]),
        }
        if rm_scores is not None:
            self.batch['rm_scores'] = rm_scores
    def __getitem__(self, idx):
        return self.items[idx]
    def __len__(self):
        return len(self.items)

def test_init_and_assertion():
    tokenizer = DummyTokenizer()
    # Should not raise
    DAPORewardManager(tokenizer, num_examine=1, max_resp_len=10, overlong_buffer_cfg=DummyOverlongCfg())
    # Should raise assertion error if overlong_buffer_cfg is set but max_resp_len is None
    with pytest.raises(AssertionError):
        DAPORewardManager(tokenizer, num_examine=1, max_resp_len=None, overlong_buffer_cfg=DummyOverlongCfg())

def test_call_with_rm_scores():
    tokenizer = DummyTokenizer()
    manager = DAPORewardManager(tokenizer, num_examine=1)
    item = DummyDataProtoItem(
        prompt_ids=torch.tensor([1,2,3]),
        response_ids=torch.tensor([4,5,6]),
        attn_mask=torch.tensor([1,1,1,1,1,1]),
        ground_truth='gt',
        data_source='ds',
    )
    data = DummyDataProto([item], rm_scores=torch.tensor([[0.5, 0.6, 0.7]]))
    out = manager(data)
    assert torch.allclose(out, torch.tensor([[0.5, 0.6, 0.7]]))
    out_dict = manager(data, return_dict=True)
    assert 'reward' in out_dict
    assert torch.allclose(out_dict['reward'], torch.tensor([[0.5, 0.6, 0.7]]))

def test_call_without_rm_scores():
    tokenizer = DummyTokenizer()
    overlong_cfg = DummyOverlongCfg(enable=True, length=1, penalty_factor=2.0, log=True)
    def dummy_score_fn(**kwargs):
        return 1.0
    manager = DAPORewardManager(tokenizer, num_examine=1, compute_score=dummy_score_fn, max_resp_len=20, overlong_buffer_cfg=overlong_cfg)
    item = DummyDataProtoItem(
        prompt_ids=torch.tensor([1,2,3]),
        response_ids=torch.tensor([4,5,6]),
        attn_mask=torch.tensor([1,1,1,1,1,1]),
        ground_truth='gt',
        data_source='ds',
    )
    data = DummyDataProto([item])
    out = manager(data)
    assert out.shape == torch.Size([1,3])
    assert out[0,2] == 1.0
    out_dict = manager(data, return_dict=True)
    assert 'reward_tensor' in out_dict
    assert out_dict['reward_tensor'][0,2] == 1.0

def test_overlong_penalty():
    tokenizer = DummyTokenizer()
    def dummy_score_fn(**kwargs):
        return 2.0
    overlong_cfg = DummyOverlongCfg(enable=True, length=1, penalty_factor=2.0, log=True)
    manager = DAPORewardManager(tokenizer, num_examine=1, compute_score=dummy_score_fn, max_resp_len=3, overlong_buffer_cfg=overlong_cfg)
    item = DummyDataProtoItem(
        prompt_ids=torch.tensor([1,2]),
        response_ids=torch.tensor([3,4,5,6]),
        attn_mask=torch.tensor([1,1,1,1,1,1]),
        ground_truth='gt',
        data_source='ds',
    )
    data = DummyDataProto([item])
    out_dict = manager(data, return_dict=True)
    print(out_dict['reward_tensor'])
    print(out_dict['reward_extra_info']['overlong_reward'])
    assert out_dict['reward_tensor'][0,3] == -2.0
    assert out_dict['reward_extra_info']['overlong_reward'][0] == -4.0
    assert out_dict['reward_extra_info']['overlong'][0].item() is True
