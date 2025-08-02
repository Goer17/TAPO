import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import requests
from .serper_topkcommon import serper_batch_search
from .code_runner import batch_exec
# from .mock_engine import get_mock_search

# mock_batch_search = get_mock_search()

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    tools: list[str]
    no_think_rl: bool=False
    search_url: str = None
    coder_url: str = None
    topk: int = 3
    search_engine_name: str = "serper"


class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']
    
    def _process_responses(self, resp, config_tools):
        """Process responses to stop at search operation or answer operation."""
        if '</search>' in resp and 'search' in config_tools:
            return resp.split('</search>')[0] + '</search>'
        elif '</code>' in resp and 'code' in config_tools:
            return resp.split('</code>')[0] + '</code>'
        elif '</answer>' in resp:
            return resp.split('</answer>')[0] + '</answer>'
        else:
            return resp

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [self._process_responses(resp, self.config.tools) for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            # actions, _ = self.env.postprocess_predictions(responses_str)
            # responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            # print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        if next_obs_ids.dtype == torch.int64:
            return next_obs_ids
        else:
            print(f"[ERROR] next_obs_ids dtype: {next_obs_ids.dtype}, converting to long")
            print(f"[ERROR] next_obs_ids shape: {next_obs_ids.shape}")
            print(f"[ERROR] next_obs_ids: {next_obs_ids}")
            return next_obs_ids.long()

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor,
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        if prompt.dtype != response.dtype:
            print(f"[ERROR] Prompt dtype: {prompt.dtype}, Response dtype: {response.dtype}")
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]

        tensors = [t.long() for t in tensors]
        tensors_with_mask = [t.long() for t in tensors_with_mask]

        if info is not None:
            # assert info.dtype == torch.int64
            if info.dtype != torch.int64:
                print(f"[ERROR] info tensor has incorrect dtype {info.dtype} : {info}")
            info = info.long()
            tensors.append(info) # [prompt, response, info]
            if info.dtype != prompt.dtype:
                print(f"[ERROR] Info dtype: {info.dtype}, Prompt dtype: {prompt.dtype}")
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            # [prompt, response, info_mask]
            tensors_with_mask.append(info_mask)
        
        assert all(tensor.dtype == torch.int64 for tensor in tensors), f"one of the tensors is not int64 type."

        concatenated = torch.cat(tensors, dim=1).long()
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1).long()
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)
        # 让内容靠右，pad 靠左

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""

        # print(f"before: {right_side['responses']}")
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        # print(f"after: {responses}")

        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        print(f"[DEBUG] In _generate_with_gpu_padding, batch size is: {batch_size}, remainder is: {remainder}")
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            # Copy the first dimention (batch)
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor, tools: List[str] = ["search", "code"]) -> Tuple[Dict, Dict]:        
        do_search = "search" in tools
        do_code = "code" in tools

        assert do_search or do_code, "At least one of 'search' and 'code' should be used"
        """Run main LLM generation loop."""
        if do_code:
            assert self.config.coder_url, "Remote CI url is empty"
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        # prompt

        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        # (bs, 0)

        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        # (bs, 1)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        # (bs, 1)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        # (bs, 1) record the number of valid action
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        # (bs, 1) record the number of valid search
        valid_code_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        # (bs, 1) record the number of valid code-exec
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            print(f"[DEBUG] In step {step}, rollings_active batch size is: {rollings_active.batch['input_ids'].shape[0]}")

            rollings_active.meta_info = rollings.meta_info

            gen_output = self._generate_with_gpu_padding(rollings_active)
            # (bs, seq_len)

            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # print(f"responses_str: {responses_str}")

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search, is_code = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=do_search, do_code=do_code
            )
            # print(f"len: {len(next_obs)}, {len(dones)}, {len(valid_action)}, {len(is_search)}, {len(is_code)}")
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)

            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            valid_code_stats += torch.tensor(is_code, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            rollings_active.meta_info = rollings.meta_info
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search, is_code = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False, do_code=False
            )
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            valid_code_stats += torch.tensor(is_code, dtype=torch.int64)

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        meta_info['valid_code_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict):
        """Compose final generation output."""
        pad_id = self.tokenizer.pad_token_id
        batch_size = right_side['responses'].shape[0]
        if right_side['responses'].shape[1] < self.config.max_response_length:
            padded_length = self.config.max_response_length - right_side['responses'].shape[1]
            device = right_side['responses'].device
            dtype = right_side['responses'].dtype
    
            padded_tensor = torch.full((batch_size, padded_length), pad_id, device=device, dtype=dtype)

            right_side['responses'] = torch.cat([right_side['responses'], padded_tensor], dim=1)
            right_side['responses_with_info_mask'] = torch.cat([right_side['responses_with_info_mask'], padded_tensor], dim=1)



        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)

        # response mask
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True, do_code=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        # print(f"cur_actions length: {len(cur_actions)}")
        # print(f"contents length: {len(contents)}")
        # print(cur_actions[:10])
        # print(contents[:10])
        next_obs, dones, valid_action, is_search, is_code = [], [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        code_quries = [content for action, content in zip(cur_actions, contents) if action == 'code']
        if do_search:
            if not search_queries:
                search_results = []
            else:
                if self.config.search_engine_name == "serper":
                    search_results: List[str] = serper_batch_search(search_queries, self.config.topk)
                else:
                    raise ValueError(f"Unsupported search engine: {self.config.search_engine_name}")
        else:
            search_results = ['[Failed] Reached the maximum search attempts.'] * sum([1 for action in cur_actions if action == 'search'])
        assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])

        if do_code:
            if not code_quries:
                code_results = []
            else:
                code_results: List[str] = batch_exec(url=self.config.coder_url, codes=code_quries)
        else:
            code_results: List[str] = ['[Failed] Reached the maximum code attempts.'] * sum([1 for action in cur_actions if action == 'code'])
        assert len(code_results) == sum([1 for action in cur_actions if action == 'code'])
        

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                is_code.append(0)
            elif action == 'answer':
                next_obs.append('')
                dones.append(1)
                valid_action.append(1)
                is_search.append(0)
                is_code.append(0)
            elif action == 'search':
                search_result = search_results.pop(0)
                if search_result:
                    next_obs.append(f'\n\n<response>{search_result.strip()}</response>\n\n')
                else:
                    # print(f"[WARN] Search result :|{search_result}| is empty for query: |{contents[i]}|")
                    next_obs.append(f'\n\n<response>Sorry, I cannot find any relevant information.</response>\n\n')
                # next_obs.append(f'\n\n<response>{search_results.pop(0).strip()}</response>\n\n')
                dones.append(0)
                valid_action.append(1)
                is_search.append(1)
                is_code.append(0)
            elif action == 'code':
                code_result = code_results.pop(0)
                next_obs.append(f"\n\n<response>{code_result.strip()}</response>\n\n")
                dones.append(0)
                valid_action.append(1)
                is_search.append(0)
                is_code.append(1)
            else:
                error_messages = ['\nMy previous action is invalid. ']
                if 'search' in self.config.tools:
                    error_messages.append('If I want to search, I should put the query between <search> and </search>.')
                if 'code' in self.config.tools:
                    error_messages.append('If I want to execute code, I should put the code between <code> and </code>.')

                error_messages.append('If I want to give the final answer, I should put the answer between <answer> and </answer>.')

                next_obs.append(' '.join(error_messages) + ' Let me try again.\n')
                dones.append(0)
                valid_action.append(0)
                is_search.append(0)
                is_code.append(0)
            
        assert len(search_results) == 0 and len(code_results) == 0
            
        return next_obs, dones, valid_action, is_search, is_code

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
        allowed_actions = []
        if 'search' in self.config.tools:
            allowed_actions.append('search')
        if 'code' in self.config.tools:
            allowed_actions.append('code')
        allowed_actions.append('answer')

        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = fr'<({"|".join(allowed_actions)})>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> List[str]:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
