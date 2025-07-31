from typing import Literal

from openai import Client
import re

sys_prompt = f"""You are an advanced reasoning assistant. Follow this structured approach: 
1. Reasoning Process: 
Analyze the question within <think>...</think> tags. Identify knowledge gaps.
2. Information Gathering:
If needed, search using <search>...</search>. Results appear in <response>...</response>.
3. Calculate:
If needed, write Python code in <code>...</code>; output appears in <response>...</response>.
4. Final Answer:
Provide the answer in <answer>...</answer> without any other explanation.
Example:
Question: What is the population density of Singapore? (Unit: number of person / km², round to one decimal place")
Response:
<think>To solve this problem, I need to search Singapore’s population and land area and then use Python to calculate the population density.</think>
<search>Singapore population 2024</search>
<response>... 5.9 million (World Bank) ...</response>
<search>Singapore land area</search>
<response>... 728 km² (official sources) ...</response>
<code>
population = 5_900_000
area = 728
density = population / area
print(f"{{density:.1f}} people/km²")
</code>
<response>8104.4 people/km²</response>
<answer>8104.4</answer>
Following the specified rules, now answer the question:
"""

from .serper_topkcommon import serper_search
from .code_runner import remote_exec

coder_url = "http://10.200.99.220:31773/run"

class ToolAgent:
    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model: str,
                 model_type: Literal["instruct", "base"] = "base"):
        self.client = Client(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.model_type = model_type
    
    def inference(self, question: str, debug: bool = False, max_turns: int = 10,  max_tokens: int = 4096, temperature: float = 1.0) -> str:
        model_type = self.model_type
        messages = [
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
        prompt = f"{sys_prompt} {question}\n"
        answer = []
        special_tags = ["search", "code", "answer"]
        pattern = fr"<({'|'.join(special_tags)})>(.*?)$"
        for i in range(max_turns):
            if model_type == "instruct":
                output = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    stop=[f"</{tag}>" for tag in special_tags],
                    max_tokens=max_tokens,
                    temperature=temperature
                ).choices[0].message.content
            elif model_type == "base":
                output = self.client.completions.create(
                    prompt=prompt,
                    model=self.model,
                    stop=[f"</{tag}>" for tag in special_tags],
                    max_tokens=max_tokens,
                    temperature=temperature
                ).choices[0].text
            else:
                raise NotImplementedError
            matches = re.search(pattern=pattern, string=output, flags=re.DOTALL)
            if matches is None:
                tag, content = None, None
            else:
                tag, content = matches.group(1), matches.group(2)
                output += f"</{tag}>"
            answer.append(output)
            if debug:
                print(output)
            if tag is None:
                continue
            if tag == "search":
                response, _ = serper_search(query=content)
            elif tag == "code":
                response = remote_exec(coder_url, code=content)
            elif tag == "answer":
                # over
                return '\n'.join(answer)
            else:
                raise RuntimeError(f"Unkonwn tag: {tag}")
            
            messages.append(
                {
                    "role": "assistant",
                    "content": output
                }
            )
            prompt += output + '\n'

            response = f"<response>{response}</response>"
            answer.append(response)
            messages.append(
                {
                    "role": "user",
                    "content": response
                }
            )
            prompt += response

            if debug:
                print(response)
        
        return '\n'.join(answer)