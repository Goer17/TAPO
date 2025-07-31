from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from shortuuid import uuid

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
import requests

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class RetrieveRequest(BaseModel):
    queries: List[str]
    topk: int
    return_scores: bool

from pprint import pprint

@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    n = len(request.queries)
    pprint(f"Received queries: {request.queries}")    
    results = []
    for i in range(n):
        result = []
        for j in range(request.topk):
            result.append(
                {
                    "document": {
                        "id": uuid(),
                        "contents": "This is a mock response."
                    },
                    "score": 0.95
                }
            )
        results.append(result)
    
    return {"result": results}


def get_mock_search(url: str = "http://127.0.0.1:8000/retrieve"):
    def mock_batch_search(queries: List[str], topk: int = 1):
        payload = {
            "queries": queries,
            "topk": topk,
            "return_scores": True
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        results = response.json()["result"]
        ans = []
        for result in results:
            contents = []
            for item in result:
                document = item["document"]
                contents.append(document["contents"])
            ans.append("\n================================\n".join(contents))
        return ans
    return mock_batch_search

if __name__ == "__main__":
    mock_batch_search = get_mock_search()
    result = mock_batch_search(
        queries=["What is the capital of Beijing?", "What is the capital of France?"],
        topk=3
    )
    print(result)