from fastapi import FastAPI
from pydantic import BaseModel
from code_server.core.code import SafeInterpretor
from code_server.utils.logger import logger, format_exec_info

from dotenv import load_dotenv, find_dotenv
import os

import warnings
from fastapi.responses import FileResponse
warnings.filterwarnings("ignore", category=SyntaxWarning)

load_dotenv(find_dotenv())

timeout = os.environ.get("timeout", 2000)
memory_limit = os.environ.get("MEMORY_LIMIT", None)
max_stdout_length = os.environ.get("MAX_STDOUT_LENGTH", 2000)
method = os.environ.get("METHOD", "inline")
debug = os.environ.get("DEBUG", "True").lower() == "true"

app = FastAPI()
interpretor = SafeInterpretor(
    timeout=timeout,
    memory_limit=memory_limit,
    max_stdout_length=max_stdout_length,
    method=method
)

class RemoteCIRequest(BaseModel):
    code: str

@app.post("/run")
def run(request: RemoteCIRequest):
    response = interpretor.run(request.code)
    info = {**{"code": request.code}, **response}
    if debug:
        logger.info(format_exec_info(info))

    return response

@app.get("/log")
def log(date: str):
    log_path = f"logs/{date}.log"
    if not os.path.exists(log_path):
        return {"error": "Log file not found"}, 404
    return FileResponse(log_path, filename=f"{date}.log", media_type="text/plain")
