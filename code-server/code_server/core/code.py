from typing import Literal
import time
from typing import Dict


class _timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        # Record execution time
        self.start_time = time.time()
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time

class ExecuteMethods:
    @staticmethod
    def _sub_process_in_tempdir(timeout: int | None = 2000, memory_limit: None = None):
        if memory_limit is not None:
            raise RuntimeError("Memory limit only supported on in this method.")
        import subprocess
        import tempfile
        def _exec(code: str) -> str:
            with tempfile.TemporaryDirectory() as tempdir:
                command = ["python", "-c", code]
                result = subprocess.run(
                    command,
                    cwd=tempdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=timeout / 1000 if timeout else None,
                )
                return result.stdout
        return _exec
    
    @staticmethod
    def _inline_exec(timeout: int | None = 2000, memory_limit: None = None):
        if memory_limit is not None:
            raise RuntimeError("Memory limit only supported on in this method.")
        from RestrictedPython import compile_restricted, safe_builtins
        from RestrictedPython.PrintCollector import PrintCollector
        from shortuuid import uuid
        import math, numpy, cmath, statistics, random, decimal, fractions, sympy
        import threading
        allowed_modules = {
            "math": math,
            "numpy": numpy,
            "cmath": cmath,
            "statistics": statistics,
            "random": random,
            "decimal": decimal,
            "fractions": fractions,
            "sympy": sympy
        }

        def _safe_import(name: str, *args, **kwargs):
            if name not in allowed_modules:
                raise ImportError(f"ImportError: The module '{name}' is not permitted for import.")
            return allowed_modules[name]
        
        def _inplacevar_(op, var, expr):
            ops = {
                "+=": lambda a, b: a + b,
                "-=": lambda a, b: a - b,
                "*=": lambda a, b: a * b,
                "/=": lambda a, b: a / b,
                "%=": lambda a, b: a % b,
                "**=": lambda a, b: a ** b,
                "<<=": lambda a, b: a << b,
                ">>=": lambda a, b: a >> b,
                "|=": lambda a, b: a | b,
                "^=": lambda a, b: a ^ b,
                "&=": lambda a, b: a & b,
                "//=": lambda a, b: a // b,
                "@=": lambda a, b: a // b,  # Note: '@=' is not standard, adjust as needed
            }
            if op in ops:
                return ops[op](var, expr)
            raise NotImplementedError(f"Operator {op} is not supported.")
        
        def _getitem_(obj, idx):
            return obj[idx]
        
        def _write_(obj):
            return obj

        _inline_exec_builtins = {**safe_builtins, **{"__import__": _safe_import}}

        basic_fns = {
            "sum": sum,
            "any": any,
            "all": all,
            "min": min,
            "max": max,
            "len": len,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "enumerate": enumerate,
            "range": range,
            "zip": zip,
            "map": map,
            "filter": filter,
            "list": list,
            "tuple": tuple,
            "set": set,
            "dict": dict,
        }
        
        def _exec(code: str) -> str:
            stdout_var = f"result_{uuid()}"
            code = code.replace("printed", "printed_") + f"\n{stdout_var} = printed\n"
            bytecode = compile_restricted(code, "<inline code>", "exec")
            glob = {
                "__builtins__": _inline_exec_builtins,
                "_print_": PrintCollector,
                "_getattr_": getattr,
                "_getiter_": iter,
                "_getitem_": _getitem_,
                "_write_": _write_,
                "_inplacevar_": _inplacevar_,
                **basic_fns
            }
            def target():
                try:
                    exec(bytecode, glob)
                except Exception as err:
                    glob["__error__"] = err

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout / 1000 if timeout else None)
            if thread.is_alive():
                raise TimeoutError(f"[TIMEOUT] Code execution exceeded {timeout} ms.")
            if "__error__" in glob:
                return str(glob["__error__"])
            return glob.get(stdout_var, "")
        
        return _exec


class SafeInterpretor:
    """Python code interpretor.
    """
    def __init__(self,
                 timeout: int | None = 2000, # ms
                 memory_limit: int | None = 200, # MB
                 max_stdout_length: int = 1024,
                 method: Literal["subprocess", "inline", "docker"] = "subprocess"
                 ):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.max_stdout_length = max_stdout_length
        if method == "subprocess":
            self._exec = ExecuteMethods._sub_process_in_tempdir(timeout=timeout, memory_limit=memory_limit)
        elif method == "inline":
            self._exec = ExecuteMethods._inline_exec(timeout=timeout, memory_limit=memory_limit)
        elif method == "docker":
            raise NotImplementedError("Not implemented yet.")
        else:
            raise NotImplementedError(f"The exec-method '{method}' is not implemented.")
    
    def run(self, code: str) -> Dict[str, str | bool | None]:
        response = {
            "stdout": None,
            "finished": False,
            "time": None
        }
        try:
            with _timer() as context:
                _stdout = self._exec(code)
                response['stdout'] = _stdout
                response['finished'] = True
        except Exception as err:
            response['stdout'] = err.__str__()
        finally:
            if 'context' in locals():
                seconds = getattr(locals()['context'], 'elapsed')
                response['time'] = f"{int(seconds * 1000)} ms"
            if len(response['stdout']) > self.max_stdout_length:
                response['stdout'] = (response['stdout'][: self.max_stdout_length // 2] + "...\n..."
                                        + response['stdout'][-self.max_stdout_length // 2 :])
        return response