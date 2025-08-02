# code-server

#### TODO 🎯

- [x] Implement a secure code interpreter with enforced timeouts and restricted write permissions.
- [x] Implement API endpoints using FastAPI.
- [x] Complete the test.



#### Methods 🔍

- [x] Sub process

	Implemented by `subprocess`

- [x] Inline

	Implemented by `RestrictedPython` and built-in `exec` function.

- [ ] Docker



#### Install ⏬

```shell
uv pip venv --python=3.10 && source .venv/bin/activate
```

```shell
uv pip install -e .
```

```shell
uv pip install -r requirements.txt
```



#### Start 🏃

**Example 1**

```shell
METHOD=subprocess TIMEOUT=2000 uvicorn code_server.server:app --workers 8
```

- Maximum: 800 seq/s

**Example 2 (recommended)**

```shell
METHOD=inline TIMEOUT=2000 uvicorn code_server.server:app --workers 8
```

- Maximum: 10,000 seq/s





#### Use Docker to Start 🐳

```shell
docker build -t . coder
```

```shell
docker run -p 8000:8000 coder
```



#### Use k6 to Test Throughput ⛰️

```shell
k6 run tests/k6-test.js
```

```plain-text
         /\      Grafana   /‾‾/  
    /\  /  \     |\  __   /  /   
   /  \/    \    | |/ /  /   ‾‾\ 
  /          \   |   (  |  (‾)  |
 / __________ \  |_|\_\  \_____/ 

     execution: local
        script: tests/k6-test.js
        output: -

     scenarios: (100.00%) 1 scenario, 128 max VUs, 40s max duration (incl. graceful stop):
              * default: 128 looping VUs for 10s (gracefulStop: 30s)



  █ TOTAL RESULTS 

    checks_total.......................: 35796   3542.677506/s
    checks_succeeded...................: 100.00% 35796 out of 35796
    checks_failed......................: 0.00%   0 out of 35796

    ✓ status is 200
    ✓ response is valid JSON
    ✓ answer is correct

    HTTP
    http_req_duration.......................................................: avg=7.52ms   min=1.54ms   med=3.41ms   max=133.22ms p(90)=15.74ms  p(95)=22.87ms 
      { expected_response:true }............................................: avg=7.52ms   min=1.54ms   med=3.41ms   max=133.22ms p(90)=15.74ms  p(95)=22.87ms 
    http_req_failed.........................................................: 0.00%  0 out of 11932
    http_reqs...............................................................: 11932  1180.892502/s

    EXECUTION
    iteration_duration......................................................: avg=107.77ms min=101.68ms med=103.64ms max=240.63ms p(90)=115.96ms p(95)=122.99ms
    iterations..............................................................: 11932  1180.892502/s
    vus.....................................................................: 128    min=128        max=128
    vus_max.................................................................: 128    min=128        max=128

    NETWORK
    data_received...........................................................: 2.1 MB 208 kB/s
    data_sent...............................................................: 2.2 MB 220 kB/s




running (10.1s), 000/128 VUs, 11932 complete and 0 interrupted iterations
default ✓ [======================================] 128 VUs  10s
```

