# 使用 Docker 自定义 Redis 服务器

该目录包含使用 Docker 构建和运行自定义 Redis 服务器所需的文件。此配置针对高并发场景进行了优化，并可根据需要进行自定义。

## 环境要求

- 您的系统上必须安装 [Docker](https://www.docker.com/get-started)。

## 配置

Redis 服务器的配置在 `redis.conf` 文件中定义。在构建 Docker 镜像之前，您可能需要调整设置以适应您的环境。

### 重要：调整 `maxmemory`

最重要的配置是 `maxmemory`。它定义了 Redis 可以使用的最大内存量。

- **文件**: `redis.conf`
- **设置**: `maxmemory 28gb`

**需要操作**: 请根据您主机的可用 RAM 或具体的项目需求，将 `28gb` 修改为适当的值。例如，如果您的机器有 16GB 内存，您可以将其设置为 `12gb`。

## 如何使用

### 1. 构建 Docker 镜像

在您的终端中，导航到此目录 (`recipe/tcg/src/redis`) 并运行以下命令来构建镜像：

```bash
docker build -t my-custom-redis .
```

这将基于所提供的 `Dockerfile` 和 `redis.conf` 创建一个名为 `my-custom-redis` 的 Docker 镜像。

### 2. 运行 Redis 容器

镜像构建完成后，您可以使用此命令启动一个容器：

```bash
docker run -d --name redis-container -p 6379:6379 my-custom-redis
```

- `-d`: 在后台（分离模式）运行容器。
- `--name redis-container`: 为容器分配一个名称，方便管理。
- `-p 6379:6379`: 将您主机上的 6379 端口映射到容器内的 6379 端口。

### 3. 验证服务

您可以检查容器是否正在正常运行：

```bash
docker ps
```

要连接到 Redis 服务器并进行测试，请在容器内使用 `redis-cli`：

```bash
docker exec -it redis-container redis-cli
```

如果您设置了密码，需要先进行身份验证：
```
> AUTH yoursecurepassword
```

然后，运行一个测试命令：
```
> PING PONG
```

如果您看到 `PONG`，说明您的自定义 Redis 服务器已成功启动并正常运行。

### 4. 在脚本中使用

不再需要直接修改 Python 文件中的 Redis 配置。

所有 Redis 配置现在都通过sh脚本进行管理。

请在该文件中找到或添加 `Redis` 部分，并设置 `SEARCH_REDIS_HOST` 和 `SEARCH_REDIS_PORT`：

```bash
export SEARCH_REDIS_HOST=10.200.99.220
export SEARCH_REDIS_PORT=32132
```

训练器启动时将自动读取此配置并应用到所有相关的模块中。

# 使用 Serper 作为搜索引擎
更改bash脚本（以`recipe/tcg/examples/test_7b_code_multi_node.sh`为例）中的以下配置：

```bash
        retriever.search_engine_name=serper \ # 使用serper
        retriever.topk=3 \ # 使用serper时建议至少为3
        do_search=True \
        do_code=True \
        coder.url="http://10.200.99.220:31773/run" \
        trainer.resume_mode=disable | tee "${CKPTS_DIR}/training_output_$(date +%Y%m%d_%H%M%S).log"
