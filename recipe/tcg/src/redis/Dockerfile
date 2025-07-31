# 使用官方的 Redis 镜像作为基础镜像
FROM redis:latest

# 将自定义的配置文件复制到镜像中的 Redis 配置目录
COPY redis.conf /usr/local/etc/redis/redis.conf

# (可选) 如果你的 Redis 配置文件中指定了密码，并且你希望在启动时就加载它
# 而不是通过命令行参数或 K8s Secret (更推荐后者)
# ENV REDIS_PASSWORD yoursecurepassword

# Redis 默认使用 6379 端口
EXPOSE 6379

# 容器启动时运行 Redis 服务器，并使用我们的配置文件
CMD [ "redis-server", "/usr/local/etc/redis/redis.conf" ]
