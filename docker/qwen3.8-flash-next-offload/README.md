# Qwen3.8-Flash-Next with KV Offload

使用 Docker Compose 在单机 8 卡环境中启动 Qwen3.8-Flash-Next，并启用 vLLM
原生 KV Offload。服务使用 host 网络，默认监听宿主机 `10669` 端口。

## 前置条件

- Docker、Docker Compose 和 NVIDIA Container Toolkit
- 8 张可用 NVIDIA GPU
- 本地存在 `vllm/vllm-openai:qwen38-flash-next` 镜像
- 模型目录中包含 `Qwen3.8-Flash-Next`
- 足够的主机内存和 `/dev/shm` 空间

## 配置环境变量

```sh
cp .env.example .env
```

编辑 `.env`：

- `VLLM_API_KEY`：API Bearer Token
- `PORT`：vLLM API 监听端口
- `MODEL_CACHE_PATH`：宿主机模型缓存目录
- `CONFIG_PATH`：宿主机上 `config.yaml` 的绝对路径

`.env` 包含敏感信息，不要提交到版本库。

host 网络不会进行端口映射；启动前应确认 `PORT` 未被宿主机上的其他服务占用。

## 准备共享内存

当前 `config.yaml` 设置了 `kv-offloading-size: 384`。启动前确认 `/dev/shm`
容量满足实际需求；配置文件中的参考命令会将其调整为 850 GiB：

```sh
sudo mount -o remount,size=850G /dev/shm
df -h /dev/shm
```

清理旧的 `vllm_offload_*.mmap` 文件会中断或破坏正在使用这些文件的进程，
只应在确认没有相关 vLLM 实例运行时执行。

## 启动

```sh
docker compose up -d
docker compose logs -f model-service
```

健康检查预留了 30 分钟冷启动时间。

## 验证

```sh
docker compose ps
curl http://localhost:10669/health
curl http://localhost:10669/v1/models \
  -H "Authorization: Bearer <VLLM_API_KEY>"
```

API 中使用的模型名为 `Qwen/Qwen3.8-Flash-Next`。当前配置启用了 MTP 推测解码、
Expert Parallel、Prefix Cache、Chunked Prefill 和 native KV Offload。

## 停止

```sh
docker compose down
```
