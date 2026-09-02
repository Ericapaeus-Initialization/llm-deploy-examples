# Qwen3.6-35B-A3B on 8 x RTX 4090

使用 Docker Compose 在单机 8 卡环境中启动 Qwen3.6-35B-A3B。服务使用 host
网络，并通过 `.env` 中的 `PORT` 直接监听宿主机端口，默认值为 `10669`。

## 前置条件

- Docker、Docker Compose 和 NVIDIA Container Toolkit
- 8 张可用 NVIDIA GPU
- 模型目录中包含 `Qwen3.6-35B-A3B`
- 宿主机存在本目录对应的 `config.yaml`

## 配置环境变量

```sh
cp .env.example .env
```

编辑 `.env`：

- `VLLM_API_KEY`：API Bearer Token
- `PORT`：vLLM API 监听端口
- `MODEL_CACHE_PATH`：宿主机模型缓存目录；其下应存在 `Qwen3.6-35B-A3B`
- `CONFIG_PATH`：宿主机上 `config.yaml` 的绝对路径

`.env` 包含敏感信息，不要提交到版本库。

host 网络不会进行端口映射；启动前应确认 `PORT` 未被宿主机上的其他服务占用。

## 启动

```sh
docker compose up -d
docker compose logs -f model-service
```

首次启动需要加载模型，健康检查预留了 5 分钟冷启动时间。

## 验证

```sh
curl http://localhost:10669/health
curl http://localhost:10669/v1/models \
  -H "Authorization: Bearer <VLLM_API_KEY>"
```

API 中使用的模型名为 `Qwen/Qwen-AgentWorld-35B-A3B`。主要推理参数位于
`config.yaml`，包括 TP=8、最大上下文 262144 和 GPU 显存利用率 0.90。

## 停止

```sh
docker compose down
```
