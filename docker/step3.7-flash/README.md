# Step 3.7 Flash

使用 vLLM 专用镜像在单机 8 卡 RTX PRO 6000D 环境中部署 Step 3.7 Flash。
服务使用 host 网络，并通过 `.env` 中的 `PORT` 直接监听宿主机端口，默认值为
`20669`。

## 前置条件

- Docker、Docker Compose 和 NVIDIA Container Toolkit
- 8 张可用 NVIDIA GPU
- 本地可拉取或已存在 `vllm/vllm-openai:stepfun37` 镜像
- 宿主机上已完整下载 `Step-3.7-Flash` 模型

## 配置环境变量

```sh
cp .env.example .env
```

编辑 `.env`：

- `VLLM_API_KEY`：API Bearer Token
- `PORT`：vLLM API 监听端口
- `MODEL_PATH`：宿主机上的 `Step-3.7-Flash` 模型目录

Hugging Face Cache 使用 Docker named volume `vllm_cache`，不需要配置宿主机
路径。`.env` 包含敏感信息，不要提交到版本库。

host 网络不会进行端口映射；启动前应确认 `PORT` 未被宿主机上的其他服务占用。

## 启动

```sh
docker compose up -d
docker compose logs -f vllm
```

当前配置使用 TP=8、Expert Parallel、Chunked Prefill、Prefix Cache、异步调度
和 3-token MTP 推测解码。最大上下文为 262144，健康检查预留约 17 分钟用于
模型加载和 KV Cache Profiling。

## 验证

```sh
docker compose ps
curl http://localhost:20669/health
curl http://localhost:20669/v1/models \
  -H "Authorization: Bearer <VLLM_API_KEY>"
```

API 模型名为 `step3.7-flash`。

## 停止

```sh
docker compose down
```

如需同时删除 Hugging Face Cache named volume，应先确认缓存不再需要，再执行：

```sh
docker compose down -v
```
