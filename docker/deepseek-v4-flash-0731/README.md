# DeepSeek V4 Flash 0731

在单机 8 卡 RTX PRO 6000 Blackwell 环境中部署 DeepSeek V4 Flash 0731。
本目录提供互斥的两种运行方案：

- `compose.vllm.yaml`：vLLM，TP=4、DP=2，并启用 Expert Parallel
- `compose.sglang.yaml`：SGLang，TP=8

两种方案使用相同的容器名和 host 网络端口，不能同时启动。OpenAI 兼容 API
通过 `.env` 中的 `PORT` 直接监听宿主机端口，默认值为 `20669`。

## 前置条件

- Docker、Docker Compose 和 NVIDIA Container Toolkit
- 8 张可用 NVIDIA GPU
- 模型文件已完整下载
- 启动前阅读两个 Compose 文件中的硬件兼容性注释

当前配置针对工作站 Blackwell 架构，并包含尚未被上游完整验证的组合。上线前
应先进行正确性、稳定性和长上下文压力测试。

## 配置环境变量

```sh
cp .env.example .env
```

编辑 `.env`：

- `API_KEY`：两种服务共用的 API Bearer Token
- `PORT`：vLLM 或 SGLang API 监听端口
- `MODEL_PATH`：宿主机上的 `DeepSeek-V4-Flash-0731` 模型目录
- `HF_CACHE_PATH`：宿主机 Hugging Face 缓存目录
- `SGLANG_EXTRA_METRIC_LABELS`：SGLang 指标附加标签 JSON

`.env` 包含敏感信息和机器信息，不要提交到版本库。

host 网络不会进行端口映射；启动前应确认 `PORT` 未被宿主机上的其他服务占用。

## 使用 vLLM

```sh
docker compose --env-file .env -f compose.vllm.yaml up -d
docker compose -f compose.vllm.yaml logs -f vllm-deepseek-v4-flash
```

当前 vLLM 配置使用 TP=4、DP=2、Expert Parallel、FP8 KV Cache、Marlin MoE，
最大上下文为 400000。DSpark 推测解码因已知兼容性问题暂未启用。

## 使用 SGLang

```sh
docker compose --env-file .env -f compose.sglang.yaml up -d
docker compose -f compose.sglang.yaml logs -f sglang-deepseek-v4-flash
```

当前 SGLang 配置使用 TP=8、FP8 E4M3 KV Cache 和 FlashInfer MXFP4 MoE，
最大上下文为 262144，并暴露运行指标。

## 验证

模型加载和编译的健康检查宽限期为 30 分钟。服务就绪后执行：

```sh
curl http://localhost:20669/health
curl http://localhost:20669/v1/models \
  -H "Authorization: Bearer <API_KEY>"
```

API 模型名为 `deepseek-v4-flash-0731`。

## 切换方案或停止

切换前必须先停止当前方案：

```sh
docker compose -f compose.vllm.yaml down
# 或
docker compose -f compose.sglang.yaml down
```
