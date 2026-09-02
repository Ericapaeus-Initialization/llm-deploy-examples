# Qwen3.8-27B-FP8 on 8 x RTX 4090

使用 vLLM 在单机 8 张 RTX 4090、384 GB CPU 内存的服务器上部署官方
`Qwen/Qwen3.8-27B-FP8`。服务使用 host 网络。

## 配置取舍

- 官方 FP8 检查点约 30.9 GB，使用 TP=8 后无需 CPU 权重 Offload。
- 模型使用原生 262144 上下文，不启用 YaRN 扩展。
- KV Cache 使用 FP8，以降低长上下文显存占用。
- 8 卡总显存约 192 GB，足以容纳权重和原生上下文 KV Cache，不启用 CPU KV
  Offload；384 GB 内存用于模型加载、文件页缓存和多模态预处理余量。
- RTX 4090 没有 NVLink，因此关闭 vLLM 自定义 all-reduce。
- 启用 3-token MTP 推测解码；若压测发现吞吐下降或不稳定，可先关闭该项。

官方检查点采用 block-size 128 的细粒度 FP8，vLLM 会从模型配置中自动识别，
不需要额外指定 `--quantization`。应使用支持 Qwen3.8 的较新 vLLM 镜像。

## 前置条件

- Docker、Docker Compose 和 NVIDIA Container Toolkit
- 8 张 24 GB RTX 4090
- 384 GB CPU 内存
- 模型缓存目录中存在 `Qwen3.8-27B-FP8`
- API 端口未被占用

## 配置环境变量

```sh
cp .env.example .env
```

编辑 `.env`：

- `VLLM_API_KEY`：API Bearer Token
- `PORT`：vLLM API 监听端口，默认 `10669`
- `MODEL_CACHE_PATH`：宿主机模型缓存目录
- `CONFIG_PATH`：宿主机上本目录 `config.yaml` 的绝对路径

`.env` 包含敏感信息，不要提交到版本库。

## 启动

```sh
docker compose config --quiet
docker compose up -d
docker compose logs -f model-service
```

模型首次加载、CUDA Kernel 初始化和图捕获可能耗时较长，健康检查预留了
15 分钟冷启动时间。

## 验证

```sh
docker compose ps
curl http://localhost:10669/health
curl http://localhost:10669/v1/models \
  -H "Authorization: Bearer <VLLM_API_KEY>"
```

加载的权重为 `Qwen3.8-27B-FP8`，API 对外模型名为 `Qwen/Qwen3.8-27B`。

## 调优与故障排查

如果启动期间显存不足，按以下顺序降低压力：

1. 将 `max-num-seqs` 从 40 降到 32。
2. 将 `max-num-batched-tokens` 从 8192 降到 4096。
3. 将 `gpu-memory-utilization` 从 0.93 降到 0.90。
4. 临时移除 `speculative-config`，排除 MTP 图捕获占用。

当前配置不会预留大块 CPU Offload 内存。该服务器启用 128 GiB native KV
Offload 时，所有 TP rank 的 `cudaHostRegister` 均失败，并导致后续 warmup 抛出
异步 `CUDA error: invalid argument`，因此不要在此配置上重新启用。

## 停止

```sh
docker compose down
```

## 参考

- [Qwen/Qwen3.8-27B-FP8 模型卡](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
- [vLLM Qwen3.8-27B Recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
