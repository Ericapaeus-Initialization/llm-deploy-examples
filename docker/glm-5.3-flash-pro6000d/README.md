# GLM-5.3-Flash / 8 × RTX PRO 6000D

参考 [vLLM 官方 recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)
及其 [配置源文件](https://github.com/vllm-project/recipes/blob/main/models/zai-org/GLM-5.3-Flash.yaml)，
核对日期：2026-09-14。此配置是针对单机八卡的适配基线，尚未在目标 GPU 上实测。

## 硬件与镜像边界

- 按本仓库已有 PRO 6000D 配置的每卡 84 GB、八卡约 672 GB 规划；实际容量以 `nvidia-smi` 为准。
- 使用 `zai-org/GLM-5.3-Flash` 原生 FP8 权重，本地目录须含完整权重、config、tokenizer 和模板文件。
  官方估计权重约 306 GiB；TP8 均摊约 38.3 GiB/卡只是权重粗估，不含复制部分、KV、运行时和 CUDA graphs。
- 官方指定 `vllm/vllm-openai:glm53-flash`，目前不要用任意旧版或 `latest` 替换。
  recipe 标注 vLLM 0.29.0+，同时仍要求专用镜像；FlashInfer 排错要求为 0.6.18+。
- **GB200/B200 的 SM100 与 PRO 6000D 的 SM120 不能视为相同内核支持。**
  官方 recipe 未列 PRO 6000D 为已验证设备。
  [SM120 FP8 社区部署](https://github.com/krzychdre/GLM-5.3-Flash-sm120)
  使用修补后的镜像，不能据此保证官方镜像在本机开箱即用。`VLLM_IMAGE` 可替换为经验证的同接口 SM120 构建。
  这里没有自动引入第三方补丁，也没有关闭依赖版本检查。
- 宿主机需要 Linux、Docker Compose、NVIDIA Container Toolkit，以及兼容镜像 CUDA 的驱动。

## 配置及启动

在此目录执行：

```sh
cp .env.example .env
# 编辑 .env，设置 MODEL_PATH 和 VLLM_API_KEY。
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
nvidia-smi topo -m
nvidia-smi topo -p2p r
docker compose config --quiet
docker compose pull
```

可先检查镜像依赖及八卡可见性（这不等于验证模型内核）：

```sh
docker compose run --rm --no-deps --entrypoint python vllm -c \
 'import torch, vllm, flashinfer; print("vLLM", vllm.__version__, "FlashInfer", flashinfer.__version__, "CUDA", torch.version.cuda); print([(torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]); assert torch.cuda.device_count() == 8'
docker compose up -d
docker compose logs -f vllm
```

初次加载及编译可能需要数十分钟，健康检查预留一小时。
Docker 的 restart 策略会重启退出的进程，但不会仅因 unhealthy 自动重启。
host 网络直接监听 `.env` 中的端口。编译缓存保存在 named volume 中。

## 默认参数与调优

| 参数 | 配置及理由 |
| --- | --- |
| 权重 / KV | 原生 FP8 / FP8 KV，沿用官方 Blackwell 配置方向 |
| 并行 | TP8、单进程组，使用 GPU 0–7 |
| 上下文 | 409600（400 × 1024），为本配置目标；不直接申请原生 1M |
| 调度 | 最多 50 条序列、每轮 8192 token、chunked prefill |
| 显存预算 | 0.90，保留运行余量 |
| 解析 | `glm47` 工具调用、`glm45` reasoning、自动工具选择 |
| Frontend | Python |
| 推测解码 | 默认关闭；高并发基线先不启用；MTP5 需确认所选 SM120 内核支持，参见调研记录 |

400K 是单请求输入加输出的上限，50 条调度序列不代表能同时容纳 50 个满长请求。
这些是期望配置，尚未实测；已有 `.env` 需手动同步 `MAX_MODEL_LEN=409600` 和 `MAX_NUM_SEQS=50`。
以启动日志的 KV 容量和最大并发估计为准。显存不足先降低 `MAX_NUM_SEQS` 或
`MAX_MODEL_LEN`，再减小 `MAX_NUM_BATCHED_TOKENS`；不要先把显存利用率推到极限。

GPU 拓扑会影响 TP8 性能；不预设关闭 NCCL P2P，也不照搬 GB200 的 NVLink/NIXL/EP 设置。
遇到 `no kernel image`、SM120 不支持或 Sparse MLA 初始化失败，应先核对镜像内核和
FlashInfer 版本；增加显存或设置 `TORCH_CUDA_ARCH_LIST` 不能修复已经缺失的预编译内核。
只在确定是 CUDA graph 问题时临时添加 `--enforce-eager` 做定位。

## 验证 API

下面示例使用默认端口，请替换 Token（自定义端口也需同步修改）：

```sh
curl -fsS http://localhost:20671/v1/models \
  -H 'Authorization: Bearer <VLLM_API_KEY>'

curl -fsS http://localhost:20671/v1/chat/completions \
  -H 'Authorization: Bearer <VLLM_API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"用中文简述张量并行。"}],"max_tokens":4096,"temperature":1,"chat_template_kwargs":{"reasoning_effort":"low"}}'
```

模型始终开启 thinking，默认 effort 为 `max`，可以使用 `low`/`high`；
不要把普通回答前的 reasoning 或过短输出预算导致的截断误判为服务故障。
上线前还需验证流式、工具调用、长输入以及真实并发；启用 MTP 后重新验证输出与性能。

```sh
docker compose down
```

停止服务不会删除缓存卷。

## PRO 6000 系列成功案例调研

见 [调研记录](RESEARCH.md)。当前 Compose 保留官方镜像作为对照基线；
该历史镜像在 SM120 上已有首轮 forward 失败报告，不能将配置校验通过视为可运行。
实际部署需要验证官方镜像是否已修复，或使用经过本机验收的 SM120 构建。
