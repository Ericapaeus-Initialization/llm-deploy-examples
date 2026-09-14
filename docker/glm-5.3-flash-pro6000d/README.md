# GLM-5.3-Flash / 8 × RTX PRO 6000D

参考 [vLLM 官方 recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)
及其 [配置源文件](https://github.com/vllm-project/recipes/blob/main/models/zai-org/GLM-5.3-Flash.yaml)，
核对日期：2026-09-14。此配置是针对单机八卡的适配基线，尚未在目标 GPU 上实测。

## 硬件与镜像边界

- 按本仓库已有 PRO 6000D 配置的每卡 84 GB、八卡约 672 GB 规划；实际容量以 `nvidia-smi` 为准。
- 使用 `zai-org/GLM-5.3-Flash` 原生 FP8 权重，本地目录须含完整权重、config、tokenizer 和模板文件。
  官方估计权重约 306 GiB；TP8 均摊约 38.3 GiB/卡只是权重粗估，不含复制部分、KV、运行时和 CUDA graphs。
- 默认使用 `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1`（CUDA 13.0）。
  [补丁源码与案例](https://github.com/chriswritescode-dev/glm-5.3-flash-sm120)
  针对原始官方镜像的 `pe_dim must be 64 for fp8_ds_mla` 报错提供 NoPE MLA 适配。
  用户的 2026-09-14 启动日志已复现该错误，因此不再默认使用原始官方镜像。
- 该社区构建报告验证了 4×96GB PRO 6000；本配置 TP8/400K/50 尚未实测。
  补丁也调整稀疏注意力候选选择，需要验证长文检索正确性。
  保持 FP8 KV，按案例关闭 FlashInfer autotune；MTP 默认关闭。
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

## 已有部署迁移

已有 `.env` 会覆盖 Compose 默认值。不要重新复制 `.env.example` 覆盖模型路径和密钥。
在服务器的当前目录仅更新以下三项：

```dotenv
VLLM_IMAGE=cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1
MAX_MODEL_LEN=409600
MAX_NUM_SEQS=50
```

使用更新后的 `compose.yaml`（新增 `--no-enable-flashinfer-autotune`）执行：

```sh
docker compose config --quiet
docker compose config --images
docker compose pull vllm
docker compose up -d --force-recreate vllm
docker compose logs -f vllm
```

`config --images` 应显示上面的补丁镜像；无需删除模型或缓存卷。
`up --force-recreate` 会替换当前服务容器。恢复服务后用 `/v1/models` 和实际生成请求验收。

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

见 [调研记录](RESEARCH.md)。当前使用针对已复现错误的社区补丁构建，
配置静态校验不代表八卡启动、400K 请求或 50 条并发已通过验收。
