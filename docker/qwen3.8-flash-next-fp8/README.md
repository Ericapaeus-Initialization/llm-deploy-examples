# Qwen3.8-Flash-Next-FP8 on 8 x RTX PRO 6000D

面向单机 8 张 RTX PRO 6000D（每卡 84 GB）和 1 TB CPU 内存的保守生产基线。
模型是 Qwen4 架构预览版本，仍有上游问题在持续修复，因此“配置可解析”不等于
“已经完成生产验收”；上线前必须执行本文的验证和灰度流程。

## 已验证依据与配置选择

- 使用官方要求的专用镜像 `vllm/vllm-openai:qwen38-flash-next`，并固定到已在
  RTX PRO 6000 Blackwell（SM120）案例中报告的镜像 digest。
- 官方 FP8 检查点约 172.78 GiB；8 卡总显存约 672 GB，容量充足。
- 8 卡 FP8 必须使用 TEP8，即 TP=8 加 Expert Parallel；普通 TP8 与 128-wide
  FP8 量化块不兼容。
- 使用官方推荐的 Triton MoE 后端，并将 GPU 内存利用率设置为 0.93。
- KV Cache 保持 BF16/auto。QSA 的 FP8 KV 支持仍是开放 RFC，不作为生产默认。
- 使用 YaRN 将上下文扩展到 400000，并将默认 reasoning effort 设置为 low。
- 启用 MTP2；上线前必须单独验证接受率、结果正确性和高并发稳定性。
- 启用 PLE CPU Offload，将 51B N-gram Embedding 放入 1 TB 主机内存。
- 并发上限为 64、批 Token 上限为 16384；混合长短请求场景必须重点压测 PLE
  Prefill 的瞬时显存占用。
- RTX PRO 6000D 为 PCIe 拓扑且无 NVLink，因此关闭自定义 all-reduce。

## 前置条件

- Linux x86_64
- Docker、Docker Compose 和 NVIDIA Container Toolkit
- 支持 CUDA 13 的 NVIDIA Driver
- 8 张 RTX PRO 6000D 均为空闲且状态正常
- 本地模型目录包含完整的 `Qwen3.8-Flash-Next-FP8`
- API 端口未被占用

先检查硬件和拓扑：

```sh
nvidia-smi
nvidia-smi topo -m
```

## 配置环境变量

```sh
cp .env.example .env
```

编辑 `.env`：

- `VLLM_IMAGE`：已固定 digest 的官方专用镜像
- `VLLM_API_KEY`：API Bearer Token
- `PORT`：API 监听端口，默认 `10669`
- `MODEL_PATH`：宿主机模型目录
- `CONFIG_PATH`：宿主机上本目录 `config.yaml` 的绝对路径
- `VLLM_CACHE_PATH`：持久化 AOT/编译缓存目录

创建缓存目录并确保 Docker 可写：

```sh
mkdir -p "$(sed -n 's/^VLLM_CACHE_PATH=//p' .env)"
```

`.env` 包含敏感信息，不要提交到版本库。

PLE Offload worker 通过 PyTorch CUDA IPC 交换张量，需要执行 `pidfd_getfd`。
为先跑通新模型，Compose 当前启用了 `privileged: true`，同时绕过 capability
和默认 seccomp 对该系统调用的限制。完成稳定性验证后，应回收权限并测试能否仅
保留 `SYS_PTRACE` capability。

## 启动

```sh
docker compose config --quiet
docker compose pull
docker compose up -d
docker compose logs -f model-service
```

首次启动会加载约 173 GiB 权重并执行 AOT/CUDA 编译，健康检查预留了 30 分钟。
编译缓存通过 `VLLM_CACHE_PATH` 持久化。

### 首次编译后的受控重启

当前专用镜像有案例显示，冷 AOT 编译峰值可能被计入 KV Cache 预算。首次启动
完全就绪且缓存落盘后，建议在维护窗口执行一次受控重启，再记录最终 KV Cache
容量：

```sh
docker compose restart model-service
docker compose logs -f model-service
```

不要在服务仍在编译或加载权重时重启。

## 上线前验证

### 1. 基础健康检查

```sh
docker compose ps
curl http://localhost:10669/health
curl http://localhost:10669/v1/models \
  -H "Authorization: Bearer <VLLM_API_KEY>"
```

API 模型名应为 `Qwen/Qwen3.8-Flash-Next-FP8`。

### 2. 正确性 Smoke Test

发送以下问题并人工确认回答同时解释 GDN 的压缩历史能力和 QSA 的稀疏检索：

> Explain how Gated DeltaNet and Qwen Sparse Attention complement each other.

同时验证：

- thinking 和 non-thinking 请求
- 带工具调用的请求
- 图片输入
- 8K、32K、128K、262K 和 400K 分档长上下文
- Prefix Cache 命中与未命中结果一致

### 3. 混合长度压力测试

必须覆盖“一个超长新请求 + 多个短续写请求”的流量形态。逐步从 4、8、16、32
提升到 64 并发，持续观察：

- CUDA OOM、worker 重启和 NCCL 错误
- 首 Token 延迟、解码吞吐和尾延迟
- GPU 显存余量与 KV Cache 使用率
- 工具调用结构化结果和长上下文正确性

如果出现 PLE Prefill OOM，依次将 `max-num-batched-tokens` 从 16384 降至 8192、
将 `max-num-seqs` 从 64 降至 32、将 `gpu-memory-utilization` 从 0.93 降至
0.85；仍不稳定时关闭 MTP，并回退到原生 262144 上下文。

## 暂不启用的优化

- `kv-cache-dtype: fp8`：QSA FP8 KV 路径尚未成为稳定默认。
- MTP3：当前只启用 MTP2，增加 draft token 前应独立测试接受率和稳定性。
- 1M YaRN：当前只扩展至 400K；静态 YaRN 可能影响短上下文质量。
- LMCache/KV Offload：新混合架构应先完成单机原生 Cache 正确性验证。

## 灰度发布

生产切流建议按 1% → 10% → 50% → 100% 逐级进行，每一级至少覆盖一次峰值
时段。出现结果异常、OOM 或 worker 重启时立即回退，不应依赖 watchdog 掩盖
持续性崩溃。

## 停止

```sh
docker compose down
```

## 参考资料

- [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
- [vLLM 官方 Recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)
- [SM120 冷 AOT 容量问题](https://github.com/vllm-project/vllm/issues/54122)
- [混合长度 PLE Prefill OOM](https://github.com/vllm-project/vllm/issues/54764)
- [QSA FP8 KV RFC](https://github.com/vllm-project/vllm/issues/54426)
- [PLE Offload Docker CUDA IPC 权限案例](https://github.com/jschmied/qwen38-flash-next-gb10)
