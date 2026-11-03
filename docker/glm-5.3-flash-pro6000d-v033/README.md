# GLM-5.3-Flash / 8 × RTX PRO 6000D — 官方 v0.33.0 试跑

在 [glm-5.3-flash-pro6000d](../glm-5.3-flash-pro6000d/README.md) 的基础上，把社区补丁镜像
`cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1` 换成官方
`vllm/vllm-openai:v0.33.0`（已固定 digest）。仅用于单节点（node70）试跑，**尚未在目标 GPU 上实测**。

## 为什么可以试

- vllm-project/vllm#55277 已合并，官方镜像原生支持 SM120 上 NoPE sparse-MLA，
  不再需要补丁镜像修 `pe_dim must be 64 for fp8_ds_mla`。
- v0.33.0 固定 flashinfer 0.7.2.post1（0.7.0 曾导致 MTP 接受率 0%，见 #59724）。
- vllm-project/vllm#54951 已合并，长上下文 indexer prefill 按 TP 分片。

## 与旧配置的差异

| 项目 | 旧配置 | 本配置 |
| --- | --- | --- |
| 镜像 | `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1` | `vllm/vllm-openai:v0.33.0@sha256:d825baf9…` |
| CUDA graph | 默认 | `VLLM_USE_BREAKABLE_CUDAGRAPH=0` + `--compilation-config '{"cudagraph_mode":"FULL"}'` |
| MTP | MTP5 | MTP5（同旧配置，与 `cudagraph_mode=FULL`、flashinfer 0.7.2.post1 的组合未经他人验证） |
| `MAX_NUM_BATCHED_TOKENS` | 16384 | 8192 |
| watchdog 标签 | `glm53-flash-autoheal` | `glm53-flash-v033-autoheal` |

- **CUDA graph：** vllm-project/vllm#64266 报告 GLM-5.3-Flash 在 SM120 + CUDA 13 上默认启动会在
  CUDA graph 捕获时 illegal memory access（`v0.32.1rc1`、`0.33.0rc1.dev174` 均复现，修复 PR 仍是 open）。
  上面两项是该 issue 给出的绕过方案，作者实测比打补丁版本慢约 10–16%。
- **批次预算：** 旧配置观察到 prefill chunk 单步约 2s，期间所有解码请求的 ITL 一起变成 2s。
  降到 8192 预计将尖峰缩短一半左右；若 TTFT 变差，可调回 16384 或试 4096 对比。
- **MTP：** 直接开启 MTP5 试跑。若启动失败、接受率掉到 0 或输出异常（reasoning 乱码、重复），
  删除 compose 中的 `--speculative-config` 及其 JSON 两行，先得到无 MTP 基线，再分别排查 MTP 与 CUDA graph。
- 同一节点上新旧栈使用同一端口和同一组 GPU，**不能同时运行**；compose 项目名取自目录名，缓存卷互相独立。

## 在 node70 上试跑

```sh
# 先停旧栈（node70 上的旧目录）
cd ../glm-5.3-flash-pro6000d && docker compose down

cd ../glm-5.3-flash-pro6000d-v033
cp .env.example .env
# 把旧目录 .env 里的 MODEL_PATH / VLLM_API_KEY / PORT 抄过来；其余保持 .env.example 的值
docker compose config --quiet
docker compose pull vllm
docker compose up -d
docker compose logs -f vllm
```

首次加载及编译可能需要数十分钟。若启动日志出现 `illegal memory access` 或
`CUBLAS_STATUS_INTERNAL_ERROR`，说明 #64266 的绕过方案没有生效，请贴出日志再定位。

## 验收

按 [RESEARCH.md](../glm-5.3-flash-pro6000d/RESEARCH.md) 的验收顺序，并与旧配置对比：

1. 启动日志：确认没有回退到别的 attention 后端，记录 KV 池 token 数和最大并发估计。
2. 约 390K 真实输入并生成，检查早/中/晚位置检索与无重复退化。
3. 50 条长短混合请求：对比 ITL P99、TTFT P99、`vllm:num_preemptions_total`、`vllm:kv_cache_usage_perc`。
4. 观察 ITL 1–2.5s 区间的 token 占比是否下降。
5. 对比 `vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_draft_tokens_total`
   （旧配置约 0.43–0.61）；接受率掉到 0 或输出异常时立即关闭 MTP。

留意风险：KDA 缓冲不在显存预估内（vllm-project/vllm#54775），`GPU_MEMORY_UTILIZATION` 保持 0.90 不要提高；
长会话 reasoning 退化（vllm-project/vllm#56868）与版本无关，升级不一定缓解。

## 回滚

```sh
docker compose down
cd ../glm-5.3-flash-pro6000d && docker compose up -d
```
