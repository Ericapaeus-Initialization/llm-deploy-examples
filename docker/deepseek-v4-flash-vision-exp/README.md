# DeepSeek V4 Flash Vision Exp

在单机 8 张 RTX PRO 6000 Blackwell（每卡按 84GB 可用显存计算）、512GB
主存的环境中，以 vLLM 提供 OpenAI 兼容的多模态 API。

## 配置取舍

- 使用官方模型专用镜像 `vllm/vllm-openai:deepseekv4-flash-vision`。模型支持虽已于
  2026-09-02 合入 vLLM `main`，截至本配置编写时仍未进入稳定版。
- 并行策略为 TP=4、DP=2、Expert Parallel。每个 TP4 副本约需 202GB 显存预算，
  可装入 4×84GB；两个副本使用全部 8 张卡，并避免工作站 PCIe 拓扑上的 TP8。
- 默认最大上下文为 524,288，可使用 Think Max。模型宣称支持 1,048,576，
  但官方 recipe 只在 GB200 上以 32K 做过基准；不要未经压测直接改到 1M。
- GPU KV cache 使用 FP8；另从 512GB 主存中预留 256GiB，交给
  `SimpleCPUOffloadConnector` 保存可复用的 prefix KV blocks。它依赖 prefix caching，
  且不是 GPU 显存的同步扩展，未命中的请求仍受 GPU KV 容量和 PCIe 带宽约束。
- DSpark 使用 checkpoint 内置 draft module，按官方已测参数生成 3 个 draft tokens。
- 每个请求最多 8 张图片，不接受视频；本地图片只能从只读 `/media` 目录读取。
- 推理参数集中在 `config.yaml`；`watchdog` 使用 autoheal，在健康检查持续失败时
  自动重启模型容器。

模型约 168GB（48 个 shard），建议模型目录位于本地 NVMe。启动和首次 JIT 编译可能
超过 20 分钟，健康检查预留了 30 分钟。

## 启动

```sh
cp .env.example .env
```

编辑 `.env`：

- `API_KEY`：API Bearer Token。
- `PORT`：宿主机监听端口，默认示例为 `20669`。
- `MODEL_PATH`：已完整下载的模型目录。
- `HF_CACHE_PATH`：Hugging Face/JIT 缓存目录，必须可写并有充足空间。
- `MEDIA_PATH`：允许客户端通过 `file:///media/...` 读取的宿主机目录。
- `CONFIG_PATH`：宿主机上本目录 `config.yaml` 的绝对路径。

启动并查看日志：

```sh
docker compose --env-file .env up -d
docker compose logs -f vllm-deepseek-v4-flash-vision
```

修改推理参数时编辑 `config.yaml`，然后重建模型容器：

```sh
docker compose up -d --force-recreate vllm-deepseek-v4-flash-vision
```

## 验证

先检查服务和模型列表：

```sh
curl http://localhost:20669/health
curl http://localhost:20669/v1/models \
  -H "Authorization: Bearer <API_KEY>"
```

再用 HTTPS 图片验证真正的视觉路径，而不只是文本路径：

```sh
curl http://localhost:20669/v1/chat/completions \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash-vision-exp",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}},
        {"type": "text", "text": "描述这张图片。"}
      ]
    }],
    "max_tokens": 512
  }'
```

Think Max 通过请求中的 `chat_template_kwargs` 开启；推荐采样参数为
`temperature=1.0`、`top_p=0.95`：

```json
{
  "temperature": 1.0,
  "top_p": 0.95,
  "chat_template_kwargs": {
    "thinking": true,
    "reasoning_effort": "max"
  }
}
```

## RTX PRO 6000 / SM120 注意事项

该模型在 SM120 上的图像 prefill 依赖 FlashInfer 的 DSV4 dual-cache sparse MLA
kernel。若文本请求正常、第一条图片请求却报以下错误：

```text
Unsupported sparse-MLA prefill configuration
```

说明镜像内的 FlashInfer 尚未包含相应 SM120 dispatch 修复。不要把服务误判为健康；
应换用包含 FlashInfer sparse-MLA runtime-topk 修复的更新版官方镜像后重新验证。
同时确认容器内 `nvcc` 可在 `PATH` 中找到，否则 JIT kernel 不会正确构建。

若显存不足，依次降低 `--max-num-seqs`、`--max-num-batched-tokens`，再把
`--max-model-len` 降至 `393216`（Think Max 下限）或 `131072`。不要优先提高
`--gpu-memory-utilization`，工作站显示/驱动和 JIT 编译都需要显存余量。

停止服务：

```sh
docker compose down
```

## 参考

- [DeepSeek 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)
- [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)
- [vLLM 支持 PR #54566](https://github.com/vllm-project/vllm/pull/54566)
- [FlashInfer SM120 dual-cache prefill PR #4850](https://github.com/flashinfer-ai/flashinfer/pull/4850)
