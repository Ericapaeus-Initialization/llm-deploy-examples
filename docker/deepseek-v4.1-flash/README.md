# DeepSeek V4.1 Flash — 8× PRO 6000D 84GB

面向 Linux x86-64、8×84GB 显存、512GB 主存。使用原版
`deepseek-ai/DeepSeek-V4.1-Flash` MXFP4/MXFP8 权重，不适用于社区 NVFP4 转换。

采用 SGLang + 固定版本 SM120 补丁和原生 FP8 Engram 卸载实现。
上游实测为四张 96GB PRO 6000；本目录改为 TP8/EP8，尚未在目标机器验证。
400000 是输入与输出合计的单请求 token 上限；50 个运行槽位不保证同时容纳
50 条 400K 请求。总 token 池先限制为 4200000，容量不足时仍会排队或重调度。

## 参数与内存

`config.yaml` 使用 SGLang 参数：`context-length: 400000`、
`max-running-requests: 50`（对应此前 vLLM max-num-seqs 的配置意图）、
`chunked-prefill-size: 2048`、`mem-fraction-static: 0.90`、DSpark block 5。
解码 CUDA Graph 初始只捕获小批次到 8，大批次使用 eager 路径以限制启动占用。

计算权重与 KV 留在 GPU；Engram 保持原始 FP8 精度，通过本地 NVMe 读取，
主存缓存预算为全机合计 128GiB，并非每卡 128GiB。512GB 主存剩余空间留给
加载、页缓存和运行时，不另设 CPU KV offload。NVMe 文件系统必须支持 direct I/O；
不要把模型挂在 NFS/对象存储文件系统。此配置的 TP8 和 50 槽位均为新增适配，
不能按上游四卡吞吐数字推算性能。

## 启动

需要 CUDA 13 兼容的 NVIDIA 驱动、Docker Compose、NVIDIA Container Toolkit。
模型目录应包含完整 48 个分片、索引、config 和 tokenizer/encoding 文件。
参考权重 revision 为 `fb2764a5cf321eaa5070ca8f9e892818f477c16d`。
启动器检查模型类型和缺失分片；不自动下载、修改或完整哈希校验权重。

```sh
cp .env.example .env
# 编辑 API_KEY、MODEL_PATH、CACHE_PATH；模型路径必须是本地完整原版目录
docker compose --env-file .env config --quiet
docker compose build
docker compose up -d
docker compose logs -f model-service
```

构建从固定提交获取上游源码，在固定 SGLang 镜像上编译 Engram adapter，
安装其 SM120 sparse-prefill 补丁，保留上游 LICENSE/NOTICE 于 `/opt/dsv41`。
不会使用上游写死四卡/最多32请求且自动下载权重的 boot.py。
健康检查有一小时加载宽限期；watchdog 只管理带本目录专用标签的容器。
host 网络监听 `.env` 的端口（默认 20670）。

## 验证与调优

```sh
curl http://localhost:20670/v1/chat/completions \
  -H 'Authorization: Bearer <API_KEY>' -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"What is 19 + 23? Reply only with the number."}],"chat_template_kwargs":{"thinking":false},"max_tokens":32,"temperature":0}'
```

预期结果 42。依次验证文本、单图片、工具调用，再用约390K输入+输出预算验证
400K请求。混合负载按并发1、8、16、32、50逐级测试，记录TTFT、吞吐、错误、
显存/主存和抢占；50路全部满400K需要2000万tokens，本配置不承诺这一容量。
多图片/视频没有沿用上游单图片结果作为可靠性证明。

若 KV 分配或工作区 OOM，先把 max-total-tokens 降至2000000，
再降低 max-running-requests 至8或16；保持400K上下文和2048分块。
调整后 `docker compose up -d --force-recreate model-service`。
若 TP8 出现SM120内核形状不支持，需针对该形状补丁验证，不能通过增大显存比例解决。
持续崩溃时 `docker compose down` 停止服务和watchdog后排查。

## 来源（2026-09-12核对）

- [上游固定源码](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000/tree/45f538a18569420721e00e37353f9a7e1af7e5da)
- [原版权重](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- [vLLM recipe，作为架构参考，本目录使用SGLang](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
