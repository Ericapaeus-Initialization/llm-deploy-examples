# PRO 6000 系列部署证据（2026-09-14）

结论：找到 SM120 / 96GB RTX PRO 6000 的可复用部署资料；未找到公开、可复现的
8×84GB PRO 6000D + GLM-5.3-Flash + 400K + 50 条满长并发验收记录。
同架构案例有参考价值，不能替代本机显存、拓扑、内核正确性和压力测试。

## 最接近现有 FP8 配置的案例

[chriswritescode-dev/glm-5.3-flash-sm120](https://github.com/chriswritescode-dev/glm-5.3-flash-sm120)
报告在四张 96GB PRO 6000 上使用原生 FP8、TP4、FP8 KV、MTP5、
`max-model-len=524288`、`max-num-seqs=10` 成功启动。
报告 KV 池 609172 token，只相当于约 1.16 个满长请求；不能解释为十条同时 512K。
没有把设置的上下文上限当作已完成同长度请求的证明。

公开镜像为 `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1`。
[Dockerfile](https://github.com/chriswritescode-dev/glm-5.3-flash-sm120/blob/main/Dockerfile)
固定了基础镜像摘要，并处理 NoPE MLA、page alignment 和 top-k 容量等问题。
其中会舍弃最低排名的一组候选 pool，给近期 tail 留空间，因此不是完全不改变计算的修补；
应验证长文检索与工具调用质量，不能直接称为无损。
原镜像 `pe_dim must be 64 for fp8_ds_mla` 的报错发生在加载权重后的首轮 forward。

作者说明 MTP5 配合 seqs=10 是为了使解码 token 数不超过 64（10×6=60）。
50×6=300 已超出其特化解码范围；不能保证该构建仍保有相同性能。
该限制是这个历史构建的说明，不能泛化成所有 vLLM 的硬限制。

## 另一条 FP8 vLLM 路线

[krzychdre/GLM-5.3-Flash-sm120](https://github.com/krzychdre/GLM-5.3-Flash-sm120)
提供 SM120 Dockerfile、探针、启动脚本和部署报告；目标为四张 96GB 卡、TP4、
262K、MTP4。其 README 报告约 172 tok/s 的短上下文单流结果以及约 1.43M-token KV 池。
这是修补依赖、KDA 和 MLA 路径的独立方案，不能混用上一方案的 page/block 参数。
它没有验证本次 TP8/400K/50 组合。

## 真正的超长输入记录

[tacos4me 512K 模型卡及探针](https://huggingface.co/tacos4me/GLM-5.3-Flash-NVFP4-FP8ATTN-512K)
报告两张 96GB 卡完成约 501K-token 的实际输入，并重复验证。
但这是额外量化的 NVFP4/FP8 checkpoint、压缩 KV 和专用补丁，512K 档只有单并发。
可借鉴其真实长输入、重启复测和检索正确性验收方法，不将其速度或 KV 占用用于原生 FP8 TP8 估算。

## 八卡高并发旁证（SGLang）

[GCP G4 实测及配置](https://shivajid.github.io/sglang-rtx-pro-6000/)
提供单机八张 96GB PRO 6000、FP8、TP8、修补版 SGLang 的结果。
GLM-5.3-Flash 的 256 客户端测试使用约 1K 输入 / 8K 输出，报告聚合输出约 2580 tok/s。
证明同类硬件可以承载高并发短请求，不能推出 400K 下的同等表现，也不是 vLLM 证据。

## 本仓库目标配置

```dotenv
MAX_MODEL_LEN=409600
MAX_NUM_SEQS=50
MAX_NUM_BATCHED_TOKENS=8192
GPU_MEMORY_UTILIZATION=0.90
```

保持 TP8、原生 FP8、FP8 KV、Python frontend，先关闭 MTP。
配置值已经同步；原有 `.env` 不会被自动覆盖。暂不把任一社区镜像设为已验证默认。

`max-num-seqs` 是每轮调度序列上限，不承诺请求全部驻留显存。
[vLLM 参数说明](https://docs.vllm.ai/en/latest/configuration/engine_args/)
50 条各 409600 token 合计 20480000 个逻辑 token（尚未考虑额外输出余量）。
MLA 的 KV 在 TP 下不一定按卡数等比分摊，不应用整机剩余显存直接除以单卡 bytes/token。
容量判断应使用目标构建启动日志中的全引擎逻辑 KV token 池：
若为 B，则不共享前缀的满长驻留数粗估上限为 floor(B/409600)，还需留运行余量。

验收分开进行：

1. 真实约 390K 输入并生成，检查早/中/晚位置检索与无重复退化；输入加输出不超 409600。
2. 50 个长短混合请求，记录运行/等待请求、KV 占用、抢占重算、TTFT 和 TPOT 的 P95/P99。
3. 多条 400K 请求从 1、2、4 逐档增加，确定满长并发边界。
4. 对通过的构建固定镜像摘要；启用 MTP 后重新做同样验收。

目前只有静态 Compose 校验，没有目标 GPU、镜像执行或性能验收结果。
