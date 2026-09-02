# Qwen3.8-27B on 2 x DGX Spark

使用 vLLM multiprocessing 后端在两台 DGX Spark 上运行 Qwen3.8-27B。
rank 0 节点提供 OpenAI 兼容 API，rank 1 节点作为无 HTTP 服务的 worker。
两台机器通过 RDMA/NCCL 通信。

## 前置条件

- 两台已安装 Docker、Docker Compose 和 NVIDIA Container Toolkit 的 DGX Spark
- 两台机器之间的 RDMA 网络和 `.env` 中的 `MASTER_PORT` 端口连通
- `/dev/infiniband` 可用
- 两台机器均能访问同一版本的模型文件

## 构建镜像

在两台机器的本目录中分别执行：

```sh
docker build -t vllm-openai-rdma:latest .
```

镜像基于官方 vLLM 镜像，并额外安装 RDMA、InfiniBand 和 OpenMPI 组件。

## 配置环境变量

```sh
cp .env.example .env
```

两台机器各自维护自己的 `.env`：

- `VLLM_API_KEY`：rank 0 API 使用的 Bearer Token
- `PORT`：rank 0 API 监听端口，默认 `10669`
- `MODEL_PATH`：本机 Qwen3.8-27B 模型目录
- `HF_CACHE_PATH`：本机 Hugging Face 缓存目录
- `HOST_IP`：当前节点用于分布式通信的本机 IP
- `MASTER_ADDR`：rank 0 节点的通信 IP；两台机器填写相同值
- `MASTER_PORT`：分布式 rendezvous 端口，默认 `29501`
- `SOCKET_IFNAME`：NCCL/Gloo 使用的网卡名称
- `NCCL_IB_HCA`：允许 NCCL 使用的 RDMA 设备列表
- `DCGM_EXPORTER_PORT`：DCGM Exporter 监听端口，默认 `9400`

`.env` 包含敏感信息和机器信息，不要提交到版本库。

## 配置 master 与 worker

当前 `compose.yaml` 默认是 master 配置：

- `--node-rank 0`
- 不启用 `--headless`
- 使用 HTTP `/health` 健康检查

在 worker 节点启动前，需要修改该节点的 `compose.yaml`：

1. 将 `--node-rank` 后面的值从 `0` 改为 `1`。
2. 取消 `--headless` 的注释。
3. 注释 HTTP healthcheck，启用文件中提供的进程 healthcheck。

如果不切换 worker 的健康检查，worker 因为没有 HTTP Server 会被判定为
unhealthy，并可能被 watchdog 反复重启。

## 启动顺序

先在 master 启动：

```sh
docker compose up -d
docker compose logs -f vllm-qwen3.8-27b
```

再在 worker 启动修改后的 Compose：

```sh
docker compose up -d
docker compose logs -f vllm-qwen3.8-27b
```

master API 地址为 `http://<master-ip>:<PORT>/v1`。DCGM Exporter 使用 host
网络，并通过 `DCGM_EXPORTER_PORT` 提供指标。启动前应确认三个端口均未被占用。

## 验证

在 master 上执行：

```sh
curl http://localhost:10669/health
curl http://localhost:10669/v1/models \
  -H "Authorization: Bearer <VLLM_API_KEY>"
```

两台机器都应检查容器日志，确认 NCCL 初始化完成且所有 rank 已加入。

## 停止

分别在 worker 和 master 上执行：

```sh
docker compose down
```
