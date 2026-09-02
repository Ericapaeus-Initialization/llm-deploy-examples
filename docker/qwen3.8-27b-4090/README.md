# Qwen3.8-27B on 8 x RTX 4090

使用 Docker Compose 在单机 8 卡环境中启动 Qwen3.8-27B，并通过独立的
LMCache 服务提供 KV Cache。所有服务使用 host 网络。

## 前置条件

- Docker、Docker Compose 和 NVIDIA Container Toolkit
- 8 张可用 NVIDIA GPU
- 模型目录中包含 `Qwen3.8-27B`
- 可用于 LMCache L1 Cache 的充足内存

## 配置环境变量

```sh
cp .env.example .env
```

编辑 `.env`：

- `VLLM_API_KEY`：API Bearer Token
- `PORT`：vLLM API 监听端口，默认 `10669`
- `LMCACHE_MP_PORT`：LMCache MP 数据通道端口，默认 `45555`
- `LMCACHE_HTTP_PORT`：LMCache HTTP 和健康检查端口，默认 `48888`
- `MODEL_CACHE_PATH`：宿主机模型缓存目录
- `CONFIG_PATH`：宿主机上 `config.yaml` 的绝对路径
- `LMCACHE_START_SCRIPT_PATH`：宿主机上 `start-lmcache.sh` 的绝对路径

`.env` 包含敏感信息，不要提交到版本库。

## 启动

```sh
docker compose up -d
docker compose logs -f lmcache-server model-service
```

Compose 会先启动 LMCache，vLLM 再通过 `LMCacheMPConnector` 连接它。相关端口
均由 `.env` 配置。host 网络不会进行端口映射，启动前应确认这些端口均未被占用：

- `10669`：vLLM OpenAI 兼容 API
- `45555`：LMCache MP 数据通道
- `48888`：LMCache HTTP/健康检查端口

## 验证

```sh
docker compose ps
curl http://localhost:10669/health
curl http://localhost:10669/v1/models \
  -H "Authorization: Bearer <VLLM_API_KEY>"
```

API 中使用的模型名为 `Qwen/Qwen-AgentWorld-35B-A3B`。LMCache 当前配置为
256 GiB L1 Cache。LMCache 与 vLLM 的连接端口已由 Compose 使用同一环境变量
传入，不需要手动同步修改两个配置文件。

## 停止

```sh
docker compose down
```
