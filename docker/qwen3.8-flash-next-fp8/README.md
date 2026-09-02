# Qwen3.8-Flash-Next FP8

此目录目前是 FP8 部署占位目录，`compose.yaml` 和 `config.yaml` 均为空，暂时
不能直接启动服务。

当前 `.env` 和 `.env.example` 不需要配置任何变量。后续补充部署配置时，应当：

1. 在 `config.yaml` 中添加模型与推理参数。
2. 在 `compose.yaml` 中添加 vLLM 服务、GPU、挂载和健康检查配置。
3. 将 API Key、宿主机路径和机器相关参数放入本目录 `.env`。
4. 在 `.env.example` 中提供不含真实敏感值的同名变量。
5. 使用以下命令验证 Compose 配置：

```sh
docker compose --env-file .env config --quiet
```
