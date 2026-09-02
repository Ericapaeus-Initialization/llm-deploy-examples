# VLLM & LLMs

Copy the stack's `.env.example` to `.env`, fill in its deployment-specific
values, then pass that environment file when starting the stack:

```sh
docker compose --env-file docker/qwen3.6-35b-a3b-4090/.env -f docker/qwen3.6-35b-a3b-4090/compose.yaml up -d
docker compose --env-file docker/qwen3.8-27b-4090/.env -f docker/qwen3.8-27b-4090/compose.yaml up -d
docker compose --env-file docker/qwen3.8-flash-next-offload/.env -f docker/qwen3.8-flash-next-offload/compose.yaml up -d
docker compose --env-file docker/qwen3.8-27b-dgx-spark-duo/.env -f docker/qwen3.8-27b-dgx-spark-duo/compose.yaml up -d
```
