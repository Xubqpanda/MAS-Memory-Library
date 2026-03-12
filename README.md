# Agent-Evolving-Library
Evolving agents through feedback-driven memory consolidation.

🔗 Access the HLE data here: [Google Drive - HLE Data](https://drive.google.com/drive/folders/1zE4jMFMlpBA8FN741mQNmBYYF9VrFfkX?usp=drive_link)

## How to run the code:
```python
python experiments/run_experiment.py \
  --env    experiments/configs/envs/hle.yaml \
  --solver experiments/configs/solver/single_agent.yaml \
  --tool   experiments/configs/tool/default.yaml \
  --memory experiments/configs/memory/empty.yaml
```

## Run SearXNG (for tool search provider)

`experiments/configs/tool/default.yaml` uses `web_search_provider: searxng`, so start SearXNG first:

```bash
cd deploy/searxng
docker compose up -d
```

Quick check:

```bash
curl "http://127.0.0.1:8080/search?q=OpenAI&format=json" | head
```

If `results` is empty and `unresponsive_engines` shows timeouts, SearXNG is up but the machine cannot reach upstream search engines. In that case, set proxy env vars before startup (`SEARXNG_HTTP_PROXY/SEARXNG_HTTPS_PROXY/SEARXNG_ALL_PROXY/SEARXNG_NO_PROXY`), then restart compose.

Stop:

```bash
docker compose down
```
