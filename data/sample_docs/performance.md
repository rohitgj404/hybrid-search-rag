# Starlight API — Performance Best Practices

Tips for keeping ingestion and queries fast and cheap.

## Batch your ingestion

Each API call has fixed overhead, so sending 10,000 points in one batch is
far cheaper than 10,000 single-point calls. The client accepts up to
10,000 points per ingest call and compresses the payload automatically.
Keep batches under the limit to avoid HTTP 413 errors.

## Use asynchronous writes

For high-throughput pipelines, use the async client and send batches
concurrently:

```python
import asyncio
from starlight import AsyncStarlightClient

async def main():
    client = AsyncStarlightClient()
    await client.points.ingest(series_id="latency", points=batch)

asyncio.run(main())
```

## Cache frequent queries

The dashboard and integrations often re-run the same query. Cache results
client-side for at least 60 seconds; the API returns a `Cache-Control`
header you can use to pick a TTL. For natural-language search, cache
embeddings for stable series names so repeated questions do not re-embed.

## Indexing for hybrid search

Hybrid search indexes your series automatically, but you can tune it:

- **Chunk size** — smaller chunks improve precision for short questions;
  larger chunks give the language model more context per answer. The
  default of 800 characters is a good starting point.
- **Reranking** — the cross-encoder reranker is the most expensive stage.
  Limit candidates with `top_k_hybrid` to keep latency predictable.

## Avoid scanning by name

Querying by series ID is O(1); searching by name triggers a scan. If your
workflow looks series up by name frequently, maintain your own name-to-ID
map at startup.

## Monitoring

The dashboard shows per-key latency percentiles (p50, p95, p99). If p99
degrades, the usual culprits are oversized time ranges, missing tag
filters, or client retries stacking under a rate limit.
