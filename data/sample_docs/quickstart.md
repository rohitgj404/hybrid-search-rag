# Starlight API — Quickstart

This page gets you from zero to your first data query in about two minutes.

## Creating a client

```python
from starlight import StarlightClient

client = StarlightClient()  # reads STARLIGHT_API_KEY from the environment
```

If your key is stored elsewhere, pass it explicitly:

```python
client = StarlightClient(api_key="your-key-here")
```

## Writing your first series

A *series* is a named collection of timestamped values. Create one with a
single call:

```python
series = client.series.create(name="server.cpu.usage", unit="percent")
print(series.id)
```

## Ingesting data points

Data points are submitted in batches for efficiency. Each point has a
timestamp and a float value:

```python
points = [
    {"timestamp": "2026-01-01T00:00:00Z", "value": 12.4},
    {"timestamp": "2026-01-01T01:00:00Z", "value": 15.1},
    {"timestamp": "2026-01-01T02:00:00Z", "value": 18.9},
]
client.points.ingest(series_id=series.id, points=points)
```

## Querying the data

The simplest query returns raw points between two timestamps:

```python
rows = client.points.query(
    series_id=series.id,
    start="2026-01-01T00:00:00Z",
    end="2026-01-02T00:00:00Z",
)
for row in rows:
    print(row.timestamp, row.value)
```

## Next steps

- Read the Searching guide for filters, aggregations, and hybrid search.
- Read the Troubleshooting guide for common errors like rate limits and
  connection timeouts.
- Read the Performance guide for batching and caching best practices.
