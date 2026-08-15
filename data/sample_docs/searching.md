# Starlight API — Searching and Querying

The Starlight query engine supports exact filters, aggregations, and a
hybrid search mode that combines semantic vector search with keyword
retrieval for natural-language questions.

## Filtering points

Every query accepts time range and tag filters. Tags are key/value pairs
attached to series at creation time:

```python
rows = client.points.query(
    series_id="cpu.usage",
    start="2026-01-01T00:00:00Z",
    end="2026-01-08T00:00:00Z",
    tags={"region": "us-east-1"},
)
```

## Aggregations

Aggregate with a window size and one of `avg`, `min`, `max`, `sum`, or
`count`:

```python
rows = client.points.query(
    series_id="cpu.usage",
    start=...,
    end=...,
    aggregation={"window": "1h", "function": "avg"},
)
```

## Hybrid natural-language search

Ask questions in plain English and the engine combines two retrievers:

1. **Semantic vector search** — embeddings capture meaning, so
   "how full are my disks" matches series about disk utilization even when
   the words do not overlap.
2. **Keyword retrieval (BM25)** — exact-term matching finds series by name,
   tag values, and identifiers, e.g. `cpu.usage`, `region=eu-west-2`, or an
   error code.

Results from both retrievers are merged with Reciprocal Rank Fusion and
then reranked by a cross-encoder model that reads the query and each
candidate together. The ranked list drives the natural-language answers in
the dashboard's Ask Starlight feature.

```python
results = client.search.query(
    "which services had the most connection timeout errors last week?",
    limit=10,
)
```

## Pagination and rate limits

Queries return at most 1,000 points per page. Use the cursor returned in
the response to fetch the next page. All endpoints are rate limited to 100
requests per minute per API key; see the Troubleshooting guide for what
happens when you exceed that.
