# Starlight API — Troubleshooting

Common errors and how to fix them.

## Connection timeouts

**Symptom:** requests hang for 30 seconds and then fail with
`ConnectionTimeoutError`.

**Causes and fixes:**

1. **Firewall blocking port 443.** Corporate networks and some cloud
   security groups block outbound HTTPS. Allow outbound traffic on port
   443 for the host running your code.
2. **Proxy configuration.** If you are behind a proxy, set `HTTP_PROXY`
   and `HTTPS_PROXY` environment variables. The client honors them.
3. **Network instability.** Retry with exponential backoff: wait 1s, then
   2s, then 4s between attempts, and give up after five tries.

## Rate limit exceeded (HTTP 429)

**Symptom:** `RateLimitError` with status 429, usually after a burst of
requests.

The API allows 100 requests per minute per key. Space out your calls or
request a higher quota from the dashboard. The client's default retry
policy backs off and retries on 429 responses automatically; you can
disable that with `client = StarlightClient(retry=False)`.

## Query is slow

**Symptom:** queries take many seconds to return.

- Narrow the time range. Scanning months of data is slower than days.
- Add tag filters so fewer series are scanned.
- Use aggregations: returning 1-hour averages of a year of data is much
  faster than returning every raw point.
- Make sure you are querying series by ID, not re-searching by name on
  every request.

## Embedding model fails to download

The hybrid search feature downloads its embedding model on first use. If
the download fails, check your internet connection or set the model cache
directory with the `STARLIGHT_MODEL_DIR` environment variable. Search
falls back to keyword-only mode until the model is available.

## Unexpected results from Ask Starlight

Natural-language answers are only as good as the data indexed. Confirm the
series you expect are tagged correctly, and rephrase the question with
specific series names or tag values to steer the hybrid search.
