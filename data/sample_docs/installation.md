# Starlight API — Installation Guide

Starlight is a hosted API for storing, querying, and visualizing
time-series data. This guide walks you through installing the official
client library and getting your first request working.

## Requirements

- Python 3.10 or newer
- pip 21 or newer
- An API key from the Starlight dashboard (Settings > API Keys)

## Installing the client

Install the Python client with pip:

```bash
pip install starlight-client
```

The client depends on `requests` and `pandas`, which pip installs
automatically. We recommend installing inside a virtual environment so the
package does not conflict with other projects.

## Verifying the installation

Run the following from a Python REPL or script:

```python
import starlight
print(starlight.__version__)
```

If you see a version number, the client is installed correctly. If you get
a `ModuleNotFoundError`, double-check that you activated the virtual
environment where you installed the package.

## Setting your API key

Set your key as an environment variable so it is not committed to source
control:

```bash
export STARLIGHT_API_KEY="your-key-here"
```

On Windows PowerShell use `$env:STARLIGHT_API_KEY="your-key-here"` instead.

## Common installation problems

- **pip not found** — Python's Scripts directory is missing from PATH.
  Reinstall Python and tick "Add Python to PATH" during setup.
- **SSL certificate errors** — usually caused by corporate proxies. Set
  `HTTPS_PROXY` and `HTTP_PROXY` environment variables to your proxy URL.
- **Version conflicts** — create a fresh virtual environment and reinstall
  the client with `pip install --upgrade starlight-client`.
