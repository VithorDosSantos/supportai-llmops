"""Benchmark de latência do POST /classify (p50/p95/p99 e throughput).

Sobe a API com os campeões (registry ou diretório exportado, conforme as
settings) num subprocesso uvicorn, aquece e dispara tickets reais do split de
teste. Mede latência do lado do cliente (inclui HTTP + validação + inferência).

Uso:
    uv run python scripts/benchmark_latency.py [--requests 1000] [--concurrency 1 4]
    uv run python scripts/benchmark_latency.py --url http://localhost:8000  # API já rodando
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import httpx
import numpy as np
import pandas as pd

from supportai.config import PROJECT_ROOT
from supportai.data.config import load_data_config

REPORTS_DIR = PROJECT_ROOT / "reports"


@contextmanager
def local_server(port: int, workers: int) -> Iterator[str]:
    env = os.environ | {"API_ALLOW_MODEL_DESERIALIZATION": "true", "API_LOG_JSON": "true"}
    cmd = [
        sys.executable, "-m", "uvicorn", "--factory", "supportai.api.main:create_app",
        "--port", str(port), "--log-level", "warning", "--workers", str(workers),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.5)
        else:
            raise TimeoutError("API não subiu em 120 s")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def run_load(url: str, texts: list[str], concurrency: int) -> dict[str, float]:
    def one(text: str, client: httpx.Client) -> float:
        start = time.perf_counter()
        response = client.post(f"{url}/classify", json={"text": text})
        response.raise_for_status()
        return time.perf_counter() - start

    with httpx.Client(timeout=30) as client:
        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            latencies = np.array(list(pool.map(lambda t: one(t, client), texts)))
        wall = time.perf_counter() - wall_start

    ms = latencies * 1000
    return {
        "requests": len(texts),
        "concurrency": concurrency,
        "p50_ms": round(float(np.percentile(ms, 50)), 2),
        "p95_ms": round(float(np.percentile(ms, 95)), 2),
        "p99_ms": round(float(np.percentile(ms, 99)), 2),
        "mean_ms": round(float(ms.mean()), 2),
        "throughput_rps": round(len(texts) / wall, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="API já em execução (padrão: sobe uma local)")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--workers", type=int, default=1, help="processos uvicorn (só para a API local)"
    )
    args = parser.parse_args()

    test = pd.read_parquet(load_data_config().paths.resolve().processed_dir / "test.parquet")
    texts = test["text"].sample(n=args.requests, replace=True, random_state=42).tolist()
    text_chars = [len(t) for t in texts]

    with local_server(args.port, args.workers) if args.url is None else _given(args.url) as url:
        run_load(url, texts[: args.warmup], 1)
        results = [run_load(url, texts, c) for c in args.concurrency]
        models = httpx.get(f"{url}/health").json()["models"]

    report = {
        "results": results,
        "models": models,
        "workers": args.workers if args.url is None else None,
        "text_chars_mean": round(float(np.mean(text_chars)), 1),
        "machine": {
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "command": "uv run python " + " ".join(sys.argv),
    }
    for r in results:
        print(
            f"c={r['concurrency']}: p50={r['p50_ms']} ms  p95={r['p95_ms']} ms  "
            f"p99={r['p99_ms']} ms  {r['throughput_rps']} req/s"
        )
    out = REPORTS_DIR / f"benchmark_workers{args.workers}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


@contextmanager
def _given(url: str) -> Iterator[str]:
    yield url


if __name__ == "__main__":
    main()
