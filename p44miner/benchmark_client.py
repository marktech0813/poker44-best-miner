"""Download and cache the public Poker44 training benchmark.

API (verified live): https://api.poker44.net/api/v1/benchmark
    GET /                                      -> status / availability
    GET /releases?limit=&before=YYYY-MM-DD     -> available release dates
    GET /chunks?sourceDate=&limit=&cursor=&split= -> chunk payloads + labels

Responses use a ``{"success": true, "data": {...}}`` envelope.

Each ``/chunks`` response contains ``data.chunks`` = a list of chunk-response objects.
Each chunk-response object contains:
    chunkHash, sourceDate, releaseVersion, schemaVersion, split,
    chunks          -> list of groups, each group is a list of hand dicts
    groundTruth     -> list of 0/1 labels, one per group (1 = bot, 0 = human)

We cache each chunk-response verbatim to ``cache_dir/<chunkHash>.json`` so experiments
are reproducible and re-runs are offline/fast.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import requests

DEFAULT_BASE_URL = "https://api.poker44.net/api/v1/benchmark"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmark_cache"


class BenchmarkClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        timeout: float = 60.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = session or requests.Session()

    # ---- low-level GET with envelope unwrap + light retry ----
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                body = resp.json()
                if isinstance(body, dict) and "data" in body:
                    return body["data"]
                return body
            except Exception as exc:  # noqa: BLE001 - network resilience
                last_exc = exc
                time.sleep(0.75 * attempt)
        raise RuntimeError(f"GET {url} failed after retries: {last_exc}")

    # ---- discovery ----
    def status(self) -> Dict[str, Any]:
        return self._get("")

    def list_releases(self, max_releases: int = 400) -> List[Dict[str, Any]]:
        """Return full release objects (incl. ``audit``), newest first."""
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        before: Optional[str] = None
        while len(out) < max_releases:
            params: Dict[str, Any] = {"limit": 100}
            if before:
                params["before"] = before
            data = self._get("/releases", params=params)
            releases = data.get("releases") or []
            if not releases:
                break
            page_dates = [r.get("sourceDate") for r in releases if r.get("sourceDate")]
            fresh = [r for r in releases if r.get("sourceDate") and r["sourceDate"] not in seen]
            if not fresh:
                break
            for r in fresh:
                seen.add(r["sourceDate"])
                out.append(r)
            before = min(page_dates)
            if len(page_dates) < 100:
                break
        return out[:max_releases]

    def list_release_dates(self, max_releases: int = 400) -> List[str]:
        """Return all available source dates, newest first, walking the cursor."""
        dates: List[str] = []
        before: Optional[str] = None
        while len(dates) < max_releases:
            params: Dict[str, Any] = {"limit": 100}
            if before:
                params["before"] = before
            data = self._get("/releases", params=params)
            releases = data.get("releases") or []
            if not releases:
                break
            page_dates = [r.get("sourceDate") for r in releases if r.get("sourceDate")]
            new_dates = [d for d in page_dates if d not in dates]
            if not new_dates:
                break
            dates.extend(new_dates)
            before = min(page_dates)  # walk further back in time
            if len(page_dates) < 100:
                break
        return dates[:max_releases]

    # ---- chunk download (paginated) ----
    def fetch_release_chunks(
        self,
        source_date: str,
        split: Optional[str] = None,
        per_page: int = 24,
    ) -> List[Dict[str, Any]]:
        """Download every chunk-response object for one source date (paginated)."""
        out: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            params: Dict[str, Any] = {"sourceDate": source_date, "limit": per_page}
            if split:
                params["split"] = split
            if cursor:
                params["cursor"] = cursor
            data = self._get("/chunks", params=params)
            chunk_objs = data.get("chunks") or []
            out.extend(chunk_objs)
            cursor = data.get("nextCursor")
            if not cursor or not chunk_objs:
                break
        return out

    def download_all(
        self,
        max_releases: int = 400,
        refresh: bool = False,
    ) -> List[Path]:
        """Download + cache all releases. Returns the list of cached file paths."""
        paths: List[Path] = []
        dates = self.list_release_dates(max_releases=max_releases)
        print(f"[benchmark] discovered {len(dates)} release date(s)")
        for date in dates:
            try:
                chunk_objs = self.fetch_release_chunks(date)
            except Exception as exc:  # noqa: BLE001
                print(f"[benchmark] WARN: failed to fetch {date}: {exc}")
                continue
            for obj in chunk_objs:
                chunk_hash = obj.get("chunkHash") or obj.get("chunkId")
                if not chunk_hash:
                    continue
                path = self.cache_dir / f"{chunk_hash}.json"
                if path.exists() and not refresh:
                    paths.append(path)
                    continue
                path.write_text(json.dumps(obj), encoding="utf-8")
                paths.append(path)
            print(f"[benchmark] cached {date}: {len(chunk_objs)} chunk-response object(s)")
        print(f"[benchmark] total cached files: {len(paths)}")
        return paths

    # ---- cache iteration ----
    def iter_cached(self) -> Iterator[Dict[str, Any]]:
        for path in sorted(self.cache_dir.glob("*.json")):
            try:
                yield json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - skip corrupt cache entries
                continue


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download Poker44 benchmark releases.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--max-releases", type=int, default=400)
    parser.add_argument("--refresh", action="store_true", help="re-download cached files")
    args = parser.parse_args()

    client = BenchmarkClient(base_url=args.base_url, cache_dir=args.cache_dir)
    status = client.status()
    print(f"[benchmark] status: {json.dumps(status)[:300]}")
    client.download_all(max_releases=args.max_releases, refresh=args.refresh)


if __name__ == "__main__":
    main()
