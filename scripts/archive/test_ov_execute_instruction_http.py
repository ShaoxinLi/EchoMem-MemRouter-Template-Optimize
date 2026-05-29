#!/usr/bin/env python
"""HTTP smoke test for OpenViking /api/v1/search/execute_instruction.

Requires a running OpenViking server (default http://localhost:1933).
Set OV_SERVER_URL env var to override.
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx

OV_URL = os.environ.get("OV_SERVER_URL", "http://localhost:1933")
OV_API_KEY = os.environ.get("OV_API_KEY", "")
OV_ACCOUNT = os.environ.get("OV_ACCOUNT", "memrouter-e2e")
OV_USER = os.environ.get("OV_USER", "default")
OV_AGENT = os.environ.get("OV_AGENT", "shared")

TEST_PAYLOAD = {
    "query": "按我的偏好来处理这个任务",
    "search_mode": "find",
    "target_uri": "viking://memories/preferences",
    "context_type": "memory",
    "limit": 5,
    "skip_intent_analysis": True,
    "typed_query": {
        "query": "按我的偏好来处理这个任务",
        "context_type": "memory",
        "intent": "preference_profile",
        "priority": 1,
        "target_directories": None,
    },
}


def main():
    url = f"{OV_URL}/api/v1/search/execute_instruction"
    headers = {
        "X-OpenViking-Account": OV_ACCOUNT,
        "X-OpenViking-User": OV_USER,
        "X-OpenViking-Agent": OV_AGENT,
    }
    if OV_API_KEY:
        headers["X-API-Key"] = OV_API_KEY

    print(f"POST {url}")
    print(f"Payload:\n{json.dumps(TEST_PAYLOAD, indent=2, ensure_ascii=False)}")
    print("-" * 60)

    try:
        start = time.time()
        resp = httpx.post(url, json=TEST_PAYLOAD, headers=headers, timeout=30.0)
        latency_ms = (time.time() - start) * 1000
    except Exception as exc:
        print(f"FAIL: Request error: {exc}")
        sys.exit(1)

    print(f"Status: {resp.status_code}")
    print(f"Latency: {latency_ms:.1f}ms")
    print("-" * 60)

    try:
        data = resp.json()
    except Exception:
        print(f"Response (raw):\n{resp.text}")
        sys.exit(1)

    print(f"Response:\n{json.dumps(data, indent=2, ensure_ascii=False)}")
    print("-" * 60)

    if resp.status_code != 200:
        print(f"FAIL: HTTP {resp.status_code}")
        sys.exit(1)

    status = data.get("status")
    result = data.get("result", {})
    memories = result.get("memories", [])
    resources = result.get("resources", [])
    skills = result.get("skills", [])
    total = result.get("total", len(memories) + len(resources) + len(skills))

    print(f"Status field:    {status}")
    print(f"Memories:        {len(memories)}")
    print(f"Resources:       {len(resources)}")
    print(f"Skills:          {len(skills)}")
    print(f"Total:           {total}")

    if status == "ok":
        print("PASS: execute_instruction returned ok")
        sys.exit(0)
    else:
        print(f"FAIL: status={status}")
        sys.exit(1)


if __name__ == "__main__":
    main()
