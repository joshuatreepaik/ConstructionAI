"""Side-by-side comparison of the v1 (original) and v2 (staged) pipelines.

Runs the known queries through both engines via the Flask test client and
prints counts + timing. v1 is untouched and serves as the control.

Usage: .venv/bin/python scripts/compare.py
"""

import sys
import time

sys.path.insert(0, ".")

QUERIES = [
    ("E4 receptacle", "/detect",
     {"page": 25, "box": [431, 346, 445, 362], "any_size": True, "veto_text": True}),
    ("E4 door", "/detect",
     {"page": 25, "box": [342, 324, 382, 372], "any_size": True, "veto_text": True}),
    ("E4 legend auto-count", "/legend_count", {"page": 25}),
]


def run(client, url, payload, engine):
    p = dict(payload)
    if engine == "v2":
        p["engine"] = "v2"
    t0 = time.time()
    j = client.post(url, json=p).get_json()
    elapsed = time.time() - t0
    return j, elapsed


def summarize(name, j):
    if j is None or j.get("error"):
        return f"ERROR: {j.get('error') if j else 'no response'}"
    if "results" in j:                       # legend count
        top = ", ".join(f"{r['label'][:24]}={r['count']}" for r in j["results"][:5])
        return f"{j['n_matched']} classes | top: {top}"
    parts = [f"count={j['count']}"]
    if j.get("review_count") is not None:
        parts.append(f"review={j['review_count']}")
    if j.get("by_scale"):
        parts.append("sizes=" + ",".join(f"{k}:{v}" for k, v in j["by_scale"].items()))
    return " ".join(parts)


def main():
    import app as appmod
    client = appmod.app.test_client()
    print(f"{'query':24} {'engine':6} {'time':>7}  summary")
    print("-" * 100)
    for name, url, payload in QUERIES:
        for engine in ("v1", "v2"):
            j, dt = run(client, url, payload, engine)
            print(f"{name:24} {engine:6} {dt:6.2f}s  {summarize(name, j)}")
        print()


if __name__ == "__main__":
    main()
