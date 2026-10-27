"""Dump every chunk of rag-hybrid's ES index to data/index/ for the in-process backend.

    python -m scripts.export_index [--es http://127.0.0.1:9200] [--index rag-docs]

Local ES, read-only, no paid calls. Both chunkers are exported (the `chunker` field is kept):
BM25 statistics in ES are computed over all of them, so the in-process backend needs them all.
"""
import argparse
import json
from pathlib import Path

import httpx
import numpy as np

FIELDS = ["doc_id", "chunker", "url", "title", "heading_path", "content", "embedding"]
DOCS_COMMIT = "6c42e266fcc1f0e0edd4805baea904e1b3cbaf9c"  # rag-hybrid data/CORPUS.md
EMBED_MODEL = "text-embedding-3-small"  # rag-hybrid ingest/index.py default; the index stores no model name


def scan(es: str, index: str):
    """Yield every hit via scroll; _id is the chunk_id."""
    with httpx.Client(timeout=120) as c:
        r = c.post(f"{es}/{index}/_search?scroll=2m", json={"size": 500, "_source": FIELDS, "sort": ["_doc"]})
        while True:
            r.raise_for_status()
            body = r.json()
            if not body["hits"]["hits"]:
                break
            yield from body["hits"]["hits"]
            r = c.post(f"{es}/_search/scroll", json={"scroll": "2m", "scroll_id": body["_scroll_id"]})
        c.request("DELETE", f"{es}/_search/scroll", json={"scroll_id": body["_scroll_id"]})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--es", default="http://127.0.0.1:9200")
    ap.add_argument("--index", default="rag-docs")
    ap.add_argument("--out", default="data/index")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    vecs, per_chunker = [], {}
    with open(out / "chunks.jsonl", "w", encoding="utf8") as f:
        for h in scan(a.es, a.index):
            s = h["_source"]
            vecs.append(s["embedding"])
            per_chunker[s["chunker"]] = per_chunker.get(s["chunker"], 0) + 1
            row = {"chunk_id": h["_id"], **{k: s[k] for k in FIELDS if k != "embedding"}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    emb = np.asarray(vecs, dtype=np.float32)
    np.save(out / "embeddings.npy", emb)
    manifest = {"docs_commit": DOCS_COMMIT, "embed_model": EMBED_MODEL, "dims": int(emb.shape[1]),
                "count": int(emb.shape[0]), "per_chunker": per_chunker, "es_index": a.index}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
