"""E8d · Agent discourse embeddings (spec §9-E8d).

Writes per-call centroids of AGENT turn embeddings (L2-normalized). `agent_discourse_similarity` (mean cosine to the
centroids of the other calls of the same group) is computed in E11, where group is first joined.
`objection_qa` clusters pass-A objection quotes typed `other` into data/validation/objection_other_clusters.md.
"""

from __future__ import annotations

import glob
from collections import Counter
from typing import Any

import numpy as np

from .. import paths
from ..cache import StageContext, file_fingerprint, read_json, run_per_call
from ..canonical import load_canonical
from ..config import load_config
from ..roles import calls_with_turns


def setup(ctx: StageContext) -> None:
    from sentence_transformers import SentenceTransformer

    ctx.models["st"] = SentenceTransformer(ctx.cfg["features"]["embedding_model"], device="cpu")


def process_call(call_id: str, ctx: StageContext) -> dict[str, Any]:
    doc = load_canonical(call_id)
    texts = [" ".join(w["w"] for w in t["words"]) for t in doc["turns"] if t["role"] == "AGENT" and t["words"]]
    if not texts:
        return {"model": ctx.cfg["features"]["embedding_model"], "n_agent_turns": 0, "centroid": None}
    emb = ctx.models["st"].encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    centroid = np.asarray(emb).mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    return {"model": ctx.cfg["features"]["embedding_model"], "n_agent_turns": len(texts),
            "centroid": (centroid / norm).round(6).tolist() if norm > 0 else None}


def run(calls: list[str] | None = None, force: bool = False):
    ids = [c for c in calls_with_turns(calls) if paths.interim("canonical", f"{c}.json").exists()]
    return run_per_call("embeddings", ids, process_call, lambda c: paths.interim("features", "embeddings", f"{c}.json"),
                        force=force, setup=setup,
                        extra_key=lambda c: file_fingerprint(paths.interim("canonical", f"{c}.json")))


def discourse_similarity(centroids: dict[str, list[float] | None], groups: dict[str, str]) -> dict[str, float | None]:
    """Mean cosine similarity of each call's centroid to the other calls' centroids in the same group."""
    out: dict[str, float | None] = {}
    for cid, c in centroids.items():
        peers = [np.asarray(centroids[o]) for o in centroids
                 if o != cid and centroids[o] is not None and groups.get(o) == groups.get(cid)]
        if c is None or not peers:
            out[cid] = None
            continue
        v = np.asarray(c)
        out[cid] = round(float(np.mean([v @ p / (np.linalg.norm(v) * np.linalg.norm(p)) for p in peers])), 4)
    return out


def objection_qa() -> int:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering

    cfg = load_config()
    items = []
    for p in sorted(glob.glob(str(paths.ANNOTATIONS / "pass_a" / "C*.json"))):
        doc = read_json(p)
        for o in doc.get("objections", []):
            if o["type"] == "other":
                items.append((doc["_meta"]["call_id"], o["turn_id"], o["quote"]))
    out = paths.ensure_dir(paths.VALIDATION) / "objection_other_clusters.md"
    lines = ["# Objection quotes typed `other` — clusters (taxonomy QA)", "",
             f"{len(items)} quotes. Model: {cfg['features']['embedding_model']}.", ""]
    if len(items) >= 2:
        model = SentenceTransformer(cfg["features"]["embedding_model"], device="cpu")
        emb = model.encode([q for _, _, q in items], normalize_embeddings=True)
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=0.6, metric="cosine",
                                         linkage="average").fit_predict(emb)
    else:
        labels = [0] * len(items)
    label_counts = Counter(labels)
    for lab in sorted(label_counts, key=lambda label: -label_counts[label]):
        members = [it for it, label in zip(items, labels) if label == lab]
        lines += [f"## Cluster {lab} ({len(members)})", ""] + [f"- {c} {t}: “{q}”" for c, t, q in members] + [""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out.relative_to(paths.ROOT)} ({len(items)} quotes)")
    return 0
