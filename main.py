"""
VectorDB Engine — Python port of the original C++ implementation.
Requires: pip install flask requests
Run:      python main.py
"""

import math
import random
import time
import threading
import heapq
from collections import defaultdict
from flask import Flask, request, jsonify, send_file, abort
import requests
import os

# =====================================================================
#  CONSTANTS
# =====================================================================

DIMS = 16  # demo vectors dimensionality

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (na * nb)

def manhattan(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))

def get_dist_fn(metric: str):
    if metric == "cosine":    return cosine
    if metric == "manhattan": return manhattan
    return euclidean

# =====================================================================
#  DATA TYPES
# =====================================================================

class VectorItem:
    def __init__(self, id: int, metadata: str, category: str, emb: list):
        self.id       = id
        self.metadata = metadata
        self.category = category
        self.emb      = emb

# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items = []

    def insert(self, item: VectorItem):
        self.items.append(item)

    def knn(self, q: list, k: int, dist_fn) -> list:
        """Returns list of (distance, id) sorted ascending."""
        results = [(dist_fn(q, v.emb), v.id) for v in self.items]
        results.sort()
        return results[:k]

    def remove(self, id: int):
        self.items = [v for v in self.items if v.id != id]

# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    def __init__(self, item: VectorItem):
        self.item  = item
        self.left  = None
        self.right = None

class KDTree:
    def __init__(self, dims: int):
        self.root = None
        self.dims = dims

    def _insert(self, node, item: VectorItem, depth: int):
        if node is None:
            return KDNode(item)
        axis = depth % self.dims
        if item.emb[axis] < node.item.emb[axis]:
            node.left  = self._insert(node.left,  item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def insert(self, item: VectorItem):
        self.root = self._insert(self.root, item, 0)

    def _knn(self, node, q: list, k: int, depth: int, dist_fn, heap: list):
        """Max-heap of (-dist, id) with size k."""
        if node is None:
            return
        d = dist_fn(q, node.item.emb)
        if len(heap) < k:
            heapq.heappush(heap, (-d, node.item.id))
        elif d < -heap[0][0]:
            heapq.heapreplace(heap, (-d, node.item.id))

        axis = depth % self.dims
        diff = q[axis] - node.item.emb[axis]
        closer  = node.left  if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left

        self._knn(closer, q, k, depth + 1, dist_fn, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist_fn, heap)

    def knn(self, q: list, k: int, dist_fn) -> list:
        heap = []
        self._knn(self.root, q, k, 0, dist_fn, heap)
        results = [(-d, id_) for d, id_ in heap]
        results.sort()
        return results

    def rebuild(self, items: list):
        self.root = None
        for item in items:
            self.insert(item)

# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class HNSW:
    class Node:
        def __init__(self, item: VectorItem, max_layer: int):
            self.item      = item
            self.max_layer = max_layer
            self.neighbors = [[] for _ in range(max_layer + 1)]  # per-layer adjacency

    def __init__(self, M: int = 16, ef_build: int = 200):
        self.M        = M
        self.M0       = 2 * M
        self.ef_build = ef_build
        self.mL       = 1.0 / math.log(M)
        self.graph    = {}           # id -> Node
        self.entry_pt = -1
        self.top_layer = -1
        random.seed(42)

    def _rand_level(self) -> int:
        return int(math.floor(-math.log(random.random()) * self.mL))

    def _search_layer(self, q, ep: int, ef: int, layer: int, dist_fn):
        """Returns sorted list of (dist, id) — closest first."""
        visited = {ep}
        d0 = dist_fn(q, self.graph[ep].item.emb)
        # min-heap for candidates, max-heap for found (negate dist)
        cands = [(d0, ep)]
        found = [(-d0, ep)]

        while cands:
            cd, cid = heapq.heappop(cands)
            if found and cd > -found[0][0] and len(found) >= ef:
                break
            node = self.graph.get(cid)
            if node is None or layer >= len(node.neighbors):
                continue
            for nid in node.neighbors[layer]:
                if nid in visited or nid not in self.graph:
                    continue
                visited.add(nid)
                nd = dist_fn(q, self.graph[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = [(-d, id_) for d, id_ in found]
        result.sort()
        return result

    def insert(self, item: VectorItem, dist_fn):
        id_  = item.id
        lvl  = self._rand_level()
        node = self.Node(item, lvl)
        self.graph[id_] = node

        if self.entry_pt == -1:
            self.entry_pt  = id_
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if ep in self.graph and lc < len(self.graph[ep].neighbors):
                W = self._search_layer(item.emb, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W    = self._search_layer(item.emb, ep, self.ef_build, lc, dist_fn)
            maxM = self.M0 if lc == 0 else self.M
            sel  = [id2 for _, id2 in W[:maxM]]
            # Extend neighbors list if needed
            while len(node.neighbors) <= lc:
                node.neighbors.append([])
            node.neighbors[lc] = sel

            for nid in sel:
                if nid not in self.graph:
                    continue
                nb = self.graph[nid]
                while len(nb.neighbors) <= lc:
                    nb.neighbors.append([])
                nb.neighbors[lc].append(id_)
                if len(nb.neighbors[lc]) > maxM:
                    # Prune: keep closest maxM
                    ds = sorted(
                        (dist_fn(nb.item.emb, self.graph[c].item.emb), c)
                        for c in nb.neighbors[lc] if c in self.graph
                    )
                    nb.neighbors[lc] = [c for _, c in ds[:maxM]]
            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt  = id_

    def knn(self, q, k: int, ef: int, dist_fn) -> list:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if ep in self.graph and lc < len(self.graph[ep].neighbors):
                W = self._search_layer(q, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist_fn)
        return W[:k]

    def remove(self, id_: int):
        if id_ not in self.graph:
            return
        for node in self.graph.values():
            for layer in node.neighbors:
                if id_ in layer:
                    layer.remove(id_)
        if self.entry_pt == id_:
            self.entry_pt = next((nid for nid in self.graph if nid != id_), -1)
        del self.graph[id_]

    def get_info(self) -> dict:
        top = max(self.top_layer, 0)
        nodes_per_layer = [0] * (top + 1)
        edges_per_layer = [0] * (top + 1)
        nodes, edges = [], []

        for id_, node in self.graph.items():
            nodes.append({
                "id":       id_,
                "metadata": node.item.metadata,
                "category": node.item.category,
                "maxLyr":   node.max_layer,
            })
            for lc in range(min(node.max_layer + 1, top + 1)):
                nodes_per_layer[lc] += 1
                if lc < len(node.neighbors):
                    for nid in node.neighbors[lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id_, "dst": nid, "lyr": lc})

        return {
            "topLayer":      self.top_layer,
            "nodeCount":     len(self.graph),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes":         nodes,
            "edges":         edges,
        }

    def __len__(self):
        return len(self.graph)

# =====================================================================
#  VECTOR DATABASE  (demo 16D index)
# =====================================================================

class VectorDB:
    def __init__(self, dims: int):
        self.dims   = dims
        self.store  = {}
        self.bf     = BruteForce()
        self.kdt    = KDTree(dims)
        self.hnsw   = HNSW(16, 200)
        self._lock  = threading.Lock()
        self._next  = 1

    def insert(self, metadata: str, category: str, emb: list, dist_fn) -> int:
        with self._lock:
            item = VectorItem(self._next, metadata, category, emb)
            self._next += 1
            self.store[item.id] = item
            self.bf.insert(item)
            self.kdt.insert(item)
            self.hnsw.insert(item, dist_fn)
            return item.id

    def remove(self, id_: int) -> bool:
        with self._lock:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.bf.remove(id_)
            self.hnsw.remove(id_)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q: list, k: int, metric: str, algo: str) -> dict:
        with self._lock:
            dist_fn = get_dist_fn(metric)
            t0 = time.perf_counter()
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dist_fn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dist_fn)
            else:
                raw = self.hnsw.knn(q, k, 50, dist_fn)
            us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for d, id_ in raw:
                if id_ in self.store:
                    v = self.store[id_]
                    hits.append({"id": id_, "metadata": v.metadata,
                                 "category": v.category, "embedding": v.emb,
                                 "distance": d})
            return {"results": hits, "latencyUs": us, "algo": algo, "metric": metric}

    def benchmark(self, q: list, k: int, metric: str) -> dict:
        with self._lock:
            dist_fn = get_dist_fn(metric)
            def timed(fn):
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1_000_000)
            return {
                "bruteforceUs": timed(lambda: self.bf.knn(q, k, dist_fn)),
                "kdtreeUs":     timed(lambda: self.kdt.knn(q, k, dist_fn)),
                "hnswUs":       timed(lambda: self.hnsw.knn(q, k, 50, dist_fn)),
                "itemCount":    len(self.store),
            }

    def all(self) -> list:
        with self._lock:
            return list(self.store.values())

    def hnsw_info(self) -> dict:
        with self._lock:
            return self.hnsw.get_info()

    def __len__(self):
        with self._lock:
            return len(self.store)

# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> list:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    step = chunk_words - overlap_words
    chunks = []
    i = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks

# =====================================================================
#  OLLAMA CLIENT
# =====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base_url    = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model   = "llama3.2"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> list:
        try:
            r = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30,
            )
            if r.status_code != 200:
                return []
            return r.json().get("embedding", [])
        except Exception:
            return []

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180,
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except Exception:
            return "ERROR: Ollama unavailable. Run: ollama serve"

# =====================================================================
#  DOCUMENT DATABASE  — HNSW over real Ollama embeddings
# =====================================================================

class DocItem:
    def __init__(self, id: int, title: str, text: str, emb: list):
        self.id    = id
        self.title = title
        self.text  = text
        self.emb   = emb

class DocumentDB:
    def __init__(self):
        self.store = {}
        self.hnsw  = HNSW(16, 200)
        self.bf    = BruteForce()
        self._lock = threading.Lock()
        self._next = 1
        self.dims  = 0

    def insert(self, title: str, text: str, emb: list) -> int:
        with self._lock:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self._next, title, text, emb)
            self._next += 1
            self.store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q: list, k: int, max_dist: float = 0.7) -> list:
        with self._lock:
            if not self.store:
                return []
            if len(self.store) < 10:
                raw = self.bf.knn(q, k, cosine)
            else:
                raw = self.hnsw.knn(q, k, 50, cosine)
            return [(d, self.store[id_]) for d, id_ in raw
                    if id_ in self.store and d <= max_dist]

    def remove(self, id_: int) -> bool:
        with self._lock:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.hnsw.remove(id_)
            self.bf.remove(id_)
            return True

    def all(self) -> list:
        with self._lock:
            return list(self.store.values())

    def __len__(self):
        with self._lock:
            return len(self.store)

# =====================================================================
#  DEMO DATA  (16D categorical vectors)
# =====================================================================

DEMO_DATA = [
    # (metadata, category, embedding)
    # Dims 0-3: CS | 4-7: Math | 8-11: Food | 12-15: Sports
    ("Linked List: nodes connected by pointers", "cs",
     [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
    ("Binary Search Tree: O(log n) search and insert", "cs",
     [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
    ("Dynamic Programming: memoization overlapping subproblems", "cs",
     [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
    ("Graph BFS and DFS: breadth and depth first traversal", "cs",
     [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
    ("Hash Table: O(1) lookup with collision chaining", "cs",
     [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
    ("Calculus: derivatives integrals and limits", "math",
     [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
    ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
     [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
    ("Probability: distributions random variables Bayes theorem", "math",
     [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
    ("Number Theory: primes modular arithmetic RSA cryptography", "math",
     [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
    ("Combinatorics: permutations combinations generating functions", "math",
     [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
    ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
     [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
    ("Sushi: vinegared rice raw fish and nori rolls", "food",
     [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
    ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
     [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
    ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
     [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
    ("Croissant: laminated pastry with buttery flaky layers", "food",
     [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
    ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
     [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
    ("Football: tackles touchdowns field goals and strategy", "sports",
     [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
    ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
     [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
    ("Chess: openings endgames tactics strategic board game", "sports",
     [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
    ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
     [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
]

def load_demo(db: VectorDB):
    dist_fn = get_dist_fn("cosine")
    for meta, cat, emb in DEMO_DATA:
        db.insert(meta, cat, emb, dist_fn)

# =====================================================================
#  FLASK APP
# =====================================================================

app = Flask(__name__)

# Global instances
db     = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

load_demo(db)

def parse_vec(s: str) -> list:
    try:
        return [float(x) for x in s.split(",") if x.strip()]
    except Exception:
        return []

def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.after_request
def after_request(response):
    return add_cors(response)

@app.route("/", methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def options_handler(path=""):
    from flask import make_response
    return make_response("", 204)

# ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────────

@app.route("/search")
def search():
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400
    k      = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    algo   = request.args.get("algo", "hnsw")
    return jsonify(db.search(q, k, metric, algo))

@app.route("/insert", methods=["POST"])
def insert():
    body = request.get_json(force=True, silent=True) or {}
    meta = body.get("metadata", "")
    cat  = body.get("category", "")
    emb  = body.get("embedding", [])
    if not meta or not emb or len(emb) != DIMS:
        return jsonify({"error": "invalid body"}), 400
    id_ = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return jsonify({"id": id_})

@app.route("/delete/<int:id_>", methods=["DELETE"])
def delete_item(id_):
    ok = db.remove(id_)
    return jsonify({"ok": ok})

@app.route("/items")
def items():
    return jsonify([
        {"id": v.id, "metadata": v.metadata, "category": v.category, "embedding": v.emb}
        for v in db.all()
    ])

@app.route("/benchmark")
def benchmark():
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400
    k      = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    return jsonify(db.benchmark(q, k, metric))

@app.route("/hnsw-info")
def hnsw_info():
    return jsonify(db.hnsw_info())

@app.route("/stats")
def stats():
    return jsonify({
        "count":      len(db),
        "dims":       DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics":    ["euclidean", "cosine", "manhattan"],
    })

# ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────────

@app.route("/doc/insert", methods=["POST"])
def doc_insert():
    body  = request.get_json(force=True, silent=True) or {}
    title = body.get("title", "")
    text  = body.get("text", "")
    if not title or not text:
        return jsonify({"error": "need title and text"}), 400

    chunks = chunk_text(text, 250, 30)
    ids    = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return jsonify({
                "error": "Ollama unavailable. Install from https://ollama.com "
                         "then run: ollama pull nomic-embed-text && ollama pull llama3.2"
            }), 503
        chunk_title = (f"{title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else title)
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.dims})

@app.route("/doc/delete/<int:id_>", methods=["DELETE"])
def doc_delete(id_):
    ok = doc_db.remove(id_)
    return jsonify({"ok": ok})

@app.route("/doc/list")
def doc_list():
    docs = doc_db.all()
    result = []
    for d in docs:
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        result.append({
            "id":      d.id,
            "title":   d.title,
            "preview": preview,
            "words":   len(d.text.split()),
        })
    return jsonify(result)

@app.route("/doc/search", methods=["POST"])
def doc_search():
    body     = request.get_json(force=True, silent=True) or {}
    question = body.get("question", "")
    k        = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"}), 400
    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"}), 503
    hits = doc_db.search(q_emb, k)
    return jsonify({"contexts": [
        {"id": d.id, "title": d.title, "distance": round(dist, 4)}
        for dist, d in hits
    ]})

@app.route("/doc/ask", methods=["POST"])
def doc_ask():
    body     = request.get_json(force=True, silent=True) or {}
    question = body.get("question", "")
    k        = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"}), 400

    # Step 1: embed question
    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"}), 503

    # Step 2: retrieve top-k chunks
    hits = doc_db.search(q_emb, k)

    # Step 3: build prompt
    ctx = ""
    for i, (_, doc) in enumerate(hits):
        ctx += f"[{i+1}] {doc.title}:\n{doc.text}\n\n"

    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{ctx}"
        f"Question: {question}\n\nAnswer:"
    )

    # Step 4: generate answer
    answer = ollama.generate(prompt)

    # Step 5: return everything
    return jsonify({
        "answer":   answer,
        "model":    ollama.gen_model,
        "contexts": [
            {"id": d.id, "title": d.title, "text": d.text, "distance": round(dist, 4)}
            for dist, d in hits
        ],
        "docCount": len(doc_db),
    })

@app.route("/status")
def status():
    up = ollama.is_available()
    return jsonify({
        "ollamaAvailable": up,
        "embedModel":      ollama.embed_model,
        "genModel":        ollama.gen_model,
        "docCount":        len(doc_db),
        "docDims":         doc_db.dims,
        "demoDims":        DIMS,
        "demoCount":       len(db),
    })

@app.route("/")
def index():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(html_path):
        abort(404)
    return send_file(html_path)

# =====================================================================
#  ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    ollama_up = ollama.is_available()
    print("=== VectorDB Engine (Python) ===")
    print("http://localhost:8080")
    print(f"{len(db)} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")
    app.run(host="0.0.0.0", port=8080, threaded=True)
