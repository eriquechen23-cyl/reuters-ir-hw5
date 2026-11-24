"""Independent BSBI and SPIMI construction, with one canonical index format."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory


@dataclass(frozen=True)
class BuildStats:
    blocks: int
    intermediate_files: int
    intermediate_bytes: int
    final_bytes: int
    merge_passes: int
    trace: dict


def _write(path, rows):
    with path.open("w", encoding="ascii", newline="\n") as out:
        out.writelines(f"{term}\t{doc}\t{count}\n" for term, doc, count in rows)


def _rows(stream):
    for line in stream:
        term, doc, count = line.split("\t")
        yield term, int(doc), int(count)


def _bsbi(rows, terms, trace):
    if not trace:
        inversion = next((i for i in range(1, len(rows)) if rows[i - 1] > rows[i]), 0)
        start = max(0, inversion - 3)
        sample = rows[start:start + 8]
        trace["排序前 (termID, docID, TF)"] = sample
        trace["termID 對應詞（樣本）"] = {tid: terms[tid] for tid, _, _ in sample}
        trace["排序後 (termID, docID, TF)"] = sorted(sample)
    rows.sort()
    starts = {}
    for i, (term_id, _, _) in enumerate(rows):
        starts.setdefault(term_id, i)
    # Sort term labels on disk so both builders can use the same external merge.
    for term_id in sorted(starts, key=terms.__getitem__):
        i = starts[term_id]
        while i < len(rows) and rows[i][0] == term_id:
            _, doc, count = rows[i]
            yield terms[term_id], doc, count
            i += 1


def _spimi(postings, trace):
    keys = sorted(postings)
    if not trace:
        trace["直接追加至 postings（第一塊樣本）"] = {
            term: postings[term][:4] for term in list(postings)[:5]
        }
    for term in keys:
        for doc, count in postings[term]:
            yield term, doc, count


def _merge(paths, output):
    with ExitStack() as stack:
        streams = [_rows(stack.enter_context(p.open(encoding="ascii"))) for p in paths]
        _write(output, heapq.merge(*streams))


def build_index(documents, mode, output, capacity=50_000, fan_in=32):
    """Capacity counts distinct (term, docID) entries per block, not RAM bytes."""
    if mode not in {"BSBI", "SPIMI"} or capacity < 1 or fan_in < 2:
        raise ValueError("Invalid algorithm, block capacity or merge fan-in")
    output = Path(output)
    ids, terms = {}, []  # BSBI's global mapping survives block flushes.
    pairs, postings = [], {}
    trace = {}
    size = 0
    with TemporaryDirectory(dir=output.parent) as directory:
        root = Path(directory)
        blocks = []

        def flush():
            block = root / f"block-{len(blocks)}.tsv"
            rows = _bsbi(pairs, terms, trace) if mode == "BSBI" else _spimi(postings, trace)
            _write(block, rows)
            blocks.append(block)
            pairs.clear()
            postings.clear()

        for doc, counts in enumerate(documents, start=1):
            for term, count in counts.items():
                if mode == "BSBI":
                    term_id = ids.get(term)
                    if term_id is None:
                        term_id = len(terms)
                        ids[term] = term_id
                        terms.append(term)
                    pairs.append((term_id, doc, count))
                else:
                    postings.setdefault(term, []).append((doc, count))
                size += 1
                if size == capacity:
                    flush()
                    size = 0
        if size:
            flush()
        block_count = len(blocks)
        files = block_count
        disk_bytes = sum(p.stat().st_size for p in blocks)
        passes = 0
        while len(blocks) > fan_in:
            next_blocks = []
            for start in range(0, len(blocks), fan_in):
                target = root / f"merge-{passes}-{start}.tsv"
                batch = blocks[start:start + fan_in]
                _merge(batch, target)
                for block in batch:
                    block.unlink()
                next_blocks.append(target)
                files += 1
                disk_bytes += target.stat().st_size
            blocks = next_blocks
            passes += 1
        if len(blocks) > 1:
            _merge(blocks, output)
            passes += 1
        elif blocks:
            blocks[0].replace(output)
        else:
            _write(output, ())
        return BuildStats(block_count, files, disk_bytes, output.stat().st_size, passes, trace)


def read_index(path):
    with path.open(encoding="ascii") as stream:
        return {term: [(doc, count) for _, doc, count in rows]
                for term, rows in groupby(_rows(stream), key=lambda row: row[0])}


@dataclass(frozen=True)
class Contribution:
    term: str
    query_tf: int
    document_tf: int
    df: int
    idf: float
    score: float


@dataclass(frozen=True)
class Hit:
    doc: int
    score: float
    terms: tuple[Contribution, ...]


class Search:
    def __init__(self, index, document_count):
        self.index = index
        self.idfs = {word: math.log((document_count + 1) / (len(items) + 1)) + 1
                     for word, items in index.items()}
        self.doc_counts = [{} for _ in range(document_count + 1)]
        norm_sq = [0.0] * (document_count + 1)
        for term, items in index.items():
            for doc, count in items:
                self.doc_counts[doc][term] = count
                norm_sq[doc] += self.weight(term, count) ** 2
        self.norms = [math.sqrt(n) for n in norm_sq]

    def weight(self, term, count):
        return (1 + math.log(count)) * self.idfs[term]

    def boolean(self, terms, mode):
        terms = list(dict.fromkeys(terms))
        if mode == "單詞":
            terms = terms[:1]
        lists = [{doc for doc, _ in self.index.get(term, ())} for term in terms]
        if not lists:
            return []
        return sorted(set.intersection(*lists) if mode == "AND" else set.union(*lists))

    def related(self, doc, limit=10):
        return self.rank(self.doc_counts[doc], limit, exclude=doc)

    def rank(self, counts, limit=10, exclude=None):
        active = {t: self.weight(t, counts[t]) for t in sorted(counts) if t in self.index}
        norm = math.sqrt(sum(w ** 2 for w in active.values()))
        scores = defaultdict(float)
        for term, query_weight in active.items():
            for doc, tf in self.index[term]:
                if doc != exclude:
                    scores[doc] += query_weight * self.weight(term, tf) / (norm * self.norms[doc])
        top = heapq.nsmallest(limit, scores, key=lambda doc: (-scores[doc], doc))
        hits = []
        for doc in top:
            contributions = [
                Contribution(term, counts[term], tf, len(self.index[term]), self.idfs[term],
                             w * self.weight(term, tf) / (norm * self.norms[doc]))
                for term, w in active.items()
                if (tf := self.doc_counts[doc].get(term))
            ]
            contributions.sort(key=lambda c: (-c.score, c.term))
            hits.append(Hit(doc, scores[doc], tuple(contributions)))
        return hits
