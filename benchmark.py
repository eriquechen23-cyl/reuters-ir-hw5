"""Reproduce construction measurements and confirm equality on real queries."""

import argparse
import json
import platform
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from corpus import QUERIES, digest, document_terms, get_archive, load_articles, tokenize
from ir import Search, build_index, read_index

MODES = ("BSBI", "SPIMI")


def benchmark(articles, capacity, repeats, directory):
    rows = []
    for size in (10_000, len(articles)):
        docs = tuple(document_terms(articles[:size]))
        paths = {mode: directory / f"{mode}-{size}.tsv" for mode in MODES}
        timings = {mode: [] for mode in MODES}
        for iteration in range(repeats):
            for mode in (MODES if iteration % 2 == 0 else MODES[::-1]):
                start = perf_counter()
                build_index(docs, mode, paths[mode], capacity)
                timings[mode].append(perf_counter() - start)
        methods = {}
        for mode in MODES:
            tracemalloc.start()
            try:
                stats = build_index(docs, mode, paths[mode], capacity)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            methods[mode] = {**asdict(stats), "seconds": timings[mode],
                             "median_s": median(timings[mode]), "peak_bytes": peak,
                             "sha256": digest(paths[mode])}
        if methods["BSBI"]["sha256"] != methods["SPIMI"]["sha256"]:
            raise RuntimeError(f"Index mismatch on {size:,} documents")
        print(f"{size:,}: " + ", ".join(f"{m} {methods[m]['median_s']:.3f}s" for m in MODES),
              flush=True)
        rows.append({"documents": size, "methods": methods})
    return rows, paths


def verify(paths, articles):
    searches = [Search(read_index(paths[mode]), len(articles)) for mode in MODES]
    first, second = searches
    if first.index != second.index:
        raise RuntimeError("Postings mismatch")
    queries = {}
    for query in QUERIES:
        counts = tokenize(query)
        queries[query] = {}
        for operation in ("單詞", "AND", "OR"):
            a, b = [s.boolean(counts, operation) for s in searches]
            if a != b:
                raise RuntimeError(f"{operation} {query} mismatch")
            queries[query][operation] = len(a)
        if first.rank(counts) != second.rank(counts):
            raise RuntimeError(f"Ranked results for {query} differ")
    return first, second, queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=50_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    args = parser.parse_args()
    if args.repeats < 1 or args.capacity < 1:
        parser.error("capacity and repeats must be positive")
    articles = load_articles(get_archive())
    with TemporaryDirectory() as name:
        cases, paths = benchmark(articles, args.capacity, args.repeats, Path(name))
        a, b, queries = verify(paths, articles)
        oil = next(article for article in articles if "OPEC" in article.title and "quota" in article.body.lower())
        selected = [oil.id, 1, 2, 3, 5, 10_000, 21_578]
        for number in selected:
            if a.related(number) != b.related(number):
                raise RuntimeError(f"Related results for {number} differ")
        examples = {}
        for number in selected[:4]:
            examples[number] = {
                "article": asdict(articles[number - 1]),
                "hits": [{"title": articles[h.doc - 1].title, **asdict(h)} for h in a.related(number, 3)]
            }
        result = {"python": platform.python_version(), "platform": platform.system(),
                  "capacity": args.capacity, "repeats": args.repeats, "runs": cases,
                  "vocabulary": len(a.index), "queries": queries,
                  "related_verified": selected, "examples": examples}
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
