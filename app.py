from dataclasses import asdict
from json import loads
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import streamlit as st

from corpus import QUERIES, digest, document_terms, get_archive, load_articles, tokenize
from ir import Search, build_index, read_index

ROOT = Path(__file__).parent
MODES = ("BSBI", "SPIMI")

st.set_page_config(page_title="Reuters 新聞檢索實驗室", page_icon="📰", layout="wide")


@st.cache_resource(show_spinner="正在讀取 Reuters-21578 並建立兩套索引…", max_entries=1)
def load_experiment(size, capacity):
    articles = load_articles(get_archive())[:size]
    docs = tuple(document_terms(articles))
    results = {}
    with TemporaryDirectory() as directory:
        paths = {mode: Path(directory) / f"{mode}.tsv" for mode in MODES}
        for mode, path in paths.items():
            start = perf_counter()
            stats = build_index(docs, mode, path, capacity)
            results[mode] = {**asdict(stats), "seconds": perf_counter() - start,
                             "sha256": digest(path)}
        del docs
        if results["BSBI"]["sha256"] != results["SPIMI"]["sha256"]:
            raise ValueError("兩套索引的 SHA-256 不一致。")
        search = Search(read_index(paths["BSBI"]), len(articles))
    return articles, search, results


def title(article):
    return article.title or "(無標題)"


def article_picker(articles, candidates, key):
    return st.selectbox("選擇文章", candidates, key=key,
                        format_func=lambda n: f"#{n} · {title(articles[n - 1])}")


def show_ranked(hits, articles):
    if not hits:
        st.info("沒有含共同索引詞的文章。")
    for rank, hit in enumerate(hits, start=1):
        article = articles[hit.doc - 1]
        words = "、".join(item.term for item in hit.terms[:3])
        with st.expander(f"{rank}. #{hit.doc} · {title(article)} · {hit.score:.4f}",
                         expanded=rank == 1):
            st.caption(f"主要貢獻詞：{words}。共同索引詞共 {len(hit.terms)} 個。")
            st.text(article.body or "此筆記錄沒有正文。")
            st.dataframe([
                {"詞": c.term, "來源 TF": c.query_tf, "結果 TF": c.document_tf,
                 "文件頻率 DF": c.df, "IDF": round(c.idf, 4), "分數貢獻": round(c.score, 6)}
                for c in hit.terms
            ], hide_index=True, width="stretch", height=190)


st.title("Reuters 新聞檢索實驗室")
st.caption("Reuters-21578 • 倒排索引、布林檢索與可解釋的文章推薦")
with st.sidebar:
    st.subheader("實驗設定")
    size = st.selectbox("文件數", (21_578, 10_000), format_func=lambda n: f"{n:,}")
    capacity = st.selectbox("區塊容量（詞－文件項目）", (50_000, 25_000, 100_000))
    st.caption("每個區塊達到容量即寫入磁碟。容量以項目數計算。")
    st.link_button("Reuters-21578, Distribution 1.0", "https://kdd.ics.uci.edu/databases/reuters21578/")
try:
    articles, search, builds = load_experiment(size, capacity)
except (OSError, ValueError) as exc:
    st.error(f"無法初始化語料或索引。請確認 Reuters 原始壓縮檔可讀，再重新載入。\n\n{exc}")
    st.stop()

columns = st.columns(3)
columns[0].metric("Reuters 文件", f"{len(articles):,}")
columns[1].metric("詞彙數", f"{len(search.index):,}")
columns[2].metric("BSBI／SPIMI 索引", "完全一致")
page = st.radio("功能", ("文章推薦", "檢索", "詞彙與 Postings", "索引效能"),
                horizontal=True, label_visibility="collapsed")

if page == "文章推薦":
    st.subheader("找相似新聞")
    st.caption("以來源文章完整向量計算 TF-IDF 餘弦相似度。點開結果可查看各詞貢獻。")
    preferred = 3
    lookup = st.text_input("依標題縮小來源範圍（可留空）", placeholder="例如 Texas、OPEC").strip()
    matches = ([a.id for a in articles if lookup.casefold() in a.title.casefold()][:100]
               if lookup else [preferred, 144, 1, 2, 5, 10_000, size])
    choose = st.radio("選文方式", ("選擇標題", "輸入 NEWID"), horizontal=True)
    if choose == "輸入 NEWID":
        doc = st.number_input("文章 NEWID", 1, size, value=preferred, step=1)
    elif matches:
        doc = article_picker(articles, list(dict.fromkeys(matches)), "source")
    else:
        st.info("找不到符合的標題。")
        st.stop()
    article = articles[doc - 1]
    st.markdown(f"**來源文章 #{doc}**")
    st.text(title(article))
    with st.expander("閱讀來源正文"):
        st.text(article.body or "此筆記錄沒有正文。")
    if doc == 3 and size == 21_578:
        st.info("案例觀察：若想追蹤 Texas Commerce，同名機構的新聞更值得先讀。"
                "若想找在相同監管環境下申設分行的公司，Texas Capital 則是有價值的對照。")
    show_ranked(search.related(doc, limit=5), articles)

elif page == "檢索":
    preset = st.selectbox("查詢範例", QUERIES)
    query = st.text_input("輸入查詢", preset, key=f"query-{preset}")
    operation = st.radio("查詢模式", ("單詞", "AND", "OR", "TF-IDF"), horizontal=True)
    terms = tokenize(query)
    if not terms:
        st.info("請輸入至少一個有效索引詞。")
    else:
        st.caption("查詢詞：" + "、".join(list(terms)[:1] if operation == "單詞" else terms))
        if operation == "TF-IDF":
            show_ranked(search.rank(terms), articles)
        else:
            ids = search.boolean(terms, operation)
            st.write(f"命中 {len(ids):,} 筆，顯示前 50 筆。")
            if ids:
                st.dataframe([{"NEWID": n, "標題": title(articles[n - 1])} for n in ids[:50]],
                             hide_index=True, width="stretch")
            else:
                st.info("無符合文章。")

elif page == "詞彙與 Postings":
    prefix = st.text_input("詞彙字首", "oil").lower().strip()
    vocabulary = [word for word in search.index if word.startswith(prefix)]
    st.caption(f"{len(vocabulary):,} 個詞符合，最多列出 100 個。Dictionary 記錄詞及文件頻率，"
               "Postings List 記錄含該詞的 NEWID 與詞頻。")
    if vocabulary:
        entries = vocabulary[:100]
        st.dataframe([{"詞": word, "DF": len(search.index[word])} for word in entries],
                     hide_index=True, width="stretch", height=250)
        term = st.selectbox("查看 Postings List", entries)
        posting = search.index[term]
        st.caption(f"{term} · {len(posting):,} 筆，顯示前 100 筆。")
        st.dataframe([{"NEWID": n, "TF": count} for n, count in posting[:100]],
                     hide_index=True, width="stretch")
    else:
        st.info("沒有符合的詞彙。")

else:
    st.subheader("索引建置實測")
    st.caption("這次啟動的計時從分塊到寫出合併後的索引。設定相同時會沿用此次建置。")
    if sys.platform == "emscripten":
        st.caption("此頁即時計時在瀏覽器執行，區塊寫入瀏覽器的虛擬檔案系統。下方另列報告使用的 Windows 實測。")
    st.dataframe([
        {"演算法": mode, "建置秒數": round(row["seconds"], 3), "區塊": row["blocks"],
         "中間檔數": row["intermediate_files"], "中間檔寫入量（位元組）": row["intermediate_bytes"],
         "最終索引（位元組）": row["final_bytes"], "合併輪數": row["merge_passes"]}
        for mode, row in builds.items()
    ], hide_index=True, width="stretch")
    st.caption("兩份最終索引的 SHA-256")
    st.code(builds["BSBI"]["sha256"], language=None)
    for mode, row in builds.items():
        with st.expander(f"{mode} 第一區塊樣本"):
            st.json(row["trace"])
    saved = ROOT / "benchmark_results.json"
    if saved.is_file():
        benchmark = loads(saved.read_text(encoding="utf-8"))
        st.subheader("報告使用的基準實驗")
        st.caption(f"每組 {benchmark['repeats']} 次取中位數。另以 tracemalloc 測量建置期間的 Python 配置峰值。")
        st.dataframe([
            {"文件數": case["documents"], "演算法": mode, "建置秒數": round(m["median_s"], 3),
             "峰值 MiB": round(m["peak_bytes"] / 2**20, 2), "中間檔數": m["intermediate_files"],
             "最終索引 MiB": round(m["final_bytes"] / 2**20, 2)}
            for case in benchmark["runs"] for mode, m in case["methods"].items()
        ], hide_index=True, width="stretch")
        st.caption("五組查詢已分別由兩份索引獨立驗證，以下為相同的命中筆數。")
        st.dataframe([{"查詢": q, **counts} for q, counts in benchmark["queries"].items()],
                     hide_index=True, width="stretch")
