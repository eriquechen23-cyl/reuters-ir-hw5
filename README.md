# Reuters-21578 倒排索引與新聞推薦

Streamlit 示範包含文章推薦、單詞／AND／OR 查詢、TF-IDF 關鍵字排序、詞彙表、postings 與索引效能。推薦預設展示 NEWID 3，展開結果可檢閱正文、雙方詞頻、文件頻率與各詞的分數貢獻。

## 本機執行

使用 Python 3.13。

```shell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

程式首次啟動會從 UCI 下載約 8 MB 的壓縮檔。離線時，可將官方 `reuters21578.tar.gz` 放在 `app.py` 旁。

## 教學部署

將程式碼、`benchmark_results.json` 與 `.streamlit/config.toml` 放入 GitHub 程式庫。於 [Streamlit Community Cloud](https://share.streamlit.io/) 使用該程式庫的分支部署 `app.py`，Python 版本選 3.13。應用程式首次啟動會下載語料並建立兩套索引。

## 重現實驗

```shell
python -m unittest -v
python benchmark.py --capacity 50000 --repeats 3
```

- 使用相同的 SGML 解析、標題與正文、英文字詞小寫化、停用詞表和每篇詞頻。原始 NEWID 保持為 1–21,578。
- BSBI 維護跨區塊 termID 字典，排序每區塊的 `(termID, docID, TF)` 再彙整。SPIMI 將詞直接加入該區塊的字典與 postings，寫檔時只排序字典鍵。
- 兩者以每區塊 50,000 個不同詞－文件項目為預設上限，共用磁碟格式與有限扇入的串流合併。上限為項目數，實際記憶體用量以獨立的 `tracemalloc` 執行測量，排除預載輸入與查詢資料結構。
- 兩套索引獨立建置。離線基準驗證完整 postings、五組查詢及七篇來源文章的推薦。Streamlit 每次建置驗證兩份檔案的 SHA-256，通過後使用已驗證的索引提供檢索與推薦。
- 向量權重為 `(1 + ln(tf)) × (1 + ln((N+1)/(df+1)))`。分數為文件與查詢或來源新聞向量的餘弦相似度。同分依 NEWID 由小到大排列，推薦排除來源本身。

資料來源為 [Reuters-21578, Distribution 1.0，UCI KDD Archive](https://kdd.ics.uci.edu/databases/reuters21578/)，供資訊檢索研究與教學實驗。索引方法參考 [Introduction to Information Retrieval，第 4 章](https://nlp.stanford.edu/IR-book/html/htmledition/index-construction-1.html)。
