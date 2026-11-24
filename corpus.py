"""Load all 21,578 Reuters records from the original SGML distribution."""

import hashlib
import html
import re
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

ARCHIVE = Path(__file__).with_name("reuters21578.tar.gz")
SOURCE = "https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz"
RECORD = re.compile(r'<REUTERS\b[^>]*\bNEWID="(\d+)"[^>]*>(.*?)</REUTERS>', re.S)
TOKEN = re.compile(r"[a-z]+(?:[-'][a-z]+)*")
STOP = frozenset("""
a an and are as at be been being but by can could did do does for from had has have
he her here him his how i if in into is it its may me more most my no not of on one
or our out over s said say she so some than that the their them then there these they
this those through to too under up us very was we were what when where which who will
with would you your reuter reuters
""".split())
QUERIES = ("crude oil", "interest rates", "wheat exports", "financial market", "computer science")


def tokenize(text):
    return Counter(w for w in TOKEN.findall(text.lower()) if w not in STOP and len(w) > 1)


@dataclass(frozen=True)
class Article:
    id: int
    title: str
    body: str


def _extract(record, tag):
    match = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", record, re.S)
    return match[1] if match else ""


def _clean(text):
    plain = html.unescape(re.sub(r"<[^>]*>", " ", text))
    return " ".join("".join(ch for ch in plain if ch.isprintable() or ch.isspace()).split())


def load_articles(archive=ARCHIVE):
    articles = []
    with tarfile.open(archive, "r:gz") as tar:
        members = sorted((m for m in tar if m.isfile() and
                          re.fullmatch(r"reut2-\d{3}\.sgm", m.name)), key=lambda m: m.name)
        for member in members:
            with tar.extractfile(member) as stream:
                for number, record in RECORD.findall(stream.read().decode("latin-1")):
                    title = _clean(_extract(record, "TITLE"))
                    body = _extract(record, "BODY")
                    if not body:
                        body = re.sub(r"<(TITLE|DATELINE|AUTHOR)>.*?</\1>", " ",
                                      _extract(record, "TEXT"), flags=re.S)
                    articles.append(Article(int(number), title, _clean(body)))
    articles.sort(key=lambda a: a.id)
    if [a.id for a in articles] != list(range(1, 21_579)):
        raise ValueError("Reuters archive does not contain the expected 21,578 consecutive NEWIDs.")
    return articles


def get_archive():
    if ARCHIVE.exists():
        return ARCHIVE
    import shutil

    with NamedTemporaryFile(dir=ARCHIVE.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            with urlopen(SOURCE, timeout=40) as response:
                shutil.copyfileobj(response, file)
            file.close()
            temporary.replace(ARCHIVE)
        finally:
            file.close()
            temporary.unlink(missing_ok=True)
    return ARCHIVE


def document_terms(articles):
    return (tokenize(f"{a.title} {a.body}") for a in articles)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
