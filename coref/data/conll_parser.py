"""
conll_parser.py — Parse CoNLL-style coreference files from the TransMuCoRes dataset.

Column layout (tab-separated, 17 cols):
  col 0:  document ID
  col 1:  part number
  col 2:  word index (0-based, resets each sentence)
  col 3:  word/token
  col 4:  POS tag
  cols 5-15: various annotations (parse, predicate, etc.)
  col 16 (last): coreference annotation

Coreference column notation:
  '-'       → not part of any mention
  '(N)'     → singleton mention of cluster N (start and end at this token)
  '(N'      → start of a multi-token mention of cluster N
  'N)'      → end of a multi-token mention of cluster N
  '(N)|(M)' → two overlapping mentions at same token (pipe-separated)
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Token:
    idx: int    # word index within its sentence (0-based)
    text: str   # surface form
    coref: str  # raw coreference column value


@dataclass
class Sentence:
    sent_idx: int                          # 0-based sentence index within document
    tokens: List[Token] = field(default_factory=list)

    def plain_text(self) -> str:
        return " ".join(t.text for t in self.tokens)


@dataclass
class Mention:
    sent_idx: int    # sentence in which the mention starts
    start_tok: int   # inclusive start token index
    end_tok: int     # inclusive end token index
    cluster_id: int  # coreference cluster ID
    is_zero: bool = False  # True if this is a zero mention (dropped pronoun)

    @property
    def position_key(self) -> Tuple[int, int, int]:

        """Unique positional identifier: (sent_idx, start_tok, end_tok)."""
        return (self.sent_idx, self.start_tok, self.end_tok)


@dataclass
class Document:
    doc_id: str                                      # unique key: '<base>#p<part>'
    language: str = ""                               # 'hi', 'ta', 'bn'
    sentences: List[Sentence] = field(default_factory=list)
    mentions: List[Mention] = field(default_factory=list)
    clusters: Dict[int, List[Mention]] = field(default_factory=dict)
    base_doc_id: str = ""                            # id as written in '#begin document (…)'
    part: int = 0                                    # part number from the same header


# ---------------------------------------------------------------------------
# Coreference column parsing
# ---------------------------------------------------------------------------

def _parse_coref_events(coref_str: str) -> List[Tuple[str, int]]:
    """
    Parse a coreference column value into a list of (event_type, cluster_id).

    event_type is one of: 'singleton', 'open', 'close'
    """
    if coref_str in ("-", "*", ""):
        return []

    events: List[Tuple[str, int]] = []
    for part in coref_str.split("|"):
        part = part.strip()
        if not part or part in ("-", "*"):
            continue
        if re.match(r"^\(\d+\)$", part):
            events.append(("singleton", int(part[1:-1])))
        elif re.match(r"^\(\d+$", part):
            events.append(("open", int(part[1:])))
        elif re.match(r"^\d+\)$", part):
            events.append(("close", int(part[:-1])))
    return events


# ---------------------------------------------------------------------------
# Mention extraction
# ---------------------------------------------------------------------------

def _extract_mentions(
    sentences: List[Sentence],
) -> Tuple[List[Mention], Dict[int, List[Mention]]]:
    """
    Walk all tokens and extract mention spans from coreference annotations.

    Returns:
        mentions   — flat list of Mention objects
        clusters   — dict mapping cluster_id → [Mention, ...]
    """
    mentions: List[Mention] = []
    # cluster_id → stack of (sent_idx, start_tok) for spans currently open.
    # A stack (not a single value) is required because the same cluster can be
    # nested inside itself, e.g. '(3(3' … '3)3)'.
    open_spans: Dict[int, List[Tuple[int, int]]] = {}

    def _close_dangling(sent: Sentence) -> None:
        """Close spans still open at the end of *sent* (mentions never cross
        sentence boundaries in this format; an unclosed span is malformed)."""
        last_idx = sent.tokens[-1].idx if sent.tokens else 0
        for cid, stack in open_spans.items():
            for s_idx, s_tok in stack:
                mentions.append(
                    Mention(sent_idx=s_idx, start_tok=s_tok,
                            end_tok=last_idx, cluster_id=cid)
                )
        open_spans.clear()

    for sent in sentences:
        for tok in sent.tokens:
            for etype, cid in _parse_coref_events(tok.coref):
                if etype == "singleton":
                    mentions.append(
                        Mention(
                            sent_idx=sent.sent_idx,
                            start_tok=tok.idx,
                            end_tok=tok.idx,
                            cluster_id=cid,
                        )
                    )
                elif etype == "open":
                    open_spans.setdefault(cid, []).append((sent.sent_idx, tok.idx))
                elif etype == "close":
                    stack = open_spans.get(cid)
                    if stack:
                        # Innermost open span closes first (LIFO).
                        s_idx, s_tok = stack.pop()
                        if not stack:
                            del open_spans[cid]
                        mentions.append(
                            Mention(
                                sent_idx=s_idx,
                                start_tok=s_tok,
                                end_tok=tok.idx,
                                cluster_id=cid,
                            )
                        )
        # Never carry an open span into the next sentence: end_tok would then be
        # an index into a *different* sentence, silently corrupting the span.
        if open_spans:
            _close_dangling(sent)

    mentions = _dedupe_span_annotations(mentions)

    clusters: Dict[int, List[Mention]] = {}
    for m in mentions:
        clusters.setdefault(m.cluster_id, []).append(m)

    return mentions, clusters


def _dedupe_span_annotations(mentions: List[Mention]) -> List[Mention]:
    """
    Keep exactly one annotation per (sent_idx, start_tok, end_tok) span.

    The source data does annotate a single span for two clusters at once —
    '(3|(4' … '3)|4)' is valid CoNLL and appears on 438 of the 67,984 test
    mentions (0.64%), in 31% of test documents. Downstream, a span is
    identified everywhere by its position key, and the task itself asks the
    model for exactly one cluster number per '<m>…</m>#MASK' — so a
    two-cluster span is not representable either in the gold structures or in
    anything the model can predict.

    Left implicit, the surplus annotation still breaks things: gold and
    predicted clusterings each keep whichever copy their own dict-building
    happened to see last, and the two can disagree, which shows up as mentions
    that appear mislinked when the linking was in fact correct.

    So the choice is made here, once, and deterministically: the lowest cluster
    id wins. Which copy survives matters less than that every consumer sees the
    same one. Mentions keep their original order; only surplus copies are
    dropped.
    """
    best: Dict[Tuple[int, int, int], int] = {}
    for m in mentions:
        key = m.position_key
        if key not in best or m.cluster_id < best[key]:
            best[key] = m.cluster_id

    kept: List[Mention] = []
    seen: set = set()
    for m in mentions:
        key = m.position_key
        if key in seen or m.cluster_id != best[key]:
            continue
        seen.add(key)
        kept.append(m)
    return kept


# ---------------------------------------------------------------------------
# File / directory loading
# ---------------------------------------------------------------------------

def parse_conll_file(filepath: str, language: str = "") -> List[Document]:
    """
    Parse one CoNLL file and return a list of Document objects.
    A single file may contain multiple documents (separated by #begin/#end).
    """
    documents: List[Document] = []

    # Mutable state for the current document being built
    current_doc_id: Optional[str] = None
    current_part: int = 0
    current_sentences: List[Sentence] = []
    current_tokens: List[Token] = []
    sent_idx: int = 0

    # ------------------------------------------------------------------
    def _flush_sentence() -> None:
        nonlocal sent_idx, current_tokens
        if current_tokens:
            current_sentences.append(
                Sentence(sent_idx=sent_idx, tokens=list(current_tokens))
            )
            current_tokens = []
            sent_idx += 1

    def _flush_document() -> None:
        nonlocal current_doc_id, current_part, current_sentences, sent_idx
        _flush_sentence()
        if current_doc_id is not None and current_sentences:
            mentions, clusters = _extract_mentions(current_sentences)
            documents.append(
                Document(
                    # Parts of the same OntoNotes document are independent
                    # documents: sentence indices and cluster ids both restart
                    # at every '#begin document'. They must therefore not share
                    # a doc_id, or their mention keys collide downstream.
                    doc_id=f"{current_doc_id}#p{current_part}",
                    language=language,
                    sentences=list(current_sentences),
                    mentions=mentions,
                    clusters=clusters,
                    base_doc_id=current_doc_id,
                    part=current_part,
                )
            )
        current_doc_id = None
        current_part = 0
        current_sentences = []
        sent_idx = 0

    # ------------------------------------------------------------------
    with open(filepath, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")

            if line.startswith("#begin document"):
                _flush_document()
                m = re.match(r"#begin document \((.+?)\)(?:;\s*part\s*(\d+))?", line)
                current_doc_id = (
                    m.group(1) if m else os.path.splitext(os.path.basename(filepath))[0]
                )
                current_part = int(m.group(2)) if (m and m.group(2)) else 0

            elif line.startswith("#end document"):
                _flush_document()

            elif line.strip() == "":
                _flush_sentence()

            else:
                cols = line.split("\t")
                if len(cols) < 4:
                    continue
                try:
                    tok_idx = int(cols[2])
                except ValueError:
                    continue
                word = cols[3]
                coref = cols[-1] if len(cols) > 4 else "-"
                current_tokens.append(Token(idx=tok_idx, text=word, coref=coref))

    # Finish any trailing document not closed by #end document
    _flush_document()
    return documents


def load_conll_dir(
    data_dir: str,
    language_filter: Optional[List[str]] = None,
    language: str = "",
    recursive: bool = False,
) -> List[Document]:
    """
    Load all .conll files from *data_dir*.

    Args:
        data_dir       — directory to search
        language_filter — if given, only load files whose name contains one
                          of these substrings (e.g. ['hin_Deva', 'tam_Taml'])
        language       — language code to attach to loaded documents
        recursive      — whether to walk sub-directories
    """
    docs: List[Document] = []
    seen_ids: Dict[str, int] = {}
    if not os.path.isdir(data_dir):
        return docs

    if recursive:
        walk_iter = (
            (dirpath, fnames)
            for dirpath, _, fnames in os.walk(data_dir)
        )
    else:
        walk_iter = [(data_dir, os.listdir(data_dir))]

    for dirpath, fnames in walk_iter:
        for fname in sorted(fnames):
            # Accept both .conll and files that end with _conll (e.g. mujadia _gold_conll)
            if not (fname.endswith(".conll") or fname.endswith("_conll")):
                continue
            if language_filter and not any(code in fname for code in language_filter):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                file_docs = parse_conll_file(fpath, language=language)
                # Two files may still declare the same (doc_id, part) pair —
                # disambiguate so every Document keeps a unique key.
                for doc in file_docs:
                    n = seen_ids.get(doc.doc_id, 0)
                    seen_ids[doc.doc_id] = n + 1
                    if n:
                        doc.doc_id = f"{doc.doc_id}#d{n}"
                docs.extend(file_docs)
            except Exception as exc:
                print(f"[warn] failed to parse {fpath}: {exc}")

    return docs
