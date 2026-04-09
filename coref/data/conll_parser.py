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
    doc_id: str
    language: str = ""                               # 'hi', 'ta', 'bn'
    sentences: List[Sentence] = field(default_factory=list)
    mentions: List[Mention] = field(default_factory=list)
    clusters: Dict[int, List[Mention]] = field(default_factory=dict)


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
    # cluster_id → (sent_idx, start_tok) for spans currently open
    open_spans: Dict[int, Tuple[int, int]] = {}

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
                    # Guard against duplicate opens (malformed data)
                    if cid not in open_spans:
                        open_spans[cid] = (sent.sent_idx, tok.idx)
                elif etype == "close":
                    if cid in open_spans:
                        s_idx, s_tok = open_spans.pop(cid)
                        mentions.append(
                            Mention(
                                sent_idx=s_idx,
                                start_tok=s_tok,
                                end_tok=tok.idx,
                                cluster_id=cid,
                            )
                        )

    # Close any spans left open (malformed annotation — treat as singletons)
    for cid, (s_idx, s_tok) in open_spans.items():
        mentions.append(
            Mention(sent_idx=s_idx, start_tok=s_tok, end_tok=s_tok, cluster_id=cid)
        )

    clusters: Dict[int, List[Mention]] = {}
    for m in mentions:
        clusters.setdefault(m.cluster_id, []).append(m)

    return mentions, clusters


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
        nonlocal current_doc_id, current_sentences, sent_idx
        _flush_sentence()
        if current_doc_id is not None and current_sentences:
            mentions, clusters = _extract_mentions(current_sentences)
            documents.append(
                Document(
                    doc_id=current_doc_id,
                    language=language,
                    sentences=list(current_sentences),
                    mentions=mentions,
                    clusters=clusters,
                )
            )
        current_doc_id = None
        current_sentences = []
        sent_idx = 0

    # ------------------------------------------------------------------
    with open(filepath, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")

            if line.startswith("#begin document"):
                _flush_document()
                m = re.match(r"#begin document \((.+?)\)", line)
                current_doc_id = (
                    m.group(1) if m else os.path.splitext(os.path.basename(filepath))[0]
                )

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
                docs.extend(file_docs)
            except Exception as exc:
                print(f"[warn] failed to parse {fpath}: {exc}")

    return docs
