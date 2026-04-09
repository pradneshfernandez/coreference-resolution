"""
preprocessor.py — Convert parsed Documents into CorefInst training/inference examples.

Pipeline per document:
  1. Pack sentences into non-overlapping frames (up to max_tokens_per_frame tokens each).
  2. For each consecutive frame pair (frame_k, frame_{k+1}):
     a. Walk mentions in both frames; assign local cluster numbers 0, 1, 2, …
        in order of first appearance. Clusters shared across the two frames get
        the SAME local number (cross-frame consistency within an input).
     b. Wrap each mention span with <m>…</m>#MASK (masked) or <m>…</m>#N (output).
     c. Join the two masked frames with ' [MID] ' to form the full masked input.
  3. Return FrameExample objects suitable for training or inference.

Frame pairs overlap at the INPUT level — frame_k appears as the "after" frame in
example (k-1) and as the "before" frame in example k.  This overlap is what
Algorithm 1 (postprocessor.py) exploits at inference time.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .conll_parser import Document, Mention, Sentence


# ---------------------------------------------------------------------------
# Instruction sets (Table 4 of the paper)
# ---------------------------------------------------------------------------

_INSTRUCTIONS: Dict[int, str] = {
    1: (
        "The text is in {language}.\n"
        "You are a coreference resolver.\n"
        "Rewrite the sentence considering these rules:\n"
        "Mentions are in <m>...</m>#MASK format.\n"
        "Group the mentions that refer to same real-world entity.\n"
        "If mentions refer to same thing write the same number instead of MASK.\n"
        "If mentions represent different things write another number.\n"
        "Where you see </z>@ there is a zero mention, which is normally not written "
        "but you also need to link them with other mentions."
    ),
    2: (
        "Identify instances of coreference where different mentions refer to the same entity.\n"
        "Rewrite the passage, tagging each term with a unique identifier and indicating "
        "coreferential relationships.\n"
        "Ensure accuracy and consider context.\n"
        "Fill MASK with unique number for every entity. Be sure that they represent exactly "
        "the same entity, not similar.\n"
        "Example output format:\n"
        "... <m>Bertrand Russell </m>#MASK is a good author, I love <m>The History of Western "
        "Philosophy </m>#MASK ...\n"
        "... <m>Bertrand Russell </m>#0 is a good author, I love <m>The History of Western "
        "Philosophy </m>#1 ...\n"
        "Where you see </z>@ there is a zero mention, which is normally not written "
        "but you also need to link them with other mentions."
    ),
    3: (
        "For every mention in <m> </m> tags examine if there is any coreferential/coherent mention.\n"
        "If two mentions represent the same entity write the same number instead of MASK "
        "after closing mention tag </m>.\n"
        "Do not change anything else other than MASK.\n"
        "Where you see </z>@ there is a zero mention, which is normally not written "
        "but you also need to link them with other mentions."
    ),
    4: (
        "For every mention in <m> </m> tags examine if there is any coreferential mention.\n"
        "If two mentions represent the same entity write the same number instead of MASK "
        "after closing mention tag </m>.\n"
        "For example: author and book represent different entities.\n"
        "Do not change anything else other than MASK.\n"
        "Where you see </z>@ there is a zero mention, which is normally not written "
        "but you also need to link them with other mentions."
    ),
    5: (
        "For every mention in <m> </m> tags examine if there is any coreferential/coherent mention.\n"
        "If two mentions represent the same entity write the same number instead of MASK "
        "after closing mention tag </m>.\n"
        "For example: author and book represent different entities.\n"
        "Do not change anything else other than MASK.\n"
        "Where you see </z>@ there is a zero mention, which is normally not written "
        "but you also need to link them with other mentions."
    ),
}


def get_instruction(instruction_id: int, language: str = "") -> str:
    template = _INSTRUCTIONS.get(instruction_id, _INSTRUCTIONS[5])
    return template.replace("{language}", language or "the given language")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameExample:
    """One training/inference example (a consecutive frame pair)."""
    doc_id: str
    language: str
    instruction: str
    masked_input: str       # '<masked_before> [MID] <masked_after>'  with #MASK tokens
    output: str             # same, but with #MASK replaced by local cluster numbers
    # Metadata needed by the postprocessor
    before_mentions: List[dict]   # [{sent_idx, start_tok, end_tok, cluster_id, local_no}, ...]
    after_mentions: List[dict]    # same for after frame
    before_sent_indices: List[int]
    after_sent_indices: List[int]


# ---------------------------------------------------------------------------
# Frame packing
# ---------------------------------------------------------------------------

def _pack_frames(sentences: List[Sentence], max_tokens: int) -> List[List[int]]:
    """
    Pack sentences into non-overlapping frames.

    Each frame is a list of sentence indices. Sentences are greedily packed
    until adding the next sentence would exceed *max_tokens*. A sentence that
    alone exceeds *max_tokens* still gets its own frame.

    Guarantee: always produces at least 2 frames when there are >= 2 sentences,
    by halving max_tokens if a naive pack would yield only 1 frame. This ensures
    every document with 2+ sentences produces at least one frame pair (example).
    """
    def _greedy_pack(sents: List[Sentence], budget: int) -> List[List[int]]:
        frames: List[List[int]] = []
        current: List[int] = []
        count = 0
        for sent in sents:
            n = len(sent.tokens)
            if current and count + n > budget:
                frames.append(current)
                current = [sent.sent_idx]
                count = n
            else:
                current.append(sent.sent_idx)
                count += n
        if current:
            frames.append(current)
        return frames

    if len(sentences) < 2:
        return _greedy_pack(sentences, max_tokens)

    frames = _greedy_pack(sentences, max_tokens)

    # If all sentences landed in a single frame, force at least 2 by halving
    # the budget until we get 2 frames or reach single-sentence frames.
    budget = max_tokens
    while len(frames) < 2 and budget > 1:
        budget = max(1, budget // 2)
        frames = _greedy_pack(sentences, budget)

    return frames


# ---------------------------------------------------------------------------
# Masked-sentence construction
# ---------------------------------------------------------------------------

def _build_masked_sentence(
    sentence: Sentence,
    mentions_in_sent: List[Mention],
    cluster_id_to_local: Dict[int, int],
    next_local_id: List[int],           # single-element mutable counter
) -> Tuple[str, str, List[dict]]:
    """
    Produce the masked and output versions of one sentence.

    Overlapping / nested mentions are resolved by taking the outermost mention
    (longest span) at each token position.  Inner mentions subsumed by an outer
    mention are silently skipped.

    Returns:
        masked_text   — sentence text with <m>…</m>#MASK inserted
        output_text   — sentence text with <m>…</m>#<local_no> inserted
        mention_info  — list of dicts with mention metadata
    """
    tokens = sentence.tokens
    n = len(tokens)

    # Sort: by start ascending, then by span length descending (outermost first)
    mentions_sorted = sorted(
        mentions_in_sent,
        key=lambda m: (m.start_tok, -(m.end_tok - m.start_tok)),
    )

    masked_parts: List[str] = []
    output_parts: List[str] = []
    mention_info: List[dict] = []

    pos = 0
    # Keep track of surviving mentions (remove those consumed as inner spans)
    remaining = list(mentions_sorted)

    while pos < n:
        # Find mentions starting at current position
        starts_here = [m for m in remaining if m.start_tok == pos]

        if starts_here:
            # Take the outermost (longest) mention starting here
            outer = max(starts_here, key=lambda m: m.end_tok - m.start_tok)

            # Assign a local cluster number (consistent within this frame pair)
            if outer.cluster_id not in cluster_id_to_local:
                cluster_id_to_local[outer.cluster_id] = next_local_id[0]
                next_local_id[0] += 1
            local_no = cluster_id_to_local[outer.cluster_id]

            if outer.is_zero:
                # Zero mention: inserted at current position as </z>#MASK or </z>#N
                masked_parts.append("</z>@MASK")
                output_parts.append(f"</z>#{local_no}")
                # Don't advance pos, zero mentions are "inserted" between tokens
                # but we need to track it. We'll advance pos in the next iteration.
            else:
                # Clamp to sentence bounds for text extraction only; keep original
                # end_tok in the position key so it matches gold annotation keys.
                text_end = min(outer.end_tok, n - 1)
                span_text = " ".join(tokens[i].text for i in range(outer.start_tok, text_end + 1))

                masked_parts.append(f"<m>{span_text}</m>#MASK")
                output_parts.append(f"<m>{span_text}</m>#{local_no}")
                pos = text_end + 1

            mention_info.append(
                {
                    "sent_idx": sentence.sent_idx,
                    "start_tok": outer.start_tok,
                    "end_tok": outer.end_tok,   # original key — must match gold
                    "cluster_id": outer.cluster_id,
                    "local_no": local_no,
                    "is_zero": outer.is_zero,
                }
            )

            # Remove all mentions fully contained within this span (if overt)
            # or just this specific zero mention.
            if outer.is_zero:
                remaining.remove(outer)
            else:
                remaining = [
                    m for m in remaining
                    if not (m.start_tok >= outer.start_tok and m.end_tok <= outer.end_tok)
                ]

        else:
            masked_parts.append(tokens[pos].text)
            output_parts.append(tokens[pos].text)
            pos += 1

    return " ".join(masked_parts), " ".join(output_parts), mention_info


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def create_frame_examples(
    doc: Document,
    instruction_id: int = 5,
    max_tokens_per_frame: int = 256,
    skip_empty: bool = True,
) -> List[FrameExample]:
    """
    Convert a Document into a list of FrameExample training instances.

    One FrameExample is created per consecutive frame pair.  The example
    covers (frame_k [MID] frame_{k+1}) with local cluster numbers assigned
    consistently across both frames.

    Args:
        doc                  — parsed Document
        instruction_id       — which instruction set to use (1–5; default 5)
        max_tokens_per_frame — approximate token budget per frame
        skip_empty           — skip examples with no mentions in either frame
    """
    if not doc.sentences:
        return []

    instruction = get_instruction(instruction_id, doc.language)

    # Build per-sentence mention lookup
    mentions_by_sent: Dict[int, List[Mention]] = {}
    for m in doc.mentions:
        mentions_by_sent.setdefault(m.sent_idx, []).append(m)

    # Build sentence index map
    sent_map: Dict[int, Sentence] = {s.sent_idx: s for s in doc.sentences}

    # Pack into frames
    frame_defs = _pack_frames(doc.sentences, max_tokens_per_frame)
    if len(frame_defs) < 2:
        return []   # Need at least two frames to form a pair

    examples: List[FrameExample] = []

    for fi in range(len(frame_defs) - 1):
        before_idxs = frame_defs[fi]
        after_idxs = frame_defs[fi + 1]

        before_sents = [sent_map[i] for i in before_idxs if i in sent_map]
        after_sents = [sent_map[i] for i in after_idxs if i in sent_map]
        if not before_sents or not after_sents:
            continue

        # Shared local cluster numbering for the entire frame pair
        cluster_id_to_local: Dict[int, int] = {}
        next_local_id = [0]

        # ---- before frame ----
        before_masked_parts: List[str] = []
        before_output_parts: List[str] = []
        before_mentions_info: List[dict] = []

        for sent in before_sents:
            mlist = mentions_by_sent.get(sent.sent_idx, [])
            m_txt, o_txt, minfo = _build_masked_sentence(
                sent, mlist, cluster_id_to_local, next_local_id
            )
            before_masked_parts.append(m_txt)
            before_output_parts.append(o_txt)
            before_mentions_info.extend(minfo)

        # ---- after frame ----
        after_masked_parts: List[str] = []
        after_output_parts: List[str] = []
        after_mentions_info: List[dict] = []

        for sent in after_sents:
            mlist = mentions_by_sent.get(sent.sent_idx, [])
            m_txt, o_txt, minfo = _build_masked_sentence(
                sent, mlist, cluster_id_to_local, next_local_id
            )
            after_masked_parts.append(m_txt)
            after_output_parts.append(o_txt)
            after_mentions_info.extend(minfo)

        # Skip examples with no mentions if requested
        if skip_empty and not before_mentions_info and not after_mentions_info:
            continue

        masked_input = " ".join(before_masked_parts) + " [MID] " + " ".join(after_masked_parts)
        output = " ".join(before_output_parts) + " [MID] " + " ".join(after_output_parts)

        examples.append(
            FrameExample(
                doc_id=doc.doc_id,
                language=doc.language,
                instruction=instruction,
                masked_input=masked_input,
                output=output,
                before_mentions=before_mentions_info,
                after_mentions=after_mentions_info,
                before_sent_indices=before_idxs,
                after_sent_indices=after_idxs,
            )
        )

    return examples
