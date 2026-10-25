"""Text annotation (gazetteer NER, relation cues, negation / hedge cues) and claim parsing."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
import numpy as np

TOKEN = re.compile(r"[a-z0-9']+")
CUES = {0: ["won", "clinched", "awarded", "took home"], 1: ["acquired", "purchase", "bought", "snapped"],
        2: ["appointed", "named", "takes over", "serves"], 3: ["visited", "arrived", "travelling", "talks"],
        4: ["launched", "unveiled", "introduced", "rolled out"], 5: ["resigned", "stepped down", "quit", "left"]}
NEG = ["false", "denied", "no evidence", "debunked", "did not", "not "]
HEDGE = ["reportedly", "sources claim", "unconfirmed"]
OBJ_TYPE_TO_REL = {"award": 0, "role": 2, "city": 3, "product": 4}


def tokenize(text: str):
    return TOKEN.findall(text.lower())


@dataclass
class Sentence:
    text: str
    ents: list            # [(type, canonical, token_position)]
    neg: bool
    hedge: bool
    rels: set


class Annotator:
    def __init__(self, gaz: dict):
        self.gaz = gaz
        self.maxlen = max(len(k.split()) for k in gaz)

    def entities(self, tokens):
        out, i, n = [], 0, len(tokens)
        while i < n:
            hit = None
            for L in range(min(self.maxlen, n - i), 0, -1):
                key = " ".join(tokens[i:i + L])
                if key in self.gaz:
                    hit = (L, self.gaz[key]); break
            if hit:
                out.append((hit[1][0], hit[1][1], i)); i += hit[0]
            else:
                i += 1
        return out

    def annotate(self, text: str):
        sents = []
        for s in re.split(r"(?<=[a-z0-9])\.\s+|\.$", text):
            s = s.strip()
            if not s:
                continue
            low = s.lower(); toks = tokenize(s)
            rels = {r for r, cs in CUES.items() if any(c in low for c in cs)}
            sents.append(Sentence(s, self.entities(toks), any(n in low for n in NEG), any(h in low for h in HEDGE), rels))
        return sents


@dataclass
class ParsedClaim:
    subject: str | None
    rel: int | None
    obj: str | None
    rel_from_cue: bool
    obj_type: str | None = None


def parse_claim(text: str, ann: Annotator) -> ParsedClaim:
    toks = tokenize(text)
    ents = ann.entities(toks)
    if len(ents) < 2:
        return ParsedClaim(None, None, None, False, None)
    subj, obj = ents[0], ents[-1]
    low = text.lower()
    cue_rels = [r for r, cs in CUES.items() if any(c in low for c in cs)]
    if obj[0] in OBJ_TYPE_TO_REL:
        rel = OBJ_TYPE_TO_REL[obj[0]]
    else:                                           # company object: resigned (person subject) or acquired (company subject)
        rel = 5 if subj[0] == "person" else 1
    from_cue = False
    if len(cue_rels) == 1:
        from_cue = True
        if obj[0] in OBJ_TYPE_TO_REL or cue_rels[0] in (1, 5):
            rel = cue_rels[0]
    return ParsedClaim(subj[1], rel, obj[1], from_cue, obj[0])
