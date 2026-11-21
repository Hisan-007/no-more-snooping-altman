"""The local vault: the ONLY place raw personal data lives.

Agents in the cloud zone never get vault contents - tools hand them placeholders
(NAME_1, ADDR_2, ...). Everything that is logged or streamed goes through
`scrub()` first, so the audit log never contains a real value either.
"""
import hashlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PREFIX = {"name": "NAME", "address": "ADDR", "birth_date": "DOB", "id_number": "ID", "phone": "PHONE",
          "email": "EMAIL", "iban": "IBAN", "date": "DATE", "company": "ORG"}


def h(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()[:12]


def _norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


@dataclass
class Doc:
    doc_id: str
    name: str
    img: Any                                  # PIL image (original)
    lines: List[dict] = field(default_factory=list)
    subject: str = ""
    items: List[dict] = field(default_factory=list)     # the subject's data (to replace)
    kept: List[dict] = field(default_factory=list)      # deliberately kept (someone else's)
    quarantined: List[dict] = field(default_factory=list)  # lines flagged as prompt injection
    llm_items: List[dict] = field(default_factory=list)
    gender: Optional[str] = None
    levels: Dict[int, str] = field(default_factory=dict)  # item id -> forced erase level
    erased: Any = None
    anonymized: Any = None
    render_info: List[dict] = field(default_factory=list)
    faker: Any = None
    verify: dict = field(default_factory=dict)
    placeholders: Dict[str, str] = field(default_factory=dict)  # normalised real value -> placeholder
    fakes: Dict[str, str] = field(default_factory=dict)         # real value -> fake value

    # ---------------------------------------------------------------- placeholders
    def placeholder(self, label: str, text: str) -> str:
        key = _norm(text)
        if key not in self.placeholders:
            p = PREFIX.get(label, "PII")
            n = sum(1 for v in self.placeholders.values() if v.startswith(p + "_")) + 1
            self.placeholders[key] = f"{p}_{n}"
        return self.placeholders[key]

    def subject_placeholder(self):
        return self.placeholder("name", self.subject) if self.subject else "SUBJECT_?"

    def secrets(self) -> List[str]:
        """Every raw value that must never leave the Mac (longest first), incl. name tokens."""
        vals = {it["text"] for it in self.items}
        names = {it["text"] for it in self.items if it["label"] == "name"}
        if self.subject:
            vals.add(self.subject)
            names.add(self.subject)
        for v in names:  # surname / first name on their own too (not city names etc.)
            for tok in re.split(r"[\s,]+", v):
                if len(tok) >= 4 and not tok.isdigit():
                    vals.add(tok)
        return sorted((v for v in vals if len(v.strip()) >= 3), key=len, reverse=True)

    # ---------------------------------------------------------------- real <-> fake
    def pairs(self) -> List[tuple]:
        """(real, fake) incl. name tokens, longest first - used to pseudonymise / re-identify text."""
        out = dict(self.fakes)
        if self.faker is not None:
            for (role, tok), fake in self.faker.tok.items():
                real = next((w for it in self.items if it["label"] == "name"
                             for w in re.split(r"[\s,]+", it["text"]) if w.lower() == tok), None)
                if real and len(real) >= 3:
                    out.setdefault(real, fake)
        return sorted(out.items(), key=lambda kv: len(kv[0]), reverse=True)

    def pseudonymise(self, text: str):
        n = 0
        for real, fake in self.pairs():
            text, k = re.subn(re.escape(real), fake, text, flags=re.I)
            n += k
        return text, n

    def reidentify(self, text: str):
        n = 0
        for real, fake in sorted(self.pairs(), key=lambda kv: len(kv[1]), reverse=True):
            text, k = re.subn(r"(?<![\w])" + re.escape(fake) + r"(?![\w])", real, text)
            n += k
        return text, n

    def scrub(self, text: str) -> str:
        """Replace every raw value with its placeholder (or a hash) - for logs and cloud payloads."""
        if not text:
            return text
        for v in self.secrets():
            ph = self.placeholders.get(_norm(v)) or h(v)
            text = re.sub(re.escape(v), ph, text, flags=re.I)
        return text

    def leaks(self, text: str) -> List[str]:
        """Raw values found in `text` (egress firewall)."""
        low = text.lower()
        return [v for v in self.secrets() if v.lower() in low]


class Vault:
    def __init__(self):
        self.docs: Dict[str, Doc] = {}

    def add(self, name, img) -> Doc:
        d = Doc(doc_id="doc_" + uuid.uuid4().hex[:8], name=name, img=img)
        self.docs[d.doc_id] = d
        return d

    def get(self, doc_id) -> Doc:
        if doc_id not in self.docs:
            raise KeyError(f"unknown doc_id {doc_id}")
        return self.docs[doc_id]

    def scrub_all(self, text: str) -> str:
        for d in self.docs.values():
            text = d.scrub(text)
        return text

    def leaks_all(self, text: str) -> List[str]:
        return [v for d in self.docs.values() for v in d.leaks(text)]


VAULT = Vault()
