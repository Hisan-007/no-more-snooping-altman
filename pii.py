"""Decide WHAT gets anonymised.

Criterion: only the personal data of the document's *data subject* - the one
private person the document is about / addressed to (patient, customer,
recipient, account holder). Everyone else stays: doctors, staff, signatories,
the sender, companies, clinics, insurers - and their addresses, phones, bank
and register numbers. Document dates, amounts and medical content stay too.

Gemma reads the OCR text (not the image) and names the data subject. Rules
then decide which spans are that person's data:

  name        every occurrence of the subject's name (full, surname, initials)
  address     the street / postcode+city lines directly under the subject's
              name, or on the same line as it
  birth_date  a date next to a birth-date label (geb., Geburtsdatum, DOB, ...)
  id_number   the value next to a personal-ID label (Versichertennummer,
              Kundennummer, Bestell-/Rechnungs-/Sendungsnummer, Steuer-ID, ...)
  email/phone only if Gemma says they're the subject's AND they sit next to the
              subject's name / address (or the email contains their name)
  iban        only if Gemma says it's the subject's, never in the footer
Every value found is then also replaced wherever it appears again on the page.
"""
import datetime as dt
import json
import re
import time
from difflib import SequenceMatcher

import requests

import detect as D

TITLES = {"herr", "herrn", "frau", "hr", "fr", "dr", "med", "prof", "dipl", "ing", "mr", "mrs", "ms",
          "miss", "patient", "patientin", "name", "familie"}
FOOTER_Y = 0.88  # below this fraction of the page height = issuer's small print

MONTHS = r"jan|feb|mär|mar|apr|mai|may|jun|jul|aug|sep|okt|oct|nov|dez|dec"
DATE_RE = re.compile(r"(?<![\d.])(\d{1,2}\.\s?\d{1,2}\.\s?(?:\d{4}|\d{2})|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}"
                     r"|\d{1,2}\.?\s?(?:%s)[a-zä]*\.?\s+\d{4})(?!\d)" % MONTHS, re.I)
DOB_LABEL = re.compile(r"(geb\.?|geboren(?:\s+am)?|geburtsdatum|geb\.?\s*-?\s*dat(?:um|\.)?|date\s+of\s+birth|"
                       r"birth\s*date|born|d\.?o\.?b\.?|\*)\s*[:.]?\s*$", re.I)
# labels whose value identifies the subject (or their case / purchase)
ID_LABEL = re.compile(
    r"(versicherten|vers\.|patienten|pat\.|kunden|kd\.|mitglieds|vertrags|fall|auftrags|bestell|rechnungs|"
    r"sendungs|personal|sozialversicherungs|sv|rentenversicherungs|order|invoice|customer|patient|member|"
    r"policy|tracking|account|case)[\s\-]*(nr\.?|nummer|number|no\.?|id)(?![a-zä])"
    r"|steuer-?\s?id|steuerliche\s+(id|identifikationsnummer)|identifikationsnummer|krankenversichertennummer",
    re.I)
# labels whose value belongs to the issuer, never the subject
DENY_LABEL = re.compile(r"ust|steuernummer|st\.?-?nr|hrb|hra|amtsgericht|registergericht|\bbic\b|swift|lanr|bsnr|"
                        r"\bik\b|institutionskennzeichen|arzt-?nr|betriebsstätte|artikel", re.I)
PHONE_RE = re.compile(r"(\+\d{1,3}[\s/]*(\(0\)\s*)?)?\(?0?\d{2,5}\)?[\s/\-]*\d[\d\s/\-]{4,}\d")
EMAIL_RE = re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+")
STREET_RE = re.compile(r"^[^\d,]{3,}?,?\s+\d{1,4}\s*[a-zA-Z]?(\s*[-/]\s*\d{1,4}\s*[a-zA-Z]?)?$")
CITY_RE = re.compile(r"^([A-Z]{1,2}-)?\d{4,5}\s+\S")
COUNTRIES = {"deutschland", "germany", "österreich", "austria", "schweiz", "switzerland", "de"}


def _n(s):
    return re.sub(r"[^0-9a-zäöüß]", "", s.lower())


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _h(ln):
    return max(1, ln["box"][3] - ln["box"][1])


# --------------------------------------------------------------------------- spans
def _item(lines, li, i, j, label, why, src="rule"):
    """Item for words i..j-1 of line li, in the shape the UI / renderer expect."""
    ln = lines[li]
    ws = ln["words"]
    text = ln["text"][ws[i]["start"]:ws[j - 1]["end"]]
    trail = re.match(r"[,;:.|]*", ln["text"][ws[j - 1]["end"]:]).group(0)
    box = _union([w["box"] for w in ws[i:j]])
    return {"label": label, "text": text, "ocr_text": text, "box": box, "gemma_box": box, "trail": trail,
            "full_line": i == 0 and j == len(ws), "snapped": True, "match": 1.0, "why": why, "src": src,
            "_span": (li, i, j)}


def _char_span(ln, a, b):
    """Words covering characters a..b of the line -> (i, j) or None."""
    ws = ln["words"]
    idx = [k for k, w in enumerate(ws) if w["end"] > a and w["start"] < b]
    return (idx[0], idx[-1] + 1) if idx else None


def _best_span(target, lines, line_ids, min_ratio=0.7):
    t = _n(target)
    if not t:
        return None
    nw = max(1, len(target.split()))
    best = None
    for li in line_ids:
        ln = lines[li]
        ws = ln["words"]
        for i in range(len(ws)):
            for j in range(i + 1, min(len(ws), i + nw + 2) + 1):
                r = SequenceMatcher(None, t, _n(ln["text"][ws[i]["start"]:ws[j - 1]["end"]])).ratio()
                if r >= min_ratio and (best is None or r > best[0]):
                    best = (r, li, i, j)
    return best


def _find(lines, text, li=None, min_ratio=0.7):
    """Locate an LLM-quoted fragment: on its line (or a neighbour), else anywhere at a stricter ratio."""
    if li is not None and 0 <= li < len(lines):
        b = _best_span(text, lines, [k for k in (li, li - 1, li + 1) if 0 <= k < len(lines)], min_ratio)
        if b:
            return b
    return _best_span(text, lines, range(len(lines)), max(min_ratio, 0.85))


def _tighten(lines, it):
    """Shrink a span to the value itself ("Telefon: 0176 555" -> "0176 555")."""
    li, i, j = it["_span"]
    ws, lab = lines[li]["words"], it["label"]
    if lab == "email":
        k = [k for k in range(i, j) if "@" in ws[k]["text"]]
        if k:
            i, j = k[0], k[-1] + 1
    elif lab in ("phone", "id_number", "iban"):
        d = [k for k in range(i, j) if re.search(r"\d", ws[k]["text"])]
        if d:
            a = d[0]
            if a - 1 >= i and re.fullmatch(r"[A-Z]{1,3}", ws[a - 1]["text"]):
                a -= 1  # "DE 124 ..." / "A 123 ..."
            i, j = a, d[-1] + 1
    elif lab == "birth_date":
        ln = lines[li]
        m = DATE_RE.search(ln["text"], ws[i]["start"], ws[j - 1]["end"])
        if m:
            i, j = _char_span(ln, m.start(), m.end())
    if (i, j) == it["_span"][1:]:
        return it
    return _item(lines, li, i, j, it["label"], it["why"], it["src"])


def _exact_spans(lines, text):
    """Spans whose text equals `text` (ignoring case / punctuation / spacing)."""
    t, nw = _n(text), max(1, len(text.split()))
    for li, ln in enumerate(lines):
        ws = ln["words"]
        for i in range(len(ws)):
            for j in range(i + max(1, nw - 1), min(len(ws), i + nw + 1) + 1):
                if _n(ln["text"][ws[i]["start"]:ws[j - 1]["end"]]) == t:
                    yield li, i, j


COMPANY_RE = re.compile(r"\b(gmbh|ag|se|kg|ohg|gbr|e\.?\s?v|mbh|ltd|inc|llc|co\.|praxis|klinik|klinikum|"
                        r"krankenhaus|zentrum|bank|versicherung|kasse|aok|dr\.|prof\.)\b", re.I)


# --------------------------------------------------------------------------- layout helpers
def _same_row_right(lines, li, x_from):
    ln = lines[li]
    cy = (ln["box"][1] + ln["box"][3]) / 2
    c = [k for k, o in enumerate(lines) if k != li and o["box"][1] <= cy <= o["box"][3] and o["box"][0] >= x_from - 5]
    return min(c, key=lambda k: lines[k]["box"][0]) if c else None


def _below(lines, li, align="left"):
    """Next line directly below li (same block)."""
    ln = lines[li]
    x1, y2, x2, h = ln["box"][0], ln["box"][3], ln["box"][2], _h(ln)
    c = [k for k, o in enumerate(lines)
         if k != li and y2 - 0.3 * h <= o["box"][1] <= y2 + 1.4 * h
         and (abs(o["box"][0] - x1) <= 1.5 * h if align == "left" else abs(o["box"][2] - x2) <= 1.5 * h)]
    return min(c, key=lambda k: lines[k]["box"][1]) if c else None


def _value_after(lines, li, char_end, kind):
    """Value following a label that ends at char_end on line li: same line, else the line to the
    right on the same row, else the line below. kind: 'id' | 'date'. Returns (li, i, j) or None."""
    def extract(lj, start):
        ln = lines[lj]
        if kind == "date":
            m = DATE_RE.search(ln["text"], start)
            if m and not ln["text"][start:m.start()].strip(" :.-\t"):  # date must come right after the label
                s = _char_span(ln, m.start(), m.end())
                return (lj, *s) if s else None
            return None
        ws = [k for k, w in enumerate(ln["words"]) if w["start"] >= start]
        run = []
        for k in ws:
            w = ln["words"][k]["text"].strip(":#")
            if re.search(r"\d", w) or (re.fullmatch(r"[A-Z]{1,3}", w) and not run):
                run.append(k)
            elif w in {":", "#", "-"} and not run:
                continue
            else:
                break
            if len(run) >= 6:
                break
        while run and not re.search(r"\d", ln["words"][run[-1]]["text"]):
            run.pop()
        return (lj, run[0], run[-1] + 1) if run else None

    v = extract(li, char_end)
    if v:
        return v
    if lines[li]["text"][char_end:].strip(" :.#-"):
        return None  # something else follows the label on this line
    lab_x2 = lines[li]["box"][2]
    for lj in (_same_row_right(lines, li, lab_x2), _below(lines, li)):
        if lj is not None:
            v = extract(lj, 0)
            if v:
                return v
    return None


# --------------------------------------------------------------------------- subject
def _subject_tokens(subject):
    """Name tokens; single letters kept as initials ("Maxim M")."""
    return [t for t in (_n(x) for x in subject.replace(",", " ").split()) if t and t not in TITLES]


def _fuzzy(a, b):
    return a == b or (len(a) >= 4 and len(b) >= 4 and SequenceMatcher(None, a, b).ratio() >= 0.8)


def _compatible(a, b):
    """Same person written differently? 'Maxim M' ~ 'Maximilian Mustermann', 'M. Mustermann' ~ 'Max Mustermann'."""
    ta, tb = _subject_tokens(a), _subject_tokens(b)
    if not ta or not tb or len(ta) != len(tb):
        return False
    full_hits = 0
    for x, y in zip(ta, tb):
        if len(x) > 1 and len(y) > 1:
            if not _fuzzy(x, y):
                return False
            full_hits += 1
        elif x[0] != y[0]:
            return False
    return full_hits >= 1


def name_spans(lines, subject):
    """Every occurrence of the subject's name: 'Max Mustermann', 'Mustermann, Max', 'Herr Mustermann',
    'M. Mustermann', and variants like 'Maxim M' for 'Maximilian Mustermann'. A lone first name is not enough."""
    toks = _subject_tokens(subject)
    full_toks = [t for t in toks if len(t) > 1]
    if not full_toks:
        return []
    surname = toks[-1] if len(toks[-1]) > 1 else None
    initials = {t[0] for t in toks}
    out = []
    for li, ln in enumerate(lines):
        ws = ln["words"]
        i = 0
        while i < len(ws):
            j, full, n_init = i, [], 0
            while j < len(ws):
                w = _n(ws[j]["text"])
                t = next((t for t in full_toks if _fuzzy(w, t)), None) if len(w) > 1 else None
                if t:
                    full.append(t)
                elif len(w) == 1 and w in initials and (full or n_init == 0):
                    n_init += 1
                else:
                    break
                j += 1
            ok = full and (len(full) >= 2 or (full[0] == surname and len(surname) >= 3) or n_init >= 1)
            if ok:
                while j > i + 1 and len(_n(ws[i]["text"])) == 1 and not _fuzzy(_n(ws[i + 1]["text"]), full[0]):
                    i += 1  # stray leading letter
                out.append(_item(lines, li, i, j, "name", "the subject's name"))
                i = j
            else:
                i += 1
    return out


def subject_aliases(lines, subject):
    """The subject plus differently-written versions of it heading other address blocks."""
    return [subject] + [b for b in block_names(lines) if _n(b) != _n(subject) and _compatible(b, subject)]


def address_block(lines, name_item):
    """Street / postcode+city lines directly under a name line (recipient block style)."""
    li = name_item["_span"][0]
    ln = lines[li]
    nb = name_item["box"]
    if (nb[2] - nb[0]) < 0.6 * (ln["box"][2] - ln["box"][0]):
        return []  # name is part of a longer line (e.g. "Patient: Max ..., geb. ...") - no block below
    align = "left"
    out, cur = [], li
    for _ in range(4):
        nxt = _below(lines, cur, align)
        if nxt is None:
            break
        t = lines[nxt]["text"].strip()
        if t.lower() in COUNTRIES:
            cur = nxt
            continue  # country stays, but the block may continue
        if STREET_RE.match(t) or CITY_RE.match(t) or t.lower().startswith(("c/o", "postfach")):
            n = len(lines[nxt]["words"])
            if n:
                out.append(_item(lines, nxt, 0, n, "address", "address under the subject's name"))
            cur = nxt
            continue
        break
    return out


def inline_address(lines, name_item):
    """'Max Mustermann, Lindenweg 12, 38106 Braunschweig' on one line."""
    li, _, j = name_item["_span"]
    ln = lines[li]
    rest_start = ln["words"][j - 1]["end"] if j else 0
    out = []
    for m in re.finditer(r"[^,;·|]+", ln["text"][rest_start:]):
        seg = m.group(0).strip()
        a = rest_start + m.start() + m.group(0).find(seg)
        if STREET_RE.match(seg) or CITY_RE.match(seg):
            s = _char_span(ln, a, a + len(seg))
            if s:
                out.append(_item(lines, li, *s, "address", "address next to the subject's name"))
    return out


# --------------------------------------------------------------------------- Gemma
PROMPT = """Below is the OCR text of one document page, one line per row:
L<id> (x%, y%): text      (x/y = where the line starts on the page, 0,0 = top-left)

This document belongs to ONE private person: the data subject. That is the person the
document is about or addressed to - the patient, the customer, the recipient of the
letter, the account holder. It is NOT the doctor, the author or signer, an employee,
or a company.

1. Who is the data subject? Give their full name as printed.
2. List the personal data on the page. For each item give the line id, the exact
   value as printed on that line (no field label), a label, and owner:
   "subject" if it belongs to the data subject, otherwise "other".
   labels: name, address, birth_date, phone, email, iban, id_number
   - address: street + house number, or postcode + city (one item per line)
   - id_number: customer / patient / insurance / order / invoice / tracking numbers,
     tax or register numbers
   Names of doctors and staff, companies, clinics and shops, their addresses,
   phone numbers, emails, bank details, tax and register numbers -> owner "other".

Answer ONLY with JSON:
{"subject": "<name>", "items": [{"line": 12, "text": "...", "label": "...", "owner": "subject"}]}

TEXT:
%s"""


def layout_text(lines, size):
    W, H = size
    order = sorted(range(len(lines)), key=lambda k: (round(lines[k]["box"][1] / 12), lines[k]["box"][0]))
    return "\n".join(f"L{k} ({100 * lines[k]['box'][0] // W}%, {100 * lines[k]['box'][1] // H}%): {lines[k]['text']}"
                     for k in order)


def ask_gemma(lines, size, model=D.DEFAULT_MODEL, think=True):
    t0 = time.time()
    r = requests.post(D.OLLAMA_URL, json={
        "model": model, "stream": False, "format": "json", "think": think,
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": PROMPT.replace("%s", layout_text(lines, size))}],
    }, timeout=600)
    r.raise_for_status()
    raw = r.json()["message"]["content"]
    data = D._parse_json(raw)
    if not isinstance(data, dict):
        data = {"items": data if isinstance(data, list) else []}
    items = [it for it in data.get("items") or [] if isinstance(it, dict)]
    return str(data.get("subject") or "").strip(), items, raw, time.time() - t0


def guess_subject(lines):
    """Fallback without the LLM: the first line that sits right above a street + postcode block."""
    b = block_names(lines)
    return b[0] if b else ""


def block_names(lines):
    """Lines that head an address block (name above street + postcode/city)."""
    out = []
    for li, ln in enumerate(lines):
        t = ln["text"].strip()
        if not t or re.search(r"\d", t) or len(t.split()) > 4 or _n(t) in TITLES:
            continue
        nxt = _below(lines, li)
        if nxt is not None and STREET_RE.match(lines[nxt]["text"].strip()):
            nn = _below(lines, nxt)
            if nn is not None and CITY_RE.match(lines[nn]["text"].strip()) and not COMPANY_RE.search(t):
                out.append(t)
    return out


def pick_subject(lines, subject, llm_items):
    """Gemma's answer unless it's clearly not a private person (a company / doctor) or not on the page;
    then a person-name item Gemma attributed to the subject, then the recipient-block guess."""
    def good(s):
        return s and not COMPANY_RE.search(s) and name_spans(lines, s)
    if good(subject):
        return subject
    for it in llm_items:
        s = str(it.get("text") or "")
        if str(it.get("label")).lower() == "name" and str(it.get("owner")).lower() == "subject" and good(s):
            return s
    g = guess_subject(lines)
    return g or subject


# --------------------------------------------------------------------------- main
def _parse_date(s):
    m = DATE_RE.search(s)
    if not m:
        return None
    for fmt in ("%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            return dt.datetime.strptime(re.sub(r"\s", "", m.group(1)), fmt).date()
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------- THE POLICY (one place)
# Replace ONLY the data subject's own: name, email, phone, home address (street / postcode+city),
# birth date, and their personal numbers (customer / order / invoice / tracking / insurance, IBAN).
# Never: other people (doctors, staff, signatories), companies, document dates, amounts, products,
# medical content, countries.
REPLACE_KINDS = {"name", "email", "phone", "address", "birth_date", "id_number", "iban"}
NEVER_RE = re.compile(r"^(deutschland|germany|österreich|austria|schweiz|switzerland|france|frankreich|"
                      r"nederland|niederlande|italia|italien|de|at|ch|eu)$", re.I)


def replaceable(it):
    t = re.sub(r"[^\wäöüß@+ ]", " ", it["text"]).strip()
    if it["label"] not in REPLACE_KINDS or len(t) < 3 or NEVER_RE.match(t):
        return False
    if it["label"] == "address" and not re.search(r"\d", t):  # street + number or postcode + city
        return False
    return True


def scope(lines, size, subject, llm_items=()):
    """-> (items to anonymise, items deliberately kept). Both carry a human-readable `why`."""
    W, H = size
    footer = lambda li: lines[li]["box"][1] > FOOTER_Y * H
    take, keep = [], []

    # 1) the subject's name everywhere + their address block(s)
    names = [nm for a in (subject_aliases(lines, subject) if subject else []) for nm in name_spans(lines, a)]
    take += names
    for nm in names:
        take += address_block(lines, nm) + inline_address(lines, nm)
    near = {it["_span"][0] for it in take}  # lines that belong to the subject's block(s)
    for li in list(near):
        for _ in range(6):  # + the lines right under them (Tel./E-Mail under the address)
            li = _below(lines, li)
            if li is None:
                break
            near.add(li)

    # 2) label rules: birth date, personal IDs
    for li, ln in enumerate(lines):
        if footer(li):
            continue
        for m in DATE_RE.finditer(ln["text"]):
            before = ln["text"][:m.start()]
            labelled = DOB_LABEL.search(before[-30:])
            if not labelled and not before.strip():  # label in the column to the left / the line above
                left = [k for k, o in enumerate(lines) if k != li and o["box"][2] <= ln["box"][0] + 5
                        and o["box"][1] <= (ln["box"][1] + ln["box"][3]) / 2 <= o["box"][3]]
                labelled = any(DOB_LABEL.search(lines[k]["text"]) for k in left)
            if labelled:
                s = _char_span(ln, m.start(), m.end())
                if s:
                    take.append(_item(lines, li, *s, "birth_date", "date after a birth-date label"))
        for m in ID_LABEL.finditer(ln["text"]):
            if DENY_LABEL.search(m.group(0)):
                continue
            v = _value_after(lines, li, m.end(), "id")
            if v:
                take.append(_item(lines, *v, "id_number", f"value of “{m.group(0).strip()}”"))

    # 2b) contact data: on the subject's own lines, or an email that contains their name
    subj_toks = [t for t in _subject_tokens(subject) if len(t) > 1]
    for li, ln in enumerate(lines):
        if footer(li):
            continue
        for m in EMAIL_RE.finditer(ln["text"]):
            local = _n(m.group(0).split("@")[0])
            if li in near or any(len(t) >= 3 and t in local for t in subj_toks):
                sp = _char_span(ln, m.start(), m.end())
                if sp:
                    take.append(_item(lines, li, *sp, "email", "email on the subject's lines" if li in near
                                      else "email contains the subject's name"))
        if li in near:
            for m in PHONE_RE.finditer(ln["text"]):
                pre = ln["text"][:m.start()]
                if re.search(r"(tel|telefon|phone|mobil|handy|fax)\.?\s*:?\s*$", pre, re.I) or not pre.strip():
                    sp = _char_span(ln, m.start(), m.end())
                    if sp:
                        take.append(_item(lines, li, *sp, "phone", "phone on the subject's lines"))

    # 3) Gemma's attributions (IBAN, extra IDs, contact data elsewhere) - checked against layout
    for it in llm_items:
        text, label = str(it.get("text") or "").strip(), str(it.get("label") or "").lower()
        owner = str(it.get("owner") or "other").lower()
        try:
            li = int(re.sub(r"\D", "", str(it.get("line", ""))) or -1)
        except ValueError:
            li = -1
        b = _find(lines, text, li if li >= 0 else None)
        if not text or not b:
            continue
        _, li, i, j = b
        x = _item(lines, li, i, j, label or "other", "", src="gemma")
        if owner != "subject":
            x["why"] = "Gemma: belongs to someone else"
            keep.append(x)
            continue
        ok, why = False, ""
        if label in ("email", "phone"):
            email_has_name = label == "email" and any(t in _n(text) for t in subj_toks if len(t) >= 3)
            ok = (li in near or email_has_name) and not footer(li)
            why = "Gemma: subject's " + label if ok else f"{label} not next to the subject's name/address"
        elif label == "iban":
            ok = not footer(li) and not DENY_LABEL.search(lines[li]["text"])
            why = "Gemma: subject's IBAN" if ok else "IBAN in the issuer's footer"
        elif label == "id_number":
            ctx = lines[li]["text"]
            ok = not footer(li) and not DENY_LABEL.search(ctx)
            why = "Gemma: subject's ID" if ok else "issuer's number (tax/register/footer)"
        elif label == "birth_date":
            d = _parse_date(text)
            ok = d is not None and d < dt.date.today() - dt.timedelta(days=365) and not footer(li)
            why = "Gemma: subject's birth date" if ok else "not a plausible birth date"
        elif label == "address":
            ok = li in near
            why = "Gemma: subject's address" if ok else "address not next to the subject's name"
        else:  # names are handled by the subject-name rule only
            why = "name doesn't match the subject" if label == "name" else "not personal data of the subject"
        x["why"] = why
        (take if ok else keep).append(x)

    take = [_tighten(lines, it) for it in take]
    keep = [_tighten(lines, it) for it in keep]

    # 4) the same value anywhere else on the page (e.g. order number repeated in the footer)
    for it in list(take):
        if it["label"] in ("name",) or len(_n(it["text"])) < 5:
            continue
        for li, i, j in _exact_spans(lines, it["text"]):
            take.append(_item(lines, li, i, j, it["label"], "same value as " + it["text"]))

    # 5) visibly KEEP other people / organisations (doctors, signatories, companies)
    other = re.compile(r"(?:Dr\.|Prof\.)(?:\s*(?:med|rer\.\s*nat|dent)\.)?\s+[A-ZÄÖÜ][\wäöüß\-]+(?:\s+[A-ZÄÖÜ][\wäöüß\-]+)?")
    for li, ln in enumerate(lines):
        for m in other.finditer(ln["text"]):
            sp = _char_span(ln, m.start(), m.end())
            if sp:
                keep.append(_item(lines, li, *sp, "name", "someone else (doctor / title holder)"))
        if COMPANY_RE.search(ln["text"]) and len(ln["words"]) <= 8 and not other.search(ln["text"]):
            keep.append(_item(lines, li, 0, len(ln["words"]), "company", "organisation, not a private person"))

    take = [it for it in take if replaceable(it)]  # the policy above, enforced last
    # rule items first, then drop overlapping / duplicate spans
    take.sort(key=lambda x: (x["src"] != "rule", -(x["_span"][2] - x["_span"][1])))
    final, claimed = [], {}
    for it in take:
        li, i, j = it["_span"]
        if any(i < b and a < j for a, b in claimed.get(li, [])):
            continue
        claimed.setdefault(li, []).append((i, j))
        final.append(it)
    keep = [k for k in keep if not any(k["_span"][1] < b and a < k["_span"][2]
                                       for a, b in claimed.get(k["_span"][0], []))]
    kept, kclaim = [], {}
    for k in sorted(keep, key=lambda x: -(x["_span"][2] - x["_span"][1])):  # widest first, no overlaps
        li, i, j = k["_span"]
        if not any(i < b and a < j for a, b in kclaim.get(li, [])):
            kclaim.setdefault(li, []).append((i, j))
            kept.append(k)
    order = lambda x: (lines[x["_span"][0]]["box"][1], x["box"][0])
    strip = lambda xs: [{k: v for k, v in x.items() if k != "_span"} for x in sorted(xs, key=order)]
    return strip(final), strip(kept)


def gender_hint(lines, items):
    """'Herrn' / 'Frau' right before or above the subject's name -> 'm' / 'f' (keeps the fake first name's gender)."""
    for it in items:
        if it["label"] != "name":
            continue
        for ln in lines:
            b, nb = ln["box"], it["box"]
            same = b[1] <= (nb[1] + nb[3]) / 2 <= b[3] and b[0] <= nb[0]
            above = 0 < nb[1] - b[3] < 1.5 * (nb[3] - nb[1]) and abs(b[0] - nb[0]) < 30
            if same or above:
                pre = ln["text"] if above else ln["text"][:max(0, ln["text"].find(it["text"]))]
                if re.search(r"\b(herrn?|hr\.|mr\.?)\s*$", pre.strip() + " ", re.I) or re.fullmatch(r"\s*herrn?\s*", pre, re.I):
                    return "m"
                if re.search(r"\b(frau|fr\.|mrs\.?|ms\.?)\s*$", pre.strip() + " ", re.I) or re.fullmatch(r"\s*frau\s*", pre, re.I):
                    return "f"
    return None


def find(lines, size, model=D.DEFAULT_MODEL, think=True):
    """Full pass: Gemma names the subject, rules pick their data."""
    try:
        subject, llm_items, raw, secs = ask_gemma(lines, size, model, think)
    except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
        subject, llm_items, raw, secs = "", [], f"Gemma failed: {e}", 0.0
    subject = pick_subject(lines, subject, llm_items)
    take, keep = scope(lines, size, subject, llm_items)
    return {"subject": subject, "items": take, "kept": keep, "llm_items": llm_items, "raw": raw, "gemma_s": secs,
            "gender": gender_hint(lines, take)}


if __name__ == "__main__":
    import sys
    from PIL import Image
    import ocr as O
    img = Image.open(sys.argv[1]).convert("RGB")
    ls = O.ocr(img)
    res = find(ls, img.size, think="--think" in sys.argv)
    print("SUBJECT:", res["subject"], f"({res['gemma_s']:.1f}s)")
    for it in res["items"]:
        print(f"  ANON {it['label']:10} {it['text']!r:40} {it['why']}")
    for it in res["kept"]:
        print(f"  keep {it['label']:10} {it['text']!r:40} {it['why']}")
