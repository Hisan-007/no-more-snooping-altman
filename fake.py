"""Believable, format-preserving fake values for detected PII.

- Same original text -> same fake everywhere in the document (consistency map).
- IDs / numbers keep their shape: digit->digit, letter->letter, separators and
  short prefixes (DE, HRB, ...) kept.
- Dates shift by one per-document offset and keep their format.
- Names / companies / streets / cities come from Faker(de_DE); `candidates()`
  returns several options so the renderer can pick the one whose width fits best.
"""
import datetime as dt
import hashlib
import random
import re
import string

from faker import Faker

COUNTRIES = {"deutschland", "germany", "österreich", "austria", "schweiz", "switzerland"}


def _case_like(src: str, out: str) -> str:
    letters = [c for c in src if c.isalpha()]
    if letters and all(c.isupper() for c in letters) and len(letters) > 1:
        return out.upper()
    return out


def _shape_preserving(text: str, rng: random.Random, keep_prefix=True, keep_years=False) -> str:
    """digit->digit, A->A, a->a; keep separators. Keep short all-letter tokens (DE, HRB) and
    a 2-letter country prefix glued to digits (DE7K41...)."""
    out = []
    for tok in re.split(r"(\s+|[-/.,:()])", text):
        if not tok or re.fullmatch(r"\s+|[-/.,:()]", tok):
            out.append(tok)
            continue
        if (keep_prefix and tok.isalpha() and len(tok) <= 4) or (keep_years and re.fullmatch(r"(19|20)\d\d", tok)):
            out.append(tok)  # HRB, DE, USt...
            continue
        keep = 2 if keep_prefix and re.match(r"^[A-Z]{2}\d", tok) else 0
        chars = list(tok[:keep])
        for i, c in enumerate(tok[keep:]):
            if c.isdigit():
                # avoid leading zero turning into something odd; keep '0' in first position as 0
                chars.append(c if (i == 0 and c == "0") else rng.choice(string.digits[1:] if i == 0 else string.digits))
            elif c.isupper():
                chars.append(rng.choice(string.ascii_uppercase))
            elif c.islower():
                chars.append(rng.choice(string.ascii_lowercase))
            else:
                chars.append(c)
        out.append("".join(chars))
    return "".join(out)


DATE_FORMATS = ["%d.%m.%y", "%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%d.%m."]
NAME_TITLES = {"herr", "herrn", "frau", "hr", "fr", "dr", "med", "prof", "dipl", "ing", "mr", "mrs", "ms", "von", "van", "de"}
MONTHS_DE = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober",
             "November", "Dezember"]
MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December"]
_MON = {m[:3].lower(): i + 1 for i, m in enumerate(MONTHS_DE)}
_MON.update({m[:3].lower(): i + 1 for i, m in enumerate(MONTHS_EN)})
_MON["mär"] = 3

_NUM_DATE = re.compile(r"(?<![\d.])(\d{1,2})(\.\s?|/)(\d{1,2})(\.\s?|/)(\d{4}|\d{2})(?!\d)")
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_TXT_DATE = re.compile(r"(?<!\d)(\d{1,2})(\.?\s?)([A-Za-zäÄ]{3,9})(\.?\s+)(\d{4})(?!\d)")


def _ascii(s):
    s = s.translate(str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"}))
    return re.sub(r"[^A-Za-z\-]", "", s)


def _ratio(a, b):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def _pad(n, like):
    return f"{n:0{len(like)}d}" if len(like) == 2 and like.startswith("0") else (f"{n:02d}" if len(like) == 2 else str(n))


def _year(y):
    y = int(y)
    if y < 100:
        cur = dt.date.today().year % 100
        y += 2000 if y <= cur + 1 else 1900
    return y


def _shift_dates(text, shift):
    """Shift every date in text by `shift`, keeping separators, padding, 2/4-digit years, month names."""
    def num(m):
        try:
            d = dt.date(_year(m.group(5)), int(m.group(3)), int(m.group(1))) + shift
        except ValueError:
            return m.group(0)
        y = f"{d.year % 100:02d}" if len(m.group(5)) == 2 else str(d.year)
        return f"{_pad(d.day, m.group(1))}{m.group(2)}{_pad(d.month, m.group(3))}{m.group(4)}{y}"

    def iso(m):
        try:
            d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) + shift
        except ValueError:
            return m.group(0)
        return d.isoformat()

    def txt(m):
        mon = _MON.get(m.group(3)[:3].lower())
        if not mon:
            return m.group(0)
        try:
            d = dt.date(int(m.group(5)), mon, int(m.group(1))) + shift
        except ValueError:
            return m.group(0)
        names = MONTHS_DE if m.group(3)[:3].lower() in [x[:3].lower() for x in MONTHS_DE] else MONTHS_EN
        name = names[d.month - 1]
        if len(m.group(3)) <= 4 and m.group(3) != names[mon - 1]:
            name = name[:len(m.group(3))]  # abbreviated "Mär" / "Okt"
        return f"{_pad(d.day, m.group(1))}{m.group(2)}{name}{m.group(4)}{d.year}"

    return _TXT_DATE.sub(txt, _ISO_DATE.sub(iso, _NUM_DATE.sub(num, text)))


class Faker2:
    def __init__(self, seed: str = "demo"):
        s = int(hashlib.sha1(seed.encode()).hexdigest()[:8], 16)
        self.rng = random.Random(s)
        self.f = Faker("de_DE")
        self.f.seed_instance(s)
        self.date_shift = dt.timedelta(days=self.rng.choice([-1, 1]) * self.rng.randint(20, 200))
        # birth dates move by up to ~10 months: same age bracket, never a weird year
        self.dob_shift = dt.timedelta(days=self.rng.choice([-1, 1]) * self.rng.randint(40, 300))
        self.map = {}  # (label, original) -> fake
        self.tok, self.init = {}, {}  # name token maps, see _name
        self.p = next(p for p in self.f.providers if hasattr(p, "first_names_male"))
        self.gender = None  # "m" / "f" once a salutation (Herr / Frau) tells us

    # ---- per-type generators (each returns ONE candidate) ----
    # names are faked token by token so every spelling of the subject stays consistent:
    # "Max Mustermann" -> "Jonas Weber", then "Herr Mustermann" -> "Herr Weber", "M. Mustermann" -> "J. Weber"
    def _name_parts(self, text):
        toks = text.split()
        comma = "," in text
        out = []
        for k, tok in enumerate(toks):
            core = re.sub(r"[^\wäöüÄÖÜß\-]", "", tok)
            if not core or core.lower().rstrip(".") in NAME_TITLES:
                role = None
            elif comma:
                role = "last" if k == 0 else "first"
            else:
                real = [x for x in toks if re.sub(r"[^\wäöüÄÖÜß\-]", "", x).lower() not in NAME_TITLES]
                role = "last" if tok == real[-1] and (len(real) > 1 or len(core) > 1) else "first"
            out.append((tok, core, role))
        return out

    def _lookup(self, role, core):
        key = core.lower()
        for (r, t), v in self.tok.items():
            if r == role and (t == key or (len(t) >= 4 and len(key) >= 4 and _ratio(t, key) >= 0.8)):
                return v
        return None

    def _fresh(self, role, core, letter=None):
        male = self.p.first_names_male
        female = self.p.first_names_female
        for _ in range(300):
            if role == "first":
                if core in female:
                    v = self.f.first_name_female()
                elif core in male or self.gender == "m":
                    v = self.f.first_name_male()
                elif self.gender == "f":
                    v = self.f.first_name_female()
                else:
                    v = self.f.first_name()
            else:
                v = self.f.last_name()
            v = v.split()[0].split("-")[0]
            if v.lower() != core.lower() and (letter is None or v[0].lower() == letter):
                return v
        return v

    def _name(self, text):
        out = []
        for tok, core, role in self._name_parts(text):
            if role is None:
                out.append(tok)
                continue
            if len(core) == 1:  # initial
                fake = self.init.get((role, core.lower())) or self._fresh(role, core)[0]
            else:
                fake = self._lookup(role, core) or self._fresh(role, core, self.init.get((role, core[0].lower()), "").lower() or None)
            fake = fake.upper() if core.isupper() and len(core) > 1 else fake
            out.append(tok.replace(core, fake, 1))
        return " ".join(out)

    def _commit_name(self, text, fake):
        src, dst = self._name_parts(text), fake.split()
        if len(src) != len(dst):
            return
        for (tok, core, role), f in zip(src, dst):
            fcore = re.sub(r"[^\wäöüÄÖÜß\-]", "", f)
            if role is None or not fcore:
                continue
            self.init[(role, core[0].lower())] = fcore[0].upper()
            if len(core) > 1:
                self.tok[(role, core.lower())] = fcore.capitalize() if not fcore.isupper() else fcore

    def _company(self, text):
        return _case_like(text, self.f.company())

    def _address(self, text):
        t = text.strip()
        if t.lower() in COUNTRIES:
            return t  # keep country - changing it would clash with the postcode
        m = re.fullmatch(r"(?:D-)?(\d{5})\s+(.+)", t)
        if m:  # "38106 BRAUNSCHWEIG"
            return _case_like(m.group(2), f"{self.f.postcode()} {self.f.city()}")
        m = re.fullmatch(r"(.+?)(\s*,?\s*)(\d+\s*[a-zA-Z]?)", t)
        if m:  # "Lindenweg, 12" -> keep the ", " style
            street = self.f.street_name()
            num = _shape_preserving(m.group(3), self.rng, keep_prefix=False)  # "11" -> "47", "3a" -> "8c"
            return _case_like(m.group(1), f"{street}{m.group(2)}{num}")
        return _case_like(t, self.f.city())

    def _phone(self, text):
        # keep +49 / (0) and the area / mobile prefix (0176, 0531 - not identifying), swap the rest
        m = re.match(r"^(\+\d{2}\s*(?:\(0\))?\s*)?(\(?0?\d{2,5}\)?[\s/\-]+)?(.*)$", text)
        prefix, area, rest = m.group(1) or "", m.group(2) or "", m.group(3)
        return prefix + area + _shape_preserving(rest, self.rng, keep_prefix=False)

    FREEMAIL = {"web.de", "gmx.de", "gmx.net", "gmail.com", "googlemail.com", "yahoo.de", "yahoo.com", "t-online.de",
                "outlook.com", "outlook.de", "hotmail.com", "hotmail.de", "icloud.com", "me.com", "posteo.de",
                "mailbox.org", "freenet.de", "aol.com", "protonmail.com", "proton.me", "tu-bs.de", "live.de"}

    def _email(self, text):
        local, _, domain = text.strip().partition("@")
        # the subject's own name in the address -> the same fake name ("max.mustermann" -> "jonas.weber")
        parts = re.split(r"([._\-])", local)
        out, hit = [], False
        for p in parts:
            if re.fullmatch(r"[._\-]", p) or not p:
                out.append(p)
                continue
            core = re.sub(r"\d", "", p)
            f = (self._lookup("first", core) or self._lookup("last", core)) if len(core) >= 2 else None
            if f:
                hit = True
                out.append(_ascii(f).lower() + re.sub(r"\D", "", p))
            else:
                out.append(p)
        if not hit:
            out = [f"{_ascii(self.f.first_name())}.{_ascii(self.f.last_name())}".lower()]
        dom = domain if domain.lower() in self.FREEMAIL else self.f.free_email_domain()
        return "".join(out) + "@" + dom

    def _iban(self, text):
        digits = "".join(self.rng.choice(string.digits) for _ in range(18))
        cc = re.sub(r"\s", "", text)[:2].upper() if re.match(r"[A-Za-z]{2}", text.strip()) else "DE"
        num = int("".join(str(int(c, 36)) for c in digits + cc + "00"))
        iban = f"{cc}{98 - num % 97:02d}{digits}"
        if " " in text.strip():
            iban = " ".join(iban[i:i + 4] for i in range(0, len(iban), 4))
        return iban

    def _date(self, text, shift=None):
        """Shift the date inside `text` (any of 14.03.1998 / 14.3.98 / 1998-03-14 / 14/03/1998 /
        14. März 1998), keep the surrounding text and the exact format. Never touches anything else."""
        shift = self.date_shift if shift is None else shift
        out = _shift_dates(text, shift)
        return out if out != text else _shape_preserving(text, self.rng, keep_years=True)

    def _birth_date(self, text):
        return self._date(text, self.dob_shift)

    def _generic(self, text):
        return _shape_preserving(text, self.rng)

    GEN = {"name": "_name", "company": "_company", "address": "_address", "phone": "_phone",
           "email": "_email", "iban": "_iban", "date": "_date", "birth_date": "_birth_date",
           "id_number": "_generic"}

    def _detect_label(self, label, text):
        """Gemma's label is sometimes off (e.g. a person's name tagged 'address') - fix obvious ones."""
        t = text.strip()
        if label == "birth_date":
            return label
        if "@" in t:
            return "email"
        if re.fullmatch(r"[A-Z]{2}\d{2}[\s\dA-Z]{10,30}", t) and label != "id_number":
            return "iban"
        if label != "id_number" and _shift_dates(t, dt.timedelta(days=1)) != t:
            return "date"
        if label == "address" and not re.search(r"\d", t) and t.lower() not in COUNTRIES and len(t.split()) <= 3:
            # "Maxim M" in an address block -> person name (cities have postcodes in front)
            return "name"
        return label

    def candidates(self, label, text, n=12):
        """Several fakes for (label, text). If we've already decided on one, only that."""
        label = self._detect_label(label, text)
        key = (label, text.strip())
        if key in self.map:
            return [self.map[key]]
        gen = getattr(self, self.GEN.get(label, "_generic"))
        deterministic = label in ("date", "birth_date") or text.strip().lower() in COUNTRIES
        k = 1 if deterministic else n
        seen, out = set(), []
        for _ in range(k * 3):
            c = gen(text)
            if c not in seen and c != text:
                seen.add(c)
                out.append(c)
            if len(out) >= k:
                break
        return out or [text]

    def commit(self, label, text, fake):
        label = self._detect_label(label, text)
        self.map[(label, text.strip())] = fake
        if label == "name":
            self._commit_name(text, fake)

    def set_gender_hint(self, lines_text):
        """'Herr'/'Herrn' or 'Frau' near the subject -> keep the fake first name's gender."""
        if re.search(r"\b(herrn?|hr\.|mr\.?)\b", lines_text, re.I):
            self.gender = "m"
        elif re.search(r"\b(frau|fr\.|mrs\.?|ms\.?)\b", lines_text, re.I):
            self.gender = "f"


def _parses(t, fmt):
    try:
        dt.datetime.strptime(t, fmt)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    F = Faker2("x")
    for lab, t in [("birth_date", "14.03.1998"), ("birth_date", "geb. 14.03.1998"), ("birth_date", "14. März 1998"),
                   ("birth_date", "1998-03-14"), ("birth_date", "3.4.98"), ("name", "Maxim M"), ("address", "Maximilian Mustermann"), ("address", "Lindenweg, 12"),
                   ("address", "38106 BRAUNSCHWEIG"), ("address", "Deutschland"), ("phone", "+49 (0) 6202 97 81 300"),
                   ("id_number", "DE7K41XQWMTA"), ("id_number", "48210050007731904"), ("id_number", "HRB 770 269"),
                   ("id_number", "USt-IdNr. DE 124 645 601"), ("date", "31.10.26"), ("company", "DECATHLON Deutschland SE & Co. KG"),
                   ("email", "max@example.com"), ("iban", "DE89 3704 0044 0532 0130 00")]:
        print(f"{lab:10} {t!r:36} -> {F.candidates(lab, t, 3)}")
