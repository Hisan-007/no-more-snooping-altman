# No More Snooping Altman

Ask AI about your documents without giving away who you are.

Upload a document image or PDF, such as a doctor's letter or an invoice. The app finds *your* personal data, replaces it with believable fake data in the same font, and gives you back the document as a PDF. You can then ask Gemini questions about it: Gemini only sees the fake version, and the answer is mapped back to your real data on your machine.

Built for the DevFest Münster 2026 Agentic AI Hackathon with Google ADK, Gemini and Gemma 4.

## What gets replaced

Only the document owner's data:

- name, email, phone
- home address (street + number, postcode + city)
- birth date
- customer, order, invoice, tracking and insurance numbers, IBAN

Everything else stays as it is: doctors and other people, companies, dates, amounts, products and medical content.

## How it works

1. **Intake gate** (code): rejects files that aren't real images or PDFs, PDFs with JavaScript, oversized files and pixel bombs.
2. **OCR** (Apple Vision, on-device): reads the text and the position of every word.
3. **Analyst and Sentinel** (Gemma 4, on-device): work out whose document it is, and flag lines that try to give instructions to an AI.
4. **Policy** (code): checks the models' answers and decides what gets replaced.
5. **Erase and redraw** (on-device): removes the values and draws the fakes in the matching font.
6. **Verify** (Gemma 4 + code): re-reads the result and checks that no pixels changed outside the replaced areas.

A **coordinator** agent (Gemini) runs these steps but only ever sees placeholders like `NAME_1`. A firewall blocks any request to Gemini that contains a real value. Every step is written to a hash-chained audit log in `audit/`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

## Requirements

- macOS on Apple Silicon (OCR uses Apple's Vision framework)
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Setup

```bash
ollama pull gemma4:e2b-it-qat
```

```bash
uv venv -p 3.12 .venv
```

```bash
VIRTUAL_ENV=.venv uv pip install -r requirements.txt
```

## Run

Make sure Ollama is running, then:

```bash
export GOOGLE_API_KEY=your-key
```

```bash
.venv/bin/uvicorn app:app --port 8000
```

Open http://localhost:8000, pick a sample and press **Run**. The first run takes longer while models load.

| Page | What |
|---|---|
| `/` | Step-by-step view of the agents |
| `/trace` | Detailed trace view |
| `/architecture` | Architecture diagram |

Terminal version:

```bash
.venv/bin/python cli.py Docs/SAMPLE_Arztbrief.png "What does my ferritin value mean?"
```

## Sample files

| File | What happens |
|---|---|
| `SAMPLE_Arztbrief.png` | Doctor's letter, normal run |
| `SAMPLE_Arztbrief_ATTACK.png` | Contains a hidden prompt injection, which gets quarantined |
| `BAD_invoice_really_a_script.png` | HTML file renamed to `.png`, blocked at intake |
| `BAD_contract_with_javascript.pdf` | PDF with JavaScript, blocked at intake |
| `BAD_scan_decompression_bomb.png` | 30000×30000 image in a 109 kB file, blocked at intake |
| `BAD_holiday_photo_no_text.png` | Photo with no text, stopped after OCR |

## Configuration

| Variable | Default | |
|---|---|---|
| `GOOGLE_API_KEY` | none | Gemini API key (required) |
| `CLOUD_MODEL` | `gemini-flash-latest` | Model for the coordinator and chat |
| `LOCAL_MODEL` | `ollama_chat/gemma4:e2b-it-qat` | Model for the on-device agents |
| `LOCAL_THINK` | `0` | Set to `1` to let Gemma think (slower) |

## Project layout

| Path | |
|---|---|
| `app.py` | Web server and intake gate |
| `stage.html` | Main UI |
| `anon_agent/` | ADK agents, tools, plugins (firewall, trace), vault, audit log |
| `pii.py` | Replacement policy |
| `ocr.py`, `erase.py`, `render.py`, `fake.py` | OCR, erasing, font-matched drawing, fake data |
| `cli.py` | Terminal version |
| `Docs/` | Sample documents |

Output is written to `uploads/<doc>_anonymised.pdf`. Audit logs are in `audit/` and can be checked with `python -m anon_agent.audit audit/<run>.jsonl`.
