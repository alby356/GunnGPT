"""Turn uploaded attachments (images, PDFs, text) into plain text.

Images → OCR via Apple Vision (ocrmac, local). PDFs → pypdf text extraction.
Text-ish files → decoded directly. Returns a short string per file; failures
are reported inline rather than raising.
"""
import base64
import io

MAX_CHARS = 6000   # cap per file so one upload can't blow the prompt


def _ocr_image(data):
    from PIL import Image
    from ocrmac import ocrmac
    img = Image.open(io.BytesIO(data)).convert("RGB")
    res = ocrmac.OCR(img).recognize()
    return "\n".join(r[0] for r in res).strip()


def _pdf_text(data):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((p.extract_text() or "") for p in reader.pages).strip()


def extract_text(att):
    """att = {name, mime, data(base64)} -> extracted text (str)."""
    name = (att.get("name") or "file").strip()
    mime = (att.get("mime") or "").lower()
    try:
        data = base64.b64decode(att.get("data", ""))
    except Exception:
        return f"[{name}: could not decode]"
    low = name.lower()
    try:
        if low.endswith(".pdf") or "pdf" in mime:
            text = _pdf_text(data)
            kind = "PDF"
        elif mime.startswith("image/") or low.endswith(
                (".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".bmp", ".tiff")):
            text = _ocr_image(data)
            kind = "image (OCR)"
        else:
            text = data.decode("utf-8", errors="ignore").strip()
            kind = "text"
    except Exception as e:
        return f"[{name}: could not read this file ({e})]"
    if not text:
        return f"[{name}: no readable text found]"
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + " …(truncated)"
    return f"File: {name} ({kind})\n{text}"


def extract_all(attachments):
    if not attachments:
        return ""
    blocks = [extract_text(a) for a in attachments[:5]]   # cap 5 files
    return "\n\n----\n\n".join(b for b in blocks if b)
