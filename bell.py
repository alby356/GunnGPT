"""Compute today's bell schedule for the live home-page widget.

The regular weekly schedule comes from WATT's canonical source of truth,
`shared/data/schedule.ts` (the same data the live gunnwatt.web.app bundles — the
generated `scripts/output/schedule.json` in the repo is stale). Special days come
from `alternates.json`, which stays in sync with WATT's Firestore. Both are cached.
Times are minutes since midnight.
"""
import datetime as dt
import json
import re
import time

import requests

import config

DAY_LETTERS = {0: "M", 1: "T", 2: "W", 3: "R", 4: "F"}
FIRST_DAY = dt.date(2026, 8, 13)
LAST_DAY = dt.date(2027, 6, 3)
PERIOD_LABELS = {
    "0": "Period 0", "1": "Period 1", "2": "Period 2", "3": "Period 3",
    "4": "Period 4", "5": "Period 5", "6": "Period 6", "7": "Period 7", "8": "Period 8",
    "B": "Brunch", "L": "Lunch", "S": "SELF", "H": "Study Hall", "P": "PRIME",
    "G": "Gunn Together", "O": "Office Hours",
}

# Canonical weekly schedule (TypeScript) + special-day overrides (JSON).
SCHEDULE_TS_URL = "https://raw.githubusercontent.com/GunnWATT/watt/main/shared/data/schedule.ts"
ALTERNATES_URL = f"{config.WATT_RAW}/alternates.json"

_cache = {"t": 0.0, "sched": None, "alts": None}


def _label(code):
    return PERIOD_LABELS.get(code, code)


def _parse_schedule_ts(txt):
    """Extract the `schedule` object literal from schedule.ts and parse it as JSON.
    The literal uses double-quoted keys, so after stripping comments and any
    trailing commas it is valid JSON. Returns {day letter: [{n, s, e}, ...]}."""
    i = txt.index("const schedule")
    start = txt.index("{", txt.index("=", i))     # skip the type annotation before "="
    depth = 0
    end = start
    for j in range(start, len(txt)):
        if txt[j] == "{":
            depth += 1
        elif txt[j] == "}":
            depth -= 1
            if depth == 0:
                end = j
                break
    obj = txt[start:end + 1]
    obj = re.sub(r"//[^\n]*", "", obj)            # strip line comments
    obj = re.sub(r",(\s*[}\]])", r"\1", obj)      # strip trailing commas
    return json.loads(obj)


def _data():
    now = time.time()
    if _cache["sched"] is None or now - _cache["t"] > 21600:      # refresh every 6h
        _cache["sched"] = _parse_schedule_ts(requests.get(SCHEDULE_TS_URL, timeout=15).text)
        _cache["alts"] = requests.get(ALTERNATES_URL, timeout=15).json()
        _cache["t"] = now
    return _cache["sched"], _cache["alts"]


def _rows(periods):
    """Turn a list of {n, s, e, grades?} periods into display rows sorted by start
    time. Grade-restricted periods (e.g. Friday's SELF / Study Hall, which differ by
    grade) each become their own row carrying the grades they apply to, so the UI can
    label them — matching how WATT shows them."""
    rows = []
    for p in sorted(periods, key=lambda x: x["s"]):
        row = {"name": _label(p["n"]), "start": p["s"], "end": p["e"]}
        if p.get("grades"):
            row["grades"] = p["grades"]
        rows.append(row)
    return rows


def today_schedule(date=None):
    date = date or dt.date.today()
    try:
        sched, alts = _data()
    except Exception:
        return {"status": "error", "message": "Couldn't load the schedule right now.", "periods": []}

    if date < FIRST_DAY or date > LAST_DAY:
        return {"status": "off", "message": "No school today!", "periods": []}

    mmdd = date.strftime("%m-%d")
    if mmdd in alts:
        periods = alts[mmdd]
        if not periods:
            return {"status": "off", "message": "No school today!", "periods": []}
        return {"status": "school", "alternate": True, "periods": _rows(periods)}

    wd = date.weekday()
    if wd >= 5:
        return {"status": "off", "message": "No school today!", "periods": []}

    return {"status": "school", "alternate": False, "periods": _rows(sched.get(DAY_LETTERS[wd], []))}
