"""Compute today's bell schedule for the live home-page widget.

Pulls WATT's schedule.json (regular weekdays) + alternates.json (special days),
caches them, and returns today's periods so the browser can tick a live clock /
countdown. Times are minutes since midnight.
"""
import datetime as dt
import time

import requests

import config

DAY_LETTERS = {0: "M", 1: "T", 2: "W", 3: "R", 4: "F"}
FIRST_DAY = dt.date(2026, 8, 13)
LAST_DAY = dt.date(2027, 6, 3)
PERIOD_LABELS = {
    "0": "Period 0", "1": "Period 1", "2": "Period 2", "3": "Period 3",
    "4": "Period 4", "5": "Period 5", "6": "Period 6", "7": "Period 7", "8": "Period 8",
    "B": "Brunch", "L": "Lunch", "S": "SELF", "P": "PRIME", "G": "Gunn Together",
}

_cache = {"t": 0.0, "sched": None, "alts": None}


def _label(code):
    return PERIOD_LABELS.get(code, code)


def _data():
    now = time.time()
    if _cache["sched"] is None or now - _cache["t"] > 21600:      # refresh every 6h
        _cache["sched"] = requests.get(f"{config.WATT_RAW}/schedule.json", timeout=15).json()
        _cache["alts"] = requests.get(f"{config.WATT_RAW}/alternates.json", timeout=15).json()
        _cache["t"] = now
    return _cache["sched"], _cache["alts"]


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
        rows = [{"name": _label(p["n"]), "start": p["s"], "end": p["e"]}
                for p in sorted(periods, key=lambda x: x["s"])]
        return {"status": "school", "alternate": True, "periods": rows}

    wd = date.weekday()
    if wd >= 5:
        return {"status": "off", "message": "No school today!", "periods": []}

    day = sched.get(DAY_LETTERS[wd], {})
    rows = [{"name": _label(code), "start": t["s"], "end": t["e"]}
            for code, t in sorted(day.items(), key=lambda kv: kv[1]["s"])]
    return {"status": "school", "alternate": False, "periods": rows}
