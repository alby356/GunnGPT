"""Fetch every Gunn data source and turn it into clean text documents.

Each document is a dict: {"id", "title", "url", "source", "text"}.
ingest.py chunks + embeds these. Nothing here needs Ollama, so you can run
`python sources.py` to eyeball the corpus.
"""
import datetime as dt
import json
import re
import sys

import requests

import config

DAY_NAMES = {"M": "Monday", "T": "Tuesday", "W": "Wednesday", "R": "Thursday", "F": "Friday"}
# Human labels for the special single-letter period codes WATT uses.
PERIOD_LABELS = {
    "0": "Period 0", "1": "Period 1", "2": "Period 2", "3": "Period 3",
    "4": "Period 4", "5": "Period 5", "6": "Period 6", "7": "Period 7", "8": "Period 8",
    "B": "Brunch", "L": "Lunch", "S": "SELF / FlexTime", "P": "PRIME", "G": "Gunn Together",
}


def _get_json(url):
    r = requests.get(url, timeout=30, headers={"User-Agent": "GunnGPT/1.0"})
    r.raise_for_status()
    return r.json()


def _fmt(minutes):
    """475 -> '7:55 AM'."""
    h, m = divmod(int(minutes), 60)
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {ampm}"


def _period_label(code):
    return PERIOD_LABELS.get(code, code)


# --------------------------------------------------------------------------- #
# Curated core facts — guarantees identity/location questions always retrieve.
# --------------------------------------------------------------------------- #
def core_facts_docs():
    text = (
        "Basic facts about Henry M. Gunn High School (also called Gunn High School "
        "or Gunn). Location and address: Gunn is located at 780 Arastradero Road, "
        "Palo Alto, California 94306. It is a public high school in the Palo Alto "
        "Unified School District (PAUSD) and is one of the two public high schools "
        "in Palo Alto; the other is Palo Alto High School (Paly). Gunn serves "
        "grades 9 through 12. It was founded / established in 1964 and is named "
        "after Henry Martin Gunn, who was the superintendent of Palo Alto schools "
        "from 1950 to 1961. The school mascot is the Titan (Timmy the Titan) and "
        "sports teams are called the Titans. Main office phone: (650) 354-8200. "
        "Website: gunn.pausd.org. How many students go to / attend Gunn: the "
        "student enrollment / population is about 1,700 students (approximately "
        "1,713 students as of the 2023-2024 school year)."
    )
    return [{
        "id": "core-facts",
        "title": "About Gunn High School (address, district, mascot)",
        "url": "https://gunn.pausd.org/",
        "source": "core-facts",
        "text": text,
    }]


def calendar_docs():
    """Official PAUSD 2026-2027 academic-calendar dates (from the district's
    published School Year Calendar PDF), which apply to Gunn High School."""
    text = (
        "Official Palo Alto Unified School District (PAUSD) academic calendar for "
        "the 2026-2027 school year, which applies to Gunn High School.\n"
        "First day of school: Thursday, August 13, 2026 (all students).\n"
        "Last day of school: Thursday, June 3, 2027 — this is an early-release / "
        "minimum day.\n"
        "Holidays and no-school days (no school on these days):\n"
        "- Labor Day: Monday, September 7, 2026 — no school.\n"
        "- Staff Development Day: Friday, October 2, 2026 — no school for students.\n"
        "- Veterans' Day: Wednesday, November 11, 2026 — no school.\n"
        "- Thanksgiving Break: Monday November 23 through Friday November 27, 2026 "
        "— no school all week.\n"
        "- Winter Break: the last day of school before break is Friday, December 18, "
        "2026; there is no school from December 21, 2026 through January 4, 2027; "
        "students return to school on Tuesday, January 5, 2027.\n"
        "- Martin Luther King Jr. Day: Monday, January 18, 2027 — no school.\n"
        "- Local Holiday: Friday, February 12, 2027 — no school. (Note: February 11 "
        "is a normal school day.)\n"
        "- Washington's Birthday / Presidents' Day (observed): Monday, February 15, "
        "2027 — no school.\n"
        "- Local Holiday: Monday, March 15, 2027 and Staff Development Day Tuesday, "
        "March 16, 2027 — no school for students.\n"
        "- Spring Break: Monday April 5 through Friday April 9, 2027 — no school all week.\n"
        "- Memorial Day: Monday, May 31, 2027 — no school.\n"
        "Early-release / minimum days (school is in session but ends early): "
        "September 4, October 30, November 20, and December 18, 2026, and April 2, "
        "June 1, June 2, and June 3, 2027.\n"
        "End of grading periods (secondary): 1st semester ends December 18, 2026; "
        "2nd semester ends June 3, 2027."
    )
    return [{
        "id": "academic-calendar-2026-2027",
        "title": "PAUSD / Gunn 2026-2027 academic calendar (holidays, breaks, no-school days)",
        "url": "https://www.pausd.org/school-life/calendar",
        "source": "academic-calendar",
        "text": text,
    }]


# --------------------------------------------------------------------------- #
# WATT: bell schedules
# --------------------------------------------------------------------------- #
def watt_schedule_docs():
    docs = []
    sched = _get_json(f"{config.WATT_RAW}/schedule.json")
    for day_code, periods in sched.items():
        name = DAY_NAMES.get(day_code, day_code)
        rows = sorted(periods.items(), key=lambda kv: kv[1]["s"])
        if not rows:
            continue
        lines = [f"Regular {name} bell schedule at Gunn High School:"]
        for code, t in rows:
            lines.append(f"  - {_period_label(code)}: {_fmt(t['s'])} to {_fmt(t['e'])}")
        end = max(t["e"] for _, t in rows)
        start = min(t["s"] for _, t in rows)
        lines.append(f"On a regular {name}, school runs from {_fmt(start)} to {_fmt(end)}. "
                     f"The school day ends at {_fmt(end)}.")
        docs.append({
            "id": f"schedule-{day_code}",
            "title": f"Regular {name} bell schedule",
            "url": "https://gunnwatt.web.app/schedule",
            "source": "bell-schedule",
            "text": "\n".join(lines),
        })
    return docs


def watt_alternate_docs():
    docs = []
    alts = _get_json(f"{config.WATT_RAW}/alternates.json")
    for mmdd, periods in alts.items():
        try:
            month, day = mmdd.split("-")
            pretty = dt.date(2000, int(month), int(day)).strftime("%B %-d")
        except Exception:
            pretty = mmdd
        if not periods:
            text = (f"On {pretty} there is NO SCHOOL at Gunn High School "
                    f"(holiday, break, or non-school day).")
            title = f"No school on {pretty}"
        else:
            lines = [f"Special / alternate bell schedule on {pretty} at Gunn High School:"]
            for p in sorted(periods, key=lambda x: x["s"]):
                seg = f"  - {_period_label(p['n'])}: {_fmt(p['s'])} to {_fmt(p['e'])}"
                if p.get("grades"):
                    seg += f" (grades {', '.join(str(g) for g in p['grades'])})"
                if p.get("note"):
                    seg += f"  [{p['note'].strip()}]"
                lines.append(seg)
            end = max(p["e"] for p in periods)
            lines.append(f"On {pretty}, the last activity ends at {_fmt(end)}.")
            text = "\n".join(lines)
            title = f"Alternate schedule on {pretty}"
        docs.append({
            "id": f"alt-{mmdd}",
            "title": title,
            "url": "https://gunnwatt.web.app/schedule",
            "source": "alternate-schedule",
            "text": text,
        })
    return docs


# --------------------------------------------------------------------------- #
# WATT: staff, courses, clubs
# --------------------------------------------------------------------------- #
def watt_staff_docs():
    data = _get_json(f"{config.WATT_RAW}/staff.json")["data"]
    docs = []
    for sid, s in data.items():
        name = s.get("name", "").strip()
        if not name:
            continue
        parts = [f"{name} is a staff member at Gunn High School."]
        if s.get("dept"):
            parts.append(f"Department / role: {s['dept']}.")
        if s.get("room"):
            parts.append(f"Room / location: {s['room']}.")
        if s.get("email"):
            parts.append(f"Email: {s['email']}.")
        if s.get("phone"):
            parts.append(f"Phone: {s['phone']}.")
        docs.append({
            "id": f"staff-{sid}",
            "title": f"Staff: {name}",
            "url": "https://gunnwatt.web.app/utilities/staff",
            "source": "staff",
            "text": " ".join(parts),
        })
    return docs


def watt_course_docs():
    data = _get_json(f"{config.WATT_RAW}/catalog.json")["data"]
    docs = []
    for i, c in enumerate(data):
        names = c.get("names", [])
        title = names[0]["title"].strip().title() if names else f"Course {i}"
        aka = [n["title"].strip().title() for n in names[1:]]
        parts = [f"Course: {title}."]
        if aka:
            parts.append(f"Also listed as: {', '.join(aka)}.")
        if c.get("section"):
            parts.append(f"Department / section: {c['section'].title()}.")
        if c.get("grades"):
            parts.append(f"Open to grades: {', '.join(str(g) for g in c['grades'])}.")
        if c.get("length"):
            parts.append(f"Length: {c['length']}.")
        if c.get("credit"):
            parts.append(f"Credit: {c['credit']}.")
        if c.get("description"):
            parts.append(f"Description: {' '.join(c['description'].split())}")
        docs.append({
            "id": f"course-{i}",
            "title": f"Course: {title}",
            "url": "https://gunnwatt.web.app/utilities/courses",
            "source": "course",
            "text": " ".join(parts),
        })
    return docs


def watt_club_docs():
    data = _get_json(f"{config.WATT_RAW}/clubs.json")["data"]
    docs = []
    for cid, c in data.items():
        name = c.get("name", "").strip()
        if not name:
            continue
        parts = [f"Club: {name} (a student club at Gunn High School)."]
        if c.get("type"):
            parts.append(f"Category: {c['type']}.")
        if c.get("tier"):
            parts.append(f"Tier: {c['tier']}.")
        advisors = ", ".join(x for x in (c.get("advisor"), c.get("coadvisor")) if x)
        if advisors:
            parts.append(f"Teacher advisor(s) in charge of the club: {advisors}.")
        emails = ", ".join(x for x in (c.get("email"), c.get("coemail")) if x)
        if emails:
            parts.append(f"Advisor email(s): {emails}.")
        if c.get("prez"):
            parts.append(f"Student president(s): {c['prez']}.")
        if c.get("day"):
            parts.append(f"Meeting day: {c['day']}.")
        if c.get("freq"):
            parts.append(f"Frequency: {c['freq']}.")
        if c.get("time"):
            parts.append(f"Meeting time: {c['time']}.")
        if c.get("room"):
            parts.append(f"Room: {c['room']}.")
        if c.get("desc"):
            parts.append(f"Description: {' '.join(c['desc'].split())}")
        docs.append({
            "id": f"club-{cid}",
            "title": f"Club: {name}",
            "url": "https://gunnwatt.web.app/clubs",
            "source": "club",
            "text": " ".join(parts),
        })
    return docs


# --------------------------------------------------------------------------- #
# Lunch (Nutrislice) — snapshot current + next month
# --------------------------------------------------------------------------- #
# Order + labels for the nutrition fields Nutrislice provides.
NUTRITION_FIELDS = [
    ("calories", "", "calories"), ("g_fat", "g", "total fat"),
    ("g_saturated_fat", "g", "saturated fat"), ("g_trans_fat", "g", "trans fat"),
    ("mg_cholesterol", "mg", "cholesterol"), ("mg_sodium", "mg", "sodium"),
    ("g_carbs", "g", "carbohydrates"), ("g_fiber", "g", "dietary fiber"),
    ("g_sugar", "g", "sugar"), ("g_added_sugar", "g", "added sugar"),
    ("g_protein", "g", "protein"), ("mg_potassium", "mg", "potassium"),
    ("mg_calcium", "mg", "calcium"), ("mg_iron", "mg", "iron"),
    ("mg_vitamin_c", "mg", "vitamin C"), ("mcg_vitamin_d", "mcg", "vitamin D"),
    ("mcg_vitamin_a", "mcg", "vitamin A"),
]


def _num(v):
    if v is None:
        return None
    return str(int(v)) if float(v) == int(v) else str(round(float(v), 1))


def _nutrition_str(food):
    n = food.get("rounded_nutrition_info") or {}
    parts = []
    for key, unit, label in NUTRITION_FIELDS:
        v = _num(n.get(key))
        if v is not None:
            parts.append(f"{v}{unit} {label}")
    return ", ".join(parts)


def _diet_flags(food):
    """Dietary + allergen labels from Nutrislice icons (Vegan, Vegetarian, Wheat...)."""
    icons = ((food.get("icons") or {}).get("food_icons")) or []
    names = []
    for ic in icons:
        name = ic.get("synced_name") or (ic.get("sprite") or {}).get("name")
        if name and name not in names:
            names.append(name)
    return names


def _item_detail(food, meal):
    name = food["name"].strip()
    flags = _diet_flags(food)
    cat = food.get("food_category") or ""
    ss = food.get("serving_size_info") or {}
    serving = f"{ss.get('serving_size_amount','')} {ss.get('serving_size_unit','')}".strip()
    lines = [f"{name} — a {meal} item served at Gunn High School" + (f" ({cat})." if cat else ".")]
    if food.get("description"):
        lines.append(food["description"].strip())
    if flags:
        lines.append(f"Dietary and allergen labels: {', '.join(flags)}.")
    veg = [f for f in flags if "vegan" in f.lower() or "vegetarian" in f.lower()]
    if veg:
        lines.append(f"This item is marked: {', '.join(veg)}.")
    if serving:
        lines.append(f"Serving size: {serving}.")
    nut = _nutrition_str(food)
    if nut:
        lines.append(f"Nutrition facts per serving: {nut}.")
    ing = (food.get("ingredients") or "").strip()
    if ing:
        lines.append(f"Ingredients: {ing}")
    return " ".join(lines)


def lunch_docs(weeks_ahead=44):
    """Load EVERY published day of the school year with full nutrition, dietary
    flags, and ingredients. Nutrislice returns one week per request, so we step
    week by week across the whole year; unpublished weeks are simply skipped.
    Produces: one summary doc per day (for 'what's for lunch on X') plus one
    detailed doc per unique menu item (for 'calories in X' / 'is X vegan')."""
    day_docs, item_details = [], {}
    monday = dt.date.today() - dt.timedelta(days=dt.date.today().weekday())
    week_starts = [monday + dt.timedelta(days=7 * i) for i in range(weeks_ahead)]
    for meal in ("lunch", "breakfast"):
        seen = set()
        for wk in week_starts:
            url = config.NUTRISLICE.format(meal=meal, year=wk.year,
                                           month=wk.month, day=wk.day)
            try:
                data = _get_json(url)
            except Exception:
                continue
            for day in data.get("days", []):
                date = day.get("date")
                foods = [it["food"] for it in day.get("menu_items", [])
                         if it.get("food") and it["food"].get("name")]
                if not date or date in seen or not foods:
                    continue
                seen.add(date)
                try:
                    pretty = dt.date.fromisoformat(date).strftime("%A, %B %-d, %Y")
                except Exception:
                    pretty = date
                # per-day summary
                lines = [f"{meal.title()} menu for {pretty} at Gunn High School.",
                         "Items served today:"]
                veg_today = []
                for f in foods:
                    flags = _diet_flags(f)
                    tag = f" [{', '.join(flags)}]" if flags else ""
                    cal = _num((f.get("rounded_nutrition_info") or {}).get("calories"))
                    calstr = f" — {cal} cal" if cal else ""
                    lines.append(f"  - {f['name'].strip()}{calstr}{tag}")
                    if any("vegan" in x.lower() or "vegetarian" in x.lower() for x in flags):
                        veg_today.append(f["name"].strip())
                    key = f["name"].strip().lower()
                    if key not in item_details:
                        item_details[key] = _item_detail(f, meal)
                if veg_today:
                    lines.append("Vegetarian/vegan options today: " + ", ".join(veg_today) + ".")
                day_docs.append({
                    "id": f"{meal}-{date}",
                    "title": f"{meal.title()} on {pretty}",
                    "url": "https://gunnwatt.web.app/",
                    "source": meal,
                    "text": "\n".join(lines),
                })
    # per-item detail docs (deduped across the whole year)
    detail_docs = []
    for i, (key, text) in enumerate(sorted(item_details.items())):
        detail_docs.append({
            "id": f"menu-item-{i}",
            "title": f"Menu item: {text.split(' — ')[0]}",
            "url": "https://gunnwatt.web.app/",
            "source": "menu-item",
            "text": text,
        })
    print(f"    lunch: {len(day_docs)} day-menus, {len(detail_docs)} unique items",
          file=sys.stderr)
    return day_docs + detail_docs


# --------------------------------------------------------------------------- #
# Gunn Wiki (MediaWiki API) — every article
# This wiki has no TextExtracts extension, so we pull raw wikitext and clean it.
# --------------------------------------------------------------------------- #
def _clean_links(wt):
    wt = re.sub(r"\[\[(?:File|Image|Category):[^\]]*\]\]", "", wt, flags=re.I)
    wt = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", wt)   # [[a|b]] -> b
    wt = re.sub(r"\[\[([^\]]*)\]\]", r"\1", wt)             # [[a]]   -> a
    wt = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", wt)    # [url text] -> text
    return wt


def _extract_infobox(wt):
    """Pull `|key = value` fields out of the leading {{Infobox ...}} as facts."""
    low = wt.lower()
    i = low.find("{{infobox")
    if i == -1:
        return [], wt
    depth, j = 0, i
    while j < len(wt) - 1:
        two = wt[j:j + 2]
        if two == "{{":
            depth += 1; j += 2
        elif two == "}}":
            depth -= 1; j += 2
            if depth == 0:
                break
        else:
            j += 1
    block, rest = wt[i:j], wt[:i] + wt[j:]
    facts = []
    for m in re.finditer(r"\|\s*([A-Za-z][\w /]*?)\s*=\s*([^|{}\n]+)", block):
        key, val = m.group(1).strip(), m.group(2).strip()
        if val and key.lower() not in ("image", "caption"):
            facts.append(f"{key}: {val}")
    return facts, rest


def clean_wikitext(wt):
    wt = re.sub(r"<!--.*?-->", "", wt, flags=re.S)
    wt = re.sub(r"<ref[^>]*/>", "", wt)
    wt = re.sub(r"<ref[^>]*>.*?</ref>", "", wt, flags=re.S)
    wt = _clean_links(wt)
    facts, wt = _extract_infobox(wt)
    prev = None                                  # strip remaining templates
    while prev != wt:
        prev = wt
        wt = re.sub(r"\{\{[^{}]*\}\}", "", wt)
    wt = re.sub(r"\{\|.*?\|\}", "", wt, flags=re.S)          # tables
    wt = re.sub(r"<[^>]+>", "", wt)                          # stray HTML
    wt = re.sub(r"^[*#:;]+\s*", "", wt, flags=re.M)          # list markers
    wt = re.sub(r"'''?", "", wt)                             # bold/italic
    wt = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1:", wt, flags=re.M)  # headings
    wt = re.sub(r"\n{3,}", "\n\n", wt).strip()
    body = ("Key facts: " + "; ".join(facts) + "\n\n" + wt) if facts else wt
    return body.strip()


def wiki_docs():
    docs = []
    titles = []
    apcontinue = None
    while True:
        params = {
            "action": "query", "list": "allpages", "aplimit": "500",
            "apnamespace": "0", "apfilterredir": "nonredirects", "format": "json",
        }
        if apcontinue:
            params["apcontinue"] = apcontinue
        r = requests.get(config.WIKI_API, params=params, timeout=30,
                         headers={"User-Agent": "GunnGPT/1.0"})
        r.raise_for_status()
        d = r.json()
        titles += [p["title"] for p in d["query"]["allpages"]]
        if "continue" in d:
            apcontinue = d["continue"]["apcontinue"]
        else:
            break

    print(f"  wiki: {len(titles)} articles; fetching wikitext...", file=sys.stderr)
    for i in range(0, len(titles), 25):
        batch = titles[i:i + 25]
        params = {
            "action": "query", "prop": "revisions", "rvprop": "content",
            "rvslots": "main", "titles": "|".join(batch), "format": "json",
        }
        r = requests.get(config.WIKI_API, params=params, timeout=60,
                         headers={"User-Agent": "GunnGPT/1.0"})
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title", "")
            revs = page.get("revisions", [])
            if not revs:
                continue
            rev = revs[0]
            raw = rev.get("slots", {}).get("main", {}).get("*") or rev.get("*", "")
            text = clean_wikitext(raw)
            if len(text) < 20:
                continue
            url = config.WIKI_BASE + "/" + title.replace(" ", "_")
            docs.append({
                "id": f"wiki-{page.get('pageid', title)}",
                "title": title,
                "url": url,
                "source": "wiki",
                "text": f"Gunn Wiki article: {title}\n\n{text}",
            })
    return docs


def app_docs():
    """Self-knowledge: what GunnGPT (this website/app) is, its pages, layout, and
    features — so it can tell users what it does and where to find things."""
    overview = (
        "About GunnGPT — this website / this app. GunnGPT is the website and app you "
        "are using right now. It is a free, student-built assistant and student "
        "portal for Henry M. Gunn High School in Palo Alto, California. GunnGPT does "
        "two main things: (1) it is a chatbot (this chat) that answers questions about "
        "Gunn — the bell schedule, when school starts and ends, lunch and brunch "
        "menus, courses, clubs, teachers and staff, the campus, and general school "
        "info; and (2) it is a student portal with pages for the daily schedule, a "
        "scannable lunch barcode, clubs, courses, and a campus map. Everything runs "
        "locally on the school's own server — nothing you type in the chat leaves that "
        "server, and there are no accounts or logins. Its schedule and campus data "
        "come from WATT (gunnwatt.web.app). GunnGPT is NOT the same thing as gunn.one: "
        "gunn.one is a separate grade-calculator website, while GunnGPT is this "
        "schedule/info assistant. If someone asks 'who made this website', 'what is "
        "this site', 'what can you do', or 'what is GunnGPT', this is the answer."
    )
    layout = (
        "GunnGPT layout — the pages and where to find things. There is a sidebar on "
        "the left with a navigation menu; you can collapse or expand it with the "
        "button at the top (or by dragging its edge). The pages are:\n"
        "- Home: the main page. It shows a live clock, today's date (tap the date to "
        "open a calendar and jump to any day), the day of the week, and today's bell "
        "schedule as colored period cards with the times, plus a live countdown to "
        "when the current period or school ends. The chat box for asking GunnGPT "
        "questions is at the bottom of the Home page.\n"
        "- Barcode: your student ID barcode for buying/scanning lunch and brunch. Type "
        "your student ID number, tap the barcode to show it fullscreen for scanning, "
        "and use the + to add more barcodes.\n"
        "- Clubs: browse every Gunn club, grouped by category, with day tabs "
        "(Monday–Friday) and a search box. Tap a club to see its room, meeting day, "
        "advisor, president, and description.\n"
        "- Courses: browse every course in the Gunn catalog with a search box. Tap a "
        "course to see its department, grade levels, length, UC approval, and "
        "description.\n"
        "- Map: an interactive map of the Gunn campus you can drag to pan and scroll "
        "or double-click to zoom, to find buildings and room numbers.\n"
        "- Settings: change the theme (light or dark mode), the time format (12-hour "
        "or 24-hour), whether to show the clock, and whether to show Period 0 and "
        "Period 8 on the schedule. Where to find X: schedule and bell times are on "
        "Home; the lunch barcode is on the Barcode page; clubs on Clubs; classes on "
        "Courses; buildings and rooms on Map; dark mode and other options in Settings."
    )
    return [
        {"id": "app-about", "title": "About GunnGPT (this website)",
         "url": "/", "source": "gunngpt-app", "text": overview},
        {"id": "app-layout", "title": "GunnGPT pages & where to find things",
         "url": "/", "source": "gunngpt-app", "text": layout},
    ]


def build_all():
    builders = [
        ("about GunnGPT app", app_docs),
        ("core facts", core_facts_docs),
        ("academic calendar", calendar_docs),
        ("bell schedules", watt_schedule_docs),
        ("alternate/holiday days", watt_alternate_docs),
        ("staff", watt_staff_docs),
        ("courses", watt_course_docs),
        ("clubs", watt_club_docs),
        ("lunch menus", lunch_docs),
        ("wiki articles", wiki_docs),
    ]
    all_docs = []
    for label, fn in builders:
        try:
            got = fn()
            print(f"  {label}: {len(got)} docs", file=sys.stderr)
            all_docs += got
        except Exception as e:
            print(f"  !! {label} failed: {e}", file=sys.stderr)
    # Scraped external sites (official site, athletics, Wikipedia).
    try:
        import scrape
        all_docs += scrape.build_scrape_docs()
    except Exception as e:
        print(f"  !! scrape failed: {e}", file=sys.stderr)
    return all_docs


if __name__ == "__main__":
    docs = build_all()
    print(f"\nTOTAL: {len(docs)} documents", file=sys.stderr)
    # Print a few samples so you can sanity-check the corpus.
    for d in docs[:3]:
        print("\n---", d["title"], "---")
        print(d["text"][:300])
