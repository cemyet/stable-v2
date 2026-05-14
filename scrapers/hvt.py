"""HVT online (Hauptverband für Traberzucht — German trotting) scraper.

The site is a server-rendered PHP app that gates everything behind a
PHPSESSID cookie. We use a single httpx.Client throughout to keep that
cookie alive.

Public entry-points:

    search_horses(client, term)               -> list[dict]
        POST /traberdaten/ with horsesearch=<term>. Returns the matched
        rows as { traberid, name, country, color, sex, year, sire, dam,
        winnings_eur }.

    fetch_grunddaten(client, traberid)        -> dict
        POST /ajax/trabersuchefirst.php returns JSON {trabername, menue,
        html}. We parse `html` into canonical horse fields.

    fetch_formenspiegel(client, traberid)     -> list[dict]
        POST /ajax/trabersuchechange.php tab=1 — race history.

    fetch_pedigree(client, traberid, gens=5)  -> dict
        POST /ajax/trabersuchechange.php tab=3 — pedigree links by id.

All HTML parsing uses BeautifulSoup. HVT's markup is very consistent
two-column tables with a German label in the first cell, so we walk
rows and map labels directly.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date as Date
from typing import Iterable

import httpx
from bs4 import BeautifulSoup

from core.config import HVT_BASE, HVT_HEADERS, REQUEST_DELAY

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def make_client() -> httpx.Client:
    """A fresh httpx.Client with our UA + a primed PHPSESSID."""
    c = httpx.Client(
        headers=HVT_HEADERS,
        follow_redirects=True,
        timeout=30.0,
        cookies={},
    )
    # First GET primes PHPSESSID; subsequent requests reuse the cookie jar.
    try:
        c.get(f"{HVT_BASE}/")
    except Exception as e:
        log.warning("hvt session prime failed: %s", e)
    return c


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

_RE_SEARCH_ROW = re.compile(
    r"<tr><td><a data-traberid='(?P<id>\d+)'[^>]*>&rsaquo;\s*"
    r"(?P<name_html>.+?)</a></td>"
    r"<td class='center'>(?P<country>[^<]*)</td>"
    r"<td class='left'>(?P<color>[^<]*)</td>"
    r"<td class='center'>(?P<sex>[^<]*)</td>"
    r"<td class='center'>(?P<year>[^<]*)</td>"
    r"<td class='left'>(?P<sire>[^<]*)</td>"
    r"<td class='left'>(?P<dam>[^<]*)</td>"
    r"<td class='right'>(?P<winnings>[^<]*)</td></tr>"
)


def _strip_html_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def _money_to_int_eur(s: str) -> int | None:
    """'6.038.408 €' -> 6038408. Returns None when blank / unparseable."""
    if not s:
        return None
    s = s.replace("&euro;", "€").replace("\xa0", " ").strip()
    s = re.sub(r"[€$£\s]", "", s)
    s = s.replace(".", "").replace(",", ".")
    if not s:
        return None
    try:
        return int(round(float(s)))
    except ValueError:
        return None


def search_horses(client: httpx.Client, term: str) -> list[dict]:
    """POST the search form and parse result rows."""
    if not term or len(term) < 2:
        return []
    r = client.post(
        f"{HVT_BASE}/traberdaten/",
        data={"horsesearch": term, "search": "1", "submitsearch": ""},
    )
    if r.status_code != 200:
        log.warning("hvt search %s -> HTTP %s", term, r.status_code)
        return []

    html = r.text
    out: list[dict] = []
    for m in _RE_SEARCH_ROW.finditer(html):
        out.append({
            "traberid":     m["id"],
            "name":         _strip_html_tags(m["name_html"]),
            "country":      m["country"].strip() or None,
            "color":        m["color"].strip() or None,
            "sex":          m["sex"].strip() or None,
            "year":         _to_int(m["year"]),
            "sire":         m["sire"].strip() or None,
            "dam":          m["dam"].strip() or None,
            "winnings_eur": _money_to_int_eur(m["winnings"]),
        })
    return out


def _to_int(s: str | None) -> int | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Grunddaten — basic horse data
# ---------------------------------------------------------------------------

_GENDER_DE_TO_CODE = {
    "Hengst": "H",          # stallion
    "Stute": "S",           # mare
    "Wallach": "V",         # gelding (German "Wallach", v1 uses 'V'/'W'; we map to V)
    "Fohlen": None,         # foal
}


def _parse_de_date(s: str | None) -> Date | None:
    """16.04.2002 -> date(2002, 4, 16)."""
    if not s:
        return None
    s = s.strip()
    m = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", s)
    if not m:
        return None
    try:
        return Date(int(m[3]), int(m[2]), int(m[1]))
    except ValueError:
        return None


def _country_from_name_suffix(s: str | None) -> str | None:
    """'Marlen (DE)' -> 'DE'. 'Nuncio (US)' -> 'US'. Returns None when missing."""
    if not s:
        return None
    m = re.search(r"\(([A-Z]{2})\)\s*$", s)
    return m[1] if m else None


def _strip_country_suffix(s: str | None) -> str | None:
    """Strip both '(XX)' and '[XX]' country suffixes (HVT mixes both styles)."""
    if not s:
        return None
    out = re.sub(r"\s*[\(\[]([A-Z]{1,3}|-)[\)\]]\s*$", "", s).strip()
    return out or None


def fetch_grunddaten(client: httpx.Client, traberid: str | int) -> dict:
    """Fetch + parse the Grunddaten (basic data) page for a single horse."""
    r = client.post(
        f"{HVT_BASE}/ajax/trabersuchefirst.php",
        data={"horseid": str(traberid), "p": "1"},
    )
    if r.status_code != 200:
        log.warning("hvt grunddaten %s -> HTTP %s", traberid, r.status_code)
        return {}
    try:
        outer = json.loads(r.text)
    except json.JSONDecodeError:
        log.warning("hvt grunddaten %s -> non-json response", traberid)
        return {}

    html = outer.get("html") or ""
    soup = BeautifulSoup(html, "lxml")

    out: dict = {
        "hvt_id": str(traberid),
        "trabername": outer.get("trabername"),
    }

    # Header span: ID / UELN / CHIP
    header = soup.find("div", class_="generalheader")
    if header:
        for span in header.find_all("span"):
            t = span.get_text(strip=True)
            if t.startswith("ID:"):
                out["hvt_id"] = t.replace("ID:", "").strip()
            elif t.startswith("UELN:"):
                out["ueln_number"] = t.replace("UELN:", "").strip()
            elif t.startswith("CHIP:"):
                out["chip_number"] = t.replace("CHIP:", "").strip()

    # Walk all the two-column tables.
    raw_kv: dict[str, str] = {}
    sire_id: str | None = None
    dam_id: str | None = None
    for table in soup.find_all("table", class_="gestuetbuch"):
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) != 2:
                continue
            label = cells[0].get_text(strip=True).rstrip(":")
            val_cell = cells[1]
            val_txt = val_cell.get_text(strip=True)
            raw_kv[label] = val_txt
            # Capture sire/dam IDs from the embedded link
            link = val_cell.find("a", attrs={"data-traberid": True})
            if link and label.startswith("Vater"):
                sire_id = link.get("data-traberid")
            elif link and label.startswith("Mutter"):
                dam_id = link.get("data-traberid")

    out["raw_grunddaten"] = raw_kv

    # Promote known fields.
    name_with_country = raw_kv.get("Name des Trabers")
    out["name"] = _strip_country_suffix(name_with_country) or outer.get("trabername")
    out["registration_country"] = _country_from_name_suffix(name_with_country)
    out["birth_country"] = out["registration_country"]
    out["bred_country"]  = out["registration_country"]

    out["gender_code"] = _GENDER_DE_TO_CODE.get(raw_kv.get("Geschlecht", "").strip())
    out["color"] = raw_kv.get("Farbe") or None
    out["date_of_birth"] = _parse_de_date(raw_kv.get("Geburtsdatum"))

    out["sire_name"] = _strip_country_suffix(raw_kv.get("Vater")) if raw_kv.get("Vater") else None
    out["dam_name"]  = _strip_country_suffix(raw_kv.get("Mutter")) if raw_kv.get("Mutter") else None
    out["sire_hvt_id"] = sire_id
    out["dam_hvt_id"]  = dam_id

    out["breeder_name"] = raw_kv.get("Züchter") or raw_kv.get("Zuchter") or None
    out["owner_name"]   = raw_kv.get("Besitzer") or None
    out["last_trainer_name"] = raw_kv.get("Letzter Trainer") or None
    out["last_driver_name"]  = raw_kv.get("Letzter Fahrer") or None

    out["scraped_prize_money_eur"] = _money_to_int_eur(raw_kv.get("Lebensgewinnsumme") or "")
    out["scraped_record"]          = raw_kv.get("Lebensrekord") or None

    starts_text = raw_kv.get("Starts / Siege / Plätze") or raw_kv.get("Starts / Siege / Platze") or ""
    starts_m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*$", starts_text)
    if starts_m:
        out["scraped_starts"]  = int(starts_m[1])
        out["scraped_wins"]    = int(starts_m[2])
        out["scraped_placed"]  = int(starts_m[3])

    return out


# ---------------------------------------------------------------------------
# Formenspiegel — race history
# ---------------------------------------------------------------------------

_RE_DRIVER_FORM = re.compile(
    r'<form[^>]*action="/fahrerdaten"[^>]*id="fahrersuche-(?P<fid>[^"]+)"[^>]*>'
    r".*?>(?P<name>[^<]+)</a>"
)


def _strip_de_thousands(s: str) -> str:
    """'2.000' -> '2000'. Distance/prize tokens use German thousands sep '.'"""
    return s.replace(".", "")


def fetch_formenspiegel(client: httpx.Client, traberid: str | int) -> list[dict]:
    """Fetch + parse the race-history table for a single horse."""
    r = client.post(
        f"{HVT_BASE}/ajax/trabersuchechange.php",
        data={"horseid": str(traberid), "tab": "1"},
    )
    if r.status_code != 200:
        log.warning("hvt formenspiegel %s -> HTTP %s", traberid, r.status_code)
        return []
    html = r.text
    if len(html) < 500:
        # No race history (Stallion-only / unraced horse).
        return []

    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="leistungenmain")
    if not table:
        return []

    out: list[dict] = []
    rows = table.find_all("tr")
    i = 0
    while i < len(rows):
        row = rows[i]
        classes = row.get("class") or []
        # Race rows have class one/onedark in the first td. Year-divider rows
        # have a single 'showyear' cell. Header rows have <th>.
        first_td = row.find("td")
        if not first_td:
            i += 1
            continue
        td_classes = first_td.get("class") or []
        if "showyear" in td_classes:
            i += 1
            continue
        if "one" not in td_classes and "onedark" not in td_classes:
            i += 1
            continue

        cells = row.find_all("td")
        if len(cells) < 11:
            i += 1
            continue

        race = {
            "race_date":      _parse_de_date(cells[0].get_text(strip=True)),
            "track_name":     cells[1].get_text(strip=True),
            # cells[2] is a marker cell (often empty)
            "placement_text": cells[3].get_text(strip=True),
            "time_text":      cells[4].get_text(strip=True),
            "distance":       _to_int(_strip_de_thousands(re.sub(r"[^\d]", "", cells[5].get_text(strip=True)))),
            "start_method":   cells[6].get_text(strip=True),
            "race_type":      cells[7].get_text(strip=True),
            "program_number": _to_int(cells[8].get_text(strip=True)),
            "evq":            cells[9].get_text(strip=True),
            "prize_eur":      _money_to_int_eur(cells[10].get_text(strip=True)),
        }
        # The next row is the driver/top3 row (also class=two/twodark).
        drv_row = rows[i + 1] if i + 1 < len(rows) else None
        if drv_row:
            drv_cells = drv_row.find_all("td")
            if len(drv_cells) >= 2:
                # driver name + top3 finishers
                drv_form = drv_cells[1].find("form", attrs={"action": "/fahrerdaten"})
                if drv_form:
                    a = drv_form.find("a")
                    fid_input = drv_form.find("input", attrs={"name": "fahrerid"})
                    race["driver_name"] = a.get_text(strip=True) if a else None
                    race["driver_hvt_id"] = fid_input["value"] if fid_input else None
                top3_cell = drv_cells[2] if len(drv_cells) >= 3 else None
                if top3_cell:
                    race["top3_text"] = top3_cell.get_text(strip=True)
            i += 2
        else:
            i += 1

        # Convert placement_text "1.", "2." etc to int when possible
        pt = race["placement_text"].rstrip(".")
        if pt.isdigit():
            race["placement"] = int(pt)
        else:
            race["placement"] = None

        out.append(race)
    return out


# ---------------------------------------------------------------------------
# Pedigree (3 or 5 generations) — used for cross-source horse discovery
# ---------------------------------------------------------------------------

def fetch_pedigree(client: httpx.Client, traberid: str | int, gens: int = 5) -> list[dict]:
    """Return list of {traberid, name, country, record, role(generation/side)}."""
    if gens not in (3, 5):
        raise ValueError("gens must be 3 or 5")
    tab = "3" if gens == 5 else "2"
    r = client.post(
        f"{HVT_BASE}/ajax/trabersuchechange.php",
        data={"horseid": str(traberid), "tab": tab},
    )
    if r.status_code != 200:
        log.warning("hvt pedigree %s gens=%s -> HTTP %s", traberid, gens, r.status_code)
        return []
    soup = BeautifulSoup(r.text, "lxml")
    out: list[dict] = []
    for a in soup.find_all("a", attrs={"data-traberid": True}):
        cls = a.get("class") or []
        # Skip the produkte links — they're offspring, not ancestors.
        # Ancestor links carry class 'vater' / 'mutter'.
        if "vater" not in cls and "mutter" not in cls:
            continue
        text = a.get_text("\n", strip=True)
        # Format: "Andover Hall [US]\n1:09,4"   or "Lindi Lana [US]\no. Rek."
        first_line = text.split("\n", 1)[0]
        country_m = re.search(r"\[([A-Z]{2}|-)\]\s*$", first_line)
        out.append({
            "traberid": a["data-traberid"],
            "name": re.sub(r"\s*\[[A-Z-]{1,2}\]\s*$", "", first_line).strip(),
            "country": country_m[1] if country_m else None,
            "record": text.split("\n", 1)[1] if "\n" in text else None,
            "side": "sire" if "vater" in cls else "dam",
        })
    return out


# ---------------------------------------------------------------------------
# CLI: `python -m scrapers.hvt search Marlen`  /  `python -m scrapers.hvt horse 120252`
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: python -m scrapers.hvt search <term> | horse <traberid>")
        return

    cmd = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) >= 3 else ""

    client = make_client()
    try:
        if cmd == "search":
            for h in search_horses(client, arg):
                print(h)
        elif cmd == "horse":
            g = fetch_grunddaten(client, arg)
            print("--- grunddaten ---")
            for k, v in g.items():
                if k != "raw_grunddaten":
                    print(f"  {k}: {v}")
            print("--- formenspiegel ---")
            races = fetch_formenspiegel(client, arg)
            print(f"  {len(races)} races")
            for r in races[:3]:
                print(f"  - {r}")
            print("--- pedigree (5 gens) ---")
            ped = fetch_pedigree(client, arg, gens=5)
            print(f"  {len(ped)} ancestor links")
            for p in ped[:6]:
                print(f"  - {p}")
        else:
            raise SystemExit(f"unknown command: {cmd!r}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
