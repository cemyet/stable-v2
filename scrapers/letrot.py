"""Le Trot (LeTROT — French trotting) scraper.

URL patterns (server-rendered HTML, light Tailwind layout):

    /courses/<YYYY-MM-DD>/<reunion-id>/<course-number>
        Race-day course page. Has the full results table (Table #1),
        partants table with musique/record/gains (Table #2), and
        per-section sectionals (Table #3 — only on PMU days).

    /stats/chevaux/<slug>/<letrot-id>/courses
        Horse identity page. Has name, sex, year, color, gains, record,
        sire/dam (with their letrot ids), trainer, owner, breeder.

    /stats/homme/<slug>/<letrot-id>/{driver|entraineur|proprietaire|eleveur}
        Person identity page (per role).

`letrot-id` is an opaque ~12-char base64-ish slug that the site treats as
stable per concept. We use it verbatim as the canonical foreign id.

The race-day course page is the high-leverage entrypoint: one HTTP call
gives us a race + every horse + every person + every result.
"""

from __future__ import annotations

import logging
import re
from datetime import date as Date
from typing import Iterable

import httpx
from bs4 import BeautifulSoup, Tag

from core.config import LETROT_BASE, LETROT_HEADERS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def make_client() -> httpx.Client:
    return httpx.Client(
        headers=LETROT_HEADERS,
        follow_redirects=True,
        timeout=30.0,
    )


def _get(client: httpx.Client, path: str) -> str | None:
    url = LETROT_BASE + path
    r = client.get(url)
    if r.status_code != 200:
        log.warning("letrot %s -> HTTP %s", url, r.status_code)
        return None
    return r.text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RE_HORSE_HREF = re.compile(r"^/stats/chevaux/([^/]+)/([^/]+)(?:/.*)?$")
_RE_PERSON_HREF = re.compile(r"^/stats/homme/([^/]+)/([^/]+)/(\w+)$")
_RE_FR_DATE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_RE_FR_TIME_KM = re.compile(r"^\s*(\d+)['\u2032](\d+)[\"\u2033](\d+)\s*$")  # 1'13"5
_RE_FR_TIME_KM2 = re.compile(r"^\s*(\d+)[\.,](\d+)[\.,](\d+)\s*$")          # 1.13.5
_RE_SA = re.compile(r"^([HFM])(\d+)$", re.IGNORECASE)


def _parse_horse_href(href: str | None) -> dict | None:
    if not href:
        return None
    m = _RE_HORSE_HREF.match(href)
    if not m:
        return None
    return {"slug": m[1], "letrot_id": m[2]}


def _parse_person_href(href: str | None) -> dict | None:
    if not href:
        return None
    m = _RE_PERSON_HREF.match(href)
    if not m:
        return None
    return {"slug": m[1], "letrot_id": m[2], "role": m[3]}


def _parse_fr_date(s: str | None) -> Date | None:
    if not s:
        return None
    m = _RE_FR_DATE.match(s.strip())
    if not m:
        return None
    try:
        return Date(int(m[3]), int(m[2]), int(m[1]))
    except ValueError:
        return None


def _parse_fr_kmtime_seconds(s: str | None) -> float | None:
    """1'13\"5 -> 73.5  (km-time in seconds)."""
    if not s:
        return None
    s = s.strip().replace("\u202f", "").replace("\xa0", "")
    if s in ("-", "DI", "DA"):
        return None
    for rx in (_RE_FR_TIME_KM, _RE_FR_TIME_KM2):
        m = rx.match(s)
        if m:
            return int(m[1]) * 60 + int(m[2]) + int(m[3]) / 10.0
    return None


def _parse_eur(s: str | None) -> int | None:
    """'7\u202f532€' or '12 345 €' -> 7532. Returns None for blanks."""
    if not s:
        return None
    s = s.replace("\xa0", "").replace("\u202f", "").replace(" ", "").replace("€", "").strip()
    if s in ("", "-"):
        return None
    s = s.replace(".", "").replace(",", ".")
    try:
        return int(round(float(s)))
    except ValueError:
        return None


def _parse_int(s: str | None) -> int | None:
    if not s:
        return None
    s = s.strip().replace("\xa0", "").replace("\u202f", "").replace(" ", "")
    if not s or s == "-":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _placement_from_rang(rang: str) -> tuple[int | None, str | None]:
    """'1' -> (1, '1'), '2e' -> (2, '2e'), '0' -> (None, '0').
    The site uses rang=0 for scratched/DNF horses."""
    raw = (rang or "").strip()
    if not raw or raw == "0":
        return None, raw
    digits = re.match(r"^(\d+)", raw)
    if digits:
        return int(digits[1]), raw
    return None, raw


def _gender_from_sa(sa: str | None) -> str | None:
    """'H10' -> 'H'  (Hongre/gelding)
       'F4'  -> 'F'  (Femelle/mare)
       'M8'  -> 'M'  (Mâle/stallion)
    """
    if not sa:
        return None
    m = _RE_SA.match(sa.strip())
    return m[1].upper() if m else None


def _age_from_sa(sa: str | None) -> int | None:
    if not sa:
        return None
    m = _RE_SA.match(sa.strip())
    return int(m[2]) if m else None


# ---------------------------------------------------------------------------
# Course (race-day) page parsing
# ---------------------------------------------------------------------------

_RE_RACE_NUMBER = re.compile(r"\bC\s*(\d+)\b", re.IGNORECASE)


def parse_course(html: str) -> dict:
    """Parse a Le Trot course page into {race, runners[]}."""
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    # Title pattern: "R1 VINCENNES C5 PRIX DU GATINAIS : partants, ..."
    track_name = None
    race_name = None
    race_number = None

    h1 = soup.find("h1")
    if h1:
        h1_text = h1.get_text(" ", strip=True)
        # h1 looks like "C5 PRIX DU GATINAIS"
        m = _RE_RACE_NUMBER.match(h1_text)
        if m:
            race_number = int(m[1])
            race_name = h1_text[m.end():].strip(" -:")
        else:
            race_name = h1_text

    # Track name is in the page title before the C# token.
    title_tokens = title.split()
    for tok in title_tokens:
        if tok.startswith("R") and tok[1:].isdigit():
            continue
        if _RE_RACE_NUMBER.match(tok):
            break
        # Track tokens are SHOUTING_CASE city names — pick the first one.
        if tok.isupper() and len(tok) >= 4:
            track_name = tok
            break

    # ARRIVÉE table — has the placements
    arrivee_table = None
    partants_table = None
    for t in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True) for th in t.find_all("th")]
        if not headers:
            continue
        if "Rang" in headers and "Cheval" in " ".join(headers):
            if arrivee_table is None:
                arrivee_table = t
        elif "Musique" in headers and "Cheval" in " ".join(headers):
            partants_table = t

    runners: list[dict] = []
    if arrivee_table is not None:
        runners = _parse_arrivee_table(arrivee_table)

    if partants_table is not None:
        _enrich_with_partants_table(runners, partants_table)

    return {
        "track_name": track_name,
        "race_name": race_name,
        "race_number": race_number,
        "runners": runners,
        "raw_h1": h1.get_text(" ", strip=True) if h1 else None,
        "raw_title": title,
    }


def _parse_arrivee_table(table: Tag) -> list[dict]:
    """Pull the placement/runner rows from the ARRIVÉE table."""
    out: list[dict] = []
    rows = table.find_all("tr")
    if len(rows) < 2:
        return out

    for tr in rows[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) < 11:
            continue
        # Headers: Rang, N°, Cheval+brand, Fer, Avis, SA, Driver+Entraîneur, Dist, Temps, Red.Km, Alloc, Rap.prob
        rang_text = cells[0].get_text(" ", strip=True)
        placement, placement_text = _placement_from_rang(rang_text)
        program_number = _parse_int(cells[1].get_text(" ", strip=True))

        # Cheval cell may contain horse link + crack-series number
        horse_cell = cells[2]
        horse_link = horse_cell.find("a", href=re.compile(r"^/stats/chevaux/"))
        horse_meta = _parse_horse_href(horse_link.get("href")) if horse_link else None
        horse_name = horse_link.get_text(" ", strip=True) if horse_link else horse_cell.get_text(" ", strip=True).split("|", 1)[0].strip()

        fer = cells[3].get_text(" ", strip=True) or None
        avis = cells[4].get_text(" ", strip=True) or None
        sa = cells[5].get_text(" ", strip=True) or None

        drv_cell = cells[6]
        person_links = drv_cell.find_all("a", href=re.compile(r"^/stats/homme/"))
        driver = _parse_person_href(person_links[0].get("href")) if len(person_links) >= 1 else None
        if driver is not None:
            driver["name"] = person_links[0].get_text(" ", strip=True)
        trainer = _parse_person_href(person_links[1].get("href")) if len(person_links) >= 2 else None
        if trainer is not None:
            trainer["name"] = person_links[1].get_text(" ", strip=True)

        distance = _parse_int(cells[7].get_text(" ", strip=True))
        temps = cells[8].get_text(" ", strip=True) or None
        red_km = cells[9].get_text(" ", strip=True) or None
        alloc = _parse_eur(cells[10].get_text(" ", strip=True))
        odds = cells[11].get_text(" ", strip=True) if len(cells) > 11 else None

        out.append({
            "placement": placement,
            "placement_text": placement_text,
            "program_number": program_number,
            "horse_name": horse_name,
            "horse_letrot_id": horse_meta["letrot_id"] if horse_meta else None,
            "horse_slug":      horse_meta["slug"]      if horse_meta else None,
            "fer": fer,
            "avis_trainer": avis,
            "sex": _gender_from_sa(sa),
            "age": _age_from_sa(sa),
            "driver": driver,
            "trainer": trainer,
            "distance": distance,
            "time_text": temps,
            "km_time_text": red_km,
            "km_time_seconds": _parse_fr_kmtime_seconds(red_km),
            "prize_eur": alloc,
            "odds_text": odds,
        })
    return out


def _enrich_with_partants_table(runners: list[dict], partants: Tag) -> None:
    """Add musique + record + gains-lifetime to existing runner rows by N°."""
    rows = partants.find_all("tr")
    if len(rows) < 2:
        return
    by_n = {r.get("program_number"): r for r in runners}
    for tr in rows[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) < 10:
            continue
        n = _parse_int(cells[0].get_text(" ", strip=True))
        target = by_n.get(n)
        if not target:
            continue
        # cells: N°, Cheval, Fer, SA, Dist, Driver+Entraîneur, Avis, Musique, Record, Gains, Moy gains
        target["musique"]        = cells[7].get_text(" ", strip=True) or None
        target["record_text"]    = cells[8].get_text(" ", strip=True).split("\n")[0].strip() or None
        target["gains_lifetime_eur"] = _parse_eur(cells[9].get_text(" ", strip=True))


# ---------------------------------------------------------------------------
# Horse identity (light) — only fetched on demand
# ---------------------------------------------------------------------------

def parse_horse_identity(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    out: dict = {}
    h1 = soup.find("h1")
    if h1:
        out["name"] = h1.get_text(" ", strip=True)

    # Walk label/value div pairs.
    for label_div in soup.select("div.text-xs.text-grey-medium"):
        label = label_div.get_text(strip=True)
        sib = label_div.find_next_sibling()
        if not sib:
            continue
        val = sib.get_text(" ", strip=True)
        link = sib.find("a")
        href = link.get("href") if link else None

        if label == "Sexe":
            out["sex"] = val
        elif label.startswith("Année"):
            try:
                out["birth_year"] = int(val)
            except ValueError:
                pass
        elif label == "Robe":
            out["color"] = val
        elif label == "Gains Totaux":
            out["gains_total_eur"] = _parse_eur(val)
        elif label.startswith("Record"):
            out["record_text"] = val
        elif label == "Père":
            out["sire_name"] = val
            ph = _parse_horse_href(href)
            out["sire_letrot_id"] = ph["letrot_id"] if ph else None
        elif label == "Mère":
            out["dam_name"] = val
            ph = _parse_horse_href(href)
            out["dam_letrot_id"] = ph["letrot_id"] if ph else None
        elif label == "Entraineur":
            out["trainer_name"] = val
            pp = _parse_person_href(href)
            out["trainer_letrot_id"] = pp["letrot_id"] if pp else None
        elif label.startswith("Propriétaire"):
            out["owner_name"] = val
            pp = _parse_person_href(href)
            out["owner_letrot_id"] = pp["letrot_id"] if pp else None
        elif label.startswith("Éleveur") or label.startswith("Eleveur"):
            out["breeder_name"] = val
            pp = _parse_person_href(href)
            out["breeder_letrot_id"] = pp["letrot_id"] if pp else None
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_course(client: httpx.Client, race_date: Date | str, reunion_id: int | str, course_number: int | str) -> dict | None:
    if isinstance(race_date, Date):
        race_date = race_date.isoformat()
    html = _get(client, f"/courses/{race_date}/{reunion_id}/{course_number}")
    if html is None:
        return None
    parsed = parse_course(html)
    parsed["race_date"] = race_date
    parsed["reunion_id"] = str(reunion_id)
    parsed["course_number"] = int(course_number)
    parsed["letrot_race_id"] = f"{race_date}_{reunion_id}_{course_number}"
    return parsed


def fetch_horse_identity(client: httpx.Client, letrot_id: str, slug: str = "x") -> dict | None:
    html = _get(client, f"/stats/chevaux/{slug}/{letrot_id}/courses")
    if html is None:
        return None
    out = parse_horse_identity(html)
    out["letrot_id"] = letrot_id
    return out


def list_today(client: httpx.Client, when: str = "aujourd-hui") -> list[dict]:
    """List courses on a given day. `when` is one of: aujourd-hui, hier, demain.

    Returns rows {race_date, reunion_id, course_number, href}."""
    html = _get(client, f"/courses/{when}")
    if html is None:
        return []
    soup = BeautifulSoup(html, "lxml")
    out: list[dict] = []
    for a in soup.select('a[href^="/courses/2"]'):
        href = a.get("href")
        m = re.match(r"^/courses/(\d{4}-\d{2}-\d{2})/(\d+)/(\d+)$", href)
        if not m:
            continue
        out.append({
            "race_date":     m[1],
            "reunion_id":    m[2],
            "course_number": int(m[3]),
            "href":          href,
        })
    # De-dup
    seen = set()
    dedup: list[dict] = []
    for r in out:
        key = r["href"]
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import sys, json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: python -m scrapers.letrot list <hier|aujourd-hui|demain> | course <YYYY-MM-DD> <reunion> <course> | horse <letrot-id>")
        return

    cmd = sys.argv[1]
    client = make_client()
    try:
        if cmd == "list":
            when = sys.argv[2] if len(sys.argv) >= 3 else "aujourd-hui"
            rows = list_today(client, when)
            print(f"{len(rows)} courses for {when}:")
            for r in rows[:10]:
                print("  ", r)
        elif cmd == "course":
            d, rid, c = sys.argv[2], sys.argv[3], sys.argv[4]
            res = fetch_course(client, d, rid, c)
            if res:
                print(f"track={res['track_name']!r} race_number={res['race_number']!r} "
                      f"race_name={res['race_name']!r} runners={len(res['runners'])}")
                for r in res["runners"][:3]:
                    print("  ", json.dumps(r, ensure_ascii=False, default=str))
        elif cmd == "horse":
            lid = sys.argv[2]
            slug = sys.argv[3] if len(sys.argv) >= 4 else "x"
            print(fetch_horse_identity(client, lid, slug))
        else:
            raise SystemExit(f"unknown command: {cmd!r}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
