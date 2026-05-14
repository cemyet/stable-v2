"""Breedly (avelstjänst.se / breedly.com) scraper.

Breedly is a Swedish breeding service. Each horse page is fully server-
rendered and embeds an Apollo GraphQL state cache in `window.App = {...}`,
which gives us:

    - The focal horse with: breedly_id, horseName, bornCountry, bornYear,
      gender, inbreeding (%), breederName, fatherId/motherId, slug, stId
      (the TravSport id when known — gold for cross-source matching!).
    - Up to ~56 ancestors in 5-7 generations of pedigree, each with their
      own breedly_id, name, year, country, record, lifetime earnings string,
      placement string ("35 (8-3-4)"), and stId when known.

URL patterns:
    /horse/<slug>                 — horse profile (served as HTML, with
                                    Apollo cache in `window.App`)
    /search-horse?q=<term>        — search (HTML; results are <a href> tags)
    /stallions/<slug>             — stallion-specific (richer record set)

We only consume the focal horse + its pedigree ancestors. Everything we
care about is in the Apollo cache so we don't bother with the rendered
HTML of the page.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from core.config import BREEDLY_BASE, BREEDLY_HEADERS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def make_client() -> httpx.Client:
    return httpx.Client(
        headers=BREEDLY_HEADERS,
        follow_redirects=True,
        timeout=30.0,
    )


# ---------------------------------------------------------------------------
# window.App extraction
# ---------------------------------------------------------------------------

def _extract_window_app(html: str) -> dict | None:
    """Find `window.App = {...};` and return the parsed object."""
    i = html.find("window.App=")
    if i < 0:
        return None
    i += len("window.App=")
    depth = 0
    in_str = False
    escape = False
    j = i
    while j < len(html):
        c = html[j]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
        j += 1
    if depth != 0:
        return None
    body = html[i:j + 1]
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        log.warning("breedly window.App json parse failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Apollo cache parsing
# ---------------------------------------------------------------------------

def _horses_from_apollo(window_app: dict) -> dict[str, dict]:
    """Return {breedly_id_str: horse_dict} for every Horse entry in apolloState."""
    apollo = (window_app or {}).get("apolloState") or {}
    out: dict[str, dict] = {}
    for k, v in apollo.items():
        if not isinstance(v, dict):
            continue
        if v.get("__typename") != "Horse":
            continue
        bid = v.get("horseId")
        if bid is None:
            continue
        out[str(bid)] = v
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_horse_by_slug(client: httpx.Client, slug: str) -> dict | None:
    """Fetch /horse/<slug> and return {focal_horse, ancestors[]}."""
    url = f"{BREEDLY_BASE}/horse/{slug}"
    r = client.get(url)
    if r.status_code != 200:
        log.warning("breedly %s -> HTTP %s", url, r.status_code)
        return None
    win = _extract_window_app(r.text)
    if not win:
        return None
    horses = _horses_from_apollo(win)

    # The focal horse is identified by URL slug.
    focal = next((h for h in horses.values() if h.get("slug") == slug), None)
    if not focal:
        # Fallback: the only horse with `inbreeding` populated is usually the focal.
        focal = next((h for h in horses.values() if h.get("inbreeding") is not None), None)

    return {
        "focal":     focal,
        "ancestors": {bid: h for bid, h in horses.items() if h is not focal},
        "raw_count": len(horses),
    }


def search_horses(client: httpx.Client, term: str) -> list[dict]:
    """Search /search-horse?q=<term> and return discovered horse links."""
    url = f"{BREEDLY_BASE}/search-horse?q={term}"
    r = client.get(url)
    if r.status_code != 200:
        log.warning("breedly search %r -> HTTP %s", term, r.status_code)
        return []
    soup = BeautifulSoup(r.text, "lxml")
    out: list[dict] = []
    seen = set()
    for a in soup.select('a[href^="/horse/"]'):
        href = a.get("href")
        if not href or "/hypothetical" in href:
            continue
        slug = href[len("/horse/"):]
        if slug in seen:
            continue
        seen.add(slug)
        out.append({
            "slug": slug,
            "name": a.get_text(" ", strip=True),
            "href": href,
        })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: python -m scrapers.breedly search <term> | horse <slug>")
        return

    cmd = sys.argv[1]
    client = make_client()
    try:
        if cmd == "search":
            for h in search_horses(client, sys.argv[2]):
                print(h)
        elif cmd == "horse":
            slug = sys.argv[2]
            res = fetch_horse_by_slug(client, slug)
            if not res:
                print("no data")
                return
            focal = res["focal"]
            print("=== focal ===")
            for k in ("horseId", "slug", "stId", "source", "horseName", "bornCountry", "bornYear",
                     "gender", "inbreeding", "breederName", "fatherId", "fatherName",
                     "motherId", "motherName"):
                print(f"  {k}: {focal.get(k)!r}")
            print(f"=== {len(res['ancestors'])} ancestors ===")
            sample = list(res["ancestors"].values())[:5]
            for a in sample:
                print(f"  H:{a.get('horseId')} {a.get('horseDisplayName')!r} "
                      f"bornYear={a.get('bornYear')} stId={a.get('stId')!r}")
        else:
            raise SystemExit(f"unknown cmd: {cmd!r}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
