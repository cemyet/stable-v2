"""Bot-protection clearance for the Svensk Travsport JSON API.

`sportapp.travsport.se` gates `/webapi/*` behind a proof-of-work bot check
(vendor: baffinbay). A plain httpx request gets one of two non-answers:

    * HTTP 403 "Threat Protection - Request Blocked" for a bare header set
    * HTTP 200 with a "Verifying..." HTML interstitial that loads
      `/.well-known/baffinbay/botprotection-resources/pow-challenge.min.js`

The challenge is solved by the site's own JavaScript, which then holds an
httpOnly `bbnvalidation` cookie. Rather than reimplement their proof-of-work
(and re-break every time they tune it), we let a real browser solve it once
per run and replay the resulting cookie on the bulk httpx fetches.

The cookie is bound to the user agent that minted it, so the cache stores
both and callers must send the cached UA.

Usage:
    from scrapers import st_clearance
    c = st_clearance.get()
    httpx.Client(cookies=c.cookies, headers={"User-Agent": c.user_agent, ...})

CLI:
    python -m scrapers.st_clearance          # mint (or reuse) and report
    python -m scrapers.st_clearance --force  # force a fresh mint
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import NamedTuple

from core import config

log = logging.getLogger(__name__)

# Cookie the bot check sets once its challenge is satisfied. Everything else in
# the jar (consent, correlation id) is decoration we keep for realism.
_CLEARANCE_COOKIE = "bbnvalidation"

# Landing page used to trigger the challenge. Any /sportinfo page works; the
# horse-search page is the cheapest one that isn't tied to a specific horse.
_WARMUP_URL = config.ST_WEBAPI_HOST + "/sportinfo/horse"


class Clearance(NamedTuple):
    cookies: dict[str, str]
    user_agent: str
    minted_at: float

    @property
    def age_s(self) -> float:
        return time.time() - self.minted_at

    @property
    def is_fresh(self) -> bool:
        return (
            bool(self.cookies.get(_CLEARANCE_COOKIE))
            and self.age_s < config.ST_CLEARANCE_TTL_S
        )


class ClearanceError(RuntimeError):
    """Raised when no clearance cookie could be obtained."""


# Serializes minting so a pool of fetchers that all hit an expired cookie
# solves the challenge once. A cookie this young counts as "someone just
# minted it" and is reused even by a forced refresh.
_MINT_LOCK = threading.Lock()
_RECENT_MINT_S = 60


def _load_cached() -> Clearance | None:
    path = config.ST_CLEARANCE_CACHE
    try:
        blob = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    cookies = blob.get("cookies")
    ua = blob.get("user_agent")
    if not isinstance(cookies, dict) or not ua:
        return None
    return Clearance(cookies, ua, float(blob.get("minted_at") or 0))


def _store(c: Clearance) -> None:
    path = config.ST_CLEARANCE_CACHE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "cookies": c.cookies,
            "user_agent": c.user_agent,
            "minted_at": c.minted_at,
        }, indent=2) + "\n")
    except OSError as exc:
        log.warning("st clearance: could not cache cookie: %r", exc)


def _mint() -> Clearance:
    """Solve the challenge in a real browser and harvest the cookie jar."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ClearanceError(
            "playwright is required to solve the TravSport bot challenge: "
            "python3 -m pip install --user playwright"
        ) from exc

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel=config.ST_CLEARANCE_BROWSER_CHANNEL, headless=True)
        try:
            ctx = browser.new_context(
                user_agent=config.DEFAULT_USER_AGENT, locale="sv-SE")
            page = ctx.new_page()
            page.goto(_WARMUP_URL, wait_until="domcontentloaded", timeout=60_000)
            # The interstitial solves its proof-of-work and reloads; poll for
            # the cookie rather than sleeping for a fixed worst case.
            jar: dict[str, str] = {}
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                jar = {c["name"]: c["value"] for c in ctx.cookies()}
                if jar.get(_CLEARANCE_COOKIE):
                    break
                page.wait_for_timeout(500)
            user_agent = page.evaluate("navigator.userAgent")
        finally:
            browser.close()

    if not jar.get(_CLEARANCE_COOKIE):
        raise ClearanceError(
            f"browser did not receive a {_CLEARANCE_COOKIE} cookie from "
            f"{_WARMUP_URL} — the bot check may have changed"
        )
    c = Clearance(jar, user_agent, time.time())
    _store(c)
    log.info("st clearance: minted %s cookie", _CLEARANCE_COOKIE)
    return c


def get(*, force_refresh: bool = False) -> Clearance:
    """Return a usable clearance, reusing the cached cookie when still fresh."""
    if not force_refresh:
        cached = _load_cached()
        if cached is not None and cached.is_fresh:
            return cached
    with _MINT_LOCK:
        # Concurrent fetchers hit the same expired cookie at once. Whoever got
        # the lock first has already minted a replacement, so re-read before
        # launching another browser.
        cached = _load_cached()
        if cached is not None:
            if force_refresh and cached.age_s < _RECENT_MINT_S:
                return cached
            if not force_refresh and cached.is_fresh:
                return cached
        return _mint()


def main() -> None:  # pragma: no cover - CLI
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    args = ap.parse_args()

    c = get(force_refresh=args.force)
    print(f"cookies:    {sorted(c.cookies)}")
    print(f"user_agent: {c.user_agent}")
    print(f"age:        {c.age_s:.0f}s (ttl {config.ST_CLEARANCE_TTL_S}s)")
    print(f"cache:      {config.ST_CLEARANCE_CACHE}")


if __name__ == "__main__":  # pragma: no cover
    main()
