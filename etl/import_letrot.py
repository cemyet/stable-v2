"""Import LeTROT (French trotting) data into stable_v2.

Given a Le Trot course-page (one HTTP call), we get a full race + every
horse + every person + every result. This module turns that payload into
upserts against the v2 master tables.

Strategy:
    - One race per (date, reunion_id, course_number); letrot_race_id =
      "<date>_<reunion>_<course>".
    - One canonical horse row per `letrot_id`. Cross-source matching by
      letrot_id only (no name-based auto-merge).
    - Driver + trainer upserted as person rows with letrot_id.
    - Track upserted by (name, country='FR'). Le Trot doesn't expose stable
      track ids — `reunion_id` is per-day, not per-track.
    - Prize money parsed in EUR and converted to SEK via core.exchange.

Public entry-points:

    import_course(conn, race_date, reunion_id, course_number)
        Scrape + upsert one race.

    import_day(conn, race_date)
        Walk every course advertised for `race_date` and import each.
"""

from __future__ import annotations

import logging
import time
from datetime import date as Date, datetime, timedelta
from datetime import timedelta as timedelta_dt
from typing import Iterable

import httpx
from psycopg2.extras import Json

from core.common import classify_letrot_placement
from core.db import buffer_prune
from core.exchange import to_sek, prefetch_range as fx_prefetch_range
from etl.matching import upsert_horse, upsert_person, upsert_race, upsert_entry, upsert_track
from core.identity import resolve_horse, resolve_person
from scrapers.letrot import (
    make_client,
    fetch_course,
    fetch_horse_identity,
    list_today,
    list_date,
    set_response_observer,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format normalisers — converge LeTrot output onto ATG/ST conventions
# ---------------------------------------------------------------------------

def km_time_to_swedish(secs: float | None) -> str | None:
    """Render a km-time (in seconds) using ATG/ST Swedish convention.

    ATG displays km-time as `SS,T` and drops the leading `1:` minute for
    the typical trot range (60-119.9 s/km):

        70.9 → '10,9'      75.3 → '15,3'
        68.0 → '08,0'      80.0 → '20,0'

    Sub-minute times (rare) use `SS,T` literally; >= 120 s/km falls back
    to `M'SS"T`. Returns None for null/invalid.
    """
    if secs is None or secs <= 0:
        return None
    tenths_total = int(round(secs * 10))
    if tenths_total < 600:
        s = tenths_total // 10
        t = tenths_total % 10
        return f"{s:02d},{t}"
    if tenths_total >= 1200:
        m = tenths_total // 600
        rem_t = tenths_total - m * 600
        s = rem_t // 10
        t = rem_t % 10
        return f"{m}'{s:02d}\"{t}"
    rem_t = tenths_total - 600
    s = rem_t // 10
    t = rem_t % 10
    return f"{s:02d},{t}"


def placement_to_swedish(text: str | None) -> str | None:
    """Strip the French 'e' ordinal suffix so leaderboards/win-rate
    queries (which key off `placement_text IN ('1','2','3')`) actually
    count LeTrot results.

        '1'  → '1'        '2e' → '2'        '15e' → '15'
        'DA' → 'DA'  (left alone — gait DQ; handled by qualifier regex)
        '0'  → '0'   (scratched)
        None → None
    """
    if text is None:
        return None
    s = text.strip()
    if len(s) >= 2 and s[-1] == 'e' and s[:-1].isdigit():
        return s[:-1]
    return s


# LeTrot returns sex as French abbreviations (H=Hongre/gelding,
# F=Femelle/mare, M=Mâle/stallion). The single canonical convention used
# everywhere in stable-v2 (horse.gender_code AND entry.sex, plus
# `web.templates._layout.swedishSex`) is the Swedish H/V/S:
#
#   H → hingst  (stallion)  ← LeTrot's M (mâle)
#   V → valack  (gelding)   ← LeTrot's H (hongre)
#   S → sto     (mare)      ← LeTrot's F (femelle)
#
# Note the letter collisions across conventions: French H (hongre) is a
# gelding, not the canonical H (hingst/stallion); French M (mâle) is a
# stallion, not a mare. We MUST translate at ingest so downstream code
# only ever sees the canonical H/V/S meaning.
_FRENCH_SEX_MAP = {"H": "V", "F": "S", "M": "H"}


def translate_french_sex(text: str | None) -> str | None:
    """French H/F/M → canonical H/V/S. Pass-through for unknown values."""
    if not text:
        return text
    t = text.strip().upper()
    return _FRENCH_SEX_MAP.get(t, text)


# ---------------------------------------------------------------------------
# Track upsert (FR — match by name+country)
# ---------------------------------------------------------------------------

def _upsert_track_fr(cur, track_name: str | None) -> int | None:
    """Upsert a French trotting track via the centralized helper.

    Le Trot doesn't expose a stable per-track id, so we synthesize one from
    the lowercased name. Cross-source matching takes care of merging this
    with any pre-existing ATG/ST row for the same place.
    """
    if not track_name:
        return None
    return upsert_track(
        cur, "letrot", f"name:{track_name.strip().lower()}",
        {"name": track_name, "country": "FR", "sport": "trot"},
        raw_payload={"discovered_via": "course_page"},
    )


# ---------------------------------------------------------------------------
# Horse upsert (light — uses runner row only; does NOT fetch identity page)
# ---------------------------------------------------------------------------

def _upsert_horse_from_runner(
    cur, runner: dict,
    *,
    race_id: int | None = None,
    race_date: Date | None = None,
) -> int | None:
    """Upsert a LeTrot horse via the strict identity protocol.

    Race-context cross-link: when an existing entry on `race_id` with the
    same `program_number` already references a horse (from ST / ATG),
    `core.identity.resolve_horse` attaches `letrot_id` to that row
    instead of inserting a new orphan. This fixes the Indy Rock class
    going forward — LeTrot will piggyback on whatever ST/ATG row exists.
    """
    if not runner.get("horse_letrot_id"):
        return None
    canonical: dict = {
        "name":        runner.get("horse_name"),
        "birth_country": "FR",
        "gender_code": translate_french_sex(runner.get("sex")),
    }
    age = runner.get("age")
    if age and race_date:
        try:
            canonical["date_of_birth"] = Date(race_date.year - int(age), 1, 1)
        except (TypeError, ValueError):
            pass
    raw = {
        "musique":           runner.get("musique"),
        "record_text":       runner.get("record_text"),
        "gains_lifetime_eur": runner.get("gains_lifetime_eur"),
        "discovered_via":    "course_page",
        "fer":               runner.get("fer"),
    }
    return resolve_horse(
        cur,
        source="letrot",
        source_id=runner["horse_letrot_id"],
        canonical_fields=canonical,
        raw_payload=raw,
        race_id=race_id,
        program_number=runner.get("program_number"),
        sire_name=runner.get("sire_name"),
        dam_name=runner.get("dam_name"),
    )


def _upsert_person(cur, person: dict | None, role: str) -> int | None:
    if not person or not person.get("name"):
        return None
    flag = {
        "driver":  {"is_driver":  True},
        "jockey":  {"is_driver":  True},
        "entraineur": {"is_trainer": True},
        "trainer": {"is_trainer": True},
        "proprietaire": {"is_owner": True},
        "eleveur": {"is_breeder": True},
    }.get(role, {"is_driver": True})

    return resolve_person(
        cur,
        source="letrot",
        source_id=person.get("letrot_id"),
        canonical_fields={"name": person.get("name"), "license_country": "FR"},
        raw_payload={"role": role, "letrot_slug": person.get("slug")},
        role_flags=flag,
        country="FR",
    )


# ---------------------------------------------------------------------------
# Race + entries
# ---------------------------------------------------------------------------

def _upsert_race(cur, parsed: dict) -> int | None:
    if not parsed.get("letrot_race_id") or not parsed.get("track_name"):
        return None

    track_id = _upsert_track_fr(cur, parsed["track_name"])
    if not track_id:
        return None

    # Distance — most runners share the same distance; pick the modal.
    distances = [r.get("distance") for r in parsed.get("runners") or [] if r.get("distance")]
    distance = max(set(distances), key=distances.count) if distances else None

    canonical = {
        "race_date":   Date.fromisoformat(parsed["race_date"]),
        "track_id":    track_id,
        "race_number": parsed.get("race_number"),
        "distance":    distance,
        "heading":     parsed.get("race_name"),
        "status":      "results",
    }
    payload = {
        "reunion_id": parsed.get("reunion_id"),
        "race_name":  parsed.get("race_name"),
        "raw_h1":     parsed.get("raw_h1"),
        "raw_title":  parsed.get("raw_title"),
    }
    return upsert_race(
        cur, "letrot", parsed["letrot_race_id"], canonical,
        raw_payload=payload,
    )


def import_course(
    conn,
    race_date: Date | str,
    reunion_id: str | int,
    course_number: int | str,
) -> dict:
    if isinstance(race_date, Date):
        race_date_str = race_date.isoformat()
    else:
        race_date_str = race_date
        race_date = Date.fromisoformat(race_date_str)

    summary = {
        "letrot_race_id":  f"{race_date_str}_{reunion_id}_{course_number}",
        "race_id":         None,
        "horses_upserted": 0,
        "entries_upserted": 0,
        "skipped":         0,
    }

    # HTTP phase: fetch + parse, then close the client BEFORE any DB work.
    # httpx 0.24 / httpcore 0.17 on macOS + Python 3.9 can leave the
    # process stuck in select() on a stale pool socket; closing the client
    # (and all its FDs) before touching psycopg2 eliminates the interaction.
    with make_client() as client:
        parsed = fetch_course(client, race_date_str, reunion_id, course_number)
    if not parsed or not parsed.get("runners"):
        summary["skipped"] = 1
        return summary

    # DB phase: all HTTP sockets are closed at this point.
    try:
        with conn.cursor() as cur:
            race_id = _upsert_race(cur, parsed)
            if not race_id:
                summary["skipped"] = 1
                conn.rollback()
                return summary
            summary["race_id"] = race_id

            for runner in parsed["runners"]:
                horse_id = _upsert_horse_from_runner(
                    cur, runner,
                    race_id=race_id, race_date=race_date,
                )
                if not horse_id:
                    summary["skipped"] += 1
                    continue
                summary["horses_upserted"] += 1

                driver_id  = _upsert_person(cur, runner.get("driver"),  "driver")
                trainer_id = _upsert_person(cur, runner.get("trainer"), "trainer")

                prize_kr, prize_orig, fx_rate, fx_date = to_sek(
                    runner.get("prize_eur"), "EUR", race_date
                )

                km_secs = runner.get("km_time_seconds")

                # Disambiguate LeTrot's overloaded "rang = 0":
                #   - 'DI' / 'DA' in km_time_text  → ran but was DQ'd
                #     ("Distancé" / "Distancé arrivée", i.e. broke gait or
                #      post-race irregular trotting). Set disqualified=True.
                #   - All other zero-rang cases stay withdrawn=False; ATG
                #     remains the authoritative source for "actually
                #     scratched at the gate" via its race-level
                #     `scratchings` array (see etl/import_atg.py). Setting
                #     withdrawn=True here is risky because the OR-merge
                #     in core.identity._ENTRY_OR_COLS would let LeTrot's
                #     ambiguous signal overwrite ATG's correct False.
                #   - For LeTrot-only French races (no ATG counterpart),
                #     `truly_silent` (no time, no placement, no prize,
                #     no musique row) is the only signal that the horse
                #     never actually appeared.
                km_text = runner.get("km_time_text")
                placement_text_raw = runner.get("placement_text")
                # Canonical result flags from the LeTrot rang code. The
                # distancé family (DA / D / D0..D9 / DM) is France's gait-break
                # DQ — there is no "galloped but placed" outcome — so this is
                # also where `galopp` gets set for French entries.
                code_flags = classify_letrot_placement(placement_text_raw)
                is_post_race_dq = (
                    code_flags.get("disqualified", False) or km_text in ("DI", "DA")
                )
                is_galopp = code_flags.get("galopp", False)
                truly_silent = (
                    runner.get("placement") is None
                    and placement_text_raw == "0"
                    and not km_text
                    and runner.get("prize_eur") is None
                    and not runner.get("musique")
                )
                # NP (non partant) is an explicit "did not start" signal.
                is_withdrawn = truly_silent or code_flags.get("withdrawn", False)

                entry_canonical = {
                    "program_number": runner.get("program_number"),
                    "post":           runner.get("program_number"),  # FR programme = post
                    "distance":       runner.get("distance"),
                    "placement":      runner.get("placement"),
                    # Normalised onto ATG/ST conventions: '2e' → '2' so the
                    # win/place SQL (placement_text IN ('1','2','3')) sees
                    # LeTrot finishes, and time_text is Swedish km-time.
                    "placement_text": placement_to_swedish(placement_text_raw),
                    "time_text":      km_time_to_swedish(km_secs),
                    "time_seconds":   km_secs,
                    "prize_kr":       prize_kr or 0,
                    "prize_currency": "EUR",
                    "prize_original": prize_orig,
                    "prize_fx_rate":  fx_rate,
                    "prize_fx_date":  fx_date,
                    "driver_id":      driver_id,
                    "trainer_id":     trainer_id,
                    "age":            runner.get("age"),
                    "sex":            translate_french_sex(runner.get("sex")),
                    "shoe_code":      runner.get("fer"),
                    "withdrawn":      is_withdrawn,
                    "disqualified":   is_post_race_dq,
                    "galopp":         is_galopp,
                }
                entry_payload = {
                    "musique":      runner.get("musique"),
                    "record_text":  runner.get("record_text"),
                    "gains_lifetime_eur": runner.get("gains_lifetime_eur"),
                    "avis_trainer": runner.get("avis_trainer"),
                    "odds_text":    runner.get("odds_text"),
                    "prize_eur":    runner.get("prize_eur"),
                    "km_time_text":      runner.get("km_time_text"),       # raw FR km-time '1\'10"9'
                    "total_time_text":   runner.get("time_text"),          # raw FR total '2\'53"6' (was overwriting canonical time_text)
                    "placement_text_raw": runner.get("placement_text"),    # raw FR '2e'
                }
                upsert_entry(cur, "letrot", race_id, horse_id,
                             entry_canonical, raw_payload=entry_payload)
                summary["entries_upserted"] += 1

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return summary


# ---------------------------------------------------------------------------
# Day import
# ---------------------------------------------------------------------------

_WHEN_BY_OFFSET = {-1: "hier", 0: "aujourd-hui", 1: "demain"}


def import_day(
    conn,
    when: str | Date = "hier",
) -> dict:
    """Walk every course Le Trot advertises for `when` and import each.

    `when` accepts 'hier'/'aujourd-hui'/'demain' (relative) or an ISO date
    string (we'll filter the listing to that date).
    """
    summary = {
        "scraped":         0,
        "imported":        0,
        "skipped":         0,
        "horses_upserted": 0,
        "entries_upserted": 0,
    }

    with make_client() as list_client:
        if isinstance(when, Date):
            target_date = when.isoformat()
            rows = list_today(list_client, "hier")
            rows = [r for r in rows if r["race_date"] == target_date]
        elif when in ("hier", "aujourd-hui", "demain"):
            rows = list_today(list_client, when)
        else:
            target_date = when
            rows = list_today(list_client, "hier")
            rows = [r for r in rows if r["race_date"] == target_date]

    # Pre-warm EUR FX cache for all race dates in one Riksbank API call
    # so that per-runner to_sek() never hits the 15 s rate limiter.
    if rows:
        race_dates = {Date.fromisoformat(r["race_date"]) for r in rows}
        d_min = min(race_dates) - timedelta(days=10)
        d_max = max(race_dates)
        try:
            fx_prefetch_range("EUR", d_min, d_max)
        except Exception as exc:
            log.warning("fx prefetch EUR %s..%s failed: %r", d_min, d_max, exc)

    for row in rows:
        summary["scraped"] += 1
        res = import_course(
            conn, row["race_date"], row["reunion_id"], row["course_number"],
        )
        if res.get("race_id"):
            summary["imported"] += 1
            summary["horses_upserted"]  += res["horses_upserted"]
            summary["entries_upserted"] += res["entries_upserted"]
        else:
            summary["skipped"] += 1
        time.sleep(0.1)

    buffer_prune(conn, "letrot")
    return summary


# ---------------------------------------------------------------------------
# Historic backfill — watchdog + safe rate-limit detection
# ---------------------------------------------------------------------------


class BackfillAborted(RuntimeError):
    """Raised by the watchdog to stop a backfill mid-flight."""


class BackfillWatchdog:
    """Observes every HTTP response and aborts the backfill if Le Trot
    starts rate-limiting, returning errors, or hanging.

    Triggers (any one fires `BackfillAborted`):
      * **HTTP 429** seen (explicit rate-limit).
      * **5xx error rate** > `error_rate_threshold` over the rolling
        window of the last `window_size` responses (default 5% of 200).
      * **Throughput collapse**: median response time in the window
        exceeds `slow_response_seconds` (default 15 s — vs the normal
        ~0.2 s) — typical signature of a soft throttle.
      * **No responses observed** for `stall_timeout_seconds` (default
        300 s, 5 min) — process is hung, network down, or LeTrot dead.

    All counters are exposed via `snapshot()` for heartbeat logging.
    """

    def __init__(
        self,
        *,
        window_size: int = 200,
        error_rate_threshold: float = 0.05,
        slow_response_seconds: float = 15.0,
        stall_timeout_seconds: float = 300.0,
        log_fn=None,
    ):
        from collections import deque
        self._deque = deque(maxlen=window_size)
        self.window_size = window_size
        self.error_rate_threshold = error_rate_threshold
        self.slow_response_seconds = slow_response_seconds
        self.stall_timeout_seconds = stall_timeout_seconds
        self.log = log_fn or log.warning
        # cumulative counters
        self.total_ok = 0
        self.total_errors = 0
        self.total_429 = 0
        self.total_5xx = 0
        self.total_4xx = 0
        self.total_other = 0
        self.last_response_at = time.monotonic()
        self.start_at = time.monotonic()
        self._aborted_reason: str | None = None

    def observe(self, status: int, elapsed: float, url: str) -> None:
        now = time.monotonic()
        self.last_response_at = now
        self._deque.append((status, elapsed, now))

        if status == 200:
            self.total_ok += 1
        elif status == 0:
            self.total_errors += 1
        elif status == 429:
            self.total_429 += 1
        elif 500 <= status < 600:
            self.total_5xx += 1
        elif 400 <= status < 500:
            self.total_4xx += 1
        else:
            self.total_other += 1

        # Trigger 1: explicit HTTP 429
        if status == 429:
            self._abort(f"HTTP 429 from {url} — Le Trot rate-limiting us")

        # Trigger 2: 5xx error rate in window
        if len(self._deque) >= self.window_size:
            err5 = sum(1 for s, _, _ in self._deque if 500 <= s < 600 or s == 0)
            rate = err5 / len(self._deque)
            if rate > self.error_rate_threshold:
                self._abort(
                    f"5xx/error rate {rate:.1%} > {self.error_rate_threshold:.0%} "
                    f"in last {len(self._deque)} responses ({err5} errors)"
                )

        # Trigger 3: median latency collapse
        if len(self._deque) >= 30:
            elapses = sorted(e for _, e, _ in self._deque)
            median = elapses[len(elapses) // 2]
            if median > self.slow_response_seconds:
                self._abort(
                    f"median response time {median:.1f}s > "
                    f"{self.slow_response_seconds:.1f}s (soft throttle?)"
                )

    def check_stall(self) -> None:
        """Call periodically to detect hangs (no responses for a while)."""
        idle = time.monotonic() - self.last_response_at
        if idle > self.stall_timeout_seconds:
            self._abort(
                f"no HTTP responses for {int(idle)}s "
                f"(threshold {int(self.stall_timeout_seconds)}s) — likely hung"
            )

    def _abort(self, reason: str) -> None:
        if self._aborted_reason is not None:
            return
        self._aborted_reason = reason
        self.log(f"[watchdog] ABORT: {reason}")
        raise BackfillAborted(reason)

    def snapshot(self) -> dict:
        elapsed = time.monotonic() - self.start_at
        total = self.total_ok + self.total_4xx + self.total_5xx + self.total_429 + self.total_other + self.total_errors
        median = None
        if self._deque:
            elapses = sorted(e for _, e, _ in self._deque)
            median = round(elapses[len(elapses) // 2], 2)
        return {
            "elapsed_s":       round(elapsed, 1),
            "total_requests":  total,
            "rps":             round(total / elapsed, 2) if elapsed > 0 else 0,
            "ok":              self.total_ok,
            "429":             self.total_429,
            "5xx":             self.total_5xx,
            "4xx":             self.total_4xx,
            "errors":          self.total_errors,
            "other":           self.total_other,
            "median_latency":  median,
            "window_size":     len(self._deque),
            "aborted":         self._aborted_reason,
        }


def import_date_range(
    conn,
    start_date: Date | str,
    end_date: Date | str,
    *,
    progress_every: int = 7,
    skip_if_present: bool = True,
    watchdog: BackfillWatchdog | None = None,
    heartbeat_path: str | None = None,
    reverse: bool = False,
) -> dict:
    """Backfill every course advertised by Le Trot between two ISO dates
    (inclusive on both ends).

    Iterates day-by-day, hitting `/courses/<YYYY-MM-DD>` for each. Each
    day's courses are imported via `import_course`.

    Args:
        skip_if_present:
            If True (default), skip an entire day if any race with that
            date is already in the DB with `letrot_race_id IS NOT NULL`.
            This makes the backfill resumable — re-running it picks up
            where the last run left off without re-scraping.

    Returns aggregate counters across the whole range.
    """
    if isinstance(start_date, str):
        start_date = Date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = Date.fromisoformat(end_date)
    if end_date < start_date:
        raise ValueError(f"end_date {end_date} < start_date {start_date}")

    # Always have a watchdog active in backfill mode — pick a default if the
    # caller didn't pass one. The observer hook attaches it to every HTTP
    # call made inside scrapers.letrot.
    own_watchdog = watchdog is None
    if own_watchdog:
        watchdog = BackfillWatchdog(log_fn=log.warning)
    set_response_observer(watchdog.observe)

    summary = {
        "start":            start_date.isoformat(),
        "end":              end_date.isoformat(),
        "days_scanned":     0,
        "days_skipped":     0,
        "scraped":          0,
        "imported":         0,
        "skipped":          0,
        "horses_upserted":  0,
        "entries_upserted": 0,
        "aborted":          None,
    }

    total_days = (end_date - start_date).days + 1

    def _fmt_duration(secs: float) -> str:
        secs = max(0, int(secs))
        if secs < 60:
            return f"{secs}s"
        if secs < 3600:
            return f"{secs // 60}m{secs % 60:02d}s"
        h, rem = divmod(secs, 3600)
        return f"{h}h{rem // 60:02d}m"

    def _heartbeat(last_date: str) -> None:
        snap = watchdog.snapshot()
        elapsed = snap["elapsed_s"]
        days_done = summary["days_scanned"] + summary["days_skipped"]
        days_remaining = max(0, total_days - days_done)
        # ETA assumes the future has the same per-day pace as the past so far.
        # Use scanned-day pace (not total) because skipped days are essentially
        # free — they finish in ms and shouldn't dilute the rate.
        scanned = max(1, summary["days_scanned"])
        sec_per_scanned_day = elapsed / scanned if elapsed > 0 else 0
        # Optimistic ETA assumes future days are the same mix of scanned/skip;
        # we don't have visibility into "future skips" so just project at the
        # scanned rate over all remaining days.
        eta_secs = sec_per_scanned_day * days_remaining if sec_per_scanned_day > 0 else 0
        eta_str = _fmt_duration(eta_secs) if eta_secs > 0 else "?"
        finish_at = (datetime.now() + timedelta_dt(seconds=eta_secs)).strftime("%a %H:%M") \
                    if eta_secs > 0 else "?"
        pct = (days_done / total_days * 100) if total_days else 0

        # Single-line for logs (greppable).
        line = (
            f"[heartbeat] @{last_date}  "
            f"days={days_done}/{total_days} ({pct:.1f}%)  "
            f"imported={summary['imported']}  "
            f"horses={summary['horses_upserted']}  entries={summary['entries_upserted']}  "
            f"| rps={snap['rps']}  median={snap['median_latency']}s  "
            f"ok={snap['ok']}  429={snap['429']}  5xx={snap['5xx']}  err={snap['errors']}  "
            f"elapsed={_fmt_duration(elapsed)}  eta={eta_str} (~{finish_at})"
        )
        log.info(line)

        # Multi-line snapshot, rewritten in-place — `watch -n 5 cat <path>`.
        if heartbeat_path:
            try:
                bar_w = 40
                filled = int(bar_w * pct / 100)
                bar = "█" * filled + "·" * (bar_w - filled)
                report = (
                    f"=== LeTrot backfill ===\n"
                    f"range:     {summary['start']} → {summary['end']}  ({total_days} days)\n"
                    f"progress:  [{bar}] {pct:5.1f}%\n"
                    f"           {days_done}/{total_days} days  "
                    f"({summary['days_scanned']} scanned, {summary['days_skipped']} skipped)\n"
                    f"imported:  {summary['imported']} courses, "
                    f"{summary['horses_upserted']} horses, "
                    f"{summary['entries_upserted']} entries\n"
                    f"elapsed:   {_fmt_duration(elapsed)}\n"
                    f"eta:       {eta_str}   (~{finish_at})\n"
                    f"throughput {snap['rps']} req/s   median latency {snap['median_latency']}s\n"
                    f"http:      ok={snap['ok']}  429={snap['429']}  "
                    f"5xx={snap['5xx']}  4xx={snap['4xx']}  err={snap['errors']}\n"
                    f"last day:  {last_date}\n"
                    f"watchdog:  {snap['aborted'] or 'healthy'}\n"
                    f"updated:   {datetime.now().isoformat(timespec='seconds')}\n"
                )
                with open(heartbeat_path, "w") as f:
                    f.write(report)
            except OSError:
                pass

    try:
        cur_date = end_date if reverse else start_date
        step = timedelta(days=-1) if reverse else timedelta(days=1)
        t0 = time.time()
        day_no = 0
        fx_warmed_year: int | None = None
        while (cur_date >= start_date) if reverse else (cur_date <= end_date):
            day_no += 1
            iso = cur_date.isoformat()

            # Pre-warm Riksbank FX cache for an entire year in one API call
            # the first time we touch that year. Without this, every runner
            # would call Riksbank individually and we'd blow past the 5/min
            # + 1000/day quota in minutes.
            if fx_warmed_year != cur_date.year:
                yr_start = Date(cur_date.year, 1, 1)
                # Cap the upper bound at the actual end of the backfill so
                # we don't fetch dates we'll never use.
                yr_end = min(Date(cur_date.year, 12, 31), end_date)
                try:
                    fx_prefetch_range("EUR", yr_start, yr_end)
                except Exception as exc:
                    log.warning("fx prefetch %s failed: %r", cur_date.year, exc)
                fx_warmed_year = cur_date.year

            already_have = False
            if skip_if_present:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM race WHERE letrot_race_id IS NOT NULL "
                        "AND race_date = %s LIMIT 1",
                        (cur_date,),
                    )
                    already_have = cur.fetchone() is not None
            if already_have:
                summary["days_skipped"] += 1
                cur_date += step
                continue

            summary["days_scanned"] += 1
            try:
                with make_client() as day_client:
                    rows = list_date(day_client, iso)
            except Exception as exc:
                log.warning("letrot list_date %s failed: %r", iso, exc)
                rows = []
            for row in rows:
                summary["scraped"] += 1
                try:
                    res = import_course(
                        conn, row["race_date"], row["reunion_id"],
                        row["course_number"],
                    )
                except Exception as exc:
                    conn.rollback()
                    log.warning("letrot import_course %s/%s/%s failed: %r",
                                row["race_date"], row["reunion_id"],
                                row["course_number"], exc)
                    summary["skipped"] += 1
                    continue
                if res.get("race_id"):
                    summary["imported"] += 1
                    summary["horses_upserted"]  += res["horses_upserted"]
                    summary["entries_upserted"] += res["entries_upserted"]
                else:
                    summary["skipped"] += 1
                time.sleep(0.05)

            if day_no % progress_every == 0:
                _heartbeat(iso)
                # Periodically also check whether the run has stalled
                # silently (no responses at all for a while — distinct
                # from the per-response watchdog triggers).
                watchdog.check_stall()
            cur_date += step

        buffer_prune(conn, "letrot")
        summary["seconds"] = round(time.time() - t0, 1)
        summary["watchdog"] = watchdog.snapshot()
    except BackfillAborted as exc:
        # Mark the summary, partially commit what we've got, propagate.
        summary["aborted"]  = str(exc)
        summary["seconds"]  = round(time.time() - t0, 1)
        summary["watchdog"] = watchdog.snapshot()
        try:
            conn.commit()
        except Exception:
            pass
        log.error("[backfill] aborted: %s", exc)
    finally:
        if own_watchdog:
            set_response_observer(None)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    from core.db import get_connection
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(
            "usage:\n"
            "  python -m etl.import_letrot course <YYYY-MM-DD> <reunion> <course>\n"
            "  python -m etl.import_letrot day    [hier|aujourd-hui|demain|YYYY-MM-DD]\n"
            "  python -m etl.import_letrot backfill <START_YYYY-MM-DD> <END_YYYY-MM-DD> [--no-skip]\n"
        )
        return

    conn = get_connection()
    try:
        cmd = sys.argv[1]
        if cmd == "course":
            print(import_course(conn, sys.argv[2], sys.argv[3], sys.argv[4]))
        elif cmd == "day":
            when = sys.argv[2] if len(sys.argv) >= 3 else "hier"
            print(import_day(conn, when))
        elif cmd == "backfill":
            if len(sys.argv) < 4:
                raise SystemExit("backfill needs <start> <end> "
                                 "[--no-skip] [--reverse] "
                                 "[--heartbeat /path]")
            extras = sys.argv[4:]
            skip = "--no-skip" not in extras
            reverse = "--reverse" in extras
            hb = None
            for i, a in enumerate(extras):
                if a == "--heartbeat" and i + 1 < len(extras):
                    hb = extras[i + 1]
            print(import_date_range(conn, sys.argv[2], sys.argv[3],
                                    skip_if_present=skip,
                                    heartbeat_path=hb,
                                    reverse=reverse))
        else:
            raise SystemExit(f"unknown command: {cmd!r}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
