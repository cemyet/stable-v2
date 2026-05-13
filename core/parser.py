"""
Parse React Query dehydrated state from Next.js SSR HTML pages.

The sportapp embeds all horse data (basic info, results, lineage, stats, etc.)
in self.__next_f.push() calls within <script> tags. The data is JSON-escaped
inside string literals. We extract and reassemble these fragments, then pull
out each React Query cache entry.
"""

from __future__ import annotations

import json
import re
import codecs
from typing import Optional

PUSH_PATTERN = re.compile(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)')

QUERY_KEYS_WE_WANT = [
    "horse-basic-information",
    "race-results",
    "lineage-small",
    "lineage-description",
    "horse-statistics",
    "breeding-evaluation-history",
    "breeding-evaluation-review",
    "breeding-evaluation-top-result",
    "horse-image",
    "horse-history",
    "horse-stack-race",
]


def _unescape_rsc_payload(raw: str) -> str:
    """Unescape the double-escaped JSON from Next.js RSC push payloads.

    The strings use JS-style escaping (\\", \\n, etc.) but may contain
    raw UTF-8 bytes. We use codecs with 'surrogateescape' to handle this
    gracefully.
    """
    try:
        return raw.encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        try:
            return raw.encode("utf-8").decode("unicode_escape")
        except Exception:
            return raw


def extract_queries_from_html(html: str) -> dict:
    """Extract all React Query cache entries from Next.js SSR HTML.

    Returns a dict mapping simplified query keys (e.g. "horse-basic-information")
    to their data payloads.
    """
    push_payloads = PUSH_PATTERN.findall(html)
    full_payload = ""
    for p in push_payloads:
        full_payload += _unescape_rsc_payload(p)

    results = {}

    for key_prefix in QUERY_KEYS_WE_WANT:
        # Match both "queryKey":["key-123"] and "queryKey":["key",{...}]
        pattern = re.compile(
            r'"queryKey":\["' + re.escape(key_prefix) + r'[^]]*\]'
        )
        match = pattern.search(full_payload)
        if not match:
            continue

        query_key_pos = match.start()
        search_region = full_payload[max(0, query_key_pos - 50000):query_key_pos]

        data_marker = '"data":{"data":'
        d_idx = search_region.rfind(data_marker)
        if d_idx < 0:
            continue

        content_start = d_idx + len(data_marker)
        rest = search_region[content_start:]

        ok_idx = rest.find(',"ok":true')
        if ok_idx < 0:
            ok_idx = rest.find(',"ok":false')
        if ok_idx < 0:
            continue

        raw_json = rest[:ok_idx]
        try:
            data = json.loads(raw_json)
            results[key_prefix] = data
        except json.JSONDecodeError:
            results[key_prefix] = raw_json

    return results


def parse_horse_page(html: str, horse_id: int) -> Optional[dict[str, object]]:
    """Parse a horse page and return extracted data, or None if parsing fails."""
    queries = extract_queries_from_html(html)
    if not queries:
        return None
    return queries
