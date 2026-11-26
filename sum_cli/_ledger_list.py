"""Row selection for `sum ledger list`, and timestamp order for `sum ledger stats`.

Kept out of main.py, whose size is frozen (Tests/test_main_py_size_frozen.py).

--since is compared as an instant, not as text, so 2026-11-25T18:19:55Z and
2026-11-25T13:19:55-05:00 select the same records as
2026-11-25T18:19:55+00:00. --limit counts rows that pass every filter, so it
is applied after --since, not as an SQL LIMIT before it.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

# One grammar on every Python version (datetime.fromisoformat reads more forms
# from 3.11 on): a date, optionally followed by T, t or a space and HH:MM, an
# optional :SS and a fraction of 1 to 9 digits, then Z, an offset +-HH:MM, or
# nothing, which means UTC. Digits are ASCII 0 to 9: \d would also take every
# Unicode decimal digit, a set that grows with the Python version.
_INSTANT = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})"
    r"(?:[Tt ]([0-9]{2}):([0-9]{2})(?::([0-9]{2})(?:\.([0-9]{1,9}))?)?(Z|z|[+-][0-9]{2}:[0-9]{2})?)?"
)


def parse_instant(value: str) -> datetime:
    """An ISO 8601 date or date-time as an aware datetime; ValueError if not.

    The fields are read by _INSTANT and range-checked by datetime itself, so
    the accepted set does not depend on the Python version.
    """
    m = _INSTANT.fullmatch(value.strip())
    if m is None:
        raise ValueError(f"not an ISO 8601 date or date-time: {value!r}")
    year, month, day, hour, minute, second, fraction, offset = m.groups()
    tz = timezone.utc
    if offset not in (None, "Z", "z"):
        hours, minutes = int(offset[1:3]), int(offset[4:6])
        if hours > 23 or minutes > 59:
            raise ValueError(f"offset out of range: {value!r}")
        sign = 1 if offset[0] == "+" else -1
        tz = timezone(sign * timedelta(hours=hours, minutes=minutes))
    return datetime(
        int(year), int(month), int(day), int(hour or 0), int(minute or 0),
        int(second or 0), int((fraction or "").ljust(6, "0")[:6]), tzinfo=tz,
    )


def _at_or_after(stamp: Any, since: datetime) -> bool:
    # A record whose timestamp cannot be read cannot be shown to be at or
    # after --since, so it is left out.
    if not isinstance(stamp, str):
        return False
    try:
        return parse_instant(stamp) >= since
    except ValueError:
        return False


def earliest_and_latest(stamps: list) -> tuple:
    """The earliest and latest of the readable timestamps, as stored.

    `sum ledger stats` orders them as instants, as --since does; a timestamp
    that parse_instant cannot read is left out. (None, None) if none is read.
    """
    read = []
    for stamp in stamps:
        if isinstance(stamp, str):
            try:
                read.append((parse_instant(stamp), stamp))
            except ValueError:
                pass
    if not read:
        return None, None
    return min(read, key=lambda r: r[0])[1], max(read, key=lambda r: r[0])[1]


def emit_rows(args: argparse.Namespace) -> int:
    """Print the matching rows as NDJSON; 0, or 2 for an unreadable --since."""
    since: Optional[datetime] = None
    if args.since:
        try:
            since = parse_instant(args.since)
        except ValueError:
            print(
                f"sum: --since {args.since!r} is not an ISO 8601 date or date-time",
                file=sys.stderr,
            )
            return 2
    where, params = ("", []) if not args.axiom else (" WHERE ap.axiom_key = ?", [args.axiom])
    limit = args.limit if args.limit and args.limit > 0 else None
    emitted = 0
    with sqlite3.connect(args.db) as conn:
        rows = conn.execute(
            "SELECT ap.prov_id, ap.axiom_key, pr.record_json "
            "FROM axiom_provenance AS ap "
            "JOIN provenance_records AS pr ON pr.prov_id = ap.prov_id"
            f"{where} ORDER BY ap.prov_id",
            params,
        )
        for prov_id, axiom_key, record_json in rows:
            if limit is not None and emitted >= limit:
                break
            rec = json.loads(record_json)
            if since is not None and not _at_or_after(rec.get("timestamp"), since):
                continue
            out = {
                "prov_id": prov_id,
                "axiom_key": axiom_key,
                "source_uri": rec.get("source_uri"),
                "byte_start": rec.get("byte_start"),
                "byte_end": rec.get("byte_end"),
                "timestamp": rec.get("timestamp"),
                "extractor_id": rec.get("extractor_id"),
            }
            json.dump(out, sys.stdout)
            sys.stdout.write("\n")
            emitted += 1
    if args.verbose:
        print(f"sum: emitted {emitted} record(s) from {args.db}", file=sys.stderr)
    return 0
