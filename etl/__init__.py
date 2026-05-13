"""ETL modules for stable-v2.

One `import_<source>.py` per source — TravSport, ATG, USTA, Le Trot, ...
Each one knows how to parse its source's raw payload and UPSERT into the
master tables (horse, person, race, entry, track) using helpers from
`etl.matching`.

Backfill mode: `import_<source>.backfill_from_v1(v1_conn, v2_conn)` reads
v1's already-parsed rows directly via a second psycopg2 connection.

Live mode: a scraper calls `import_<source>.upsert_horse(v2_conn, ...)`
in-memory after parsing.
"""
