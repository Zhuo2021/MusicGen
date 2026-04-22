import sqlite3
import os
import sys
from pathlib import Path

DB_PATH = Path('chroma_db') / 'chroma.sqlite3'

if not DB_PATH.exists():
    print(f"ERROR: {DB_PATH} not found")
    sys.exit(1)

print(f"Chroma DB file: {DB_PATH}  (size={DB_PATH.stat().st_size} bytes)\n")

conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()

print("SQLite master entries:")
cur.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','index','view') ORDER BY type,name")
items = cur.fetchall()
for name, typ in items:
    print(f"- {name} ({typ})")

# For each table, show columns and up to 5 rows
for name, typ in items:
    if typ != 'table':
        continue
    print('\n' + '='*60)
    print(f"Table: {name}")
    try:
        cur.execute(f"PRAGMA table_info('{name}')")
        cols = cur.fetchall()
        col_info = [(c[1], c[2]) for c in cols]
        print("Columns:")
        for ci in col_info:
            print(f"  - {ci[0]} : {ci[1]}")
    except Exception as e:
        print("  (couldn't get columns)", e)

    try:
        cur.execute(f"SELECT rowid, * FROM '{name}' LIMIT 5")
        rows = cur.fetchall()
        print("Sample rows:")
        if rows:
            for r in rows:
                print("  ", r)
        else:
            print("  (no rows)")
    except Exception as e:
        print("  (couldn't read rows)", e)

conn.close()

print('\n' + '='*60)
print('Listing chroma_db directory contents:')
root = Path('chroma_db')
for entry in sorted(root.iterdir(), key=lambda p: p.name):
    if entry.is_dir():
        print(f"- DIR: {entry.name}")
        try:
            for f in sorted(entry.iterdir(), key=lambda p: p.name)[:200]:
                st = f.stat()
                print(f"    - {f.name}  (size={st.st_size})")
        except Exception as e:
            print(f"    (couldn't list dir) {e}")
    else:
        st = entry.stat()
        print(f"- FILE: {entry.name}  (size={st.st_size})")

print('\nDone.')
