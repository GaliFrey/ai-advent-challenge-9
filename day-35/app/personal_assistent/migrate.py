"""Offline migration with the same exclusive lock as the bot and a SQLite backup.

    python -m personal_assistent.migrate --database PATH --backup NEW_PATH
"""
import argparse
import json
import os
import fcntl
import sqlite3
from pathlib import Path
from .storage import Store
from .event_schema import VERSION


def requires_migration(database):
    path=Path(database).resolve()
    if not path.exists() or path.stat().st_size==0:
        return False
    connection=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    try:
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='meta'").fetchone():
            return True
        row=connection.execute("SELECT value FROM meta WHERE key='events_schema_version'").fetchone()
        return row is None or json.loads(row[0])!=VERSION
    finally:
        connection.close()


def migrate_database(database, backup):
    database,backup=Path(database).resolve(),Path(backup).resolve()
    if not database.is_file() or database.stat().st_size==0:
        raise ValueError('Нужна существующая непустая база; личную базу инструмент не ищет.')
    if backup==database:
        raise ValueError('Резервная копия должна быть отдельным новым файлом.')
    with open(str(database)+'.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Exclusive creation prevents accidental replacement of an earlier snapshot.
        descriptor=os.open(backup,os.O_CREAT | os.O_EXCL | os.O_WRONLY,0o600)
        os.close(descriptor)
        try:
            source=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)
            try:
                target=sqlite3.connect(backup)
                try:
                    source.backup(target)
                    if target.execute('PRAGMA integrity_check').fetchone()[0]!='ok':
                        raise ValueError('Резервная копия не прошла integrity_check')
                finally:
                    target.close()
            finally:
                source.close()
        except BaseException:
            backup.unlink(missing_ok=True)
            raise
        store=Store(database)
        try:
            if store.db.execute('PRAGMA integrity_check').fetchone()[0]!='ok' or store.db.execute('PRAGMA foreign_key_check').fetchone():
                raise ValueError('Миграция не прошла проверку целостности; восстановите резервную копию при остановленном боте.')
        finally:
            store.db.close()
    return backup


def main():
    parser=argparse.ArgumentParser(description='Миграция остановленной БД с обязательной резервной копией')
    parser.add_argument('--database',required=True,type=Path)
    parser.add_argument('--backup',required=True,type=Path)
    args=parser.parse_args()
    migrate_database(args.database,args.backup)
    print('Миграция и проверки целостности завершены; резервная копия сохранена.')


if __name__=='__main__':
    main()
