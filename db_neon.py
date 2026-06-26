"""Neon PostgreSQL database layer for the Coeur app.

This module provides a thin connection helper plus CRUD functions that
provide a connection pool plus CRUD functions for the Neon PostgreSQL database.
Auth is handled locally via werkzeug password hashing + Flask sessions.

The connection pool is created lazily on first use and reused for the
lifetime of the process. Each call checks out a connection, runs its
transaction, and returns it, so the pool stays small and safe under the
threading async mode used by Flask-SocketIO.
"""

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from uuid import UUID

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

logger = logging.getLogger(__name__)

_pool = None


def _get_pool():
    """Lazily create (or return the existing) thread-local connection pool."""
    global _pool
    if _pool is not None:
        return _pool
    # Use NEON_DATABASE_URL (preferred) so the Coeur app is not affected by
    # the parent shell's DATABASE_URL, which in this sandbox is set to the
    # Next.js project's SQLite path (file:/home/z/my-project/db/custom.db).
    # Fall back to DATABASE_URL for standard environments.
    dsn = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not dsn or not dsn.startswith("postgres"):
        raise RuntimeError(
            "NEON_DATABASE_URL is not set or is not a PostgreSQL DSN. "
            "Set NEON_DATABASE_URL to the Neon connection string."
        )
    # Small pool: the app is single-process with a threading worker.
    _pool = ThreadedConnectionPool(
        minconn=1, maxconn=5, dsn=dsn, connect_timeout=15
    )
    logger.info("Neon connection pool created (min=1, max=5)")
    return _pool


def _checkout_healthy_conn():
    """Get a healthy connection from the pool, retrying once if stale.

    Neon (and any managed Postgres) may close idle connections server-side.
    We probe the connection with a trivial SELECT; if it's dead we discard
    it and try a fresh one (up to 2 attempts).
    """
    pool = _get_pool()
    last_exc = None
    for attempt in range(2):
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return conn  # healthy
        except psycopg2.OperationalError as e:
            # Stale/dead connection — discard and retry once.
            last_exc = e
            logger.warning("DB connection stale (attempt %d), retrying: %s",
                           attempt + 1, str(e)[:120])
            try:
                pool.putconn(conn, close=True)
            except Exception:
                pass
            conn = None
    raise last_exc or RuntimeError("Could not acquire a DB connection")


@contextmanager
def get_conn():
    """Context manager that yields a RealDictCursor and returns the conn.

    Acquires a healthy connection (probed with SELECT 1 to avoid the
    "connection already closed" error on stale pooled connections), runs
    the caller's block in a single transaction, commits on success, and
    always returns the connection to the pool.
    """
    conn = _checkout_healthy_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _get_pool().putconn(conn)


# ---------------------------------------------------------------------------
# JSON / value helpers
# ---------------------------------------------------------------------------

def _to_jsonable(value):
    """Make a Python value JSON-serializable for JSONB columns."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


def _parse_uuid(value):
    """Coerce a string/UUID to a string suitable for the UUID column."""
    if value is None:
        return None
    return str(value)


# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------

def upsert_user(user_id, email=None, name=None, is_volunteer=None, location=None):
    """Insert or update a user row. Only supplied fields are written.

    
    Returns the full row as a dict.
    """
    fields = []
    values = []
    if email is not None:
        fields.append("email"); values.append(email)
    if name is not None:
        fields.append("name"); values.append(name)
    if is_volunteer is not None:
        fields.append("is_volunteer"); values.append(is_volunteer)
    if location is not None:
        fields.append("location"); values.append(_to_jsonable(location))

    # Always update updated_at on write.
    fields.append("updated_at"); values.append(datetime.now().isoformat())

    # Build an UPSERT keyed on id. If no profile fields are supplied beyond
    # updated_at, we still ensure the row exists.
    set_clause = ", ".join(f"{f} = EXCLUDED.{f}" for f in fields if f != "id")
    insert_cols = ", ".join(["id"] + fields)
    insert_ph = ", ".join(["%s"] * (1 + len(values)))
    upsert_set = set_clause if set_clause else "updated_at = EXCLUDED.updated_at"

    sql = (
        f"INSERT INTO users ({insert_cols}) VALUES ({insert_ph}) "
        f"ON CONFLICT (id) DO UPDATE SET {upsert_set} "
        f"RETURNING *"
    )
    params = [_parse_uuid(user_id)] + values
    with get_conn() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    return dict(row) if row else None


def get_user(user_id):
    """Return a user row by id, or None."""
    with get_conn() as cur:
        cur.execute("SELECT * FROM users WHERE id = %s", (_parse_uuid(user_id),))
        row = cur.fetchone()
    return dict(row) if row else None


def get_user_by_email(email):
    """Return a user row by email, or None. Used by the local auth login."""
    with get_conn() as cur:
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        row = cur.fetchone()
    return dict(row) if row else None


def create_user(email, password_hash, name=None):
    """Create a new auth user with a hashed password.

    Generates a UUID id. Returns the new row (without password_hash) or
    raises psycopg2.IntegrityError if the email is already registered
    (caught by the caller to return a 409 to the client).
    """
    from uuid import uuid4
    uid = str(uuid4())
    with get_conn() as cur:
        cur.execute(
            "INSERT INTO users (id, email, name, password_hash) "
            "VALUES (%s, %s, %s, %s) RETURNING id, email, name, "
            "is_volunteer, location, created_at, updated_at",
            (uid, email, name, password_hash),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def authenticate_user(email, password):
    """Verify email + password against Neon. Returns the user row (without
    password_hash) on success, or None on bad credentials / missing user."""
    from werkzeug.security import check_password_hash
    user = get_user_by_email(email)
    if not user or not user.get('password_hash'):
        return None
    if not check_password_hash(user['password_hash'], password):
        return None
    user.pop('password_hash', None)
    return user


# ---------------------------------------------------------------------------
# emergencies
# ---------------------------------------------------------------------------

def insert_emergency(user_id, type="Emergency", description="", location=None, status="active"):
    """Insert an emergency and return the new row (including generated id).

    Ensures the user row exists first (the emergencies.user_id FK requires it).
    Auth is local, so the user row is created here.
    """
    uid = _parse_uuid(user_id)
    with get_conn() as cur:
        # Ensure the user row exists (no-op if already present).
        cur.execute(
            "INSERT INTO users (id) VALUES (%s) ON CONFLICT (id) DO NOTHING",
            (uid,),
        )
        sql = (
            "INSERT INTO emergencies (user_id, type, description, location, status) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING *"
        )
        params = (uid, type, description, _to_jsonable(location), status)
        cur.execute(sql, params)
        row = cur.fetchone()
    return dict(row) if row else None


def update_emergency(emergency_id, user_id, status="resolved"):
    """Update an emergency owned by user_id. Returns the updated row or None."""
    sql = (
        "UPDATE emergencies SET status = %s, updated_at = %s "
        "WHERE id = %s AND user_id = %s RETURNING *"
    )
    params = (status, datetime.now().isoformat(),
              _parse_uuid(emergency_id), _parse_uuid(user_id))
    with get_conn() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# emergency_contacts
# ---------------------------------------------------------------------------

def list_emergency_contacts(user_id):
    """Return all emergency contacts for a user as a list of dicts."""
    with get_conn() as cur:
        cur.execute(
            "SELECT * FROM emergency_contacts WHERE user_id = %s ORDER BY created_at",
            (_parse_uuid(user_id),),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def insert_emergency_contact(user_id, name, phone):
    """Insert a contact and return the new row (including generated id).

    Ensures the user row exists first (the emergency_contacts.user_id FK
    requires it). Auth is local, so the user row is created here.
    """
    uid = _parse_uuid(user_id)
    with get_conn() as cur:
        cur.execute(
            "INSERT INTO users (id) VALUES (%s) ON CONFLICT (id) DO NOTHING",
            (uid,),
        )
        sql = (
            "INSERT INTO emergency_contacts (user_id, name, phone) "
            "VALUES (%s, %s, %s) RETURNING *"
        )
        params = (uid, name, phone)
        cur.execute(sql, params)
        row = cur.fetchone()
    return dict(row) if row else None


def delete_emergency_contact(contact_id, user_id):
    """Delete a contact owned by user_id. Returns True if a row was deleted."""
    sql = (
        "DELETE FROM emergency_contacts WHERE id = %s AND user_id = %s"
    )
    params = (_parse_uuid(contact_id), _parse_uuid(user_id))
    with get_conn() as cur:
        cur.execute(sql, params)
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# medical_info (one row per user: medications, allergies)
# ---------------------------------------------------------------------------

def upsert_medical_info(user_id, medications=None, allergies=None):
    """Insert or update the user's medical info row. Returns the row."""
    uid = _parse_uuid(user_id)
    with get_conn() as cur:
        # Ensure the user row exists (FK requirement).
        cur.execute(
            "INSERT INTO users (id) VALUES (%s) ON CONFLICT (id) DO NOTHING",
            (uid,),
        )
        cur.execute(
            """
            INSERT INTO medical_info (user_id, medications, allergies)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE
              SET medications = EXCLUDED.medications,
                  allergies = EXCLUDED.allergies,
                  updated_at = NOW()
            RETURNING id, user_id, medications, allergies, updated_at
            """,
            (uid, medications, allergies),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def get_medical_info(user_id):
    """Return the user's medical info row, or None."""
    uid = _parse_uuid(user_id)
    with get_conn() as cur:
        cur.execute(
            "SELECT id, user_id, medications, allergies, updated_at "
            "FROM medical_info WHERE user_id = %s",
            (uid,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# medical_records (uploaded files: name, type, content)
# ---------------------------------------------------------------------------

def insert_medical_record(user_id, name, type_=None, content=None):
    """Insert a medical record. Returns the new row (without content to keep
    it light for list views — use get_medical_record for the full content)."""
    uid = _parse_uuid(user_id)
    with get_conn() as cur:
        cur.execute(
            "INSERT INTO users (id) VALUES (%s) ON CONFLICT (id) DO NOTHING",
            (uid,),
        )
        cur.execute(
            "INSERT INTO medical_records (user_id, name, type, content) "
            "VALUES (%s, %s, %s, %s) "
            "RETURNING id, user_id, name, type, uploaded_at",
            (uid, name, type_, content),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def list_medical_records(user_id):
    """Return all medical records for a user (metadata only, no content)."""
    uid = _parse_uuid(user_id)
    with get_conn() as cur:
        cur.execute(
            "SELECT id, user_id, name, type, uploaded_at "
            "FROM medical_records WHERE user_id = %s ORDER BY uploaded_at DESC",
            (uid,),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_medical_record(record_id, user_id):
    """Return a single medical record (including content) owned by user_id."""
    with get_conn() as cur:
        cur.execute(
            "SELECT id, user_id, name, type, content, uploaded_at "
            "FROM medical_records WHERE id = %s AND user_id = %s",
            (_parse_uuid(record_id), _parse_uuid(user_id)),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def delete_medical_record(record_id, user_id):
    """Delete a medical record owned by user_id. Returns True if deleted."""
    with get_conn() as cur:
        cur.execute(
            "DELETE FROM medical_records WHERE id = %s AND user_id = %s",
            (_parse_uuid(record_id), _parse_uuid(user_id)),
        )
        return cur.rowcount > 0
