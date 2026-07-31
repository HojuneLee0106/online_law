"""사용자 인증 + 대화 목록 관리 (SQLite)."""
import secrets
import sqlite3
import time
from contextlib import contextmanager

import bcrypt

DB_PATH = "users.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS conversations (
            thread_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS tokens (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """)


# ── 회원가입 / 로그인 ──
def register_user(username: str, password: str) -> tuple[bool, str]:
    if not username or not password:
        return False, "아이디와 비밀번호를 입력하세요."
    if len(password) < 4:
        return False, "비밀번호는 4자 이상이어야 합니다."
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, pw_hash, time.time()),
            )
        return True, "가입 완료"
    except sqlite3.IntegrityError:
        return False, "이미 존재하는 아이디입니다."


def login_user(username: str, password: str) -> str | None:
    """성공 시 토큰 반환, 실패 시 None."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
    if not row:
        return None
    if not bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
        return None
    # 토큰 발급
    token = secrets.token_urlsafe(32)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO tokens (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, row["id"], time.time()),
        )
    return token


def get_user_by_token(token: str) -> int | None:
    """토큰으로 user_id 조회. 없으면 None."""
    if not token:
        return None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT user_id FROM tokens WHERE token = ?", (token,)
        ).fetchone()
    return row["user_id"] if row else None


# ── 대화 목록 관리 ──
def upsert_conversation(thread_id: str, user_id: int, title: str):
    """대화 생성/갱신. 새 대화면 insert, 있으면 updated_at만 갱신."""
    now = time.time()
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT thread_id FROM conversations WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE thread_id = ?",
                (now, thread_id),
            )
        else:
            conn.execute(
                "INSERT INTO conversations (thread_id, user_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (thread_id, user_id, title[:50], now, now),
            )


def list_conversations(user_id: int) -> list[dict]:
    """유저의 대화 목록 (최신순)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT thread_id, title, updated_at FROM conversations "
            "WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def owns_thread(user_id: int, thread_id: str) -> bool:
    """이 유저가 이 thread의 소유자인지 (남의 대화 접근 방지)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE thread_id = ? AND user_id = ?",
            (thread_id, user_id),
        ).fetchone()
    return row is not None