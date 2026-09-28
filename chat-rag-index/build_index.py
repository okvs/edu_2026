"""DB에서 대화를 읽어 청킹하고 임베딩을 만들어 저장한다.

사용:
    set EMBED_API_URL=http://embed.internal/v1/embeddings   # 사내 bge-m3 서빙 주소
    python build_index.py                 # chat.db 읽어서 chunks 테이블에 저장 (증분)
    python build_index.py --rebuild       # 처음부터 다시
    python build_index.py --backend local # 서빙 없이 로컬에서 모델 직접 실행 (sentence-transformers 필요)

증분 동작:
    방마다 마지막 청크를 지우고 그 청크의 start_id부터 다시 청킹한다.
    그래서 새 메시지가 몇 개 추가되든 기존 청크 경계가 흔들리지 않는다.
"""
import argparse
import json
import sqlite3
import sys
import time
from typing import Iterator

import numpy as np

from chunker import Chunk, Message, stream_chunks
from embedder import make_embedder

# 실제 DB에 맞출 때는 이 SQL만 바꾸면 된다. 컬럼 순서: id, room, speaker, text
MESSAGES_SQL = "SELECT id, room, speaker, text FROM messages WHERE id >= ? ORDER BY id"


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id   INTEGER PRIMARY KEY,
            room       TEXT NOT NULL,
            start_id   INTEGER NOT NULL,
            end_id     INTEGER NOT NULL,
            n_messages INTEGER NOT NULL,
            speakers   TEXT NOT NULL,      -- JSON 배열
            text       TEXT NOT NULL,
            embedding  BLOB NOT NULL,      -- float32 배열 bytes
            UNIQUE(room, start_id)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_room ON chunks(room, start_id);
        CREATE TABLE IF NOT EXISTS index_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM index_meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO index_meta(key, value) VALUES (?, ?)", (key, value))


def resume_points(conn: sqlite3.Connection) -> dict[str, int]:
    """방별로 다시 청킹을 시작할 메시지 id. 마지막 청크를 지우고 그 start_id를 돌려준다."""
    rows = conn.execute("SELECT room, MAX(start_id) FROM chunks GROUP BY room").fetchall()
    points = {}
    for room, start_id in rows:
        conn.execute("DELETE FROM chunks WHERE room = ? AND start_id >= ?", (room, start_id))
        points[room] = start_id
    return points


def iter_messages(conn: sqlite3.Connection, points: dict[str, int]) -> Iterator[Message]:
    """시간순으로 메시지를 스트리밍. 이미 인덱싱된 방은 재시작 지점 이후만 흘려보낸다."""
    min_id = min(points.values()) if points else 0
    cur = conn.execute(MESSAGES_SQL, (min_id,))
    while True:
        rows = cur.fetchmany(5000)
        if not rows:
            return
        for id_, room, speaker, text in rows:
            if id_ < points.get(room, 0):
                continue
            if not text or not text.strip():
                continue
            yield Message(id=id_, room=room, speaker=speaker, text=text.strip())


def insert_chunks(conn: sqlite3.Connection, chunks: list[Chunk], vectors: np.ndarray) -> None:
    conn.executemany(
        "INSERT INTO chunks(room, start_id, end_id, n_messages, speakers, text, embedding) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (c.room, c.start_id, c.end_id, c.n_messages, json.dumps(c.speakers, ensure_ascii=False),
             c.text, vectors[i].tobytes())
            for i, c in enumerate(chunks)
        ],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="chat.db")
    ap.add_argument("--size", type=int, default=12, help="청크당 메시지 수")
    ap.add_argument("--overlap", type=int, default=4, help="인접 청크와 겹치는 메시지 수")
    ap.add_argument("--backend", choices=["http", "local", "voyage"], default="http")
    ap.add_argument("--model", default=None, help="임베딩 모델 이름 (백엔드 기본값 사용 시 생략)")
    ap.add_argument("--batch-size", type=int, default=64, help="한 번에 DB에 저장할 청크 수")
    ap.add_argument("--rebuild", action="store_true", help="기존 청크를 모두 지우고 처음부터")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_schema(conn)

    # 청크 크기나 모델이 바뀌면 기존 인덱스와 섞이면 안 되므로 강제로 rebuild
    config = json.dumps({"size": args.size, "overlap": args.overlap, "backend": args.backend, "model": args.model})
    prev = get_meta(conn, "config")
    if prev is not None and prev != config and not args.rebuild:
        print(f"설정이 바뀌었습니다 (이전: {prev}). --rebuild 로 다시 만드세요.", file=sys.stderr)
        sys.exit(1)
    if args.rebuild:
        conn.execute("DELETE FROM chunks")
        conn.commit()

    embedder = make_embedder(args.backend, args.model)
    set_meta(conn, "config", config)
    set_meta(conn, "embedding_model", embedder.name)
    set_meta(conn, "embedding_dim", str(embedder.dim))

    points = resume_points(conn)
    if points:
        print(f"증분 모드: {len(points)}개 방을 마지막 청크부터 다시 처리")

    t0 = time.time()
    n_chunks = n_msgs = 0
    batch: list[Chunk] = []

    def flush():
        nonlocal n_chunks
        if not batch:
            return
        vectors = embedder.embed_passages([c.text for c in batch])
        insert_chunks(conn, batch, vectors)
        conn.commit()
        n_chunks += len(batch)
        print(f"  청크 {n_chunks}개 저장 ({time.time() - t0:.1f}s)")
        batch.clear()

    def counted(msgs):
        nonlocal n_msgs
        for m in msgs:
            n_msgs += 1
            yield m

    for chunk in stream_chunks(counted(iter_messages(conn, points)), size=args.size, overlap=args.overlap):
        batch.append(chunk)
        if len(batch) >= args.batch_size:
            flush()
    flush()

    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    print(f"완료: 메시지 {n_msgs}개 읽음, 청크 {n_chunks}개 생성, 총 {total}개 (모델 {embedder.name}, dim {embedder.dim})")


if __name__ == "__main__":
    main()
