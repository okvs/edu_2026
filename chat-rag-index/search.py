"""임베딩이 제대로 만들어졌는지 확인하는 검색 스크립트.

    python search.py --list-rooms            # 대화방 목록 (방 이름 검색: --list-rooms 프로젝트)
    python search.py "회식 어디서 하기로 했어?"
    python search.py "회의 몇 시" --room 팀프로젝트 --context 1
    python search.py "약속 언제" --room 팀프로젝트 동아리      # 여러 방 선택

실제 서비스 흐름: 방 이름으로 검색 -> 방 여러 개 선택 -> 질문. 여기서는 --room 에 방을 여러 개 주면 된다.

--context N : 검색된 청크 앞뒤로 N개 청크를 같이 출력 (경계 오류 보정용)
"""
import argparse
import json
import sqlite3

import numpy as np

from embedder import make_embedder


def neighbors(conn: sqlite3.Connection, room: str, start_id: int, n: int) -> list[tuple[int, int, str]]:
    """같은 방에서 start_id 기준 앞뒤 n개 청크를 순서대로 돌려준다."""
    before = conn.execute(
        "SELECT start_id, end_id, text FROM chunks WHERE room = ? AND start_id < ? ORDER BY start_id DESC LIMIT ?",
        (room, start_id, n),
    ).fetchall()
    center = conn.execute(
        "SELECT start_id, end_id, text FROM chunks WHERE room = ? AND start_id = ?", (room, start_id)
    ).fetchall()
    after = conn.execute(
        "SELECT start_id, end_id, text FROM chunks WHERE room = ? AND start_id > ? ORDER BY start_id LIMIT ?",
        (room, start_id, n),
    ).fetchall()
    return list(reversed(before)) + center + after


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default=None)
    ap.add_argument("--db", default="chat.db")
    ap.add_argument("--list-rooms", nargs="?", const="", default=None, metavar="KEYWORD",
                    help="대화방 목록 출력. 키워드를 주면 이름에 포함된 방만")
    ap.add_argument("--room", nargs="+", default=None, help="대화방 필터 (여러 개 가능)")
    ap.add_argument("--speaker", default=None, help="발화자 필터")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--context", type=int, default=0, help="앞뒤로 함께 보여줄 청크 수")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    if args.list_rooms is not None:
        rows = conn.execute(
            "SELECT room, COUNT(*), MIN(start_id), MAX(end_id) FROM chunks "
            "WHERE room LIKE ? GROUP BY room ORDER BY room", (f"%{args.list_rooms}%",)
        ).fetchall()
        for room, n, lo, hi in rows:
            print(f"{room}	청크 {n}개	메시지 {lo}~{hi}")
        return
    if not args.query:
        ap.error("query 가 필요합니다 (또는 --list-rooms)")

    cfg = json.loads(conn.execute("SELECT value FROM index_meta WHERE key='config'").fetchone()[0])
    dim = int(conn.execute("SELECT value FROM index_meta WHERE key='embedding_dim'").fetchone()[0])
    embedder = make_embedder(cfg["backend"], cfg["model"])

    sql = "SELECT chunk_id, room, start_id, end_id, speakers, text, embedding FROM chunks"
    where, params = [], []
    if args.room:
        where.append(f"room IN ({','.join('?' * len(args.room))})")
        params.extend(args.room)
    if args.speaker:
        where.append("speakers LIKE ?")
        params.append(f'%"{args.speaker}"%')
    if where:
        sql += " WHERE " + " AND ".join(where)
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        print("조건에 맞는 청크가 없습니다")
        return

    matrix = np.frombuffer(b"".join(r[6] for r in rows), dtype=np.float32).reshape(len(rows), dim)
    q = embedder.embed_query(args.query)
    scores = matrix @ q  # 정규화된 벡터이므로 내적 = 코사인 유사도
    order = np.argsort(-scores)[: args.top]

    for rank, i in enumerate(order, 1):
        chunk_id, room, start_id, end_id, speakers, text, _ = rows[i]
        print(f"\n#{rank} score={scores[i]:.3f}  {room}  메시지 {start_id}~{end_id}  참여 {json.loads(speakers)}")
        if args.context:
            for s, e, t in neighbors(conn, room, start_id, args.context):
                mark = " <== 검색된 청크" if s == start_id else ""
                print(f"--- 메시지 {s}~{e}{mark}\n{t}")
        else:
            print(text)


if __name__ == "__main__":
    main()
