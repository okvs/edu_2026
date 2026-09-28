"""예시 대화 DB 생성.

실제 데이터는 (대화방, 발화자, 대화) 행이 시간순으로 쌓인 구조라고 했으니
같은 모양의 SQLite 테이블을 만들고 여러 방의 대화를 시간순으로 섞어서 넣는다.
id 컬럼이 곧 시간 순서다. 실제 DB에 맞출 때는 build_index.py의 SQL 한 줄만 바꾸면 된다.
"""
import random
import sqlite3
import sys

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "chat.db"

# 방마다 몇 개의 "주제 스크립트"를 두고, 이것들을 섞어서 시간순 스트림을 만든다.
SCRIPTS = {
    "팀프로젝트": [
        [("철수", "회의 내일로 미루자"), ("영희", "몇 시?"), ("철수", "3시 어때"), ("민수", "나 3시 수업"),
         ("철수", "그럼 5시"), ("영희", "5시 좋아"), ("민수", "ㅇㅋ 5시 확정")],
        [("영희", "발표 자료 누가 만들어?"), ("민수", "내가 초안 잡을게"), ("철수", "디자인은 내가"),
         ("영희", "그럼 난 데모 준비"), ("민수", "금요일까지 공유하자")],
        [("철수", "깃 충돌 났는데 도와줄 사람"), ("민수", "어느 파일?"), ("철수", "main.py"),
         ("민수", "내가 방금 푸시한 거랑 겹친 듯, 내 브랜치 먼저 머지해"), ("철수", "됐다 고마워")],
        [("영희", "교수님이 보고서 분량 10장 이상이래"), ("철수", "헐 너무 많은데"), ("민수", "그림 넣으면 금방 채워"),
         ("영희", "표지랑 목차도 포함이래"), ("철수", "그럼 할 만하네")],
    ],
    "동아리": [
        [("지훈", "이번 주 회식 어디로?"), ("수진", "지난번 삼겹살집 어때"), ("지훈", "거기 예약 안 됨"),
         ("현우", "그럼 역 앞 치킨집"), ("수진", "치킨 좋다"), ("지훈", "그럼 토요일 7시 치킨집으로"), ("현우", "예약은 내가 할게")],
        [("수진", "동아리방 열쇠 누가 갖고 있어?"), ("현우", "나"), ("수진", "내일 오전에 쓸 수 있어?"),
         ("현우", "9시에 갖다 놓을게"), ("수진", "고마워")],
        [("지훈", "MT 장소 투표 올렸어"), ("현우", "가평 한 표"), ("수진", "나도 가평"),
         ("지훈", "가평으로 거의 확정이네"), ("현우", "펜션은 내가 알아볼게")],
        [("현우", "회비 아직 안 낸 사람 3명"), ("수진", "나 냈어"), ("지훈", "나도 어제 보냄"),
         ("현우", "확인했어 고마워")],
    ],
    "가족": [
        [("엄마", "저녁 뭐 먹을래"), ("나", "김치찌개"), ("아빠", "나도 찬성"), ("엄마", "그럼 두부 사와"), ("나", "알겠어")],
        [("아빠", "주말에 할머니 댁 갈 거다"), ("나", "몇 시에 출발?"), ("아빠", "아침 9시"),
         ("엄마", "선물 뭐 사갈까"), ("나", "홍삼 어때"), ("엄마", "좋다")],
        [("엄마", "택배 왔는데 니 거야?"), ("나", "응 책이야"), ("엄마", "방에 갖다 놨어"), ("나", "고마워")],
        [("나", "이번 달 용돈 좀 일찍 줄 수 있어?"), ("아빠", "왜"), ("나", "교재 사야 해서"),
         ("아빠", "얼마"), ("나", "5만원"), ("아빠", "보냈다")],
    ],
}

# 주제 사이에 끼어드는 잡담. 실제 채팅처럼 주제가 섞이게 만든다.
FILLER = {
    "팀프로젝트": [("영희", "ㅋㅋㅋ"), ("민수", "점심 뭐 먹지"), ("철수", "졸리다"), ("영희", "오늘 비 온대")],
    "동아리": [("지훈", "ㅋㅋ"), ("수진", "배고파"), ("현우", "다들 시험 잘 봤어?"), ("지훈", "망했어")],
    "가족": [("엄마", "밥 먹었어?"), ("나", "응"), ("아빠", "일찍 들어와라"), ("나", "네")],
}


def build_stream(rounds: int, seed: int = 0):
    """모든 방의 스크립트를 여러 번 반복하며 시간순으로 섞은 메시지 스트림을 만든다."""
    rng = random.Random(seed)
    rows = []
    for _ in range(rounds):
        for room, scripts in SCRIPTS.items():
            for script in scripts:
                # 주제 스크립트 앞뒤에 잡담을 붙이고, 중간에도 하나 끼워 넣는다
                seq = [rng.choice(FILLER[room])] + list(script)
                seq.insert(rng.randint(1, len(seq) - 1), rng.choice(FILLER[room]))
                rows.append((room, seq))
    rng.shuffle(rows)  # 방 순서를 섞어서 여러 방의 대화가 시간순으로 교차되게
    stream = []
    for room, seq in rows:
        for speaker, text in seq:
            stream.append((room, speaker, text))
    return stream


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(
        """
        DROP TABLE IF EXISTS messages;
        CREATE TABLE messages (
            id      INTEGER PRIMARY KEY,   -- 시간순 순번
            room    TEXT NOT NULL,         -- 대화방 이름
            speaker TEXT NOT NULL,         -- 발화자
            text    TEXT NOT NULL          -- 대화 내용
        );
        CREATE INDEX idx_messages_room ON messages(room, id);
        """
    )
    stream = build_stream(rounds=8)
    conn.executemany("INSERT INTO messages(room, speaker, text) VALUES (?, ?, ?)", stream)
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    rooms = conn.execute("SELECT room, COUNT(*) FROM messages GROUP BY room").fetchall()
    conn.close()
    print(f"{DB_PATH}: 메시지 {n}개 생성")
    for room, c in rooms:
        print(f"  {room}: {c}개")


if __name__ == "__main__":
    main()
