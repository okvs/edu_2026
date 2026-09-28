"""메시지 스트림을 대화방별 청크로 묶는다.

- 시각 정보가 없으므로 id(순번)만 믿는다.
- 여러 방의 메시지가 시간순으로 섞여 들어와도 방별 버퍼로 한 번에 처리한다 (메모리 상한 = 방 수 × size).
- 청크는 size개 메시지, 인접 청크와 overlap개 겹친다. 작은 청크로 검색하고
  검색 후 start_id/end_id 앞뒤로 확장해서 문맥을 붙이는 전략을 전제로 한다.
"""
from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Message:
    id: int
    room: str
    speaker: str
    text: str


@dataclass(frozen=True)
class Chunk:
    room: str
    start_id: int          # 청크에 포함된 첫 메시지 id
    end_id: int            # 마지막 메시지 id
    n_messages: int
    speakers: tuple[str, ...]
    text: str              # 임베딩에 넣을 본문


def render_chunk(room: str, msgs: list[Message]) -> Chunk:
    body = "\n".join(f"{m.speaker}: {m.text}" for m in msgs)
    speakers = tuple(sorted({m.speaker for m in msgs}))
    text = f"[대화방: {room}] [참여: {', '.join(speakers)}]\n{body}"
    return Chunk(
        room=room,
        start_id=msgs[0].id,
        end_id=msgs[-1].id,
        n_messages=len(msgs),
        speakers=speakers,
        text=text,
    )


def stream_chunks(messages: Iterable[Message], size: int = 12, overlap: int = 4) -> Iterator[Chunk]:
    """시간순 메시지 스트림을 받아 완성되는 대로 청크를 내보낸다."""
    if not 0 <= overlap < size:
        raise ValueError("0 <= overlap < size 여야 합니다")

    buffers: dict[str, list[Message]] = {}
    flushed_once: set[str] = set()
    for m in messages:
        buf = buffers.setdefault(m.room, [])
        buf.append(m)
        if len(buf) == size:
            yield render_chunk(m.room, buf)
            flushed_once.add(m.room)
            # 마지막 overlap개는 다음 청크 앞부분으로 넘긴다
            buffers[m.room] = buf[size - overlap:] if overlap else []

    # 스트림 끝: 아직 청크로 안 나간 새 메시지가 있으면 짧은 청크로 내보낸다.
    # 버퍼 길이가 정확히 overlap이고 이미 한 번 내보낸 방이면 직전 청크의 꼬리만 남은 것이므로 건너뛴다.
    for room, buf in buffers.items():
        if not buf:
            continue
        if room in flushed_once and len(buf) == overlap:
            continue
        yield render_chunk(room, buf)
