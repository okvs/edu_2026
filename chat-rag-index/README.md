# chat-rag-index

메신저 대화내역(대화방, 발화자, 대화가 시간순으로 쌓인 구조)을 읽어
대화방별로 청킹하고 임베딩 벡터를 만들어 저장하는 인덱싱 파이프라인.

```
DB(messages) --> 방별 스트리밍 청킹 --> 사내 bge-m3 서빙 API --> chunks 테이블
                                                              |
        방 목록 조회 --> 방 다중 선택 --> 질문 --> 벡터 검색 <-------+
```

## 파일

| 파일 | 역할 |
|---|---|
| `create_sample_db.py` | 예시 SQLite DB 생성 (실제 DB 있으면 불필요) |
| `chunker.py` | 시간순 메시지 스트림을 방별 청크로 묶는 순수 함수 |
| `embedder.py` | 임베딩 백엔드 (http: 사내 서빙 API, local: sentence-transformers, voyage: Voyage AI) |
| `build_index.py` | DB 읽기 -> 청킹 -> 임베딩 -> 저장. 증분 갱신 지원 |
| `search.py` | 방 목록 조회, 방 다중 선택 검색, 앞뒤 청크 확장. 검증용 |

## 설치

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

기본 백엔드는 `http`: 사내 GPU에서 서빙되는 bge-m3 API를 호출한다. 모델 다운로드 없음.

```
set EMBED_API_URL=http://embed.internal/v1/embeddings    # Windows
export EMBED_API_URL=http://embed.internal/v1/embeddings # macOS / Linux
set EMBED_API_KEY=...        # 인증이 있으면 (Authorization: Bearer 로 전송)
set EMBED_MODEL=BAAI/bge-m3  # 요청 본문의 model 필드. 기본값이 이거라 보통 생략
```

응답 형식은 두 가지를 자동 판별한다.

| 서버 종류 | URL 끝 | 요청 | 응답 |
|---|---|---|---|
| OpenAI 호환 (vLLM, Infinity 등) | `/v1/embeddings` | `{"input": [...], "model": ...}` | `{"data": [{"embedding": [...]}]}` |
| TEI (Text Embeddings Inference) | `/embed` | `{"inputs": [...]}` | `[[...], ...]` |

사내 서빙이 다른 형식이면 `embedder.py` 의 `HttpEmbedder._call` (요청) 과 `_parse` (응답) 만 고친다.
벡터는 클라이언트에서 정규화하므로 서버가 정규화를 하든 안 하든 상관없다.
한 요청에 32개 텍스트를 보내는데, 서버 제한이 있으면 `HttpEmbedder(batch_size=...)` 를 줄인다.

서빙 없이 로컬에서 돌리려면 `pip install sentence-transformers` 후 `--backend local` (약 2.2GB 다운로드).

## 실행

```
python create_sample_db.py                 # 예시 DB chat.db 생성 (704개 메시지, 3개 방)
python build_index.py                      # 인덱싱 (두 번째부터는 증분)
python build_index.py --rebuild            # 처음부터 다시
python search.py --list-rooms              # 방 목록
python search.py --list-rooms 프로          # 이름에 "프로"가 들어간 방
python search.py "회의 몇 시로 정했어" --room 팀프로젝트 동아리 --context 1
```

Windows 콘솔에서 한글이 깨지면 환경변수 `PYTHONUTF8=1` 을 켠다.

로컬 실행 시 가벼운 모델로 빠른 테스트: `python build_index.py --backend local --model intfloat/multilingual-e5-small --rebuild`

## 실제 DB에 붙일 때 고칠 곳

1. **`EMBED_API_URL`**: 사내 서빙 주소. 응답 형식이 위 표와 다르면 `embedder.py` 의 `HttpEmbedder` 수정.
2. **`build_index.py` 의 `MESSAGES_SQL`**: 실제 테이블/컬럼명으로 변경. 컬럼 순서는 `id, room, speaker, text` 유지.
   `id` 는 시간순 정렬 가능한 정수여야 한다 (없으면 뷰나 ROW_NUMBER로 만들어서 넘긴다).
3. **DB 드라이버**: SQLite가 아니면 `sqlite3.connect` 를 해당 드라이버로 교체.
   원본 DB와 `chunks` 저장 DB를 분리하려면 `build_index.py` 에서 커넥션을 두 개로 나눈다.
4. **저장소**: 운영에서는 `chunks` 를 pgvector 등으로 옮긴다. `embedding` 컬럼은 float32 배열이며
   `index_meta.embedding_dim` 에 차원이 기록되어 있다 (bge-m3는 1024).
5. **청크 크기**: `--size` / `--overlap` 기본값 12 / 4. 바꾸면 `--rebuild` 필요 (설정 변경은 자동 감지).
6. **필터링**: 시스템 메시지, 이모티콘만 있는 메시지 등은 `build_index.py` 의 `iter_messages` 에서 걸러낸다.

## 스키마

```sql
-- 입력 (예시 DB 기준)
messages(id INTEGER PRIMARY KEY, room TEXT, speaker TEXT, text TEXT)

-- 출력
chunks(chunk_id, room, start_id, end_id, n_messages, speakers JSON, text, embedding BLOB)
index_meta(key, value)   -- config, embedding_model, embedding_dim
```

청크는 항상 한 대화방 안에서만 만들어진다. `(room, start_id)` 가 유일하다.

## 아직 없는 것 (다음 단계)

- 청크별 문맥 요약 붙이기 (contextual retrieval)
- BM25 키워드 검색과의 하이브리드, 리랭킹
- 검색 결과를 Claude에 넘겨 답변 생성 (citations)
- 평가 세트
