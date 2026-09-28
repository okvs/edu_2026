"""임베딩 백엔드. 사내 서빙 API(http), 로컬(sentence-transformers), Voyage AI.

Anthropic은 임베딩 모델을 제공하지 않으므로 셋 중 하나를 쓴다.
- http  : 사내 GPU 서빙 서비스 호출 (기본). 환경변수로 설정한다.
            EMBED_API_URL   예) http://embed.internal/v1/embeddings  또는  http://embed.internal/embed
            EMBED_API_KEY   (선택) Authorization: Bearer 로 전송
            EMBED_MODEL     (선택) 요청 본문의 model 필드. 기본 BAAI/bge-m3
          OpenAI 호환(vLLM, Infinity 등: {"input": [...]} -> {"data": [{"embedding": [...]}]}) 과
          TEI(Text Embeddings Inference: {"inputs": [...]} -> [[...], ...]) 두 형식을 자동 판별한다.
          다른 형식이면 HttpEmbedder._parse 만 고치면 된다.
- local : sentence-transformers 로 직접 실행. 기본 BAAI/bge-m3 (약 2.2GB 다운로드).
- voyage: VOYAGE_API_KEY 필요.

검색은 내적으로 유사도를 계산하므로 모든 백엔드가 정규화된 벡터를 돌려준다.
"""
from __future__ import annotations

import json
import os
import urllib.request

import numpy as np


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-12)).astype(np.float32)


class Embedder:
    name: str
    dim: int

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class LocalEmbedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.dim = self.model.get_sentence_embedding_dimension()
        # e5 계열은 passage:/query: 접두어를 붙여야 제 성능이 나온다
        self._prefixed = "e5" in model_name.lower()

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if self._prefixed:
            texts = [f"passage: {t}" for t in texts]
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        if self._prefixed:
            text = f"query: {text}"
        vec = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return vec.astype(np.float32)


class HttpEmbedder(Embedder):
    def __init__(self, model_name: str | None = None, batch_size: int = 32, timeout: float = 120.0):
        self.url = os.environ.get("EMBED_API_URL")
        if not self.url:
            raise RuntimeError("EMBED_API_URL 환경변수가 필요합니다 (예: http://host/v1/embeddings)")
        self.name = model_name or os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
        self.api_key = os.environ.get("EMBED_API_KEY")
        self.batch_size = batch_size
        self.timeout = timeout
        # TEI 는 {"inputs": [...]}, OpenAI 호환은 {"input": [...], "model": ...}
        self._tei = self.url.rstrip("/").endswith("/embed")
        self.dim = len(self._call(["dim probe"])[0])

    def _call(self, texts: list[str]) -> np.ndarray:
        if self._tei:
            body = {"inputs": texts, "normalize": True, "truncate": True}
        else:
            body = {"input": texts, "model": self.name}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return self._parse(payload, len(texts))

    @staticmethod
    def _parse(payload, n: int) -> np.ndarray:
        if isinstance(payload, list):                      # TEI: [[...], ...]
            vectors = payload
        elif isinstance(payload, dict) and "data" in payload:   # OpenAI 호환
            items = sorted(payload["data"], key=lambda d: d.get("index", 0))
            vectors = [d["embedding"] for d in items]
        elif isinstance(payload, dict) and "embeddings" in payload:  # 일부 서버
            vectors = payload["embeddings"]
        else:
            raise ValueError(f"알 수 없는 응답 형식: {str(payload)[:200]}")
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != n:
            raise ValueError(f"응답 벡터 수가 맞지 않습니다: 요청 {n}, 응답 {arr.shape}")
        return _normalize(arr)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        out = [self._call(texts[i:i + self.batch_size]) for i in range(0, len(texts), self.batch_size)]
        return np.concatenate(out, axis=0)

    def embed_query(self, text: str) -> np.ndarray:
        return self._call([text])[0]


class VoyageEmbedder(Embedder):
    def __init__(self, model_name: str = "voyage-multilingual-2"):
        import voyageai

        if not os.environ.get("VOYAGE_API_KEY"):
            raise RuntimeError("VOYAGE_API_KEY 환경변수가 필요합니다")
        self.name = model_name
        self.client = voyageai.Client()
        self.dim = len(self.embed_query("dim probe"))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        out = self.client.embed(texts, model=self.name, input_type="document").embeddings
        return np.asarray(out, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        out = self.client.embed([text], model=self.name, input_type="query").embeddings
        return np.asarray(out[0], dtype=np.float32)


def make_embedder(backend: str, model_name: str | None = None) -> Embedder:
    if backend == "http":
        return HttpEmbedder(model_name)
    if backend == "local":
        return LocalEmbedder(model_name) if model_name else LocalEmbedder()
    if backend == "voyage":
        return VoyageEmbedder(model_name) if model_name else VoyageEmbedder()
    raise ValueError(f"알 수 없는 backend: {backend}")
