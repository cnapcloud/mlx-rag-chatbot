# Simple RAG-powered Chatbot (Apple Silicon, MLX)

로컬 LLM을 서빙하고, 그 위에 챗봇(LibreChat)과 문서 검색(RAG)을 얹는 개발 환경입니다.

Apple Silicon Mac에서 MLX로 Qwen3-8B와 bge-m3 임베딩 모델을 직접 서빙하므로, 문서와 대화가 외부 API로 나가지 않습니다.
PDF, TXT, Markdown 문서를 올리면 청크로 나눠 임베딩한 뒤 Postgres(pgvector)에 저장하고, 챗봇이 MCP 도구로 이를 검색해 문서를 근거로 답합니다.
로컬 모델에 장애가 나면 LiteLLM이 Gemini로 넘겨 대화가 끊기지 않게 합니다(`GEMINI_API_KEY` 설정 시).

```
                         ┌────────────────────── Docker ──────────────────────┐
 브라우저 ──▶ LibreChat ─┼─▶ LiteLLM(:4000) ─▶ vllm-mlx(호스트 :8100) / Gemini │
   :3080        │        │                                                    │
                │ MCP    │   simple-rag(:3000) ─▶ Postgres+pgvector(:5434)    │
                └────────┼──────────▲─────────────────┐                      │
                         │          └ 임베딩 ─ vllm-mlx(호스트 :8100)          │
                         │   MongoDB (LibreChat 사용자/대화 저장)               │
                         └────────────────────────────────────────────────────┘
```

| 구성요소 | 위치 | 포트 | 역할 |
|---|---|---|---|
| vllm-mlx | 호스트(macOS) | 8100 | 채팅·임베딩 모델 서빙 (`make run`) |
| LibreChat | Docker | 3080 | 챗봇 웹 UI, 사용자 등록/로그인 |
| LiteLLM | Docker | 4000 | 모델 게이트웨이 (qwen3-8b → 장애 시 Gemini fallback) |
| simple-rag | Docker | 3000 | 문서 업로드·벡터 검색 API + MCP 서버 |
| Postgres(pgvector) | Docker | 127.0.0.1:5434 | 문서·청크·임베딩, LiteLLM DB |
| MongoDB | Docker | (미공개) | LibreChat 데이터 |
| mcp-everything | Docker | 3001 | MCP 테스트용 서버 |

## 목차

- [0. 사전 준비](#0-사전-준비)
- [1. vllm-mlx 올리기 (호스트)](#1-vllm-mlx-올리기-호스트)
- [2. 환경 변수 설정](#2-환경-변수-설정)
- [3. docker compose 실행](#3-docker-compose-실행)
- [4. 문서 inject (업로드)](#4-문서-inject-업로드)
- [5. 챗봇 사용자 등록과 로그인](#5-챗봇-사용자-등록과-로그인)
- [6. RAG 검색](#6-rag-검색)
  - [6-1. API로 직접](#6-1-api로-직접)
  - [6-2. 챗봇에서 (MCP)](#6-2-챗봇에서-mcp)
- [7. 중지와 초기화](#7-중지와-초기화)
- [8. 문제 해결](#8-문제-해결)
- [참고 사항](#참고-사항)

## 0. 사전 준비

- Apple Silicon Mac, [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [uv](https://docs.astral.sh/uv/) — `brew install uv`
- 디스크 여유 공간: 모델 3개 다운로드용 (수 GB)
- (선택) Gemini API 키 — LiteLLM fallback용

## 1. vllm-mlx 올리기 (호스트)

Docker가 아니라 **호스트에서 직접** 실행합니다. MLX는 macOS에서만 동작하기 때문입니다.

```bash
make install     # hf, vllm-mlx 설치 (uv tool)
make download    # 채팅/임베딩/리랭크 모델 다운로드 (최초 1회)
make run         # 서버 기동 (포그라운드, 터미널 하나를 점유)
```

다른 터미널에서 확인합니다.

```bash
curl http://localhost:8100/v1/models
```

- 기본 모델: `mlx-community/Qwen3-8B-4bit` (채팅), `mlx-community/bge-m3-mlx-fp16` (임베딩, 1024차원)
- 종료: `make stop`
- 설정(포트, 컨텍스트, 캐시 등)은 [Makefile](Makefile) 상단 변수를 수정합니다.
- `VLLM_HOST`는 `0.0.0.0`이어야 합니다. Docker 컨테이너가 `host.docker.internal`로 접근하기 때문입니다.

> **임베딩은 vllm-mlx가 떠 있어야 합니다.** 문서 업로드와 검색 모두 8100의 임베딩 API를 호출하므로, 꺼져 있으면 실패합니다.

## 2. 환경 변수 설정

```bash
cp .env.example .env   # 이미 .env가 있으면 건너뜁니다
```

| 변수 | 설명 |
|---|---|
| `LITELLM_MASTER_KEY` | LiteLLM 마스터 키. LibreChat이 이 값으로 LiteLLM에 접속하고, LiteLLM Admin UI 비밀번호로도 쓰입니다. |
| `GEMINI_API_KEY` | Gemini fallback용. 비워 두면 `docker compose` 실행 시 경고가 나오고, Gemini 모델 호출만 실패합니다. |

`simple-rag`는 로컬(`npm run dev`)로 따로 띄울 때만 `simple-rag/.env`를 씁니다. Docker로 띄우면 [compose.yaml](compose.yaml)의 `environment`가 우선합니다.

> `compose.yaml`의 `CREDS_KEY`, `JWT_SECRET`, DB 비밀번호는 개발용 placeholder입니다. 외부에 노출하기 전에 바꾸세요.

## 3. docker compose 실행

```bash
docker compose up -d --build
docker compose ps
```

- 첫 기동에는 이미지 pull과 `mcp-everything`의 `npm install`로 몇 분 걸립니다. LibreChat은 `mcp-everything`가 healthy가 된 뒤에 시작합니다.
- Postgres의 `rag`, `litellm` DB와 role은 [config/initdb/01-init.sh](config/initdb/01-init.sh)가 **`data/postgres`가 비어 있을 때만** 만듭니다.
- `simple-rag` 컨테이너는 시작할 때 테이블을 자동 생성(`npm run migrate`)합니다.

상태 확인:

```bash
curl http://localhost:3000/health        # {"status":"ok"}
curl http://localhost:4000/health/liveliness
docker compose logs -f librechat         # 에러 확인
```

## 4. 문서 inject (업로드)

`simple-rag`의 `POST /documents`로 파일을 올리면 텍스트 추출 → 청크 분할 → 임베딩 → pgvector 저장이 이뤄집니다.

```bash
curl -F "file=@simple-rag/samples/sample-policy.txt;type=text/plain" http://localhost:3000/documents
curl -F "file=@simple-rag/samples/cheongajin-notice.pdf;type=application/pdf" http://localhost:3000/documents
```

성공하면 `201`과 함께 문서 ID와 청크 수가 돌아옵니다.

```json
{ "data": { "documentId": "…uuid…", "chunksCreated": 12 } }
```

- 지원 형식: PDF, TXT, Markdown (`;type=`으로 mimetype을 명시하세요)
- 제한: 파일 1개, 최대 10MB
- 같은 파일을 두 번 올리면 **중복 문서가 생깁니다.** 중복 방지는 아직 없습니다.
- 임베딩 서버(8100)가 꺼져 있으면 실패하고, 이때 청크가 일부만 들어간 문서가 남을 수 있습니다. 아래 "문제 해결"을 참고하세요.

업로드된 문서 확인:

```bash
docker exec postgres psql -U rag -d rag -c \
  "SELECT d.id, d.filename, count(c.id) AS chunks
   FROM documents d LEFT JOIN document_chunks c ON c.document_id = d.id
   GROUP BY d.id ORDER BY d.created_at;"
```

문서 삭제(청크도 함께 삭제):

```bash
curl -X DELETE http://localhost:3000/documents/<document-id>
```

## 5. 챗봇 사용자 등록과 로그인

LibreChat은 `ALLOW_REGISTRATION: "true"`로 설정돼 있어 누구나 가입할 수 있습니다.

1. 브라우저에서 <http://localhost:3080/register> 접속
2. 이름, 이메일, 비밀번호를 입력해 가입 (이메일 인증 없음)
3. <http://localhost:3080/login> 에서 로그인

- **처음 가입한 계정이 관리자**가 됩니다.
- 모델 목록에서 `qwen3-8b`(로컬)와 `gemini-3.5-flash-lite`를 고를 수 있습니다. 기본값은 `qwen3-8b`입니다.
- 사용자 계정은 MongoDB(`data/mongodb`)에 저장되어 컨테이너를 재기동해도 유지됩니다.

LiteLLM 관리 UI는 <http://localhost:4000/ui> 입니다. 사용자명은 `admin`, 비밀번호는 `LITELLM_MASTER_KEY` 값입니다.

## 6. RAG 검색

### 6-1. API로 직접

```bash
# 검색된 청크와 유사도만 반환
curl -X POST http://localhost:3000/search \
  -H 'content-type: application/json' \
  -d '{"question":"입주자 모집 자격이 뭐야?"}'

# 검색 + 답변 생성 (출처 포함)
curl -X POST http://localhost:3000/ask \
  -H 'content-type: application/json' \
  -d '{"question":"입주자 모집 자격이 뭐야?"}'

# 특정 문서로 범위 제한
curl -X POST http://localhost:3000/ask \
  -H 'content-type: application/json' \
  -d '{"question":"...", "documentId":"<uuid>"}'
```

- `/ask`는 유사도 `0.35` 미만(`RAG_MIN_SIMILARITY`)이면 `I could not find this in the provided documents.`를 돌려줍니다.
- 벡터 검색 상위 `5`개(`RAG_TOP_K`)를 리랭크해, 그중 상위 `3`개(`RAG_RERANK_TOP_N`)를 근거로 씁니다.

### 6-2. 챗봇에서 (MCP)

`simple-rag`는 `/mcp`로 MCP 도구 3개를 제공하고, [config/librechat.yaml](config/librechat.yaml)에 `simple-rag` 서버로 등록돼 있습니다.

| 도구 | 설명 |
|---|---|
| `search_documents` | 질문과 유사한 청크를 벡터 검색 |
| `list_documents` | 업로드된 문서 목록 |
| `delete_document` | 문서 하나 삭제 (되돌릴 수 없음) |

1. LibreChat에 로그인해 새 대화를 엽니다.
2. 입력창의 MCP(도구) 메뉴에서 `simple-rag`를 활성화합니다. 보이지 않으면 에이전트를 만들어 도구로 `simple-rag`를 추가합니다. (LibreChat 버전에 따라 메뉴 위치가 다릅니다.)
3. 업로드한 문서에 대해 질문합니다. 예: `업로드한 모집공고에서 입주 자격을 문서에서 찾아서 알려줘`

모델이 `search_documents`를 호출하고, 반환된 청크를 근거로 답합니다.

## 7. 중지와 초기화

```bash
docker compose down                 # 컨테이너 중지/삭제 (데이터는 유지)
make stop                           # vllm-mlx 종료

# 완전 초기화 — 사용자, 대화, 업로드한 문서가 모두 사라집니다
docker compose down
rm -rf data/mongodb data/postgres
```

`data/postgres`를 지우면 다음 기동 때 `initdb`가 다시 실행됩니다.

## 8. 문제 해결

| 증상 | 확인 |
|---|---|
| 업로드/검색이 `Embedding failed` 또는 연결 오류 | vllm-mlx가 떠 있는지(`curl localhost:8100/v1/models`), `VLLM_HOST=0.0.0.0`인지 확인 |
| 업로드는 실패했는데 문서가 남아 있음 | 임베딩 실패 시 청크가 일부만 들어간 문서가 남습니다. 위 SQL로 `chunks`가 비정상인 문서를 찾아 `DELETE`한 뒤 다시 올리세요. |
| LibreChat이 오래 시작하지 않음 | `mcp-everything`의 `npm install` 대기 중일 수 있습니다. `docker compose logs -f mcp-everything` |
| 챗봇에서 MCP 도구를 호출하지 않음 | 로컬 `qwen3-8b`는 도구 호출이 불안정할 수 있습니다. 같은 대화를 `gemini-3.5-flash-lite`로 시도해 보세요. |
| `qwen3-8b`가 응답 없이 Gemini로 넘어감 | LiteLLM fallback 동작입니다. vllm-mlx 장애 또는 컨텍스트 초과 여부를 확인하세요. |
| `GEMINI_API_KEY is not set` 경고 | `.env`에 키를 넣거나 셸에서 export 하세요. |


## 참고 사항

`simple-rag/`는 [sunflowerIU/ai-simple-rag](https://github.com/sunflowerIU/ai-simple-rag)(ISC 표기, LICENSE 파일 없음)를 가져와 **수정한 것**입니다. 원본 파이프라인 설명은 [simple-rag/README.md](simple-rag/README.md)에 있고, 설정값이 다르면 이 README가 기준입니다.

- **LLM·임베딩:** Groq + Ollama(nomic, 768차원)를 vllm-mlx(Qwen3-8B, bge-m3 1024차원)로 교체했습니다. nomic용 prefix는 제거했습니다.
- **기능 추가:** 리랭크(bge-reranker-v2-m3, `RAG_RERANK_TOP_N`), 문서 목록·삭제, MCP 서버(`/mcp`, 도구 3개), `Dockerfile`, 샘플 PDF를 추가했습니다.
- **설정 변경:** 루트 [compose.yaml](compose.yaml)에서 LibreChat, LiteLLM과 함께 실행합니다.

## 라이선스

이 저장소는 [MIT License](LICENSE)로 공개합니다. 다만 `simple-rag/`는 [sunflowerIU/ai-simple-rag](https://github.com/sunflowerIU/ai-simple-rag)를 가져와 수정한 것이고, 원본이 ISC로 표기되어 있어 `simple-rag/package.json`에도 ISC를 그대로 두었습니다.
