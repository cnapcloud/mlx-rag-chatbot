# vllm-mlx 기반 RAG 챗봇 구축 (3) - 문서 검색 API와 MCP 서버

[2편](02-vllm-mlx.md)에서 모델 서버의 메모리를 설계했다. 이번 편은 그 서버를 호출하는 simple-rag를 다룬다. 문서를 청크로 나눠 pgvector에 저장하고, 질문이 오면 관련 청크를 찾아 답을 만들며, 같은 검색을 MCP 도구로 챗봇에 내놓는 서비스다. 코드가 작아서 RAG의 각 단계를 눈으로 따라갈 수 있고, 단계마다 점수와 응답이 어떻게 나오는지도 직접 확인해 볼 수 있다.

이 글은 GitHub 저장소에 포함된 코드를 기준으로 설명한다. 저장소를 아직 클론하지 않았다면 다음과 같이 클론한다. simple-rag는 임베딩과 답변 생성을 vllm-mlx에 맡기므로, 모델 서버를 포함한 전체 스택을 띄우는 방법은 [1편](01-overview.md)의 3~4절을 따른다.

```bash
git clone https://github.com/cnapcloud/mlx-rag-chatbot.git
cd mlx-rag-chatbot
```

---

## 1. RAG 소개

RAG(Retrieval-Augmented Generation, 검색 증강 생성)는 LLM이 답하기 전에 외부 문서에서 관련 내용을 먼저 검색하고, 그 내용을 프롬프트에 함께 넣어 답을 생성하게 하는 방식이다. 이 프로젝트에서는 문서를 임베딩해 pgvector에 저장해 두고, 질문이 오면 유사한 청크를 찾아 vllm-mlx 모델에 컨텍스트로 넘긴다. 모델을 다시 학습시키지 않고도 내 문서를 근거로 최신 정보를 답하게 할 수 있으며, 근거 없는 답(환각)도 줄일 수 있다.

이 글에서 다루는 simple-rag는 [sunflowerIU/ai-simple-rag](https://github.com/sunflowerIU/ai-simple-rag)(ISC 라이선스)를 가져와 이 시리즈의 구성에 맞게 고친 것이다. 원본은 RAG 프레임워크 없이 Fastify, PostgreSQL(pgvector), AI SDK만으로 `문서 → 텍스트 추출 → 청크 → 임베딩 → 벡터 검색 → LLM 답변`을 구현한 프로젝트다. 청크 분할도, 프롬프트 조립도, 출처 표기도 모두 한 파일(`rag.ts`)에 드러나 있다. 변경 후 [`src/`](../simple-rag/src/) 전체가 약 700줄이라 한 번에 읽을 수 있고, 핵심은 [`rag.ts`](../simple-rag/src/rag.ts)와 [`mcp.ts`](../simple-rag/src/mcp.ts)다.

원본 코드는 답변 생성을 Groq, 임베딩을 Ollama에 맡기는 구성이었다. 이 시리즈에서는 모델 호출을 모두 vllm-mlx로 옮기고, 리랭크와 문서 목록·삭제, MCP 서버를 추가했다.

| 항목 | 원본 | 이 프로젝트 |
|---|---|---|
| 답변 LLM | Groq (`llama-3.1-8b-instant`) | vllm-mlx의 Qwen3-8B-4bit (OpenAI 호환 API) |
| 임베딩 | Ollama `nomic-embed-text` (768차원) | vllm-mlx의 bge-m3 (1024차원) |
| 임베딩 접두어 | nomic용 접두어(`search_documents:`, `search_query:`) 부착 | 제거 |
| 벡터 차원 | 테이블에 768 고정 | `EMBEDDING_DIMENSIONS` 환경 변수 |
| 리랭크 | 없음 | bge-reranker-v2-m3 (`RAG_RERANK_TOP_N`) |
| 문서 목록 | 없음 | `GET /documents` |
| 문서 삭제 | 없음 | `DELETE /documents/:id` |
| MCP 서버 | 없음 | `/mcp` (도구 3개) |
| 실행 | 로컬 `npm run dev` | Dockerfile, 시작 시 자동 migrate |

---

## 2. 시작하기

### 2.1 서버 실행

가장 먼저 호스트에서 vllm-mlx(`:8100`)가 올라와 있어야 한다. simple-rag는 임베딩, 리랭크, 답변 생성을 모두 vllm-mlx의 모델에 맡기므로, vllm-mlx가 꺼져 있으면 업로드와 검색, `/ask`가 실패한다. 1편의 `docker compose up -d --build`로 전체를 이미 띄웠다면 `:3000`의 API를 바로 호출할 수 있다.

소스에서 직접 개발 모드로 실행하려면 다음과 같이 한다. compose의 `simple-rag` 컨테이너가 `:3000`을 쓰고 있다면 `docker compose stop simple-rag`로 먼저 내려야 포트가 충돌하지 않는다.

```bash
make run                          # 1. vllm-mlx 시작 (저장소 루트, 터미널을 점유하므로 다른 터미널에서 계속)
curl http://localhost:8100/v1/models   # 모델 3개가 보이면 준비 완료

docker compose up -d postgres     # 2. Postgres(pgvector)
cd simple-rag
cp .env.example .env
npm install
npm run migrate                   # 3. 테이블 생성
npm run dev                       # 4. simple-rag 시작 (:3000)
```

### 2.2 문서 등록

PDF, TXT, Markdown 파일을 한 번에 하나씩(최대 10MB) 올리면 텍스트 추출부터 청크 분할, 임베딩, 저장까지 끝낸 뒤 `201`을 돌려준다.

```bash
curl -F "file=@samples/sample-policy.txt;type=text/plain" http://localhost:3000/documents
# {"data":{"documentId":"5665e76b-8119-41a1-bf21-b1f8100b6cd4","chunksCreated":1}}

curl -F "file=@samples/cheongajin-notice.pdf;type=application/pdf" http://localhost:3000/documents
# {"data":{"documentId":"62ad33b8-becf-47e6-8d35-7686c2ddc921","chunksCreated":26}}
```

### 2.3 목록 조회

```bash
curl http://localhost:3000/documents
```

```json
{
  "data": [
    {
      "id": "62ad33b8-becf-47e6-8d35-7686c2ddc921",
      "filename": "cheongajin-notice.pdf",
      "mimeType": "application/pdf",
      "chunks": 26,
      "createdAt": "2026-10-04T11:05:05.766Z"
    },
    {
      "id": "5665e76b-8119-41a1-bf21-b1f8100b6cd4",
      "filename": "sample-policy.txt",
      "mimeType": "text/plain",
      "chunks": 1,
      "createdAt": "2026-10-04T11:05:04.854Z"
    }
  ]
}
```

같은 파일을 다시 올리면 중복 문서가 생긴다(중복 방지 없음).

### 2.4 검색

`/search`는 질문과 가까운 청크를 점수와 함께 돌려줄 뿐 답변은 만들지 않는다. 검색 결과를 확인하는 용도이며, MCP `search_documents`도 같은 검색을 쓴다. `documentId`를 주면 그 문서 안에서만 찾는다.

```bash
curl -X POST http://localhost:3000/search -H 'content-type: application/json' \
  -d '{"question":"환불은 며칠 이내에 가능해?"}'
```

```json
{ "data": [
  { "chunkIndex": 0,  "similarity": 0.8117, "rerankScore": 0.1347, "content": "Amit Store Policy Refunds are available within 7 days of delivery. …" },
  { "chunkIndex": 14, "similarity": 0.5413, "rerankScore": 0.0110, "content": "임차인으로 선정되신 분은 임대차 계약 종료일 이후 …" },
  { "chunkIndex": 13, "similarity": 0.5397, "rerankScore": 0.0084, "content": "2026.07.17.(금) ~ 2026.07.20.(월) - 청약방법 : …" }
] }
```

### 2.5 답변

`/ask`는 `/search`와 같은 검색을 한 뒤, 찾은 청크를 LLM에 넣어 문장으로 된 답변(`answer`)을 만든다. 응답에는 답변과 함께 프롬프트에 넣은 청크가 `sources`로 따라온다. 즉 `/search`는 청크와 점수를, `/ask`는 답변과 근거 청크를 돌려준다. 프롬프트 템플릿은 [rag.ts](../simple-rag/src/rag.ts)의 `askDocument`를 참고한다.

```bash
curl -X POST http://localhost:3000/ask -H 'content-type: application/json' \
  -d '{"question":"환불은 며칠 이내에 가능해?"}'
```

```json
{ "data": {
    "answer": "환불은 7일 이내에 가능합니다. [Source 1]",
    "sources": [
      { "label": "Source 1", "chunkIndex": 0,  "similarity": 0.8117, "preview": "Amit Store Policy Refunds are available within 7 days …" },
      { "label": "Source 2", "chunkIndex": 14, "similarity": 0.5413, "preview": "임차인으로 선정되신 분은 임대차 계약 종료일 이후 …" },
      { "label": "Source 3", "chunkIndex": 13, "similarity": 0.5397, "preview": "2026.07.17.(금) ~ 2026.07.20.(월) - 청약방법 : …" }
    ]
} }
```

서버의 thinking 예산(`VLLM_THINK_BUDGET`, 128토큰)에서 간헐적으로 `/ask` 답변이 한 줄로 짧게 생성되는 경우가 있었다. 서버 설정을 1024토큰으로 바꾸거나 vllm-mlx에 직접 요청해 예산을 늘리자 정상 답변이 나왔다(한 질문 기준).

### 2.6 삭제

```bash
curl -X DELETE http://localhost:3000/documents/5665e76b-8119-41a1-bf21-b1f8100b6cd4     # 204
curl -X DELETE http://localhost:3000/documents/5665e76b-8119-41a1-bf21-b1f8100b6cd4     # 404 {"error":"Document not found"}
```

청크와 임베딩도 함께 지워진다(`ON DELETE CASCADE`). 삭제 후 목록에는 `cheongajin-notice.pdf`만 남는다.

---

## 3. 파이프라인 구성

```text
[업로드]
 PDF / TXT / MD  ─►  텍스트 추출  ─►  documents 저장 ─► 청크 분할  ─►  청크마다 임베딩  ─►  document_chunks 저장
                                            (450단어, 70단어 겹침)  (bge-m3, 1024차원)    (pgvector)

[질문]
 질문 ─► 질문 임베딩  ─►  벡터 검색  ─►  유사도 임계값   ─►  리랭크  ─►  프롬프트 조립 ─►  LLM  ─►  답변 + 출처
                      (TOP_K)  (MIN_SIMILARITY) (RERANK_TOP_N)   └──── /ask만 ────┘
```

위쪽(업로드)은 이 절과 4절에서, 아래쪽(질문)은 5절에서 다룬다.

### 3.1 텍스트 추출

업로드 요청의 mimetype(클라이언트가 보낸 Content-Type)으로 추출 방식을 고른다. PDF는 `pdf-parse`로 텍스트 레이어를 읽고, TXT와 Markdown은 UTF-8로 그대로 읽는다. 이 세 가지가 아니면 추출 전에 `400 "Only PDF, TXT, and Markdown are supported"`로 거절한다. curl 예시에서 `;type=`을 붙인 이유도 이 때문이다. 확장자만 `.md`인 파일을 `;type=` 없이 올리면 400이 나왔다.

텍스트가 비어 있으면 `Document has no readable text` 오류가 나고, 이 경우 응답은 `500`이다. OCR은 없다. 이 검사는 `documents`에 저장하기 전에 하므로 빈 문서 행은 남지 않는다.

### 3.2 청크 분할

공백으로 나눈 단어를 450개씩 묶고, 다음 청크는 70단어를 겹쳐 시작한다. 즉 380단어씩 앞으로 나간다.

```ts
const chunkSize = 450;
const overlap = 70;

for (let start = 0; start < words.length; start += chunkSize - overlap) {
  const chunk = words.slice(start, start + chunkSize).join(" ");
  if (chunk.trim()) chunks.push(chunk);
}
```

겹침은 경계에 걸린 문장이 두 청크에 모두 들어가게 해서 의미가 끊기는 것을 줄인다. 샘플 PDF는 9,517단어였고 26개 청크가 나왔다. 다만 단어 수 기준이라 문장이나 표 경계를 보지 않아, 검색된 청크가 문장 중간에서 시작하기도 한다.

### 3.3 임베딩과 pgvector 저장

청크마다 vllm-mlx의 `/v1/embeddings`를 호출해 1024차원 벡터를 받고, `document_chunks`에 저장한다. 받은 벡터의 길이가 `EMBEDDING_DIMENSIONS`와 다르면 오류로 중단한다. 벡터는 `[0.1,0.2,…]` 문자열로 만들어 SQL에서 `::vector`로 변환한다.

---

## 4. 데이터 모델

```text
documents                         document_chunks
─────────────────────────         ───────────────────────────────
id          uuid  PK        1    id           uuid  PK
filename    text            ──►   document_id  uuid  FK (ON DELETE CASCADE)
mime_type   text            N     chunk_index  int
text_content text                 content      text
created_at  timestamptz           embedding    vector(1024)
                                  created_at   timestamptz
```

- 문서를 지우면 `ON DELETE CASCADE`로 청크와 임베딩도 함께 지워진다.
- `documents.text_content`에 원문 전체를 저장하지만 현재 코드는 읽지 않는다.
- `document_chunks`에는 벡터 인덱스가 없고 기본키 인덱스만 있다(pgvector 0.8.7에서 확인). 검색은 모든 청크와 거리를 계산하는 정확 검색이다. 청크가 수십 개인 이 데모에서는 문제가 없고, 규모가 커지면 HNSW 같은 인덱스를 추가해야 한다.

테이블은 simple-rag 컨테이너가 시작할 때 만든다. Dockerfile이 `npm run migrate && npm start`로 실행하고, migrate는 `CREATE ... IF NOT EXISTS`라서 몇 번을 실행해도 안전하다. 단, 이미 만든 테이블의 벡터 차원은 바뀌지 않는다. 임베딩 모델을 바꿔 차원이 달라지면 테이블을 다시 만들고 문서를 다시 올려야 한다.

---

## 5. 검색과 답변 생성 흐름

### 5.1 처리 순서

```text
 질문
  │
  ▼
 ① 질문 임베딩 ········ bge-m3
  │
  ▼
 ② 벡터 검색 ·········· 코사인 유사도 상위 RAG_TOP_K(5)개
  │                     (documentId를 주면 해당 문서만)
  ▼
 ③ 유사도 임계값 ······ RAG_MIN_SIMILARITY(0.35) 미만 제외 (코사인 기준)
  │
  ▼
 ④ 리랭크 ············· bge-reranker-v2-m3로 재정렬, 상위 RAG_RERANK_TOP_N(3)개
  │                     (0이면 생략)
  ├──► /search, MCP search_documents: 청크 반환 (여기까지)
  ▼
 ⑤ 답변 생성 (/ask만) ·· 청크가 없으면 LLM 호출 없이
                        "I could not find this in the provided documents."
```

②의 쿼리는 다음과 같다.

```sql
SELECT id, document_id, chunk_index, content,
       1 - (embedding <=> $1::vector) AS similarity
FROM document_chunks
ORDER BY embedding <=> $1::vector
LIMIT $2
```

### 5.2 검색 점수 분석

저장된 문서 2건(한국어 PDF, 영어 TXT)에 질문을 보내 `/search`가 돌려준 점수를 비교했다. 아래는 결과 3개의 범위다.

| 질문 | 결과 | 코사인 | 리랭크 점수 |
|---|---|---|---|
| 입주자 모집 자격이 뭐야? | 답이 있는 청크 | 0.639~0.643 | 0.73~0.88 |
| 자동차 소유 기준이 어떻게 돼? | 답이 있는 청크 | 0.646~0.676 | 0.09~0.33 |
| 반품 가능 기간 | 1위: 답이 있는 영문 청크 | 0.796 | 0.415 |
| 반품 가능 기간 | 2~3위: 답과 무관한 PDF 청크 | 0.518~0.524 | 0.004~0.007 |
| 오늘 서울 날씨 어때? | 문서에 답이 없음(PDF 청크 3개) | 0.488~0.500 | 0.0003~0.0011 |

답이 없는 질문에도 top-k 검색은 가장 가까운 청크를 항상 반환한다. RAG의 기본 성질이며, 이 구현도 날씨 질문에 청크 3개를 반환한다.

- 코사인 임계값(0.35)으로는 막지 못하고, 관련 여부와 일치하는 것은 리랭크 점수
- 현재는 리랭크 점수로 거르지 않으며, 도입하려면 질문 세트로 기준값 검증 필요

---

## 6. MCP 서버

챗봇과 연결하기 위해 문서 검색, 목록 조회, 삭제 기능을 `/mcp` 엔드포인트 하나에서 MCP 도구 세 개로 제공한다.

| 도구 | 인자 | 하는 일 |
|---|---|---|
| `search_documents` | `question`, `documentId`(선택) | 질문과 가까운 청크를 유사도, 리랭크 점수와 함께 반환 (임계값 적용) |
| `list_documents` | 없음 | 문서 id, 파일명, 청크 수, 업로드 시각. 내용은 없음 |
| `delete_document` | `documentId` | 문서와 임베딩을 삭제 |

도구는 JSON-RPC로 호출한다. 다음은 `search_documents` 호출 예이며, 응답은 SSE 형식(`event: message` / `data: {...}`)으로 오고 `result.content[0].text`에 검색 결과 JSON이 문자열로 들어 있다(`/search`의 `data`와 같은 청크 3개).

```bash
curl -s -X POST http://localhost:3000/mcp \
  -H 'content-type: application/json' \
  -H 'accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"search_documents","arguments":{"question":"환불은 며칠 이내에 가능해?"}}}'
```

`list_documents`와 `delete_document`도 `name`과 `arguments`만 바꿔 같은 방식으로 호출한다.

### 6.1 stateless streamable-http

요청이 올 때마다 MCP 서버와 트랜스포트를 새로 만들고, 응답이 끝나면 닫는다. 세션 id를 만들지 않는다(`sessionIdGenerator: undefined`).

```ts
app.post("/mcp", async (request, reply) => {
  const server = createMcpServer();
  const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
  reply.raw.on("close", () => { void transport.close(); void server.close(); });
  await server.connect(transport);
  reply.hijack();
  await transport.handleRequest(request.raw, reply.raw, request.body);
});
```

세 도구가 모두 요청 하나로 끝나는 조회나 삭제라서 세션에 보관할 상태가 없다. 대신 서버가 클라이언트에 먼저 보내는 SSE 스트림(`GET /mcp`)과 세션 종료(`DELETE /mcp`)는 지원하지 않아 `405`를 돌려준다(`GET`으로 확인했다).

---

## 7. 벡터 저장소

벡터 검색은 Postgres 확장인 pgvector로도, 전용 벡터 저장소로도 구현할 수 있다. 이 프로젝트는 별도 벡터 DB 없이 pgvector를 쓴다. pgvector의 특징과 규모가 커질 때 부딪히는 한계를 보고, 전용 저장소(Qdrant, Milvus 등)로 옮겨야 하는 경우를 정리한다.

### 7.1 pgvector

pgvector는 Postgres에 벡터 타입과 거리 연산자를 더하는 확장이다. 이 프로젝트에서는 별도 벡터 DB 없이 이미 쓰는 Postgres에 청크와 임베딩을 함께 저장한다.

- 문서, 청크, 벡터를 한 DB에서 관리
- `ON DELETE CASCADE`와 트랜잭션으로 정합성 유지
- `WHERE`, `JOIN` 등 SQL 필터를 그대로 사용
- LiteLLM과 컨테이너를 공유해 운영 대상 추가 없음

반면 인덱스가 없으면 질문마다 모든 청크와 거리를 계산하므로 검색 시간이 청크 수에 비례해 늘어난다. 청크가 수만 개 수준까지는 문제가 되지 않지만, 규모가 커지면 근사 검색 인덱스를 추가한다.

```sql
CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops);
SET hnsw.iterative_scan = relaxed_order;  -- 0.8.0부터 지원, 필터 결과 부족 완화
```

인덱스를 추가해도 다음 한계가 남는다.

- 인덱스 차원 상한: `vector` 2,000, `halfvec` 4,000
- 근사 검색의 recall 손실과 HNSW의 메모리·빌드 비용
- 필터는 인덱스 스캔 뒤에 적용되어 결과 부족 가능
- 단일 노드 중심 구조, 샤딩 미내장
- 주 DB와 CPU·메모리 공유

필터 문제는 iterative scan 외에 부분 인덱스(필터 값이 적을 때)나 파티셔닝(값이 많을 때)으로 줄인다.

### 7.2 전용 벡터 저장소

Qdrant, Milvus 같은 전용 저장소는 벡터 검색만을 위한 확장 모델과 기능을 제공한다. 일반적으로 다음 경우에 검토한다.

- 단일 Postgres가 감당하기 어려운 벡터 규모
- 높은 동시 질의와 낮은 지연 요구
- 벡터 검색이 주 DB의 트랜잭션 성능에 영향
- 테넌트, 권한 필터를 검색 단계에서 즉시 적용
- PQ 등 고급 양자화, GPU 인덱싱, 스트리밍 수집

전용 저장소는 문서 삭제 같은 변경을 Postgres와 따로 동기화해야 하고 운영 대상도 늘어난다. 이미 Postgres를 쓰고 있고 벡터 규모가 크지 않다면 pgvector로 시작하고, 위 조건이 실제로 문제가 될 때 옮기는 편이 낫다. 검색은 `rag.ts`의 `searchDocuments` 한 곳에 모여 있어 저장소를 바꿔도 수정 범위가 이 함수의 쿼리로 한정된다.

---

## 8. 마무리

이 글에서는 문서를 청크로 나눠 pgvector에 저장하고, 벡터 검색과 리랭크를 거쳐 답을 만들며, 같은 검색을 MCP 도구로 챗봇에 연결하는 구조를 살펴보았다. 이 구현은 RAG의 흐름을 이해하기 위한 PoC이며, 운영 수준으로 가려면 다음을 고려해야 한다.

- **수집 파이프라인**: 비동기 큐, 재시도, 문서 상태 관리
- **데이터 정합성**: 중복 업로드 방지, 트랜잭션 저장
- **청크 전략**: 문장·제목 경계와 토큰 기준 분할
- **검색 품질**: 질문 세트 기반 평가, 임계값 재설계, 벡터 인덱스 또는 전용 저장소
- **답변 품질**: thinking 예산에 따른 답변 단축 검증
- **보안**: REST와 MCP의 인증 및 접근 제어

참고로, 상용 수준의 RAG 서비스는 [rag-docs.cnapcloud.com](https://rag-docs.cnapcloud.com)을 사용하여 구축할 있다.

---

## 다음 편

[4편](04-litellm.md)에서는 LiteLLM을 다룬다. 챗봇과 vllm-mlx 사이에서 모델을 라우팅하고, 로컬 모델이 실패하거나 컨텍스트를 넘으면 Gemini로 넘기는 fallback과 그 프라이버시 트레이드오프를 설명한다.