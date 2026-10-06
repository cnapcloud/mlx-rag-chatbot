# vllm-mlx 기반 RAG 챗봇 구축 (5) - LibreChat UI와 MCP 도구 연동

[4편](04-litellm.md)에서는 챗봇과 모델 서버 사이의 LiteLLM 게이트웨이를 구성했다. 이번 편에서는 사용자가 실제로 마주하는 챗봇 UI인 LibreChat을 다룬다. LibreChat은 앞 편에서 만든 구성요소를 하나로 묶는 지점이다. 모델 호출은 LiteLLM으로, 문서 검색은 MCP로 simple-rag에 연결되고, 이 연결은 모두 [`config/librechat.yaml`](../config/librechat.yaml) 한 파일에 정의된다. 이 글은 시작하기, 사용자 등록, 모델 메뉴 구성, LiteLLM 연동, MCP 서버 연동의 순서로 설정을 살펴본 뒤, 챗봇에서 업로드한 문서를 근거로 답하게 하는 과정과 에이전트의 capabilities 설정까지 다룬다.

이 글은 GitHub 저장소에 포함된 설정을 기준으로 설명한다. 저장소를 아직 클론하지 않았다면 다음과 같이 클론한다. LibreChat은 LiteLLM, simple-rag와 함께 동작하므로 전체 스택을 띄우는 방법은 [1편](01-overview.md)의 3~4절을 따른다.

```bash
git clone https://github.com/cnapcloud/mlx-rag-chatbot.git
cd mlx-rag-chatbot
```

---

## 1. LibreChat 소개

[LibreChat](https://github.com/danny-avila/LibreChat)은 오픈소스 챗봇 웹 UI다. 여러 LLM 제공자와 OpenAI 호환 API를 한 화면에서 고를 수 있고, 대화 이력 저장, 다중 사용자 계정, 에이전트, MCP 도구 연동을 갖추고 있다. 직접 서버에 올려 운영하는 셀프 호스팅 방식이며, 대화와 사용자 정보는 MongoDB에 저장된다.

- 여러 제공자와 OpenAI 호환 API를 하나의 모델 메뉴로 통합
- 사용자 계정과 로그인, 사용자별 대화 이력 보관
- 에이전트 빌더를 통한 도구와 지침의 묶음 구성
- MCP 서버에 등록된 도구의 대화 중 호출
- `librechat.yaml` 한 파일로 하는 엔드포인트, 모델, 도구 설정

이 프로젝트에서 LibreChat은 RAG를 MCP로 연결해, 업로드한 문서를 대화로 묻고 답하는 지식베이스 서비스의 화면 역할을 한다. 사용자는 로그인해 대화하고, 모델 호출은 LiteLLM 하나를 가리키는 custom 엔드포인트로 보낸다. 모델 메뉴는 `modelSpecs`로 직접 구성한다. 문서 검색은 simple-rag의 MCP 도구를 등록해 해결하며, 이 도구를 지침과 함께 묶어 두는 용도로 에이전트를 쓴다.

참고로, LibreChat에도 자체 RAG 기능이 있다. 다만 사용자가 대화에 올린 파일을 대상으로 하는 방식이라 여러 사용자가 함께 쓰는 공용 문서 지식베이스로 운영하기에는 맞지 않고, 별도 RAG API 서비스를 두어야 하며 임베딩 제공자도 정해진 범위에서 골라야 한다. 문서 저장소를 챗봇 밖에 두면 공용 문서를 대상으로 검색할 수 있고, 로컬 모델로 임베딩을 처리할 수 있으며, MCP를 지원하는 다른 클라이언트에서도 같은 검색을 재사용할 수 있다.

---

## 2. 시작하기

LibreChat은 [1편](01-overview.md)의 `docker compose up -d --build`로 전체 스택과 함께 올라간다. 첫 시작에는 `mcp-everything`의 `npm install` 때문에 몇 분이 걸린다. 로그에서 시작을 확인한 뒤 <http://localhost:3080>에 접속한다.

```bash
docker compose logs -f librechat
```

설정 파일 [`config/librechat.yaml`](../config/librechat.yaml)은 컨테이너에 읽기 전용으로 마운트되고 시작할 때 한 번 읽힌다. 수정한 뒤에는 재시작해야 반영된다.

```bash
docker compose restart librechat
```

---

## 3. 사용자 등록과 로그인

가입 허용 여부는 `compose.yaml`의 환경 변수 `ALLOW_REGISTRATION`이 정한다. 이 프로젝트는 시연을 위해 `"true"`로 두었다.

```yaml
# compose.yaml (librechat 발췌)
environment:
  MONGO_URI: mongodb://mongodb:27017/LibreChat
  ALLOW_REGISTRATION: "true"
```

스택이 올라온 뒤 다음 순서로 가입하고 로그인한다.

1. <http://localhost:3080/register>에서 이름, 이메일, 비밀번호를 입력해 가입한다. 이메일 인증은 없다.
2. <http://localhost:3080/login>에서 로그인한다.

**처음 가입한 계정이 관리자**가 된다. 따라서 여러 사람이 쓰는 환경이라면 본인 계정을 먼저 만들어 두어야 한다.

계정과 대화는 MongoDB 컨테이너가 `data/mongodb`에 저장한다. 컨테이너를 재시작하거나 다시 만들어도 계정과 대화가 유지되고, `data/mongodb`를 지우면 모두 사라진다. MongoDB 포트는 호스트에 공개하지 않으며 LibreChat 컨테이너에서만 접근한다.

이 방식은 누구나 이메일과 비밀번호로 가입하는 시연용 구성이다. 실제 운영에서는 조직의 IdP(Identity Provider)와 연계해 계정 발급과 회수를 중앙에서 관리해야 한다. LibreChat은 [OpenID Connect](https://www.librechat.ai/docs/configuration/authentication/OAuth2-OIDC), [SAML](https://www.librechat.ai/docs/configuration/authentication/SAML), [LDAP/AD](https://www.librechat.ai/docs/configuration/authentication/ldap) 연동을 지원하고, OpenID Connect는 Keycloak, Microsoft Entra ID, Auth0, AWS Cognito에 대한 설정 가이드를 제공한다.

---

## 4. 모델 노출 설정

LibreChat의 모델 선택 메뉴는 두 가지 설정으로 구성한다. 하나는 LiteLLM에 어떻게 연결하는지를 정하는 `endpoints.custom` 항목(5절)이고, 다른 하나는 메뉴에 무엇을 어떤 이름으로 보여 줄지를 정하는 `modelSpecs`다. 이 절에서는 후자를 다룬다. 두 설정은 모두 [`config/librechat.yaml`](../config/librechat.yaml)에 있다.

### 4.1 modelSpecs로 메뉴 구성

`modelSpecs`를 쓰면 엔드포인트 단계 없이 모델을 메뉴 최상단에 바로 노출할 수 있다.

```yaml
modelSpecs:
  enforce: false
  prioritize: true
  addedEndpoints: ["agents"]
  list:
    - name: qwen3-8b
      label: qwen3-8b
      default: true
      preset:
        endpoint: LiteLLM
        model: qwen3-8b
        modelLabel: qwen3-8b
    # gemini-3.5-flash-lite 항목 생략 (같은 형식)
```

- `list`: 메뉴에 나타나는 모델 항목, `preset`의 `endpoint`와 `model`이 실제 호출 대상
- `default`: 새 대화를 열 때 자동으로 선택되는 항목
- `prioritize`: 새 대화에서 기본 항목을 우선 선택
- `enforce`: `false`로 두어 목록 밖의 엔드포인트와 모델 선택도 허용
- `addedEndpoints`: 모델 항목과 함께 메뉴에 노출할 엔드포인트, 여기서는 에이전트
- `modelLabel`: 답변 작성자로 표시되는 이름

`modelLabel`을 지정하지 않으면 엔드포인트의 `modelDisplayLabel`인 `LiteLLM`이 답변 작성자로 표시되어, 어느 모델이 답했는지 알 수 없다. 그래서 항목마다 모델 이름을 그대로 적었다.

### 4.2 노출할 모델 고르기

LiteLLM에는 모델이 세 개 등록되어 있지만([4편](04-litellm.md) 3절) 메뉴에는 두 개만 나온다. `gemini-3.1-flash-lite`는 `gemini-3.5-flash-lite`가 과부하일 때 쓰는 예비 모델이므로 사용자가 직접 고르지 않게 했다. 메뉴에 나오는 항목은 4.1절의 `modelSpecs.list`가 정하므로 예비 모델은 이 목록에 넣지 않았다. 엔드포인트의 모델 목록도 LiteLLM에서 받아 오지 않고 같은 두 모델만 직접 적는다.

```yaml
endpoints:
  custom:
    - name: LiteLLM
      models:
        default:
          - qwen3-8b
          - gemini-3.5-flash-lite
        fetch: false
```

`fetch: false`는 LiteLLM에서 모델 목록을 받아 오지 않고 `default`에 적은 값을 쓴다는 뜻이다. 목록에 적는 이름은 `litellm.yaml`의 `model_name`과 정확히 같아야 한다. 다르면 LiteLLM이 해당 모델을 찾지 못해 오류를 돌려준다.

---

## 5. LiteLLM 연동

LibreChat은 LiteLLM을 OpenAI 호환 custom 엔드포인트로 등록해 사용한다. LibreChat이 호출하는 대상은 LiteLLM 하나이고, 그 뒤의 vllm-mlx와 Gemini는 LibreChat이 알지 못한다.

```yaml
endpoints:
  custom:
    - name: LiteLLM
      apiKey: "${LITELLM_MASTER_KEY}"
      baseURL: "http://litellm:4000/v1"
      # models 생략 (4.2절)
```

### 5.1 접속 주소와 키

- `baseURL`: Docker 네트워크 안의 `litellm` 서비스, 포트 4000
- `apiKey`: 환경 변수 `${LITELLM_MASTER_KEY}` 참조, 파일에는 키 미기재
- 환경 변수: `compose.yaml`이 `.env` 값을 컨테이너에 주입

```yaml
# compose.yaml (librechat 발췌)
environment:
  LITELLM_MASTER_KEY: ${LITELLM_MASTER_KEY} # librechat.yaml 의 apiKey 에서 참조
```

같은 compose 안이므로 `host.docker.internal` 없이 서비스 이름으로 접근한다. `litellm` 서비스에 주입하는 값도 같은 `.env` 값이므로 두 값이 항상 일치한다. 키를 바꾸면 두 컨테이너를 모두 다시 만들어야 한다.

### 5.2 컨텍스트 한도

LibreChat에서는 모델별 컨텍스트 한도를 `tokenConfig`로 지정한다.

```yaml
tokenConfig:
  qwen3-8b:
    prompt: 0
    completion: 0
    context: 24000 # litellm.yaml max_input_tokens(28672)보다 작게
```

- `context`: 24000, LiteLLM `max_input_tokens`(28672)보다 작게 둔 값
- `prompt`, `completion`: 100만 토큰당 비용, 로컬 모델이므로 0

토큰 수는 구성요소마다 세는 방식이 달라 같은 입력도 개수가 다르게 나온다. 한도를 앞단일수록 낮게 두어, LibreChat이 보기에 한도 안인 요청이 LiteLLM에서 거절되지 않게 했다. 한도 설계의 전체 근거는 [4편](04-litellm.md) 4절에 있다.

### 5.3 제목 생성

```yaml
titleConvo: true
titleModel: current_model
```

대화가 시작되면 LibreChat이 첫 메시지로 대화 제목을 만든다. `titleModel: current_model`은 지금 대화에 쓰는 모델에게 제목도 맡긴다는 뜻이다. 제목 생성도 LiteLLM을 거치는 별도 요청이므로 LiteLLM 로그에서는 대화 한 번에 요청이 둘로 보일 수 있다.

### 5.4 사용자 식별 헤더

```yaml
headers:
  x-litellm-end-user-id: "{{LIBRECHAT_USER_EMAIL}}"
```

요청마다 사용자 이메일을 `x-litellm-end-user-id` 헤더로 전달한다. LiteLLM이 이 값으로 사용량을 사용자별로 집계하며, 자세한 내용은 [4편](04-litellm.md) 8절에서 다루었다.

### 5.5 연동 확인

LibreChat에서 `qwen3-8b`를 고르고 메시지를 보낸 뒤, LiteLLM 로그에서 요청이 도착했는지 확인한다.

```bash
docker compose logs -f litellm
```

요청이 보이면 연동이 끝난 것이다. LiteLLM 관리 UI(<http://localhost:4000/ui>)의 Logs에서는 사용자 이메일이 `end_user`로 기록된 것도 볼 수 있다. 답변이 오지 않고 오류가 난다면 다음 항목을 확인한다.

- `.env`의 `LITELLM_MASTER_KEY`와 컨테이너에 주입된 값의 일치 여부
- 모델 이름과 `litellm.yaml`의 `model_name`의 일치 여부
- `docker compose logs litellm`의 오류 메시지

---

## 6. MCP 서버 연동

LibreChat이 simple-rag의 검색 도구를 쓰려면 `librechat.yaml`에 MCP 서버를 등록해야 한다. 이 프로젝트는 서버를 둘 등록한다. 하나는 문서 검색용 `simple-rag`([3편](03-simple-rag.md) 6절), 다른 하나는 동작 확인용 `everything`이다.

### 6.1 서버 등록

```yaml
mcpServers:
  everything:
    type: streamable-http
    url: "http://host.docker.internal:3001/mcp"
    sseReadTimeout: 3600000

  simple-rag:
    type: streamable-http
    url: "http://host.docker.internal:3000/mcp"
```

- `type`: 전송 방식, 두 서버 모두 `streamable-http`
- `url`: MCP 엔드포인트, 호스트 포트를 `host.docker.internal`로 접근
- `sseReadTimeout`: SSE 스트림 유휴 허용 시간(밀리초), 3600000은 1시간

`everything`은 MCP 공식 참조 서버(`@modelcontextprotocol/server-everything`)로, 시험용 도구를 모아 둔 서버다. 원래 표준 입출력으로 동작하는 서버이므로 `mcp-proxy`로 감싸 HTTP 엔드포인트로 노출했다. 문서 검색 없이 MCP 연결 자체가 되는지 확인할 때 쓴다.

### 6.2 전송 방식 선택

LibreChat은 원격 MCP 서버에 SSE, streamable-http 등으로 연결한다. 처음에는 `everything`을 SSE로 연결했는데, 유휴 상태에서 연결이 끊겨 계속 재연결하는 현상이 있었다. 그래서 `streamable-http`로 바꾸었고, `sseReadTimeout`으로 SSE 스트림의 유휴 허용 시간을 1시간으로 늘려 두었다.

`simple-rag`는 [3편](03-simple-rag.md) 6.1절에서 본 것처럼 stateless 서버라 세션과 SSE 스트림이 없고 요청마다 `POST /mcp`로 처리된다. 유휴 연결이 끊기는 문제가 생기지 않으므로 `sseReadTimeout`을 두지 않았다.

### 6.3 접근 허용 주소

LibreChat은 MCP 서버가 내부망 주소를 가리키면 기본적으로 막는다. 내부 서비스로 요청을 보내게 하는 우회 공격을 막기 위한 설정이다. 그런데 이 프로젝트의 MCP 서버는 모두 호스트의 `host.docker.internal`에 있으므로, 이 주소를 명시적으로 허용해야 한다.

```yaml
mcpSettings:
  allowedAddresses:
    - "host.docker.internal:3001"
    - "host.docker.internal:3000" # simple-rag
```

- 항목 형식: `host:port`
- 새 서버 추가 시: `mcpServers` 등록과 별개로 이 목록에도 주소 추가
- 목록에서 빠진 경우: 서버를 등록해 두어도 연결 차단

MCP 서버를 새로 등록했는데 도구가 나타나지 않는다면 이 목록부터 확인한다.

### 6.4 기동 순서

LibreChat은 시작할 때 등록된 MCP 서버에 연결을 시도한다. `everything`이 아직 준비되지 않은 상태에서 LibreChat이 먼저 뜨면 연결이 실패한다. 그래서 `compose.yaml`에서 `mcp-everything`가 healthy가 된 뒤에 LibreChat이 시작하도록 했다.

```yaml
# compose.yaml (librechat 발췌)
depends_on:
  mcp-everything:
    condition: service_healthy # MCP 서버가 뜬 뒤에 librechat 시작
```

`mcp-everything`는 시작할 때 `npm install`을 하므로 첫 기동에는 몇 분이 걸리고, 그동안 LibreChat은 올라오지 않는다([1편](01-overview.md) 4.4절). `simple-rag`에는 이 조건을 걸지 않았다. 요청마다 새로 연결하는 stateless 서버라 유지해야 할 연결이 없다.

### 6.5 연결 확인

설정을 고친 뒤에는 컨테이너를 재시작하고 로그에서 MCP 연결을 확인한다.

```bash
docker compose restart librechat
docker compose logs librechat | grep -i mcp
```

LibreChat 화면에서는 새 대화를 열었을 때 입력창에 MCP 서버를 고르는 메뉴가 나타나고, 거기에 `simple-rag`와 `everything`이 보이면 등록이 끝난 것이다. 메뉴의 위치와 이름은 LibreChat 버전에 따라 다르다.

---

## 7. 챗봇에서 RAG 사용하기

### 7.1 문서 준비와 도구 켜기

검색할 문서가 있어야 하므로, 아직 올리지 않았다면 [1편](01-overview.md) 5.1절의 명령으로 샘플 문서를 올린다.

```bash
curl -F "file=@simple-rag/samples/cheongajin-notice.pdf;type=application/pdf" http://localhost:3000/documents
```

그다음 LibreChat에서 다음 순서로 진행한다.

1. 로그인해 새 대화를 연다.
2. 모델 메뉴에서 `qwen3-8b`를 고른다(기본값이다).
3. 입력창의 MCP 메뉴에서 `simple-rag`를 활성화한다.
4. 문서에 대해 질문한다.

```text
업로드한 모집공고에서 입주 자격을 문서에서 찾아서 알려줘
```

도구를 켜지 않으면 모델은 도구가 있는 줄 모르고 자신의 지식으로만 답한다.

### 7.2 호출 흐름

질문을 보내면 다음 순서로 처리된다.

```text
① 질문 ─► LibreChat ─► LiteLLM ─► qwen3-8b       (질문 + 도구 목록)
② qwen3-8b: search_documents 호출 결정
③ LibreChat ─► simple-rag (POST /mcp)            ─► 검색된 청크 반환
④ LibreChat ─► LiteLLM ─► qwen3-8b               (청크를 근거로 답변 생성)
```

모델은 질문을 읽고 도구를 부를지 스스로 정한다. 검색이 필요하다고 판단하면 `search_documents`를 호출하고, LibreChat이 이를 simple-rag로 전달해 얻은 청크를 다시 모델에 넘긴다. 모델은 그 청크를 근거로 답한다. 이 과정에서 LiteLLM 요청이 두 번 이상 발생한다.

<p align="center">
  <img src="images/rag-mcp.png" alt="LibreChat에서 simple-rag MCP 도구로 search_documents를 호출한 화면" width="720">
</p>

대화 화면에는 도구 호출 내역이 표시되므로, 어떤 질문으로 검색했고 어떤 청크가 돌아왔는지 펼쳐 볼 수 있다. 답변이 이상할 때는 먼저 이 내역으로 검색이 잘못됐는지 모델이 잘못 답했는지 가른다.

### 7.3 도구 설명이 호출을 이끈다

모델이 어떤 도구를 부를지는 MCP 서버가 알려 주는 도구 설명에 달려 있다. simple-rag의 설명은 언제 쓰고 언제 쓰지 말아야 하는지를 분명히 적어 두었다([`mcp.ts`](../simple-rag/src/mcp.ts)).

```ts
// search_documents 의 description (발췌)
description:
  // 앞부분 생략: 검색 대상과 반환값 설명
  "Use it for any question about what the documents say. Searches all documents unless documentId is given. Not for listing files (use list_documents). " +
  "Answer only from the returned chunks; if nothing relevant is found, say so.",
```

작은 모델은 선택 인자에 빈 문자열이나 임의의 값을 채워 보내는 경우가 있다. 이를 막으려고 `documentId`는 스키마에서 선택 항목으로 두고 설명에 생략하라고 적었으며, 핸들러에서 UUID 형식을 다시 검증한다.

### 7.4 도구를 부르지 않을 때

로컬 `qwen3-8b`는 도구 호출이 불안정할 수 있다. 도구를 켰는데도 호출하지 않고 일반 지식으로 답하는 경우가 있다.

- 질문에 "문서에서 찾아서"처럼 검색을 직접 지시하는 표현 추가
- 같은 대화에서 모델을 `gemini-3.5-flash-lite`로 바꿔 재시도
- LibreChat 화면에 도구 호출 내역이 있는지 확인

모델을 Gemini로 바꾸면 검색된 문서 청크가 외부 API로 전송된다. 이 프로젝트의 전제와 충돌하는 부분이므로 민감한 문서에서는 피해야 하며, 이 상충 관계는 [4편](04-litellm.md) 6절에서 다루었다.

### 7.5 삭제 도구 주의

simple-rag는 검색 외에 `list_documents`, `delete_document`도 MCP 도구로 노출한다. 도구를 켜면 이 세 개가 함께 켜지고, 사용자가 "문서를 지워 줘"라고 하면 모델이 삭제를 호출할 수 있다. 삭제는 되돌릴 수 없다. `delete_document`의 설명에 사용자가 명시적으로 요청할 때만 호출하라고 적었지만, 이것은 모델의 판단에 기대는 안내일 뿐 접근 통제가 아니다. 지금 구성에서는 simple-rag에 인증이 없고 문서도 사용자별로 나뉘지 않아, 도구를 켠 어느 사용자든 모든 문서를 지울 수 있다.

---

## 8. 에이전트 capabilities 설정

에이전트는 모델, 지침, 도구를 하나로 묶어 저장해 두는 기능이다. 매번 대화마다 MCP 도구를 켜는 대신, 문서 검색용 에이전트를 만들어 두고 그것을 골라 대화할 수 있다. MCP 메뉴에서 도구가 보이지 않을 때의 대안이기도 하다.

### 8.1 설정

에이전트 엔드포인트의 기능은 `librechat.yaml`의 `endpoints.agents`에서 정한다.

```yaml
endpoints:
  agents:
    disableBuilder: false
    capabilities: ["tools"]
```

- `disableBuilder`: `false`, 사용자의 에이전트 직접 생성 허용
- `capabilities`: 에이전트에 부여할 수 있는 기능의 범위, 목록 밖의 기능은 선택 불가

4.1절의 `addedEndpoints: ["agents"]` 덕분에 모델 메뉴에서 모델 항목과 함께 에이전트를 고를 수 있다.

### 8.2 capabilities 항목

이 프로젝트는 MCP 도구만 필요하므로 `tools` 하나만 열었다. 필요한 기능만 열어 두는 편이 안전하다. 제외한 기능은 다음과 같다.

- `web_search`: 웹 검색, 검색어가 외부 검색 서비스로 전달되므로 제외
- `execute_code`: 모델이 만든 코드의 실행, 문서와 대화를 다루므로 제외
- `file_search`: 대화에 올린 파일 검색, 문서 검색은 simple-rag가 담당
- `actions`, `skills`, `artifacts`: 이 시리즈에서 쓰지 않는 기능

### 8.3 문서 검색 에이전트 만들기

1. 에이전트 빌더를 연다. 메뉴 위치는 LibreChat 버전에 따라 다르다.
2. 이름과 지침을 적는다. 예: "업로드한 문서에서 답을 찾는다. 항상 `search_documents`로 검색하고, 검색 결과에 없는 내용은 모른다고 답한다."
3. 모델은 엔드포인트 `LiteLLM`과 `qwen3-8b`를 고른다.
4. 도구에서 MCP 서버 `simple-rag`를 추가한다.
5. 저장한 뒤 새 대화에서 에이전트를 골라 질문한다.

지침에 검색을 강제하는 문장을 넣어 두면, 7.4절에서 본 도구 미호출을 줄일 수 있다. 에이전트는 도구가 이미 붙어 있으므로 대화마다 MCP 메뉴를 켤 필요도 없다.

### 8.4 chatMenu로 도구 숨기기

MCP 서버 항목에는 `chatMenu` 설정이 있다. `false`로 두면 일반 대화의 MCP 메뉴에서는 서버가 사라지고, 저장된 에이전트에 추가된 경우에만 쓸 수 있다.

```yaml
mcpServers:
  simple-rag:
    type: streamable-http
    url: "http://host.docker.internal:3000/mcp"
    chatMenu: false # 에이전트에서만 사용
```

이 프로젝트의 설정에는 넣지 않았다. 다만 7.5절의 삭제 문제 때문에, 도구를 관리자가 만든 에이전트로만 노출하려는 환경에서는 쓸 만한 선택지다.

---

## 9. 마무리

이 글에서는 LibreChat을 시리즈의 프런트엔드로 두고, 사용자 등록, `modelSpecs`로 구성한 모델 메뉴, LiteLLM custom 엔드포인트, simple-rag와 `everything`의 MCP 연동, 에이전트 capabilities를 설정했다. 이것으로 브라우저에서 질문하면 LiteLLM을 거쳐 vllm-mlx가 답하고, 문서가 필요하면 MCP로 simple-rag의 검색 결과를 근거로 삼는 RAG 챗봇의 전체 흐름이 완성되었다.

실제 운영 환경을 구성한다면 다음을 고려해야 한다.

- 조직 IdP 연계를 통한 계정 관리(3절)
- `CREDS_KEY`, `JWT_SECRET`, DB 비밀번호 등 placeholder 교체([1편](01-overview.md) 4.3절)
- simple-rag의 REST와 MCP 인증, 사용자별 문서 접근 범위 분리
- 삭제 도구의 사용 제한 또는 에이전트 전용 노출
- 도구 호출이 불안정한 로컬 모델의 대안 검토와 그 경우 외부 전송 범위 점검
- 동시 사용자 증가에 따른 vllm-mlx 처리량과 LibreChat 한도 재조정
- 시험용 MCP 서버(`everything`)의 운영 환경 제외
