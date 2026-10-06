# vllm-mlx 기반 RAG 챗봇 구축 (2) - 모델 서버 구성과 메모리 설계

[1편](01-overview.md)에서 전체 구성을 띄워 봤다. 이번 편은 그중 호스트에서 도는 vllm-mlx를 다룬다. 모델을 띄우는 일은 쉬웠고, 어려웠던 것은 **한정된 메모리 안에서 긴 RAG 프롬프트를 감당하게 만드는 일**이었다. 각 설정값이 왜 그 값인지, 메모리를 어떻게 나눠 썼는지를 정리한다.

이 글은 GitHub 저장소에 포함된 코드를 기준으로 설명한다. 저장소를 아직 클론하지 않았다면 다음과 같이 클론한다. 사전 준비와 전체 스택 실행은 [1편](01-overview.md)의 3~4절을 따른다.

```bash
git clone https://github.com/cnapcloud/mlx-rag-chatbot.git
cd mlx-rag-chatbot
```

> 이 글의 설정은 M4 MacBook Pro(48GB 메모리, GPU 16코어)를 기준으로 한다. 모든 설정과 테스트는 이 환경을 기준으로 진행했다.

---

## 1. vllm-mlx 소개

vllm-mlx는 Apple Silicon에서 LLM을 서빙하는 추론 서버다. GPU 서버에서 쓰는 [vLLM](https://github.com/vllm-project/vllm)의 서빙 방식을 Mac에서 쓸 수 있게 하되, 연산은 CUDA 대신 Apple의 [MLX](https://github.com/ml-explore/mlx) 위에서 Metal GPU로 처리한다. Hugging Face의 MLX 모델을 변환 없이 그대로 받아 쓴다.

Mac용 로컬 LLM 도구가 대체로 한 사람의 채팅에 맞춰져 있는 데 비해, vllm-mlx는 서버로 쓰기 위한 다음 기능을 제공한다.

- continuous batching, paged KV cache, prefix caching 등 서버급 추론 최적화
- OpenAI `/v1/*`와 Anthropic `/v1/messages` 호환 API. 기존 클라이언트를 주소만 바꿔 연결할 수 있다.
- 텍스트, 이미지, 영상, 오디오, 임베딩, 리랭크를 한 서버에서 서빙
- 도구 호출 파서와 reasoning 모델 지원

참고로 vllm-mlx는 vLLM과는 별개 프로젝트다. OpenAI 호환 API와 설계 개념(continuous batching, KV cache)은 닮았지만, 서버 옵션 이름(vLLM의 `--max-model-len`에 해당하는 옵션이 여기서는 `--max-kv-size`다)과 모델 포맷(MLX 변환본), 실행 환경(macOS 전용)이 달라 vLLM 설정을 그대로 옮길 수 없다.

이 글에서는 vllm-mlx를 로컬 RAG의 모델 서버로 사용한다. 위 기능 중 채팅, 임베딩, 리랭크 모델을 사용하며(2절), 이미지·영상·오디오는 사용하지 않는다. Apple Silicon은 CPU와 GPU가 Unified Memory를 공유하기 때문에 별도의 VRAM 용량에 묶이지 않고 비교적 큰 모델을 노트북에서도 구동할 수 있다. 다만 이 메모리를 모델, KV cache, OS가 나눠 써야 하므로, 긴 RAG 프롬프트를 감당하려면 메모리 설계가 필요하다.

---

## 2. 서빙 모델

vllm-mlx는 서버 하나에 채팅, 임베딩, 리랭크 모델을 함께 올려 `:8100` 하나로 응답하도록 구성할 수 있다.

### 2.1 Qwen3-8B-4bit

`mlx-community/Qwen3-8B-4bit`는 답변을 생성하는 채팅 모델이다. 답변 전에 `<think>` 구간으로 추론을 먼저 출력하는 reasoning 모델이라, 컨텍스트와 지연을 설계할 때 추론 토큰까지 계산에 넣어야 한다.

### 2.2 bge-m3

`mlx-community/bge-m3-mlx-fp16`는 문서 청크와 질문을 1024차원 벡터로 바꾸는 임베딩 모델이다. 문서 업로드와 검색이 모두 이 모델을 거친다.

### 2.3 bge-reranker-v2-m3

`BAAI/bge-reranker-v2-m3`는 검색 결과를 재정렬하는 리랭크 모델이다. 검색에 연결해 사용한다.

세 모델은 `/v1/models`에 함께 나온다.

```bash
curl http://localhost:8100/v1/models
```

RAG는 질문을 벡터로 바꾸고(임베딩), 검색 결과를 재정렬하고(리랭크), 답변을 생성하는(채팅) 세 역할이 모두 필요하다. 이 세 모델을 한 서버에 올리면 프로세스 하나와 포트 하나(`:8100`)로 RAG에 필요한 모델 호출을 모두 받을 수 있다. LiteLLM과 simple-rag가 같은 주소를 바라보므로 모델 서버를 따로 관리할 필요가 없다.

---

## 3. 시작하기

### 3.1 Makefile

실행 설정은 저장소 루트의 [Makefile](../Makefile) 하나에 모아 두었다.

```bash
make install     # hf, vllm-mlx 설치 (uv tool)
make download    # 세 모델 다운로드 (최초 1회)
make run         # 서버 시작 (포그라운드)
make stop        # 서버 종료
```

`make run`은 상단 변수를 `vllm-mlx serve`의 옵션으로 넘긴다.

```makefile
run:
	vllm-mlx serve $(VLLM_MODEL) --port $(VLLM_PORT) --host $(VLLM_HOST) \
		--continuous-batching --max-num-seqs $(VLLM_MAX_SEQS) \
		--max-kv-size $(VLLM_CONTEXT) \
		--chunked-prefill-tokens $(VLLM_CHUNK) \
		--prefill-step-size $(VLLM_PREFILL_STEP) --mllm-prefill-step-size $(VLLM_PREFILL_STEP) \
		--reasoning-parser qwen3 \
		--default-thinking-token-budget $(VLLM_THINK_BUDGET) \
		--max-tokens $(VLLM_MAX_TOKENS) \
		--cache-memory-mb $(VLLM_CACHE_MB) \
		--embedding-model $(VLLM_EMBED_MODEL) \
		--rerank-model $(VLLM_RERANK_MODEL)
```

`--continuous-batching`으로 여러 요청을 한 배치로 처리하고, `--reasoning-parser qwen3`으로 `<think>` 구간을 답변과 분리한다.

### 3.2 모델별 역할

세 모델은 Makefile의 변수 하나씩에 대응하고, 서버에 서로 다른 방식으로 전달된다.

| 변수 | 전달 방식 | 역할 |
|---|---|---|
| `VLLM_MODEL` | `vllm-mlx serve` 인자 | 채팅. LiteLLM을 거쳐 LibreChat과 `/ask`가 사용 |
| `VLLM_EMBED_MODEL` | `--embedding-model` | 임베딩. simple-rag가 업로드와 검색에서 사용 |
| `VLLM_RERANK_MODEL` | `--rerank-model` | 리랭크. 검색 결과 재정렬에 사용 |

채팅 모델이 서버의 기본 모델이고, 임베딩과 리랭크는 같은 서버에 추가로 올라가는 모델이다. 채팅과 임베딩은 OpenAI 호환 API로 바로 호출해 볼 수 있다.

```bash
# 채팅
curl http://localhost:8100/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mlx-community/Qwen3-8B-4bit",
       "messages":[{"role":"user","content":"안녕"}],
       "max_tokens":256}'

# 임베딩 (bge-m3, 1024차원)
curl http://localhost:8100/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"model":"mlx-community/bge-m3-mlx-fp16","input":"입주자 모집 자격"}'
```

`/ask`나 챗봇에서 문제가 생겼을 때 이렇게 서버를 직접 불러 보면 모델 쪽 문제인지 그 위 계층의 문제인지 가를 수 있다.

### 3.3 포트 및 환경 설정

나머지 변수와 옵션의 대응은 다음과 같다. 아래 절들은 이 값들이 어떻게 정해졌는지를 따라간다.

| 변수 | 값 | 옵션 | 의미 |
|---|---|---|---|
| `VLLM_PORT` | 8100 | `--port` | 서비스 포트 |
| `VLLM_HOST` | 0.0.0.0 | `--host` | 바인딩 주소 (4절 참고) |
| `VLLM_MAX_SEQS` | 4 | `--max-num-seqs` | 동시 처리 시퀀스 수 |
| `VLLM_CONTEXT` | 32768 | `--max-kv-size` | 시퀀스당 KV cache 한도(토큰) |
| `VLLM_CHUNK` | 2048 | `--chunked-prefill-tokens` | 스케줄러 step당 prefill 토큰 상한 |
| `VLLM_PREFILL_STEP` | 4096 | `--prefill-step-size` | prefill 계산 청크 크기 |
| `VLLM_MAX_TOKENS` | 4096 | `--max-tokens` | 요청에 `max_tokens`가 없을 때의 생성 상한 |
| `VLLM_CACHE_MB` | 12288 | `--cache-memory-mb` | prefix cache 메모리 상한 |
| `VLLM_THINK_BUDGET` | 128 | `--default-thinking-token-budget` | reasoning 토큰 상한 |

---

## 4. 서버 바인딩과 로컬 네트워크 구성

### 4.1 0.0.0.0 바인딩

vllm-mlx만 Docker 밖에서 돈다. MLX가 macOS의 Metal을 쓰기 때문에 컨테이너 안에서는 GPU를 쓸 수 없다. 그러면 컨테이너(LiteLLM, simple-rag)가 호스트의 서버를 불러야 한다. 문제는 기본 바인딩인 `localhost`로는 컨테이너가 들어올 수 없다는 점이다. 그래서 `VLLM_HOST`를 `0.0.0.0`으로 열었다.

대가가 있다. `0.0.0.0`은 **같은 네트워크의 다른 기기에서도 8100에 접근할 수 있다**는 뜻이다. 서버를 띄우면 이런 경고가 나온다.

```text
SECURITY WARNING: Server running without API key authentication. Anyone can access the API.
```

바인딩 주소는 리스닝 소켓으로 확인할 수 있다. `*:8100`이면 모든 인터페이스에 열려 있다는 뜻이다.

```bash
lsof -nP -iTCP:8100 -sTCP:LISTEN
```

`--api-key` 옵션으로 인증을 걸 수 있지만, 그러면 LiteLLM 쪽 `api_key: not-needed`도 같이 바꿔야 한다. 이 프로젝트는 개발용이라 인증 없이 두었다. 카페나 공용 Wi-Fi에서는 방화벽으로 8100을 막거나 서버를 끄는 편이 안전하다.

### 4.2 컨테이너에서 접근

컨테이너는 호스트의 서버를 `host.docker.internal`로 부른다.

```text
컨테이너 ──► host.docker.internal:8100 ──► 호스트의 vllm-mlx
```

`compose.yaml`은 `extra_hosts: host.docker.internal:host-gateway`로 이 이름을 풀어 준다. 이 경로를 쓰는 쪽은 채팅 요청을 보내는 LiteLLM과 임베딩 요청을 보내는 simple-rag다.

컨테이너에서 실제로 닿는지는 임시 컨테이너로 확인할 수 있다. 이 명령이 실패하면 `VLLM_HOST`가 `0.0.0.0`인지부터 본다.

```bash
docker run --rm --add-host host.docker.internal:host-gateway \
  curlimages/curl -s http://host.docker.internal:8100/v1/models
```

---

## 5. 메모리 설계 (48GB 예시)

Apple Silicon은 CPU와 GPU가 메모리를 나눠 쓰는 Unified Memory 구조라서, 모델과 KV Cache, Prefix Cache, OS가 모두 같은 메모리에서 경쟁한다. 이 절에서는 RAM 48GB에서 Qwen3-8B-4bit, bge-m3, bge-reranker-v2-m3를 한 서버에 올릴 때 메모리를 어떻게 나눠 쓰는지 보여 준다. 고정되는 것은 모델 가중치뿐이고, KV Cache와 Prefix Cache의 크기는 설정으로 정해야 한다.

### 5.1 전체 현황

| 요소 | 크기 | 크기를 정하는 것 |
|---|---|---|
| 모델 가중치 (3종) | 약 8~9GiB (추정) | 파라미터 수 × 정밀도. 파일 용량은 Qwen3-8B-4bit 4.3, bge-m3 1.1, bge-reranker 2.1로 합계 약 7.5GiB이고, 여기에 런타임 오버헤드를 더했다. |
| KV Cache | 약 18GiB (최악) | 토큰당 KV × 컨텍스트 × 동시 시퀀스 (32K × 4) |
| Prefix Cache | 12GiB (상한) | `--cache-memory-mb` |
| OS 및 기타 | 수 GiB (미측정) | macOS, Docker와 컨테이너, 다른 앱 |
| **합계 (최악)** | **약 45GiB 안팎** | |

합계는 네 시퀀스가 동시에 32K를 채우고 Prefix Cache도 가득 차는 **최악의 경우**를 가정한 값이며, 평소 사용량은 이보다 훨씬 적다. 참고로 서버를 띄우고 몇 번 요청한 직후 `vllm-mlx` 프로세스의 RSS(프로세스가 점유한 물리 메모리)는 약 9.7GB였고, 한 시점의 값이라 최대 사용량은 아니다.

### 5.2 KV Cache와 Prefill

KV Cache는 처리하는 토큰마다 쌓이므로 **컨텍스트가 길수록, 동시 요청이 많을수록** 커진다. 토큰을 생성하는 채팅 모델(Qwen3-8B)만 계산한다. 4bit 양자화는 가중치에만 적용되며, KV Cache는 모델의 연산 정밀도인 `bfloat16`(2바이트)로 저장된다(mlx-lm에서 확인했고, `--kv-cache-quantization`은 사용하지 않았다). 토큰당 KV는 모델 구조(`config.json`)에서 계산한다.

```python
layers, kv_heads, head_dim = 36, 8, 128                          # Qwen3-8B
per_token = 2 * layers * kv_heads * head_dim * 2                 # K,V × bfloat16 2바이트
print(per_token / 1024, "KiB/token")                             # 144.0
```

여기에 컨텍스트와 동시 시퀀스 수를 곱한 크기는 다음과 같다(구조에서 계산한 추정치).

| 컨텍스트 | 시퀀스 1개 | 2개 | 4개 |
|---|---|---|---|
| 8K | 1.1GiB | 2.3GiB | 4.5GiB |
| 16K | 2.3GiB | 4.5GiB | 9.0GiB |
| 32K | 4.5GiB | 9.0GiB | **18GiB** |

RAG는 검색된 문서가 프롬프트에 들어가고 Qwen3는 추론 토큰까지 쓰므로 컨텍스트를 `VLLM_CONTEXT=32768`로 잡았다. 개인용이라 동시 요청이 많지 않다고 보고 `VLLM_MAX_SEQS`는 4로 정했다. 모든 요청이 32K를 쓰는 것은 아니지만, 최악의 경우(표의 굵은 칸)를 알아야 값을 정할 수 있다.

RAG는 입력이 수천~수만 토큰이라 첫 토큰이 나오기까지의 시간(TTFT)을 Prefill이 좌우한다. 검색이 가져오는 청크 수와 길이가 곧 응답 지연이자 KV Cache 사용량이며, `RAG_TOP_K`와 청크 크기가 이 서버의 부하와 직결되는 이유다. 긴 프롬프트 하나가 Prefill을 독점해 생성 중인 다른 요청이 멈추지 않도록 두 값을 두었다.

```makefile
VLLM_CHUNK := 2048         # --chunked-prefill-tokens: 스케줄러 step당 prefill 토큰 상한
VLLM_PREFILL_STEP := 4096  # --prefill-step-size: prefill 계산 청크 크기
```

앞의 값은 긴 입력을 나눠 스케줄링해 다른 요청의 Decode가 끼어들 틈을 만들고, 뒤의 값은 실제 계산 단위다. 이 두 값은 정답을 찾은 값이 아니라 출발점이며, 입력 길이별 TTFT를 재 보며 확인해야 한다.

### 5.3 Prefix Cache와 생성 제어

| 값 | 선택 | 이유 |
|---|---|---|
| `VLLM_CACHE_MB` | 12288 | 기본 한도로는 30K 토큰 프롬프트가 저장되지 않음 |
| `VLLM_MAX_TOKENS` | 4096 | 반복 루프로 출력이 끝나지 않는 것을 방지 |
| `VLLM_THINK_BUDGET` | 128 | 추론 길이에 상한을 두어 응답 시간을 제한 |

- **Prefix Cache**
  - 반복되는 시스템 프롬프트와 문서 컨텍스트의 KV 재사용
  - 기본 한도는 시작 시점 여유 메모리의 20%이며, 한도를 넘는 항목(예: 30K 토큰, 약 4.1GiB)은 저장 거부
  - 컨텍스트 크기와 보관할 프리픽스 수를 고려한 직접 설정 필요
  - 이 구성: 32K 시퀀스 KV(약 4.5GiB) 2개 이상을 수용하는 12GiB
- **출력 토큰 (`max_tokens`)**
  - 모델이 생성하는 출력 토큰의 상한 (답변과 `<think>` 추론 포함)
  - 입력 토큰 + 출력 토큰 ≤ 컨텍스트 (서버가 강제하지는 않으므로 앞단에서 관리, 6.4절 참고)
  - 서버 기본값 32768은 컨텍스트와 동일하여, 반복 루프 시 창이 찰 때까지 생성이 이어질 수 있음 → 4096으로 제한
  - 요청에 `max_tokens`가 없을 때만 적용
- **Thinking Budget**
  - 추론은 정확도를 높이지만 답변보다 먼저 생성되어 응답 시간 증가
  - 추론이 128토큰을 넘으면 end-think를 강제로 넣고 답변으로 전환
  - 예산을 줄이면 답변이 얕아질 수 있음 (영향 미측정)
  - 요청의 `thinking_token_budget`이 우선하므로 복잡한 질문에만 예산 확대 가능

```bash
curl http://localhost:8100/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mlx-community/Qwen3-8B-4bit",
       "messages":[{"role":"user","content":"두 공고의 자격 조건을 비교해 줘"}],
       "thinking_token_budget": 1024}'
```

### 5.4 컨텍스트 한도 정렬

서버만 설정해서는 끝나지 않는다. LiteLLM과 LibreChat도 같은 한도를 알아야 사용자가 한도 초과 에러를 보지 않는다.

| 구성요소 | 설정 | 값 | 계산 | 다루는 편 |
|---|---|---|---|---|
| vllm-mlx | `VLLM_CONTEXT` | 32768 | 모델 컨텍스트 창 | 2편 (이 글) |
| LiteLLM | `max_input_tokens` | 28672 | 32768 − 4096 (출력 예약) | 4편 |
| LibreChat | `tokenConfig` context | 24000 | LiteLLM 한도보다 작게, 여유분 | 5편 |

구성요소마다 토큰을 세는 방식이 달라 같은 입력도 토큰 수가 다르게 계산되므로, 앞단일수록 한도를 낮게 잡아 여유분을 두어 설정했다. 다만 이 여유분이 적절한지는 아직 확인하지 못했다. 참고로 LibreChat은 한도에 도달하면 대화 요약(summarize)이 실행되도록 구성했다.

클라이언트가 4096보다 큰 `max_tokens`를 보내면 입력 한도 계산이 깨진다. 각 한도가 구성요소에서 어떻게 쓰이는지는 4편(LiteLLM)과 5편(LibreChat)에서 이어서 다룬다.

---

## 6. 성능 측정과 이슈

### 6.1 테스트 환경

- 기기: **M4 / 48GB / GPU 16코어**
- 서버: vllm-mlx v0.4.1, Qwen3-8B-4bit, `VLLM_CONTEXT` 32768, `VLLM_MAX_SEQS` 4, `VLLM_CACHE_MB` 12288
- 호출: 8100 직접 호출 (LiteLLM, LibreChat 우회)
- 동시 사용자: **3명까지만** 측정 (동시 요청은 2명, 멀티턴 세션은 3명)

### 6.2 테스트 시나리오

| 시나리오 | 방법 |
|---|---|
| 입력 길이별 | 1K~30K 무작위 텍스트(캐시 미적중)를 1명, 2명이 같은 길이로 동시에 요청, 출력 최대 256토큰 |
| 멀티턴 세션 | 이전 대화 전체를 다시 보내며 턴마다 약 3.1K토큰씩 31.7K까지 증가, 1~3명 동시, 출력 최대 128토큰 |

칸마다 1회 측정이다. 입력은 출력을 포함해 32768을 넘지 않도록 32K 대신 약 30K까지 올렸다.

### 6.3 결과

**입력 길이별 (캐시 미적중, 1명 vs 2명 동시)**

| 입력 토큰 | TTFT 1명 | TTFT 2명 | Decode 1명 | Decode 2명(사용자당) | 합산 Prefill 1명 → 2명 |
|---|---|---|---|---|---|
| 1,035 | 2.9s | 6.4s | 60.0 t/s | 23~26 t/s | 355 → 323 t/s |
| 4,035 | 11.7s | 23.5s | 49.5 t/s | 31~40 t/s | 344 → 344 t/s |
| 8,034 | 25.0s | 50.1s | 41.1 t/s | 26~33 t/s | 322 → 321 t/s |
| 16,034 | 56.7s | 113.5s | 28.6 t/s | 21 t/s | 283 → 283 t/s |
| 30,035 | 130.4s | 276.2s | 19.7 t/s | 11~14 t/s | 230 → 218 t/s |

- 입력이 길수록 Prefill과 Decode 속도 모두 저하
- 동시 사용자 증가 시 Prefill을 나눠 쓰므로 사용자당 TTFT가 사용자 수에 비례해 증가
- 캐시 적중이 없는 긴 입력의 동시 처리는 이 환경에서 실익이 작을 것으로 추정 (3명은 미측정)

**멀티턴 세션과 prefix cache**

| 사용자 | 적중 | 최종 턴 TTFT |
|---|---|---|
| 1명 (27K까지) | 매 턴 적중 | 18.4s |
| 2명 동시 | 매 턴 2/2 적중 | 약 45s |
| 3명 동시 | 매 턴 3/3 적중 | 약 89s |

- 직전 프롬프트 전체 재사용(적중 토큰이 직전 프롬프트 토큰 수와 일치), 새로 붙은 약 3.1K토큰만 계산
- 캐시 없이 2명이 30K를 보낸 경우(276초) 대비 2명 31.7K의 최종 턴은 약 45초로 단축 (입력 길이는 다름)
- 캐시가 가득 차 오래된 턴의 엔트리가 밀려나도 마지막 턴 전까지 현재 세션의 적중에는 영향 없음

**메모리**

- 캐시 엔트리는 토큰당 144KiB와 일치 (31.7K 토큰 엔트리 2개 8,938MB, 32K 시퀀스 하나 약 4.5GiB)
- 3명이 31.7K까지 갈 때 활성 메모리(prefix cache 포함) 최대 36.7GB, 피크 39.95GB (48GB 중)

### 6.4 이슈

- **입력 길이 제한 미지원**
  - `--max-kv-size`를 넘는 입력도 오류 없이 처리되고 앞부분이 유지됨 (37K 확인)
  - 설계상으로는 한도를 넘으면 오래된 앞쪽 토큰이 밀려나야 하므로, 동작 원인은 추가 조사가 필요함
  - [#795](https://github.com/waybarrios/vllm-mlx/issues/795)는 설정하지 않은 경우만 다룸
- **12GiB 캐시는 3명의 31K 세션에 부족:** 엔트리 3개(약 13.4GB)가 안 들어가 2개만 남음, 재계산 가능성(추정)

---

## 7. 마무리

이 글에서는 채팅(Qwen3-8B-4bit), 임베딩(bge-m3), 리랭크(bge-reranker-v2-m3) 세 모델을 vllm-mlx 서버 하나에 구성하고, 동시 사용자에 맞춘 컨텍스트, 캐시, 메모리 설정을 살펴보았다.

실제 vllm-mlx 기반의 운영 환경을 구성한다면, 모델 구성, 컨텍스트 길이, 예상 동시 사용자 수를 고려해 서버를 선정하고 구성해야 한다.

- 메모리 용량: 모델 가중치, 동시 사용자 × 컨텍스트 × 토큰당 KV, prefix cache, OS 여유를 합산해 산정
- 서버 수: 서버 한 대의 Prefill 처리량이 거의 일정하므로 목표 TTFT와 평균 입력 길이로 계산
- 앞단 통제: 서버가 입력 길이를 막지 않고 기본 인증도 없으므로 게이트웨이에서 인증, 입력 한도, 동시 처리 제한, 대기열 처리
- 관측: `/v1/status`의 캐시 적중률, 밀어내기, 활성 메모리 상시 수집
- 라우팅: 서버를 여러 대로 늘리면 prefix cache가 서버별일 수 있어 세션 고정 검토

---

## 다음 편

[3편](03-simple-rag.md)에서는 simple-rag를 다룬다. 문서를 청크로 나눠 pgvector에 저장하고, 검색해서 MCP 도구로 내놓는 부분이다.