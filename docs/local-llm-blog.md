# Apple Silicon에서 vLLM-MLX로 Qwen3-8B 서빙하기

RAG 서비스에 쓸 로컬 LLM을 GPU 서버 없이 노트북에서 돌려 보기로 했다. `vllm-mlx`로 Qwen3-8B를 띄우고, 같은 프로세스에서 Embedding과 Reranker까지 서빙해서 LiteLLM과 LibreChat 뒤에 붙였다.

모델을 띄우는 것 자체는 어렵지 않았다. 어려웠던 부분은 **한정된 메모리 안에서 긴 RAG 프롬프트를 감당하게 만드는 일**이었고, 이 글은 그 과정에서 값을 어떻게 정했는지에 대한 기록이다.

> **먼저 밝혀 둘 것.** 이 글의 숫자(`32768`, `12288`, `4` 등)는 **M4 MacBook Pro, 48GB 메모리, GPU 16코어** 기준으로 정한 값이다. 이 기기의 메모리 예산에 맞춘 선택이고, 범용 권장값도 벤치마크 결과도 아니다. 사양이 다르면 같은 방식으로 다시 계산해야 한다.

---

## 구성

```text
 LibreChat ──► LiteLLM ──► vLLM-MLX (Mac 호스트, :8100)
 (Docker)      (Docker)     ├─ Qwen3-8B-4bit
                  │         ├─ BGE-M3 (embedding)
                  │         └─ BGE Reranker v2 m3
                  │
                  └─ 장애 시 ──► Gemini Flash Lite
```

vLLM-MLX는 컨테이너가 아니라 Mac에서 직접 실행한다. MLX는 Metal을 쓰기 때문에 Docker 안에서는 GPU를 쓸 수 없다. 대신 LiteLLM이 `host.docker.internal:8100`으로 호스트에 접근한다. 컨테이너에서 접근해야 하므로 서버는 `localhost`가 아닌 `0.0.0.0`에 바인딩했다.

실행 설정은 전부 Makefile 하나에 모았다.

```makefile
VLLM_MODEL := mlx-community/Qwen3-8B-4bit
VLLM_EMBED_MODEL := mlx-community/bge-m3-mlx-fp16
VLLM_RERANK_MODEL := BAAI/bge-reranker-v2-m3
VLLM_PORT := 8100
VLLM_HOST := 0.0.0.0
VLLM_MAX_SEQS := 4
VLLM_CONTEXT := 32768
VLLM_CHUNK := 2048
VLLM_PREFILL_STEP := 4096
VLLM_MAX_TOKENS := 4096
VLLM_CACHE_MB := 12288
VLLM_THINK_BUDGET := 128

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

`make install && make download && make run`이면 서빙이 시작된다. 이 설정의 값들은 대부분 한 가지 질문에서 나왔다. **48GB를 어떻게 나눠 쓸 것인가.**

---

## 모델보다 KV Cache가 메모리를 먹는다

8B 모델을 4bit로 양자화하면 가중치는 4.5~5GB 정도다. Embedding과 Reranker까지 합쳐도 7~9GB 안팎으로 추정한다. 48GB에서는 모델 가중치가 큰 문제가 아니다.

문제는 컨텍스트다. RAG는 검색된 문서가 프롬프트에 들어가고 Qwen3는 추론 토큰까지 쓰기 때문에 컨텍스트를 넉넉히 잡아야 했고, 그래서 `VLLM_CONTEXT`를 32768로 정했다. 이때 KV Cache 크기를 Qwen3-8B 구조(36 layers, KV head 8개, head_dim 128)로 계산해 보면 이렇다.

```text
토큰당 KV ≈ 2(K,V) × 36 × 8 × 128 × 2 bytes ≈ 144 KiB
32K 토큰   ≈ 4.5 GiB   ← sequence 하나
```

직접 측정한 값이 아니라 구조에서 뽑은 추정치다. 그런데 이 숫자가 동시성을 정하는 기준이 됐다. 네 요청이 모두 32K를 채운다고 가정하면 KV만 약 18GiB다. 개인 RAG 서버라서 동시 요청이 많지 않다고 보고 `VLLM_MAX_SEQS`는 4로 잡았다. 32K를 설정했다고 모든 요청이 32K를 쓰는 것은 아니지만, 최악의 경우를 알고 있어야 값을 정할 수 있다.

---

## RAG의 병목은 Decode가 아니라 Prefill

일반 채팅은 입력이 짧아서 생성 속도(tokens/sec)가 체감 성능을 좌우한다. RAG는 다르다. 질문은 몇십 토큰이지만 시스템 프롬프트, 대화 이력, 검색된 청크가 붙으면 입력이 수천~수만 토큰이 된다. 첫 토큰이 나오기까지 기다리는 시간(TTFT)이 곧 체감 속도이고, 이는 Prefill이 결정한다.

```text
질문 → 검색 → [시스템 프롬프트 + 이력 + 검색 문서 + 질문] → Prefill → Decode
                                                             ▲ 여기가 병목
```

긴 프롬프트 하나가 Prefill을 독점하면 이미 생성 중인 다른 요청이 멈춘다. 그래서 두 값을 조정했다.

```makefile
VLLM_CHUNK := 2048         # 스케줄러 step당 prefill 토큰 상한
VLLM_PREFILL_STEP := 4096  # prefill 계산 청크 크기
```

`--chunked-prefill-tokens`는 긴 입력을 나눠서 스케줄링해 다른 요청의 Decode가 끼어들 틈을 만든다. `--prefill-step-size`는 실제 계산 청크 크기다. 클수록 빠르지만 중간 계산에 메모리를 더 쓴다. 너무 잘게 쪼개면 스케줄링 오버헤드가 늘어난다.

솔직히 이 두 값은 아직 **정답을 찾은 값이 아니라 출발점**이다. 두 값이 내부에서 어떻게 맞물리는지(청크 4096 > step 상한 2048)는 입력 길이별 TTFT를 재 보면서 확인할 계획이다.

---

## 12GB 캐시는 문제를 겪고 나서 정한 값

RAG에서는 같은 시스템 프롬프트나 같은 문서 컨텍스트가 여러 요청에 반복되므로 Prefix Cache가 효과적이다. 이미 계산한 prefix는 다시 Prefill하지 않아도 된다.

그런데 기본 설정에서는 긴 프롬프트가 캐시에 **저장되지 않았다.** `--cache-memory-mb`의 기본값은 가용 RAM의 20%(약 4GB)인데, 30K 토큰짜리 프롬프트의 KV가 약 4.1GB라서 한도에 걸려 저장이 거부됐다. 앞에서 계산한 토큰당 144KiB로 보면 30K 토큰이 4.1GiB 정도이니 숫자도 맞아떨어진다.

그래서 `VLLM_CACHE_MB`를 12288(12GiB)로 올려 30K급 프롬프트가 들어가도록 했다. 대신 이 캐시도 Unified Memory를 쓴다. 지금까지의 값을 합치면 48GB는 이렇게 나뉜다.

```text
48GB Unified Memory
 ├─ 모델 3종                         약  7~9 GB
 ├─ KV Cache (4 seq × 32K, 최악)     약 18 GB
 ├─ Prefix Cache 상한                    12 GB
 └─ OS / Docker / 기타                   수 GB
                              최악의 경우 합계가 45GB 안팎
```

네 sequence가 동시에 32K를 꽉 채울 일은 드물지만, 그 경우에는 여유가 거의 없다. 또 macOS에는 GPU가 쓸 수 있는 메모리 상한(`iogpu.wired_limit_mb`)이 있고 기본값이 전체 메모리보다 작은 것으로 알고 있는데, 이 기기의 값은 아직 확인하지 못했다. 48GB를 전부 쓸 수 있다고 가정해서는 안 된다.

---

## 폭주를 막는 두 가지 안전장치

로컬 모델은 클라우드와 달리 토큰마다 과금되지 않지만, **통제 없이 생성하면 메모리와 응답 시간이 예측 불가능해진다.** 그래서 생성 쪽에 상한을 두었다.

```makefile
VLLM_THINK_BUDGET := 128   # reasoning 토큰 상한
VLLM_MAX_TOKENS := 4096    # 요청에 max_tokens가 없을 때의 기본 생성 상한
```

**Thinking Budget.** Qwen3는 `<think>`로 추론을 먼저 출력한다. RAG에서는 검색된 문서가 이미 근거를 주므로 매번 길게 생각할 필요가 없다. 128토큰을 넘기면 서버가 end-think를 강제로 넣고 답변으로 넘어간다. 요청에 `thinking_token_budget`이 있으면 그 값이 우선하므로 복잡한 질문만 예산을 늘릴 수 있다. 품질을 최대화하려는 설정이 아니라 **지연을 예측 가능하게 만드는 설정**이다.

**Max Tokens.** 서버 기본 생성 상한이 32768이어서, 모델이 반복 루프에 빠지면 KV 창을 넘길 때까지 끝나지 않을 수 있다. 이를 막으려고 기본값을 4096으로 낮췄다. 다만 이것은 `max_tokens`가 **없는 요청에만** 적용되는 기본값이다. 클라이언트가 직접 큰 값을 보내면 그대로 적용된다.

---

## 앞단과 숫자를 맞추기

서버만 설정해서는 끝나지 않았다. LiteLLM과 LibreChat도 같은 한도를 알아야 사용자가 한도 초과 에러를 보지 않는다.

```text
vllm-mlx   VLLM_CONTEXT     = 32768
LiteLLM    max_input_tokens = 28672   (= 32768 − 4096)
LibreChat  context          = 20000   (LiteLLM 한도보다 작게, 여유분)
```

입력 한도는 컨텍스트에서 생성용 4096토큰을 뺀 값이다. 그래서 클라이언트가 4096보다 큰 `max_tokens`를 보내면 이 계산이 깨진다는 점은 알고 있어야 한다.

**Fallback은 프라이버시와의 트레이드오프다.** LiteLLM은 vLLM-MLX가 죽거나 타임아웃이 나면 Gemini로 넘어가게 해 두었다. 반면 컨텍스트 초과 요청은 일부러 넘기지 않고 에러로 끝나게 했다. 문제는 서버 장애 시 **RAG로 가져온 문서가 Gemini로 전송된다**는 점이다. 로컬 LLM을 쓰는 이유가 데이터를 밖으로 내보내지 않는 것이라면 이 fallback은 끄는 편이 맞다.

보안도 하나 짚어 둔다. 컨테이너 접근을 위해 연 `0.0.0.0`은 같은 네트워크의 다른 기기에서도 8100 포트에 접근할 수 있다는 뜻이고, 인증도 걸어 두지 않았다. 노트북을 외부 Wi-Fi에 연결할 때는 방화벽을 확인해야 한다.

---

## 아직 측정하지 못한 것

이 글에는 벤치마크 수치가 없다. 값들은 메모리 예산 계산과 겪은 문제에서 나온 것이고, 체계적으로 측정한 결과가 아니다. 다음에는 입력을 1K, 4K, 8K, 16K, 32K로 바꿔 가며 TTFT, Prefill/Decode tokens/sec, 동시 요청 시 처리량, 메모리 사용량을 재 볼 계획이다. 결과는 **M4 / 48GB / GPU 16코어**라는 사양과 함께 기록해야 의미가 있다.

---

## 정리

Apple Silicon에서 로컬 LLM을 서빙하는 일은 **가장 큰 모델을 올리는 문제가 아니라 메모리를 나누는 문제**였다. 8B 4bit 모델은 가볍고, 메모리는 KV Cache와 Prefix Cache에서 갈린다. 컨텍스트 길이, 동시성, 캐시 크기, 생성 상한은 서로 연결된 값이라 하나만 바꿔서는 의미가 없었다.

48GB 기기에서 이 값들이 맞았다고 해서 다른 기기에도 맞는 것은 아니다. 숫자를 복사하기보다, 자기 기기에서 **토큰당 KV 크기 → sequence당 메모리 → 동시성 → 캐시 크기** 순서로 계산해 보는 것을 권한다.
