# Apple Silicon(MLX) 기준 vllm-mlx 서빙 설정
HF_VERSION := 1.28.0       # huggingface-hub (hf CLI)
VLLM_VERSION := 0.4.1      # vllm-mlx
VLLM_MODEL := mlx-community/Qwen3-8B-4bit
VLLM_EMBED_MODEL := mlx-community/bge-m3-mlx-fp16
VLLM_RERANK_MODEL := BAAI/bge-reranker-v2-m3
VLLM_PORT := 8100
VLLM_HOST := 0.0.0.0      # 클러스터(LiteLLM)에서 접근하려면 0.0.0.0 필요, 기본값 localhost는 외부 접근 불가
VLLM_MAX_SEQS := 4        # 동시 처리 시퀀스 수
VLLM_CONTEXT := 32768     # 시퀀스당 KV cache 한도(토큰), reasoning 모델은 32768 이상
VLLM_CHUNK := 2048        # 스케줄러 step당 prefill 토큰 상한, 긴 프롬프트가 생성 중인 요청을 막지 않게
VLLM_PREFILL_STEP := 4096 # prefill 청크 크기(텍스트/멀티모달 경로 공통), 클수록 빠르지만 메모리 더 사용
VLLM_MAX_TOKENS := 4096   # 요청에 max_tokens 가 없을 때의 생성 상한(기본 32768), 반복 루프로 KV 창을 넘겨 폭주하는 것 방지
VLLM_CACHE_MB := 12288    # prefix cache 메모리 상한(MB), 기본값(가용 RAM 20% ≈ 4GB)은 30k 토큰 프롬프트(약 4.1GB) 저장이 거부됨
VLLM_THINK_BUDGET := 128  # reasoning 토큰 상한, 초과 시 end-think 강제 삽입(요청의 thinking_token_budget 이 우선)

.PHONY: help install download run stop

help: ## 사용 가능한 타겟 목록 표시
	@echo "vllm-mlx 서빙 (macOS Apple Silicon / MLX 전용)"
	@echo ""
	@echo "사용법: make <target>"
	@echo ""
	@awk -F':.*## ' '/^[a-z]+:.*## / {printf "  %-10s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "현재 설정: MODEL=$(VLLM_MODEL) HOST=$(VLLM_HOST) PORT=$(VLLM_PORT)"

# hf, vllm-mlx가 없을 때만 uv tool로 설치
install: ## hf, vllm-mlx 설치 (uv tool)
	@command -v uv >/dev/null || { echo "uv가 필요합니다: brew install uv"; exit 1; }
	@command -v hf >/dev/null || uv tool install huggingface-hub==$(HF_VERSION)
	@command -v vllm-mlx >/dev/null || uv tool install vllm-mlx==$(VLLM_VERSION)

download: install ## 채팅/임베딩/리랭크 모델 다운로드
	hf download $(VLLM_MODEL)
	hf download $(VLLM_EMBED_MODEL)
	hf download $(VLLM_RERANK_MODEL)

run: ## Run vllm-mlx server (chat + embedding + rerank)
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

stop: ## vllm-mlx 서버 종료
	pkill -f "vllm-mlx serve $(VLLM_MODEL) --port $(VLLM_PORT)" || true