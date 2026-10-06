# vllm-mlx 기반 RAG 챗봇 구축 시리즈

Apple Silicon Mac에서 vllm-mlx, pgvector, LiteLLM, LibreChat을 활용해 로컬 환경에서 동작하는 RAG 챗봇을 구축하는 시리즈다. 각 편 아래에는 본문의 `##` 제목을 순서대로 목차로 나열한다.

---

## [vllm-mlx 기반 RAG 챗봇 구축 (1) - 전체 구성과 설치](01-overview.md)

1. 프로젝트 소개
2. 시스템 구성
3. 사전 준비
4. 시작하기
5. 동작 테스트
6. 중지와 초기화
7. 마무리

## [vllm-mlx 기반 RAG 챗봇 구축 (2) - 모델 서버 구성과 메모리 설계](02-vllm-mlx.md)

1. vllm-mlx 소개
2. 서빙 모델
3. 시작하기
4. 서버 바인딩과 로컬 네트워크 구성
5. 메모리 설계 (48GB 예시)
6. 성능 측정과 이슈
7. 마무리

## [vllm-mlx 기반 RAG 챗봇 구축 (3) - 문서 검색 API와 MCP 서버](03-simple-rag.md)

1. RAG 개요
2. 시작하기
3. 파이프라인 구성
4. 데이터 모델
5. 검색과 답변 생성 흐름
6. MCP 서버
7. 벡터 저장소
8. 마무리

## [vllm-mlx 기반 RAG 챗봇 구축 (4) - LiteLLM 게이트웨이와 Fallback 전략](04-litellm.md)

1. LiteLLM 소개
2. 시작하기
3. 모델 등록
4. 모델 컨텍스트 한도 설계
5. Fallback 설계
6. Fallback과 프라이버시의 상충 관계
7. Admin UI
8. 사용자 식별 헤더
9. 마무리

## [vllm-mlx 기반 RAG 챗봇 구축 (5) - LibreChat UI와 MCP 도구 연동](05-librechat.md)

1. LibreChat 소개
2. 시작하기
3. 사용자 등록과 로그인
4. 모델 노출 설정
5. LiteLLM 연동
6. MCP 서버 연동
7. 챗봇에서 RAG 사용하기
8. 에이전트 capabilities 설정
9. 마무리

---

## 작성 기준

- 글 제목은 `vllm-mlx 기반 RAG 챗봇 구축 (N) - 소제목` 형식으로 쓴다.
- 도구 이름은 제목을 포함해 항상 소문자 `vllm-mlx`로 표기한다.
- 명령 예제는 저장소 루트에서 실행하는 것을 기준으로 쓴다.
- 각 편 끝에는 다음 편 링크를 넣는다. 글을 추가하거나 제목·절 구성을 바꾸면 이 목차와 1편의 "시리즈 안내", 이전·다음 편 링크도 함께 고친다.

