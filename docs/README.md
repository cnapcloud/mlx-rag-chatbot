# vllm-mlx 기반 RAG 챗봇 구축 시리즈

Apple Silicon Mac에서 vllm-mlx, pgvector, LiteLLM, LibreChat으로 문서와 대화가 외부로 나가지 않는 RAG 챗봇을 구축하는 시리즈다. 편마다 `##` 제목과 같은 1레벨 목차를 적는다.

---

## [vllm-mlx 기반 RAG 챗봇 구축 (1) - 전체 구성과 설치](01-overview.md)

1. 프로젝트 소개
2. 시스템 구성
3. 사전 준비
4. 설치 및 실행
5. 동작 테스트
6. 중지와 초기화
7. 문제 해결
8. 시리즈 안내
9. 참고

## [vllm-mlx 기반 RAG 챗봇 구축 (2) - 모델 서버 구성과 메모리 설계](02-vllm-mlx.md)

1. vllm-mlx 소개
2. 서빙 모델
3. 설치 및 실행
4. 서버 바인딩과 로컬 네트워크 구성
5. 메모리 설계 (48GB 예시)
6. 성능 측정과 이슈
7. 마무리

## [vllm-mlx 기반 RAG 챗봇 구축 (3) - 문서 검색 API와 MCP 서버](03-simple-rag.md)

1. 개요
2. RAG 실행
3. 파이프라인 구성
4. 데이터 모델
5. 검색과 답변 생성 흐름
6. MCP 서버
7. 마무리

## [vllm-mlx 기반 RAG 챗봇 구축 (4) - LiteLLM 게이트웨이와 Fallback 전략](04-litellm.md)

1. LiteLLM 소개
2. LiteLLM 실행
3. 모델 등록
4. 모델 컨텍스트 한도 설계
5. Fallback 설계
6. Fallback과 프라이버시의 상충 관계
7. Admin UI
8. 사용자 식별 헤더
9. 마무리

## vllm-mlx 기반 RAG 챗봇 구축 (5) - LibreChat UI와 MCP 도구 연동 (작성 예정)

1. LibreChat 개요와 구성
2. 사용자 등록과 로그인
3. 모델 노출 설정
4. LiteLLM 연동
5. MCP 서버 연동
6. 챗봇에서 RAG 사용하기
7. 에이전트 capabilities 설정
8. 보안 점검 사항
9. 문제 해결

---

## 작성 기준

- 한도 정렬(32768 → 28672 → 24000)은 4편 4.4절의 구성요소별 표에 정리했다. LibreChat 쪽 설정(24000)은 5편에서 이어서 쓰고 서로 링크한다.
- 편마다 "다음 편" 링크와 "M4 / 48GB 기준" 안내를 넣어 단독으로 읽어도 오해가 없게 한다.
- 도구 이름은 실제 이름에 맞춰 vllm-mlx로 표기한다.
