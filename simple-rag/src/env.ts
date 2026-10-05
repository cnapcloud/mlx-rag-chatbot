import "dotenv/config";
import z from "zod";

const envSchema = z.object({
  PORT: z.coerce.number().default(3000),
  DATABASE_URL: z.url(),
  VLLM_BASE_URL: z.url().default("http://localhost:8100/v1"),
  VLLM_API_KEY: z.string().default("not-needed"),
  VLLM_MODEL: z.string().default("mlx-community/Qwen3-8B-4bit"),
  EMBEDDING_MODEL: z.string().default("mlx-community/bge-m3-mlx-fp16"),
  EMBEDDING_DIMENSIONS: z.coerce.number().default(1024),
  RERANK_MODEL: z.string().default("BAAI/bge-reranker-v2-m3"),
  RAG_TOP_K: z.coerce.number().default(5),
  // 리랭크 후 남길 청크 수. 0 이면 리랭크를 건너뛰고 벡터 검색 결과(RAG_TOP_K)를 그대로 전달한다
  RAG_RERANK_TOP_N: z.coerce.number().int().min(0).default(3),
  RAG_MIN_SIMILARITY: z.coerce.number().default(0.35),
});

const parsed = envSchema.safeParse(process.env);

if (!parsed.success) {
  const errors = z.treeifyError(parsed.error);
  console.error("errors: ");
  console.dir(errors, { depth: null });
  process.exit(1);
}

export const env = parsed.data;
