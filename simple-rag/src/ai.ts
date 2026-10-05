import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { env } from "./env.js";
import { generateText } from "ai";

const vllm = createOpenAICompatible({
  name: "vllm",
  baseURL: env.VLLM_BASE_URL,
  apiKey: env.VLLM_API_KEY,
});

export async function askLlm(input: { system: string; prompt: string }) {
  const result = await generateText({
    model: vllm.chatModel(env.VLLM_MODEL),
    system: input.system,
    prompt: input.prompt,
    temperature: 0,
    maxOutputTokens: 700,
  });

  return result.text;
}

export async function rerank(input: {
  query: string;
  documents: string[];
  topN: number;
}): Promise<Array<{ index: number; score: number }>> {
  const response = await fetch(`${env.VLLM_BASE_URL}/rerank`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.VLLM_API_KEY}`,
    },
    body: JSON.stringify({
      model: env.RERANK_MODEL,
      query: input.query,
      documents: input.documents,
      top_n: input.topN,
    }),
  });

  if (!response.ok) {
    throw new Error(`Rerank failed:${response.status}`);
  }

  const data = (await response.json()) as {
    results: Array<{ index: number; relevance_score: number }>;
  };

  return data.results.map((r) => ({ index: r.index, score: r.relevance_score }));
}

export async function createEmbedding(text: string): Promise<number[]> {
  const response = await fetch(`${env.VLLM_BASE_URL}/embeddings`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.VLLM_API_KEY}`,
    },
    body: JSON.stringify({ model: env.EMBEDDING_MODEL, input: text }),
  });

  if (!response.ok) {
    throw new Error(`Embedding failed:${response.status}`);
  }

  const data = (await response.json()) as {
    data: Array<{ embedding: number[] }>;
  };

  const embedding = data.data[0]?.embedding;

  if (!embedding) {
    throw new Error("No embedding returned");
  }

  if (embedding.length !== env.EMBEDDING_DIMENSIONS) {
    throw new Error(
      `Embedding dimension mismatch. Expected ${env.EMBEDDING_DIMENSIONS}, got ${embedding.length}`,
    );
  }
  return embedding;
}
