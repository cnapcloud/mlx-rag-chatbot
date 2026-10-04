import { PDFParse } from "pdf-parse";
import { db } from "./db.js";
import { askLlm, createEmbedding } from "./ai.js";
import { env } from "./env.js";

function vectorToSql(vector: number[]) {
  return `[${vector.join(",")}]`;
}

function chunkText(text: string) {
  const words = text.replace(/\s+/g, " ").trim().split(" ").filter(Boolean);

  const chunks: string[] = [];

  const chunkSize = 450;
  const overlap = 70;

  for (let start = 0; start < words.length; start += chunkSize - overlap) {
    const chunk = words.slice(start, start + chunkSize).join(" ");

    if (chunk.trim()) {
      chunks.push(chunk);
    }
  }
  return chunks;
}

async function extractText(input: { buffer: Buffer; mimeType: string }) {
  if (input.mimeType === "application/pdf") {
    const parser = new PDFParse({ data: input.buffer });
    const result = await parser.getText();
    await parser.destroy();
    return result.text.trim();
  }
  if (input.mimeType === "text/plain" || input.mimeType === "text/markdown") {
    return input.buffer.toString("utf8").trim();
  }

  throw new Error("Unsupported file type");
}

export async function ingestDocument(input: {
  fileName: string;
  mimeType: string;
  buffer: Buffer;
}) {
  const text = await extractText({
    buffer: input.buffer,
    mimeType: input.mimeType,
  });

  if (!text) {
    throw new Error("Document has no readable text");
  }

  const documentResult = await db.query<{ id: string }>(
    `
    INSERT INTO documents (filename,mime_type,text_content)
    VALUES ($1, $2, $3)
    RETURNING id
    `,
    [input.fileName, input.mimeType, text],
  );

  const documentId = documentResult.rows[0]?.id;

  if (!documentId) {
    throw new Error("Failed to create document");
  }

  const chunks = chunkText(text);

  for (let i = 0; i < chunks.length; i++) {
    const content = chunks[i];

    if (!content) continue;

    const embedding = await createEmbedding(content);

    await db.query(
      `
        INSERT INTO document_chunks( document_id,chunk_index,content,embedding )
        VALUES ($1, $2, $3, $4::vector)
        `,
      [documentId, i, content, vectorToSql(embedding)],
    );
  }

  return {
    documentId,
    chunksCreated: chunks.length,
  };
}

export async function listDocuments() {
  const result = await db.query<{
    id: string;
    filename: string;
    mime_type: string;
    chunks: string;
    created_at: Date;
  }>(
    `
    SELECT d.id, d.filename, d.mime_type, count(c.id) AS chunks, d.created_at
    FROM documents d
    LEFT JOIN document_chunks c ON c.document_id = d.id
    GROUP BY d.id
    ORDER BY d.created_at DESC
    `,
  );

  return result.rows.map((row) => ({
    id: row.id,
    filename: row.filename,
    mimeType: row.mime_type,
    chunks: Number(row.chunks),
    createdAt: row.created_at.toISOString(),
  }));
}

export async function deleteDocument(id: string) {
  // document_chunks 는 ON DELETE CASCADE 로 함께 삭제된다
  const result = await db.query(`DELETE FROM documents WHERE id = $1`, [id]);
  return (result.rowCount ?? 0) > 0;
}

export async function searchDocuments(input: {
  question: string;
  documentId?: string;
}) {
  const queryEmbedding = await createEmbedding(input.question);

  const queryVector = vectorToSql(queryEmbedding);

  const result = input.documentId
    ? await db.query<{
        id: string;
        document_id: string;
        chunk_index: number;
        content: string;
        similarity: string;
      }>(
        `
        SELECT
          id,
          document_id,
          chunk_index,
          content,
          1 - (embedding <=> $1::vector) AS similarity
        FROM document_chunks
        WHERE document_id = $2
        ORDER BY embedding <=> $1::vector
        LIMIT $3
        `,
        [queryVector, input.documentId, env.RAG_TOP_K],
      )
    : await db.query<{
        id: string;
        document_id: string;
        chunk_index: number;
        content: string;
        similarity: string;
      }>(
        `
        SELECT
          id,
          document_id,
          chunk_index,
          content,
          1 - (embedding <=> $1::vector) AS similarity
        FROM document_chunks
        ORDER BY embedding <=> $1::vector
        LIMIT $2
        `,
        [queryVector, env.RAG_TOP_K],
      );

  return result.rows.map((row) => ({
    id: row.id,
    documentId: row.document_id,
    chunkIndex: row.chunk_index,
    content: row.content,
    similarity: Number(row.similarity),
  }));
}

export async function askDocument(input: {
  question: string;
  documentId?: string;
}) {
  const chunks = await searchDocuments(input);

  const strongChunks = chunks.filter(
    (chunk) => chunk.similarity >= env.RAG_MIN_SIMILARITY,
  );

  if (strongChunks.length === 0) {
    return {
      answer: "I could not find this in the provided documents.",
      sources: [],
    };
  }

  const context = strongChunks
    .map((chunk, index) => {
      return `[Source ${index + 1}]
${chunk.content}`;
    })
    .join("\n\n---\n\n");

  const answer = await askLlm({
    system: `
You are a document Q&A assistant.

Rules:
1. Answer only using the provided context.
2. If the answer is not in the context, say: "I could not find this in the provided documents."
3. Do not use outside knowledge.
4. Cite sources like [Source 1].
`.trim(),

    prompt: `
Context:

${context}

Question:
${input.question}

Answer:
`.trim(),
  });

  return {
    answer,
    sources: strongChunks.map((chunk, index) => ({
      label: `Source ${index + 1}`,
      documentId: chunk.documentId,
      chunkId: chunk.id,
      chunkIndex: chunk.chunkIndex,
      similarity: chunk.similarity,
      preview: chunk.content.slice(0, 250),
    })),
  };
}
