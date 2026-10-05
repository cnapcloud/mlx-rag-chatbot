import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import type { FastifyInstance, FastifyReply } from "fastify";
import z from "zod";
import { deleteDocument, listDocuments, searchDocuments } from "./rag.js";

const toolInput = {
  question: z.string().min(1).max(1000).describe("Natural-language question"),
  // 작은 모델은 빈 문자열/플레이스홀더를 채워 보내므로 스키마는 string 으로 두고 핸들러에서 검증한다
  documentId: z
    .string()
    .optional()
    .describe(
      "OPTIONAL. Omit this field to search all documents (default). Set only to restrict to one document, with an exact UUID from list_documents. Never send an empty string or a made-up id.",
    ),
};

const uuidSchema = z.uuid();

function createMcpServer() {
  const server = new McpServer({ name: "simple-rag", version: "1.0.0" });

  server.registerTool(
    "list_documents",
    {
      title: "List documents",
      description:
        "List uploaded files (id, filename, chunk count, upload time). Metadata only, no document content, so it cannot answer questions about what the documents say. " +
        "Use it when asked which documents exist, or to look up a document id. To answer questions about content, use search_documents instead; no need to call this first.",
    },
    async () => {
      const documents = await listDocuments();
      return { content: [{ type: "text", text: JSON.stringify(documents) }] };
    },
  );

  server.registerTool(
    "delete_document",
    {
      title: "Delete document",
      description:
        "Permanently delete one document and its embeddings. Irreversible: call only when the user explicitly asks, with an exact id from list_documents. Bulk delete is not supported.",
      inputSchema: {
        documentId: z.uuid().describe("Document id to delete (from list_documents)"),
      },
    },
    async ({ documentId }) => {
      const deleted = await deleteDocument(documentId);
      return {
        isError: !deleted,
        content: [
          {
            type: "text",
            text: deleted
              ? `Deleted document ${documentId}`
              : `Document not found: ${documentId}`,
          },
        ],
      };
    },
  );

  server.registerTool(
    "search_documents",
    {
      title: "Search documents",
      description:
        "Semantic search over the content of uploaded documents; returns the most relevant text chunks with similarity scores. " +
        "Use it for any question about what the documents say. Searches all documents unless documentId is given. Not for listing files (use list_documents). " +
        "Answer only from the returned chunks; if nothing relevant is found, say so.",
      inputSchema: toolInput,
    },
    async ({ question, documentId }) => {
      const id = documentId?.trim() || undefined; // "" → 전체 검색
      if (id && !uuidSchema.safeParse(id).success) {
        return {
          isError: true,
          content: [
            {
              type: "text",
              text: `Invalid documentId "${id}". Omit documentId to search all documents, or use an exact id from list_documents.`,
            },
          ],
        };
      }
      const chunks = await searchDocuments({ question, documentId: id });
      return { content: [{ type: "text", text: JSON.stringify(chunks) }] };
    },
  );

  return server;
}

// stateless streamable-http: 요청마다 서버/트랜스포트를 새로 만든다
export function registerMcp(app: FastifyInstance) {
  app.post("/mcp", async (request, reply) => {
    const server = createMcpServer();
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
    });

    reply.raw.on("close", () => {
      void transport.close();
      void server.close();
    });

    await server.connect(transport);
    reply.hijack();
    await transport.handleRequest(request.raw, reply.raw, request.body);
  });

  // stateless 모드에서는 SSE 스트림(GET)과 세션 종료(DELETE)를 지원하지 않는다
  const notAllowed = async (_req: unknown, reply: FastifyReply) =>
    reply.code(405).header("Allow", "POST").send({
      jsonrpc: "2.0",
      error: { code: -32000, message: "Method not allowed." },
      id: null,
    });
  app.get("/mcp", notAllowed);
  app.delete("/mcp", notAllowed);
}
