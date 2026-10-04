import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import type { FastifyInstance, FastifyReply } from "fastify";
import z from "zod";
import { deleteDocument, listDocuments, searchDocuments } from "./rag.js";

const toolInput = {
  question: z.string().min(1).max(1000).describe("자연어 질문"),
  documentId: z
    .uuid()
    .optional()
    .describe("특정 문서로 검색 범위를 제한할 때의 문서 ID"),
};

function createMcpServer() {
  const server = new McpServer({ name: "simple-rag", version: "1.0.0" });

  server.registerTool(
    "list_documents",
    {
      title: "List documents",
      description:
        "임베딩되어 검색 가능한 문서 목록(id, 파일명, 청크 수, 업로드 시각)을 반환한다. search_documents 의 documentId 로 범위를 좁힐 때 id 를 사용한다.",
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
        "문서 하나와 그 임베딩(청크)을 영구 삭제한다. 되돌릴 수 없으므로 반드시 사용자가 삭제를 명시적으로 요청한 경우에만, list_documents 로 확인한 정확한 id 로 호출한다. 전체 삭제는 지원하지 않는다.",
      inputSchema: {
        documentId: z.uuid().describe("삭제할 문서 ID (list_documents 의 id)"),
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
        "업로드된 문서에서 질문과 유사한 청크를 벡터 검색해 원문과 유사도를 반환한다. 답변은 반환된 청크만 근거로 작성하고, 관련 내용이 없으면 문서에서 찾지 못했다고 답한다.",
      inputSchema: toolInput,
    },
    async (input) => {
      const chunks = await searchDocuments(input);
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
