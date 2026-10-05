import fastifyMultipart from "@fastify/multipart";
import Fastify from "fastify";
import {
  askDocument,
  deleteDocument,
  ingestDocument,
  listDocuments,
  searchDocuments,
} from "./rag.js";
import z from "zod";
import { env } from "./env.js";
import { registerMcp } from "./mcp.js";

const app = Fastify({
  logger: true,
});

//for uploading files. like we used multer in express
await app.register(fastifyMultipart, {
  limits: {
    fileSize: 10 * 1024 * 1024,
    files: 1,
  },
});

//routes
app.get("/health", async () => {
  return {
    status: "ok",
  };
});

//
app.post("/documents", async (request, reply) => {
  const file = await request.file();

  if (!file) {
    return reply.code(400).send({
      error: "File is required",
    });
  }

  const allowedTypes = ["application/pdf", "text/plain", "text/markdown"];

  if (!allowedTypes.includes(file.mimetype)) {
    return reply.code(400).send({
      error: "Only PDF, TXT, and Markdown are supported",
    });
  }

  const buffer = await file.toBuffer();

  const result = await ingestDocument({
    buffer,
    fileName: file.filename,
    mimeType: file.mimetype,
  });

  return reply.code(201).send({
    data: result,
  });
});

app.get("/documents", async () => {
  return { data: await listDocuments() };
});

const documentParamsSchema = z.object({
  id: z.uuid(),
});

app.delete("/documents/:id", async (request, reply) => {
  const parsed = documentParamsSchema.safeParse(request.params);
  if (!parsed.success) {
    const errors = z.treeifyError(parsed.error);
    return reply.code(400).send({
      error: errors,
    });
  }

  const deleted = await deleteDocument(parsed.data.id);

  if (!deleted) {
    return reply.code(404).send({
      error: "Document not found",
    });
  }

  return reply.code(204).send();
});

///schema for asking question
const askSchema = z.object({
  question: z.string().min(1).max(1000),
  documentId: z.uuid().optional(),
});

//
app.post("/ask", async (request, reply) => {
  const parsed = askSchema.safeParse(request.body);
  if (!parsed.success) {
    const errors = z.treeifyError(parsed.error);
    return reply.code(400).send({
      error: errors,
    });
  }

  const result = await askDocument(parsed.data);

  return reply.send({
    data: result,
  });
});

//
app.post("/search", async (request, reply) => {
  const parsed = askSchema.safeParse(request.body);

  if (!parsed.success) {
    const errors = z.treeifyError(parsed.error);
    return reply.code(400).send({
      error: errors,
    });
  }

  const result = await searchDocuments(parsed.data);

  return reply.send({
    data: result,
  });
});

registerMcp(app);

app.setErrorHandler((error: Error, request, reply) => {
  request.log.error(error);

  return reply.code(500).send({
    error: "Internal server error",
    message: error.message,
  });
});

await app.listen({
  port: env.PORT,
  host: "0.0.0.0",
});
