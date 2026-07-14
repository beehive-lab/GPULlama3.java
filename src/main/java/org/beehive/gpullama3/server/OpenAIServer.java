package org.beehive.gpullama3.server;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.format.ChatFormat;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;

import static org.beehive.gpullama3.model.loader.ModelLoader.loadModel;

/**
 * OpenAI-compatible HTTP server for GPULlama3, built on the JDK {@link HttpServer} (no external
 * dependencies). Exposes the loaded model behind the endpoints an OpenAI client already speaks:
 *
 * <ul>
 *   <li>{@code POST /v1/chat/completions} — chat, streaming (SSE) or full JSON.</li>
 *   <li>{@code POST /v1/completions} — text completion (prompt as a single user turn).</li>
 *   <li>{@code GET  /v1/models} — the one served model.</li>
 *   <li>{@code GET  /health} — liveness.</li>
 * </ul>
 *
 * <p>Generation is serialized on a single {@link InferenceService} (the GPU is one context); HTTP
 * accept is multi-threaded so clients queue cleanly. Run:</p>
 * <pre>
 *   java ... org.beehive.gpullama3.server.OpenAIServer --model model.gguf --port 8080 --gpu
 * </pre>
 */
public final class OpenAIServer {

    private final InferenceService service;
    private final String servedModel;
    private final AtomicLong seq = new AtomicLong();

    public OpenAIServer(InferenceService service, String servedModel) {
        this.service = service;
        this.servedModel = servedModel;
    }

    public static void main(String[] args) throws IOException {
        String modelPath = null;
        int port = 8080;
        boolean gpu = false;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--model", "-m" -> modelPath = args[++i];
                case "--port", "-p" -> port = Integer.parseInt(args[++i]);
                case "--gpu" -> gpu = true;
                default -> { }
            }
        }
        if (modelPath == null) {
            System.err.println("usage: OpenAIServer --model <model.gguf> [--port 8080] [--gpu]");
            System.exit(1);
        }
        System.setProperty("llama.enableTornadoVM", String.valueOf(gpu));

        Path path = Paths.get(modelPath);
        // interactive=true bypasses the --prompt-required check; the server never uses it.
        Options options = new Options(path, "server", null, null, true, 0.0f, 0.95f, 1234L, 512, false, false, gpu, false, 1);
        System.err.println("[server] loading " + path.getFileName() + " (gpu=" + gpu + ") ...");
        Model model = loadModel(options);
        InferenceService service = new InferenceService(model, gpu);
        String served = path.getFileName().toString().replaceAll("\\.gguf$", "");

        OpenAIServer server = new OpenAIServer(service, served);
        server.start(port);
    }

    public void start(int port) throws IOException {
        HttpServer http = HttpServer.create(new InetSocketAddress(port), 0);
        http.createContext("/health", this::handleHealth);
        http.createContext("/v1/models", this::handleModels);
        http.createContext("/v1/chat/completions", ex -> handleCompletion(ex, true));
        http.createContext("/v1/completions", ex -> handleCompletion(ex, false));
        // Accept concurrently; generation itself is serialized inside InferenceService.
        http.setExecutor(Executors.newFixedThreadPool(8));
        Runtime.getRuntime().addShutdownHook(new Thread(service::close));
        http.start();
        System.err.println("[server] listening on http://localhost:" + port + "  model=" + servedModel);
    }

    // ── Endpoints ─────────────────────────────────────────────────────────────

    private void handleHealth(HttpExchange ex) throws IOException {
        sendJson(ex, 200, Map.of("status", "ok"));
    }

    private void handleModels(HttpExchange ex) throws IOException {
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("id", servedModel);
        entry.put("object", "model");
        entry.put("created", 0);
        entry.put("owned_by", "gpullama3");
        sendJson(ex, 200, Map.of("object", "list", "data", List.of(entry)));
    }

    @SuppressWarnings("unchecked")
    private void handleCompletion(HttpExchange ex, boolean chat) throws IOException {
        if (!"POST".equals(ex.getRequestMethod())) {
            sendError(ex, 405, "Method not allowed — use POST");
            return;
        }
        Map<String, Object> body;
        try {
            String raw = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            body = Json.parseObject(raw);
        } catch (Exception e) {
            sendError(ex, 400, "Invalid JSON body: " + e.getMessage());
            return;
        }

        List<ChatFormat.Message> messages = new ArrayList<>();
        try {
            if (chat) {
                Object msgs = body.get("messages");
                if (!(msgs instanceof List) || ((List<?>) msgs).isEmpty()) {
                    sendError(ex, 400, "'messages' must be a non-empty array");
                    return;
                }
                for (Object o : (List<Object>) msgs) {
                    Map<String, Object> m = (Map<String, Object>) o;
                    String role = Json.str(m, "role", "user");
                    String content = Json.str(m, "content", "");
                    messages.add(new ChatFormat.Message(new ChatFormat.Role(role), content));
                }
            } else {
                Object prompt = body.get("prompt");
                String text = prompt instanceof String s ? s : prompt == null ? "" : prompt.toString();
                if (text.isEmpty()) {
                    sendError(ex, 400, "'prompt' must be a non-empty string");
                    return;
                }
                messages.add(new ChatFormat.Message(ChatFormat.Role.USER, text));
            }
        } catch (Exception e) {
            sendError(ex, 400, "Malformed request: " + e.getMessage());
            return;
        }

        int maxTokens = Json.intVal(body, "max_tokens", chat ? Json.intVal(body, "max_completion_tokens", 256) : 256);
        float temperature = (float) Json.num(body, "temperature", 0.0);
        float topP = (float) Json.num(body, "top_p", 0.95);
        long seed = (long) Json.num(body, "seed", 1234);
        boolean stream = Json.bool(body, "stream", false);

        var req = new InferenceService.Request(messages, maxTokens, temperature, topP, seed);
        String id = (chat ? "chatcmpl-" : "cmpl-") + seq.incrementAndGet();
        long created = System.currentTimeMillis() / 1000;

        if (stream) {
            streamResponse(ex, req, id, created, chat);
        } else {
            fullResponse(ex, req, id, created, chat);
        }
    }

    // ── Non-streaming ─────────────────────────────────────────────────────────

    private void fullResponse(HttpExchange ex, InferenceService.Request req, String id, long created, boolean chat) throws IOException {
        InferenceService.Result r;
        try {
            r = service.generate(req, null);
        } catch (Exception e) {
            sendError(ex, 500, "Generation failed: " + e);
            return;
        }
        String finish = r.stopped() ? "stop" : "length";
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        if (chat) {
            choice.put("message", Map.of("role", "assistant", "content", r.text()));
        } else {
            choice.put("text", r.text());
        }
        choice.put("finish_reason", finish);

        Map<String, Object> resp = new LinkedHashMap<>();
        resp.put("id", id);
        resp.put("object", chat ? "chat.completion" : "text_completion");
        resp.put("created", created);
        resp.put("model", servedModel);
        resp.put("choices", List.of(choice));
        resp.put("usage", Map.of(
                "prompt_tokens", r.promptTokens(),
                "completion_tokens", r.completionTokens(),
                "total_tokens", r.promptTokens() + r.completionTokens()));
        sendJson(ex, 200, resp);
    }

    // ── Streaming (Server-Sent Events) ────────────────────────────────────────

    private void streamResponse(HttpExchange ex, InferenceService.Request req, String id, long created, boolean chat) throws IOException {
        ex.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
        ex.getResponseHeaders().set("Cache-Control", "no-cache");
        ex.getResponseHeaders().set("Connection", "keep-alive");
        ex.sendResponseHeaders(200, 0);
        OutputStream os = ex.getResponseBody();

        String object = chat ? "chat.completion.chunk" : "text_completion";
        try {
            if (chat) {
                // First chunk carries the assistant role.
                writeSse(os, chunk(id, object, created, roleDelta(), null));
            }
            InferenceService.Result r = service.generate(req, piece -> {
                try {
                    writeSse(os, chunk(id, object, created, chat ? contentDelta(piece) : textField(piece), null));
                } catch (IOException io) {
                    throw new RuntimeException(io); // client disconnected — abort generation
                }
            });
            String finish = r.stopped() ? "stop" : "length";
            writeSse(os, chunk(id, object, created, chat ? Map.of() : textField(""), finish));
            os.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
            os.flush();
        } catch (Exception e) {
            // Best-effort: nothing more we can send on a half-written SSE stream.
        } finally {
            os.close();
        }
    }

    private Map<String, Object> roleDelta() {
        Map<String, Object> d = new LinkedHashMap<>();
        d.put("role", "assistant");
        d.put("content", "");
        return d;
    }

    private Map<String, Object> contentDelta(String piece) {
        return Map.of("content", piece);
    }

    private Map<String, Object> textField(String piece) {
        return Map.of("text", piece);
    }

    /** One SSE data object. {@code delta} is the chat delta map or the completion text map. */
    private String chunk(String id, String object, long created, Map<String, Object> delta, String finish) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        if (object.startsWith("chat")) {
            choice.put("delta", delta);
        } else {
            choice.putAll(delta); // {"text": ...}
        }
        choice.put("finish_reason", finish);
        Map<String, Object> obj = new LinkedHashMap<>();
        obj.put("id", id);
        obj.put("object", object);
        obj.put("created", created);
        obj.put("model", servedModel);
        obj.put("choices", List.of(choice));
        return Json.write(obj);
    }

    private void writeSse(OutputStream os, String jsonData) throws IOException {
        os.write(("data: " + jsonData + "\n\n").getBytes(StandardCharsets.UTF_8));
        os.flush();
    }

    // ── Wire helpers ──────────────────────────────────────────────────────────

    private void sendJson(HttpExchange ex, int status, Map<String, Object> body) throws IOException {
        byte[] bytes = Json.write(body).getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private void sendError(HttpExchange ex, int status, String message) throws IOException {
        Map<String, Object> err = Map.of("error", Map.of(
                "message", message,
                "type", "invalid_request_error"));
        sendJson(ex, status, err);
    }
}
