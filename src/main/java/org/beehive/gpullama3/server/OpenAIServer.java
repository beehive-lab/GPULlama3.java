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
    private final boolean gpu;
    private int port;
    private final AtomicLong seq = new AtomicLong();

    public OpenAIServer(InferenceService service, String servedModel, boolean gpu) {
        this.service = service;
        this.servedModel = servedModel;
        this.gpu = gpu;
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

        OpenAIServer server = new OpenAIServer(service, served, gpu);
        server.start(port);
    }

    public void start(int port) throws IOException {
        this.port = port;
        HttpServer http = HttpServer.create(new InetSocketAddress(port), 0);
        http.createContext("/", this::handleIndex);
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

    private void handleIndex(HttpExchange ex) throws IOException {
        if (!"/".equals(ex.getRequestURI().getPath())) {
            sendError(ex, 404, "Not found");
            return;
        }
        byte[] bytes = INDEX_HTML
                .replace("{{model}}", servedModel)
                .replace("{{backend}}", gpu ? "GPU (TornadoVM)" : "CPU")
                .replace("{{port}}", String.valueOf(port))
                .getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "text/html; charset=utf-8");
        ex.sendResponseHeaders(200, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static final String INDEX_HTML = """
            <!doctype html>
            <html lang="en">
            <head>
            <meta charset="utf-8">
            <title>GPULlama3.java server</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
              body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
                     max-width: 780px; margin: 40px auto; padding: 0 20px; line-height: 1.5; color: #1a1a1a; }
              h1 { font-size: 1.5rem; margin-bottom: 0; }
              .sub { color: #666; margin-top: 4px; }
              .badges span { display: inline-block; background: #eee; border-radius: 4px; padding: 2px 8px;
                             margin: 4px 6px 0 0; font-size: 0.85rem; }
              table { border-collapse: collapse; margin: 12px 0; }
              td { padding: 3px 10px 3px 0; vertical-align: top; }
              td.k { color: #666; }
              code, pre { background: #f4f4f4; border-radius: 4px; }
              code { padding: 1px 5px; }
              pre { padding: 10px; overflow-x: auto; }
              a { color: #0969da; }
              hr { border: none; border-top: 1px solid #eee; margin: 24px 0; }
              ul { padding-left: 20px; }
            </style>
            </head>
            <body>
              <h1>GPULlama3.java</h1>
              <div class="sub">Local OpenAI-compatible inference server</div>

              <table>
                <tr><td class="k">Model</td><td><code>{{model}}</code></td></tr>
                <tr><td class="k">Backend</td><td>{{backend}}</td></tr>
                <tr><td class="k">Port</td><td>{{port}}</td></tr>
              </table>

              <p>
                This is a local instance of
                <a href="https://github.com/beehive-lab/GPULlama3.java" target="_blank">GPULlama3.java</a>,
                a Llama3-family inference engine written in native Java and automatically accelerated on
                GPUs with <a href="https://github.com/beehive-lab/TornadoVM" target="_blank">TornadoVM</a>.
                It supports Llama3, Mistral, Devstral 2, Qwen2.5, Qwen3, Phi-3, IBM Granite 3.2+, and
                IBM Granite 4.0 models in GGUF format, and is also used as the GPU inference engine behind
                the <a href="https://docs.quarkiverse.io/quarkus-langchain4j/dev/gpullama3-chat-model.html" target="_blank">Quarkus</a>
                and <a href="https://docs.langchain4j.dev/integrations/language-models/gpullama3-java" target="_blank">LangChain4j</a>
                integrations.
              </p>

              <hr>

              <h3>API endpoints</h3>
              <ul>
                <li><code>GET  /health</code> — liveness check</li>
                <li><code>GET  /v1/models</code> — list the served model</li>
                <li><code>POST /v1/chat/completions</code> — chat completions (supports <code>"stream": true</code> SSE)</li>
                <li><code>POST /v1/completions</code> — text completions (supports <code>"stream": true</code> SSE)</li>
              </ul>

              <h3>Quick test</h3>
              <pre><code>curl http://localhost:{{port}}/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{
            "model": "{{model}}",
            "messages": [{"role": "user", "content": "Hello!"}]
          }'</code></pre>

              <p class="sub">
                Any OpenAI-compatible client (e.g. the <code>openai</code> Python/Node SDK) can point its
                base URL at this server. Generation runs on a single serialized GPU/CPU context, so
                concurrent requests are queued and processed one at a time.
              </p>
            </body>
            </html>
            """;

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
