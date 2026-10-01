# Aura Voice Backend

Local AI voice-calling backend: FastAPI + SQLite persistence, an in-process
Live Call pipeline (mic audio in → local Whisper → local LLM → local TTS →
audio back out, streamed sentence-by-sentence), and local semantic document
search for grounding answers in uploaded content.

## Structure

```text
backend/
  main.py           # Application entrypoint
  api/              # HTTP + WebSocket routes
  models/           # Domain DTOs (domain.py) and SQLAlchemy tables (orm.py)
  schemas/          # Request validation
  repositories/     # SQLite-backed persistence (repositories/sql.py)
  services/
    voice.py          # LLM chat/streaming, STT, TTS, provider status
    live_session.py   # The Live Call WebSocket pipeline: VAD, turn orchestration
    documents.py       # Document upload, chunking, semantic + lexical retrieval
    embeddings.py       # Local multilingual embedding model (ONNX Runtime)
    overview.py, campaigns.py, factory.py
  db.py             # SQLAlchemy engine/session, table creation
  config.py         # Runtime settings (pydantic-settings, reads .env)
```

## Run

```bash
python backend/main.py
```

The API runs at `http://127.0.0.1:4000` by default. Run from the workspace root (or `python main.py` from inside `backend`).

For development with automatic reload:

```bash
uvicorn backend.api.server:app --reload --app-dir D:\ai_local_model --host 127.0.0.1 --port 4000
```

## Requirements

```powershell
pip install -r backend\requirements.txt
```

`coqui-tts` (XTTS v2 voice cloning) needs PyTorch, which isn't always pulled in automatically:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu130   # match your CUDA version
pip install torch                                                      # CPU-only fallback
```

Semantic document search runs through ONNX Runtime + the standalone `tokenizers` library
rather than `sentence-transformers`, specifically because `sentence-transformers` (and even a
bare `transformers` import) pulls in scikit-learn's compiled extensions as a side effect, and
some locked-down Windows setups (Application Control / WDAC policies) block those unsigned
native DLLs outright. If the embedding model can't load for any reason, document search
degrades gracefully to lexical keyword matching rather than failing — check the backend logs
if retrieval quality seems off.

## Endpoints

- `GET /api/health`, `GET /api/overview` — real aggregates from the `calls` table, zero-safe when empty
- `GET /api/calls`, `GET /api/campaigns`, `POST /api/campaigns`
- `GET /api/voice/providers` — actual configured/reachable state of LLM/STT/TTS
- `GET /api/voice/languages`, `GET /api/voice/voices` — config-driven lists (no hardcoded frontend copies)
- `GET /api/voice/documents`, `POST /api/voice/documents/upload`
- `POST /api/voice/chat`, `POST /api/voice/session/turn`, `POST /api/voice/session/stream` — one-shot text paths
- `WS /api/voice/session/live` — the real-time pipeline: client streams mic PCM16 frames, server streams back transcript/assistant-text/audio messages sentence-by-sentence. See `services/live_session.py` for the wire protocol.
- `POST /api/voice/transcribe`, `POST /api/voice/synthesize`, `POST /api/voice/test-tts`

## Local AI configuration

Copy `.env.example` to `.env` before starting the backend.

- **LLM**: Ollama + Qwen3 8B by default (`LLM_URL`, `LLM_MODEL`).
- **STT**: local Faster-Whisper (`STT_MODE=local`), loaded once at startup instead of per-request.
- **TTS**: disabled by default (`TTS_ENABLED=false`). To enable XTTS v2 voice cloning, set
  `TTS_ENABLED=true` and `TTS_VOICE_PROFILES` to a JSON map of voice name → reference WAV
  (6-15s, clean audio), e.g. `{"default":"D:\\path\\to\\reference.wav"}`. Multiple named
  profiles are supported — the frontend's voice picker is populated from whatever you configure,
  not a hardcoded list.
- **Persistence**: SQLite at `DATABASE_URL` (default `backend/data/aura.db`). Calls, campaigns,
  and uploaded documents all survive a restart.
- **Live Call latency tuning**: `VAD_SILENCE_MS` (how much trailing silence ends an utterance),
  `VAD_MIN_SPEECH_MS` (minimum speech duration to avoid triggering on noise bursts).

API keys are optional: leave `LLM_API_KEY`, `STT_API_KEY`, and `TTS_API_KEY` empty for local
providers. When set, they're sent as `Authorization: Bearer ...`.

### Local model links

- Ollama for Windows: https://ollama.com/download/windows
- Qwen3 models: https://ollama.com/library/qwen3
- Faster-Whisper: https://github.com/SYSTRAN/faster-whisper
- Coqui XTTS v2: https://github.com/coqui-ai/TTS
- Piper TTS (lighter fallback, no cloning): https://github.com/rhasspy/piper

## Tests

```powershell
python -m unittest discover -s backend\tests -t .
```

## Known scope boundaries

Contacts, AI Agents, the Knowledge Base page's own UI (separate from the working document
upload/retrieval feature), Analytics, Infrastructure, and Settings are still the original
static prototype screens — they need new data models and product decisions beyond this pass.
There's no authentication or multi-tenancy: all uploaded documents and calls are shared,
process-wide state, appropriate for local single-user testing before a real telephony
integration (Plivo) is wired in.
