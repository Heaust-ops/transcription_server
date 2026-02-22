# Transcription Server

A lightweight HTTP server that provides speech-to-text transcription powered by [Whisper](https://github.com/ggerganov/whisper.cpp).

## Endpoints

### `POST /bytes`

Transcribe audio from raw bytes.

Send raw 32-bit float little-endian audio samples in the request body and receive the transcription as plain text.

```bash
curl -X POST http://127.0.0.1:3277/bytes \
  --data-binary @audio.f32le
```

### `GET /speech/autocut`

Record from microphone with automatic voice activity detection (VAD).

Starts a microphone recording session. The server listens until VAD detects the end of speech, then returns the transcription automatically.

```bash
curl http://127.0.0.1:3277/speech/autocut
```

## Usage

```
transcription_service [OPTIONS] --model <WHISPER_MODEL_PATH>
```

### Options

| Flag                               | Description                    | Default      |
| ---------------------------------- | ------------------------------ | ------------ |
| `-m, --model <WHISPER_MODEL_PATH>` | Path to the Whisper model file | _(required)_ |
| `-n, --host <HOST>`                | Hostname or IP to bind to      | `127.0.0.1`  |
| `-p, --port <PORT>`                | Port to bind to                | `3277`       |
| `-h, --help`                       | Print help                     |              |
| `-V, --version`                    | Print version                  |              |

### Example

```bash
transcription_service --model /home/user/ggml-distil-large-v3.bin
```

```bash
transcription_service -m /home/user/ggml-distil-large-v3.bin -n 0.0.0.0 -p 8080
```

## Model

Download a compatible GGML Whisper model from [huggingface.co/ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp). Distil models like `ggml-distil-large-v3.bin` offer a good balance of speed and accuracy.
