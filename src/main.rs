mod recorder;
mod transcriber;

use std::{ffi::c_void, process::exit, sync::Arc};

use axum::{
    Router,
    body::Bytes,
    extract::State,
    response::{Html, IntoResponse},
    routing::{get, post},
};
use clap::Parser;
use std::path::PathBuf;

use recorder::SoundRecorder;

use crate::transcriber::Transcriber;

#[derive(Parser)]
#[command(author, version, about = "Start the transcription server.")]
struct Args {
    /// Path to the Whisper model file [example: /home/user/ggml-distil-large-v3.bin]
    #[arg(short = 'm', long, value_name = "WHISPER_MODEL_PATH")]
    model: PathBuf,

    /// Hostname or IP to bind/connect to
    #[arg(short = 'n', long, default_value = "127.0.0.1", value_name = "HOST")]
    host: String,

    /// Port to bind/connect to
    #[arg(short = 'p', long, default_value_t = 3277, value_name = "PORT")]
    port: u16,
}

#[derive(Clone)]
struct AppState {
    transcriber: Arc<tokio::sync::Mutex<Transcriber>>,
    sound_recorder: Arc<tokio::sync::Mutex<SoundRecorder>>,
}

#[tokio::main]
async fn main() {
    // silence whisper logs
    unsafe {
        unsafe extern "C" fn silent_cb(_level: u32, _msg: *const i8, _user: *mut c_void) {
            // do nothing => fully silent
        }
        whisper_rs::set_log_callback(Some(silent_cb), std::ptr::null_mut());
    }

    let args = Args::parse();

    if !args.model.exists() || !args.model.is_file() {
        eprintln!(
            "'{}' does not exist or is not the whisper model file!",
            args.model.to_string_lossy()
        );
        exit(1);
    }

    let transcriber;
    match args.model.to_str() {
        Some(e) => {
            transcriber = Transcriber::new(&e);
        }
        None => {
            eprintln!("path argument is formatted in a weird way!");
            exit(1);
        }
    }
    let transcriber = Arc::new(tokio::sync::Mutex::new(transcriber));
    let sound_recorder = Arc::new(tokio::sync::Mutex::new(SoundRecorder::new()));

    let state = AppState {
        transcriber,
        sound_recorder,
    };

    let router = Router::new()
        .route("/", get(handler))
        .route("/bytes", post(bytes_transcriber))
        .route("/speech/autocut", get(vad_transcriber))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to start listener");

    println!("listening on {}:{}", args.host, args.port);
    let _ = axum::serve(listener, router)
        .await
        .expect("failed to serve");
}

fn bytes_to_f32_vec(b: &Bytes) -> Vec<f32> {
    let bs = b.as_ref();
    let n = bs.len() / 4;
    let mut v = Vec::with_capacity(n);
    for i in (0..bs.len()).step_by(4) {
        let arr = [bs[i], bs[i + 1], bs[i + 2], bs[i + 3]];
        v.push(f32::from_le_bytes(arr));
    }
    v
}

async fn bytes_transcriber(State(state): State<AppState>, body: Bytes) -> impl IntoResponse {
    let floats = bytes_to_f32_vec(&body);
    let mut t = state.transcriber.lock().await;
    let text = t.transcribe(floats);
    text
}

async fn vad_transcriber(State(state): State<AppState>) -> impl IntoResponse {
    let recorder = state.sound_recorder.lock().await;
    let rec = recorder.start_with_vad(0.75).await;

    if let Ok(r) = rec {
        state.transcriber.lock().await.transcribe(r)
    } else {
        eprintln!("error recording sound");
        String::from("")
    }
}

async fn handler() -> Html<&'static str> {
    Html("<h1>server is running</h1>")
}
