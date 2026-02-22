use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use cpal::traits::StreamTrait;
use cpal::traits::{DeviceTrait, HostTrait};

pub struct SoundRecorder {}

impl SoundRecorder {
    pub fn new() -> Self {
        return SoundRecorder {};
    }

    pub async fn start_with_vad(&self, vad_threshold: f32) -> Result<Vec<f32>, String> {
        let accumulator = Arc::new(Mutex::new(Vec::new()));
        let is_completed = Arc::new(AtomicBool::new(false));
        let worker_is_completed = Arc::clone(&is_completed);

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .expect("no output device available");

        let mut supported_configs_range = device
            .supported_input_configs()
            .expect("error while querying configs");
        let supported_config = supported_configs_range
            .next()
            .expect("no supported config?!")
            .with_sample_rate(cpal::SampleRate(16_000));

        let acc_vad = Arc::clone(&accumulator);

        let sr = supported_config.sample_rate().0;

        let mut vad = voice_activity_detector::VoiceActivityDetector::builder()
            .sample_rate(sr)
            .chunk_size(512usize)
            .build()
            .expect("failed making vad");

        // vad
        let mut handle = Some(thread::spawn(move || {
            let mut vad_level = 0; // 0 = not started talking, 1 = talking, 2 = stopped talking
            // maybe?, 3 yeah definitely stopped talking

            loop {
                let data: Vec<f32> = {
                    let gaurd = acc_vad.lock().expect("unlock failed");
                    let len = gaurd.len();

                    // only get the last 512 'cuz that's what this package supports
                    gaurd[len.saturating_sub(512)..].to_vec()
                };

                let probability = vad.predict(data);

                if probability > vad_threshold {
                    match vad_level {
                        0 => vad_level = 1,
                        2 => vad_level = 1,
                        _ => {}
                    }
                }

                if probability < vad_threshold {
                    match vad_level {
                        0 => {
                            // clear buffer before some past once detection starts to minimize data
                            // and only have the speech parts in the recording
                            let keep_past_seconds = 5;

                            let mut gaurd = acc_vad
                                .lock()
                                .expect("failed to get lock while pruning buffer");
                            let len = gaurd.len();

                            if len > (sr as usize) * keep_past_seconds {
                                gaurd.drain(0..(len - (sr as usize) * keep_past_seconds));
                            }
                        }
                        1 => vad_level = 2,
                        2 => vad_level = 3,
                        _ => {}
                    }
                }

                if vad_level == 3 {
                    worker_is_completed.store(true, Ordering::Relaxed);
                    break;
                }

                // this is how often we poll for cutoff
                thread::sleep(Duration::from_millis(300));
            }
        }));

        let acc = Arc::clone(&accumulator);

        let stream = device.build_input_stream(
            &supported_config.config(),
            move |data: &[f32], _| {
                if let Ok(mut gaurd) = acc.lock() {
                    gaurd.extend_from_slice(data);
                }
            },
            move |err| {
                eprintln!("an error occurred on the output audio stream: {}", err);
            },
            None,
        );

        match stream {
            Ok(value) => {
                let is_play = value.play();

                if is_play.is_err() {
                    eprintln!("error recording");
                }

                // keep alive
                loop {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

                    let flag = is_completed.load(Ordering::Acquire);
                    if flag && let Some(h) = handle.take() {
                        h.join().expect("failed to reunify the thread");

                        return Ok(accumulator
                            .lock()
                            .expect("failed to get accumulator while returning from vad")
                            .to_vec());
                    }
                }
            }

            Err(err) => {
                eprintln!("error making stream: {}", err);
                return Err("error making stream".to_string());
            }
        }
    }
}
