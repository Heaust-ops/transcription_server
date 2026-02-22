pub(crate) use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

pub struct Transcriber {
    whisper_state: WhisperState,
}

impl Transcriber {
    pub fn new(path_to_whisper: &str) -> Self {
        let whisper_context =
            WhisperContext::new_with_params(path_to_whisper, WhisperContextParameters::default())
                .expect("failed to load whisper context");

        let whisper_state = whisper_context
            .create_state()
            .expect("failed to create state");

        Self { whisper_state }
    }

    pub fn transcribe(&mut self, samples: Vec<f32>) -> String {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("en"));

        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        self.whisper_state
            .full(params, &samples[..])
            .expect("failed to run model");

        // fetch the results
        let num_segments = self.whisper_state.full_n_segments();

        let mut t = "".to_string();

        for i in 0..num_segments {
            // fetch the results
            let segment = self
                .whisper_state
                .get_segment(i)
                .expect("failed to get segment");

            t += &segment.to_string();
        }

        t
    }
}
