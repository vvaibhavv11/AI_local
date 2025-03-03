use std::num::NonZero;

use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    sampling::LlamaSampler,
};
use serde::Serialize;
use tauri::{ipc::Channel, Manager, State};
use tokio::{sync::Mutex, task::spawn_blocking};

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase", tag = "event", content = "data")]
enum AiResponse {
    Streaming { response: String },
}

struct ModelInfo {
    ctx_params: Option<LlamaContextParams>,
    backend: Option<LlamaBackend>,
    model: Option<LlamaModel>,
}

impl Default for ModelInfo {
    fn default() -> Self {
        ModelInfo {
            ctx_params: None,
            backend: None,
            model: None,
        }
    }
}

#[tauri::command(rename_all = "snake_case")]
async fn load_model(state: State<'_, Mutex<ModelInfo>>, model_path: String) -> Result<bool, ()> {
    let result: (LlamaContextParams, LlamaModel, LlamaBackend) = spawn_blocking(move || {
        let backend = LlamaBackend::init()
            .map_err(|err| format!("Backend init error: {:?}", err))
            .unwrap();
        let params = LlamaModelParams::default();
        if params.use_mmap() {
            println!("ok it is using");
        }
        let ctx_params = LlamaContextParams::default();
        let model = LlamaModel::load_from_file(&backend, &model_path, &params)
            .map_err(|err| format!("Model load error: {:?}", err))
            .unwrap();
        Ok((ctx_params, model, backend))
    })
    .await
    .map_err(|join_err| format!("Thread panicked: {:?}", join_err))
    .unwrap()?; // join error
    let mut info = state.lock().await;
    let (ctx_params, model, backend) = result;

    info.ctx_params = Some(ctx_params);
    info.model = Some(model);
    info.backend = Some(backend);

    Ok(true)
}

#[tauri::command(rename_all = "snake_case")]
async fn response(
    state: State<'_, Mutex<ModelInfo>>,
    input: String,
    on_event: Channel<AiResponse>,
) -> Result<(), ()> {
    let prompt = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        input
    );
    let app_state = state.lock().await;
    let state_model = app_state.model.as_ref().unwrap();
    let state_backend = app_state.backend.as_ref().unwrap();
    let state_ctx_params = app_state.ctx_params.as_ref().unwrap();
    let mut ctx = state_model
        .new_context(
            &state_backend,
            state_ctx_params
                .clone()
                .with_n_ctx(NonZero::new(4096))
                .with_n_threads(4),
        )
        .expect("unable to create the llama_context");

    let tokens_list = state_model
        .str_to_token(&prompt, AddBos::Always)
        .unwrap_or_else(|_| panic!("failed to tokenize {prompt}"));
    let mut batch = LlamaBatch::new(4096, 1);
    // let n_len = 1024;

    let last_index = tokens_list.len() as i32 - 1;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        // llama_decode will output logits only for the last token of the prompt
        let is_last = i == last_index;
        batch.add(token, i, &[0], is_last).unwrap();
    }

    ctx.decode(&mut batch).expect("llama_decode() failed");

    let mut n_cur = batch.n_tokens();

    // The `Decoder`
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut sampler = LlamaSampler::greedy();

    loop {
        // sample the next token
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        {
            sampler.accept(token);

            // is it an end of stream?

            let output_bytes = state_model
                .token_to_bytes(token, Special::Tokenize)
                .unwrap();
            // use `Decoder.decode_to_string()` to avoid the intermediate buffer
            let mut output_string = String::with_capacity(32);
            let _decode_result = decoder.decode_to_string(&output_bytes, &mut output_string, false);
            print!("{output_string}");
            let _ = on_event.send(AiResponse::Streaming {
                response: output_string,
            });

            batch.clear();
            batch.add(token, n_cur, &[0], true).unwrap();
        }

        if token == state_model.token_eos() {
            eprintln!();
            break;
        }

        n_cur += 1;

        ctx.decode(&mut batch).expect("failed to eval");
    }

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![load_model, response])
        .setup(|state| {
            state.manage(Mutex::new(ModelInfo::default()));
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
