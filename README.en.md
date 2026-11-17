# CosLuma

**An AI photo finishing workspace for cosplay photography.** Discuss a direction, compare real samples, then finish a batch. Original identity details are composited into the generated image to preserve the person and photographic texture.

[简体中文](README.md) · English

[Download for Windows](https://github.com/Magnus0100/cosluma/releases/tag/v0.27.5) · [Third-party licenses](LICENSES/README.md) · [Development guidelines](AGENTS.md)

Windows x64 · Chinese UI · Source / local build 0.44.0 (not publicly released) · Public release 0.27.5

## What it does

- **Work through chat or controls.** Explicitly request three directions, a revision from a named sample, sample approval or batch production in chat. All manual controls remain available.
- **Explore with real samples.** Use the character, original photos and discussion to create three complete directions from one representative photo. Refine a selected sample individually.
- **Finish a consistent batch.** Apply the approved direction and sample to selected photos, with four concurrent photos by default. Reuse the approved sample when its original is included.
- **Preserve identity.** Align the original identity region with a similarity transform, match its lighting and color, and composite in floating point. Hair editing and subject relighting are independent options, both off by default.
- **Retouch saved results.** Optionally adjust skin, slim the face, repair or reshape hair, or edit the neck. After painting a local selection, use chat to record requirements, generate or save while keeping the base result and version history.
- **Adjust light and color through chat or controls.** Apply fixed full-frame exposure, luminance curves, warmth, green–magenta tint and saturation to a saved version, then save a separate tone version.
- **Manage projects and delivery.** Keep photos, discussions, samples and versions together. Choose a delivery version per photo or explicitly adopt a saved batch-color round, then copy existing PNG / JPEG files through controls or chat with a per-photo manifest.
- **Continue discussing saved photos.** Discuss the current photo on the delivery page, explicitly compare, adopt, adjust tone with fixed algorithms, or make one new revision along a selected branch.

## Get started

Download `CosLuma-0.27.5-windows-x64.zip`, **extract the entire archive**, and launch `CosLuma/CosLuma.exe`. The desktop package includes Python and image processing dependencies.

**Model services, weights and accounts are configured separately and are not bundled.** Basic finishing requires:

| Service | Requirement |
| --- | --- |
| Photography Agent | Chat Completions-compatible endpoint with system messages, image input, tool calling and JSON content output |
| Full-image editing | Image editing endpoint accepting multipart `image[]`; original plus style reference requires two input images; returns an image URL or `b64_json` |
| Segmentation and upscaling | ComfyUI with SAM3.1 and an upscaling model; both can share a service |

Services may run locally or remotely. Remote services receive the photos and prompts sent to them.

### Workflow

1. **Create a project and connect services.** Follow the setup wizard or import an existing configuration. “模型下载” in connection settings provides upstream links, target paths and commands.
2. **Import photos and enter the character.** Common image formats, HEIC / HEIF and camera RAW are supported. Checkboxes select photos for production; row highlighting only changes the preview. Choose a representative photo separately.
3. **Discuss and sample.** Describe what to keep and change. Enable hair editing or relighting if needed, then say “探索三个方向” or use its button. Ask for a revision from the viewed sample or explore three new directions.
4. **Approve and run the batch.** Finalize a complete sample, or explicitly ask to adopt the second sample and produce the batch. Production uses checked photos. A failed photo does not stop the others; its error appears at the relevant step.
5. **Choose versions and export.** Open “全部作品” from “作品” and select each photo's version or explicitly adopt a saved batch-color round. Set a delivery location, then export existing files through controls or chat with a per-photo manifest. Start optional beauty retouching, local editing or a single-photo revision when needed.

Opening projects, switching pages and browsing load saved records. Production, revisions and optional retouching start on user action.

### Interface layout

Connection settings switch directly between Photography assistant, Image editing, Segmentation and upscaling, and Save location, with Save always visible. Switching sections does not validate, save or call a service. Disabled relighting connections collapse their fields while retaining edits until save or cancel. When current production results exist, approval information and batch-production controls expand under Production settings; without production records they remain directly visible. Production records group optional edits under Adjust photo and saved-file actions under File; the photo list shows filenames and status, and retains failure details.

A fixed left rail provides Projects, Photos, Create and Works, with Settings at the bottom for connections and configuration import. The header shows the current page and project. The project home searches by name or character and displays recorded project status and a representative photo. Empty libraries, unmatched searches, projects without photos and unreadable projects have separate guidance. Photo filenames stay on one line with middle elision; production checkboxes, highlighted rows and the sample source remain independent. Organize and File menus keep selection and file actions available; save location and output size are under More settings. Creative requirements, saved rounds and comparisons expand when needed.

Delivery keeps the contact sheet on the left, the photo in the center and conversation on the right. Adjust photo groups optional edits; Delivery settings holds destination, formats, comparison choices and saved exports, while export stays at the bottom. Beauty, tone and local editing open with Conversation; sliders, references and detailed requirements remain under Manual adjustment / Manual settings. Required face selection, painting and generation actions remain directly available, and save stays outside the scrolling parameters. Photos open singly, with explicit comparison and true 100% viewing available.

Creative, delivery and all three editing pages share message bubbles and a composer. Messages retain their send-time viewing context and operation status. The UI keeps functional labels, actual status and necessary instructions, without promotional headings or suggested requests. Empty conversations show a short no-messages label. Ctrl + Enter sends and Enter inserts a newline. Navigation, expansion and browsing never call a model.

## Output and processing

The default requested **long edge is 6000 pixels**. Results are saved as **16-bit PNG with an sRGB profile** and a matching **8-bit JPEG**. Originals are read-only; results are saved separately.

```text
Original + character and direction → template prompt writer → independent prompt review → full-image generation
→ one model upscale if needed → SAM3.1 original identity region
→ SIFT similarity alignment → MKL color match / optional relighting
→ float32 mask composite → optional luminance curve → PNG + JPEG
```

- With hair editing off, the original entire head is preserved. With it on, the face, ears and specific solid headwear are preserved while generated hair is used. Original clothing and prop designs are retained by default.
- The base pipeline does not locally warp the face. Optional relighting modulates original pixels with a predicted lighting field, replacing normal MKL color matching.
- Full-image generation reconstructs the scene; the body, background and other regions may change. Fine hair, translucent materials, hard shadows and large geometry changes can leave visible seams.
- Model stages include 8-bit inputs and outputs; 16-bit saving does not imply original precision throughout. Upscaling runs once; insufficient coverage is saved at the actual available size.
- Failures keep existing files and error records. The user initiates another run. Models and devices follow connection settings.

The chat Agent receives reference-library and photography-operation tools for its current stage. Local Python calls run the existing pipeline; no code editing, Shell or arbitrary file tools are provided. Each message can request one photography operation. Questions and browsing do not start production, and restored conversation records never execute old operations. Creative chat supports exploration, revision, approval and batch production. Delivery chat supports comparing, adopting, adjusting tone, batch-saving light/color edits and making single-photo revisions. Tone-editor chat supports exposure, curves, warmth, green–magenta tint, saturation and explicitly requested saves. Beauty editing and local selection remain in their respective interfaces; local-editor chat operates within the painted selection.

Use the right-side **对话** tab in the project delivery view. Each message freezes its photo, right-hand delivery version, left-hand comparison and actual viewing mode; only images shown in that mode are attached. Browsing during a call never changes its target. The `request_saved_photo` tool accepts only existing version IDs for that photo. Comparison and adoption share the manual entry points; adoption preserves delivery inclusion. Tone requests pass directly to the existing editor without another model conversation, using the specified saved PNG as their base. Preview is the default; saving requires an explicit request. The saved result appears as the left-hand comparison while the right-hand delivery version stays unchanged. Discussion, input drafts and actual execution status are saved with the project. Restoration and browsing never replay operations.

An explicit request such as “make a revision from the left version with a darker background” uses the same single-photo production entry as **调整照片 → 重新创作这张**. The current message is passed unchanged as the new requirement. The selected version's underlying base result supplies its direction, JPEG style reference and previously adopted branch requirements; other branches and project requirements stay independent. Hair editing and subject relighting inherit the base settings unless this message explicitly changes their switches. Beauty, local-retouch and tone pixels are not carried into full-image revisions. After prompt writing and independent review, `finish_photo` runs once with the original photo as the identity source. The full image is regenerated, so other regions are not pixel-locked. The new result is automatically compared on the left for the photo captured when the message was sent; the right-hand delivery version and inclusion stay unchanged. Failures show the actual step and preserve files and records. Restoring a conversation never resumes old work.

The local editor opens its **对话** tab by default; use it after manually choosing hair repair, hair shaping or neck editing and painting a selection. `request_local_edit` can only append the current user message verbatim to the requirements, generate within the current selection or save an existing result. It cannot change the task, mask, shape guide, reference, engine or arbitrary files. Discussion attaches a selection-and-guide preview of the saved base; when viewing a retouched result, it also attaches the written-back result crop. A selected reference is labeled separately. Each message records its base, task, requirements, selection and guide hashes, and viewing mode. Ordinary discussion does not change the draft. “Record this first” only appends the original message, with later requirements overriding conflicts. Generation reuses `generate_local`, task-template writing and independent review, previewing by default. An explicit “generate and save” continues into saving, or a later “save the current result” saves without regenerating. Manual controls remain available, pixels outside the selection keep the base values, and saving does not adopt a delivery version. Each task keeps its own discussion, input and painted draft. Switching tasks, returning or restoring a project never replays operations; failure never automatically resumes generation or saving.

Click **光色调整** from a saved batch or project delivery to edit the selected saved PNG. Exposure spans ±2 EV. Warmth and green–magenta tint each span −1 to 1; positive values are warmer and more magenta respectively. These are relative RGB balance controls, not Kelvin temperature. Saturation spans 0 to 2, with 0 for monochrome and 1 for the original value. The fixed sequence is: decode to linear light → exposure → RGB gains normalized to preserve neutral-gray luminance → saturation around linear luminance → clip out-of-gamut values and encode to sRGB → full-frame luminance curve. The curve keeps fixed black/white endpoints and output controls at 25%, 50% and 75% luminance; their ranges are 0.05–0.40, 0.40–0.60 and 0.60–0.95. The same parameters apply throughout the image, including the face, without changing geometry or separately beautifying it. Strong exposure, color balance or saturation can clip channels. Manual editing needs no model. The editor's `adjust_tone` and delivery's `request_saved_photo` share parameters and the processing function; only `llm` interprets chat, with no image generation. Parameters are relative to the opened version; resetting preserves its pixels exactly: exposure, warmth and tint 0, saturation 1, curve 0.25/0.5/0.75. Changes save a draft; a separate version requires the save button or an explicit chat request. Preview and saving start from full-resolution float32 pixels, with native-size display images for 100% viewing. Hidden editors release full-resolution buffers; returning reloads the saved image and draft without replaying chat operations. A new version records its parent and all parameters and saves a 16-bit PNG plus matching JPEG. Delivery changes only when the user explicitly adopts a version.

In the delivery-opened color editor, **按此参数另存交付勾选照片** applies all current parameters to the saved version selected for each delivery-checked photo at click time. The ordinary single-photo editor opened from a batch does not offer this button. An explicit delivery-chat request to apply parameters and save checked photos uses `request_saved_photo` with `tone_batch`; targets, versions and parameters are frozen when the message is sent. Browsing or changing checkboxes during the call does not change that request. The batch reuses `save_tone`, reading, processing and saving one full-resolution float32 photo at a time and releasing its buffers before the next. There is no image generation, per-photo LLM call or automatic color matching. Identical parameters may not suit photos with different exposure or lighting; individual adjustment remains available. Successful versions appear on the comparison side while delivery choices and inclusion remain unchanged. Partial failures retain successful files and per-photo errors, without retries or claiming complete success. Project-local `tone-batches/*/batch.json` records actual bases, parents, parameters, files and status. **最近批量光色记录** in the delivery page’s **管理作品** menu reads the latest round; project restoration never reruns it.

**From saved results to delivery:** “设置交付位置” in the right-side “交付设置” tab stores a project-specific destination parent without exporting. “导出所选” then reuses it. “采用最近光调成功版” in “管理作品”, or an explicit chat request to adopt the latest saved round, sets each successful saved result as its delivery version, preserves inclusion and compares the previous version. Failed or missing results are reported individually; older versions do not stand in for them.

The bounded `complete_delivery` tool offers `adopt_batch` for a named recorded round, `export` for the delivery versions checked when the message was sent, and `adopt_batch_and_export` for adopting a round and exporting only its successfully adopted photos that were checked at send time. Each discussion includes the latest three actual round IDs, parameters and per-photo status. Targets and versions are frozen at send time; browsing or changing checkboxes during the call does not rewrite the request. The destination comes from the UI setting, with no arbitrary path argument for the Agent. Explicit PNG / JPEG requests affect only that export; otherwise the captured UI formats apply. Questions, comparisons and expressions of satisfaction do not authorize adoption or export, and do not start subsequent tasks.

Adoption and export share the manual local functions, with no image generation or re-encoding. Each new delivery directory contains `delivery.json`, updated after every photo. It records the captured selection, version IDs, actual saved and copied paths, dimensions, formats and success or failure; combined adoption/export also records adoption results. Partial failures keep copied files and actual errors while cleaning task temporary files. Later production or version selection does not change old deliveries. Project restoration only reads the location, choices, discussions and delivery records, without adopting or exporting again.

Full-image generation, Qwen / H3 local generation and BeautyGRPO skin-base generation use built-in task templates and separate prompt-writing and prompt-review subagents. Both calls use the `llm` connection with independent contexts. Review checks the prompt against the task, without scoring the finished photo or asking users to perform extra validation. A rejected prompt preserves responses and displays specific issues; it is not automatically rewritten or retried. BeautyGRPO retains its exact official instruction. Analysis, upscaling, sliders, compositing and saving keep their existing fixed implementations and bypass these prompt calls. Full-image generation adds one review to its existing writer call; local and skin-base generation each add two LLM calls. Records are saved as `generation-prompt.json` and raw responses in the operation's working directory.

The optional face-beauty editor opens **对话** by default, with sliders in **手动调整**. Select and analyze a face manually first. Chat cannot change the face range, model or prompt. The bounded `request_face_beauty` tool's `adjust` action sets skin strength, slimming strength and feature protection; unspecified values retain their send-time settings. Strengths 0–1 correspond to manual sliders 0–100. Adjustments reuse existing analysis, the saved skin base and fixed algorithms. Missing skin bases never trigger generation automatically. Only an explicit request to generate or regenerate uses `generate_skin`, through the existing BeautyGRPO entry, template writer and independent prompt review. Generation preserves current strengths unless the user explicitly changes them. Saving requires an explicit request and writes a native-size 16-bit PNG plus matching JPEG without adopting it for delivery. Resetting restores base pixels; manual controls remain available.

Each beauty discussion freezes its base, analysis session, parameters and actual viewing mode. Attachments contain only the before, after or both face crops selected in that mode; history retains the original context and actual operation status. Questions and expressions of satisfaction do not trigger edits. Discussion failures preserve existing usable results; generation failures cancel subsequent saving and retain the error without retrying. Neither the generation button nor chat automatically increases strength. Discussion and unsent text persist with the base. Hidden editors release full-resolution buffers; returning reads records and updates display without replaying operations. Explicitly started generation and saving can finish after leaving the page.

<details>
<summary><strong>Model service deployment and configuration</strong></summary>

### Configuration paths

The default configuration is `~/CosplayWorkbench/config/cosluma.json`. Override it with `--config` or `COSLUMA_CONFIG`. Projects default to `~/CosLuma/projects/`, each with its own `project.json` and working directory.

See [settings.example.json](providers/settings.example.json). Credentials can come from environment variables or separate files. When saved through the UI, the JSON stores only the credential file path.

| Setting | Purpose |
| --- | --- |
| `llm` / `image` | Photography Agent / full-image editing endpoint, model and credentials |
| `head` / `sr` | Segmentation / upscaling ComfyUI root URL and model name |
| `relight` / `beauty` / `local_edit` | Optional relighting / beauty / local editing |
| `workers` / `output_long_edge` | Concurrent photos / output long edge; defaults `4` / `6000` |
| `prompt_library` | Empty or `null` uses the built-in categorized reference library; a custom library is a JSON array with `title`, `text` and `source` per entry |

### Base ComfyUI service

The service needs `SAM3_Detect` and standard image upload, job submission and result endpoints. Configure model names to match the files actually loaded by the service:

| Directory | Default file |
| --- | --- |
| `models/checkpoints/` | `sam3.1_multiplex_fp16.safetensors` |
| `models/upscale_models/` | `4xNomos2_hq_atd.safetensors` |

For services that do not yet include the SAM3 text encoder 512-dimensional projection fix, apply the [projection patch](providers/patches/comfyui-sam3-clip-projection.patch) from the ComfyUI directory: `git apply --ignore-space-change <CosLuma>/providers/patches/comfyui-sam3-clip-projection.patch`, then restart.

### Optional models

Copy the required `providers/comfyui_cosluma_*` folders into the service's `ComfyUI/custom_nodes/`, install missing dependencies with its existing Python, and restart. Service nodes must be updated separately when updating the desktop app. The model download UI provides exact files and commands.

| Feature / node folder | Weights and default inference settings |
| --- | --- |
| Relighting · `comfyui_cosluma_relight` | Full PIXLRelight + Marigold-IID-Lighting v1.1; 512 analysis canvas, BF16 network, single-step FP16 Marigold, float32 lighting field; `cuda:0`, seed 42 |
| Beauty · `comfyui_cosluma_beauty` | Full BeautyGRPO LoRA + official `FluxKontextPipeline`; 1024 input, 28 steps, guidance 2.5, seed 42; FGTBT WFLW 98 landmarks and FaceXFormer skin parsing |
| Local editing · `comfyui_cosluma_local` | Default Qwen-Image-2.1 INT8 ConvRot, BF16 Qwen3-VL 8B and VAE; 1024 area resolution, 40 steps, cfg 1, Euler/simple, seed 42 |
| Local editing · `comfyui_cosluma_h3` | Optional MiniMax-H3-Image INT8 ConvRot, Qwen3-VL 32B NVFP4 AWQ, FP16 VAE; 2048 area resolution, 6 steps, cfg 1, Euler/simple, seed 42 |

Area resolution follows the input aspect ratio: approximately 1 MP / 4 MP respectively. Model files reside on the service machine and are not bundled with the desktop app.

**Relighting:** default directories are `models/cosluma_relight/pixlrelight` (upstream source, `model.safetensors`, `config.yaml`) and `models/cosluma_relight/marigold-iid-lighting-v1-1`. Configure `relight.model_directory`, `intrinsics_directory`, `device` and `seed`; `cpu` can also be explicitly selected. The node additionally needs `addict==2.4.0` and OmegaConf.

**Beauty:** place the full official Diffusers model in `models/diffusers/FLUX.1-Kontext-dev/`, the complete LoRA at `models/loras/beautygrpo.safetensors`, and `facexformer.safetensors` / `fgtbt.safetensors` in `models/cosluma_face/`. Extract inference weights from original analysis checkpoints with the [FaceXFormer converter](providers/convert_facexformer.py) and [FGTBT converter](providers/convert_fgtbt.py).

`beauty.memory=bnb_nf4` is the default layout for 16 GB VRAM: NF4 / double quant / BF16 computation for the Transformer, full BF16 T5 with Accelerate CPU offload, and BF16 CLIP / VAE; the complete LoRA is retained. The `cuda` layout puts the entire pipeline in VRAM. Set the analysis device through `beauty.analysis_device`; generation uses `beauty.inference_device=cuda:0`. On Windows, set `PYTORCH_ALLOC_CONF=backend:native` and `PYTORCH_CUDA_ALLOC_CONF=backend:native`, and start the service with `--disable-dynamic-vram --disable-cuda-malloc`. The combined desktop and participating services are required to stay within 30 GB peak system RAM.

**Local editing files:** place generation weights in `models/diffusion_models/`, text encoders in `models/text_encoders/`, and VAEs in `models/vae/`.

| Engine | Generation weights / text encoder / VAE |
| --- | --- |
| `local_edit.engine=qwen` | `qwen_image_2.1_int8_convrot.safetensors` / `qwen3vl_8b_bf16.safetensors` / `qwen_image_2.1_vae_bf16.safetensors` |
| `local_edit.engine=h3` | `minimax_h3_image_pruned_int8_convrot.safetensors` / `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` / `minimax_h3_image_vae_fp16.safetensors` |

Qwen's separate inference process explicitly initializes the service's installed ComfyUI DynamicVRAM and loads file-backed weights without duplicating the complete text encoder. The BF16 text encoder uses CUDA with CPU offloading and is released before loading the diffusion model; configured weights, precision and 40-step inference parameters are retained. VAE tiles are 512 with overlap of 64. H3 settings are under `local_edit.h3` and require native H3 support. Neck edits use the full photo; hair edits use a detail crop plus full-photo context at a 1536-pixel long edge. Both engines write back only within the painted selection at original size; H3 uses gradient-domain writeback. Local editing and beauty release the models used for each task. Saved results remain available for comparison, adjustment and saving as new versions.

</details>

## Source and interfaces

Use Python 3.12 and deploy model services separately:

```powershell
python -m pip install -r scripts/requirements-desktop.txt
python -B scripts/qt_workbench.py
```

The CLI provides `styles` (three samples), `stylize` (batch), `compose` (reuse a generated image) and `beautify` (retouch a saved result). Place `--root` and `--config` before the subcommand:

```powershell
python -B scripts/project_cli.py --root D:/output --config D:/config/cosluma.json styles --source D:/photos --sample D:/photos/a.jpg --character "Character name"
```

Use `CosLuma.exe --cli` for the packaged CLI. Source 0.28.0 removes the application's own HTTP server and `--server` entry point. The desktop calls Python APIs directly; external model services still use their configured HTTP interfaces.

- [photo_pipeline.py](scripts/photo_pipeline.py): sole single-photo entry `finish_photo` and batch entry `finish_batch`.
- [model_clients.py](scripts/model_clients.py): photography Agent, image editing and model service calls.
- [photography_tools.py](scripts/photography_tools.py): photography operations bound to sample IDs in the message context.
- [delivery_chat.py](scripts/delivery_chat.py): frozen saved-photo context and bounded comparison, adoption, tone, revision and group-delivery tools.
- [delivery_store.py](scripts/delivery_store.py): saved versions, explicit adoption and unchanged file-copy delivery with per-photo manifests.
- [local_chat.py](scripts/local_chat.py): discussion, generation and saving bound to the current local task and painted selection.
- [beauty_chat.py](scripts/beauty_chat.py) / [beauty_editor.py](scripts/beauty_editor.py): optional discussion after manual face selection, sharing parameter, generation and save entries with the controls.
- [tone_edit.py](scripts/tone_edit.py) / [tone_editor.py](scripts/tone_editor.py): fixed exposure, luminance curves and color adjustment, bounded chat tools, full-resolution editing and separate versions.
- [generation_prompts.py](scripts/generation_prompts.py) / [task templates](providers/generation_templates.json): prompt writing and independent review for generation.
- [image_io.py](scripts/image_io.py) / [head_composite.py](scripts/head_composite.py): image I/O, identity alignment and compositing.
- [providers/](providers/): service nodes and configuration template.

To build the desktop app, install the [build dependencies](scripts/requirements-build.txt) and run `python -m PyInstaller --noconfirm CosLuma.spec`. See [AGENTS.md](AGENTS.md) for development rules.

## License

Code is licensed under [GPL-3.0-or-later](LICENSE). This is a noncommercial, nonprofit project. Model weights and third-party components retain their own licenses; some weights have noncommercial or territorial restrictions. See [third-party licenses](LICENSES/README.md).
