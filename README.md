# ComfyUI-NoBG

ComfyUI nodes for [nobg](https://github.com/feyninc/nobg), an open-source library for background removal and image matting.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/feyninc/ComfyUI-NoBG
pip install -r ComfyUI-NoBG/requirements.txt
```

Requires Python ≥ 3.11 and `transformers` ≥ 5.5. Uses ComfyUI's existing `torch` and `torchvision` installation.

## Nodes

### NoBG Model Loader

Loads a model and outputs `NOBG_MODEL`.

- **model**: a Hub preset, or any folder in `ComfyUI/models/nobg/` (listed as `local: <name>`). A folder needs a `config.json`.
- **architecture**: `auto` works out the model type from the preset or folder name. If it guesses a local folder wrong, set it yourself.
- **device** / **dtype**: `auto` uses ComfyUI's device and keeps the checkpoint's own dtype.
- **download_to_models_dir**: downloads Hub presets into `models/nobg/` instead of the Hugging Face cache.
- **sam3_aggregate**: SAM3 only. `semantic` gives the sharpest matte. `max` and `mean` build the matte from instance masks.

### NoBG Remove Background

Takes `IMAGE` (batches and video frames work) and outputs:

- **cutout**: RGBA `IMAGE`. Feed it to *Save Image* to get a transparent PNG.
- **foreground**: RGB `IMAGE`. These are the refined colors when `refine_foreground` is on.
- **mask**: the alpha matte as a `MASK`.

Options:

- **refine_foreground** / **refine_radius**: unmixes leftover background color from soft edges, so you don't get halos when you composite onto a new background.
- **prompt**: SAM3 only, e.g. `the dog`. Leave it empty to use the model's default "main subject" prompt.
- **boxes**: SAM3 only. Write `x1,y1,x2,y2` pixel boxes, one per line. Every image in the batch uses the same boxes.
- **score_threshold**: SAM3 with the `max`/`mean` aggregate only. `-1` uses the model's config value.
- **keep_model_loaded**: turn this off to move the model off the GPU after each run.

### NoBG Refine Foreground

Runs nobg's foreground color estimation on any `IMAGE` + `MASK` pair, including masks from other nodes. It outputs the refined RGB image and an RGBA cutout.

## Example

```
Load Image ─► NoBG Remove Background ─► Save Image   (cutout → transparent PNG)
                 ▲                └──► mask → ImageCompositeMasked, etc.
NoBG Model Loader┘
```

Open a workflow from [`example_workflows/`](example_workflows/) in ComfyUI's
template browser and choose your image in **Load Image**. For SAM3, set the prompt
to the subject you want to cut out, such as `the dog`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests, and pull-request guidelines.

## License

Apache-2.0. Model weights have their own licenses: SAM3 uses Meta's SAM License.
