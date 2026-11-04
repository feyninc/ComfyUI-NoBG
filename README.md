# ComfyUI-NoBG

ComfyUI nodes for [nobg](https://github.com/feyninc/nobg), an open-source library for background removal and image matting.

## Installation

Install from the [Comfy Registry](https://registry.comfy.org/publishers/chonk-lain/nodes/comfyui-nobg)
with [comfy-cli](https://github.com/Comfy-Org/comfy-cli):

```bash
comfy node install comfyui-nobg
```

Or install manually:

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

Device selection follows ComfyUI, the checkpoint keeps its own dtype, and Hub
models use the Hugging Face cache. Architecture is detected from the preset or
folder name; include `sam3` in a local SAM3 folder's name. SAM3 uses semantic matting.

### NoBG Remove Background

Takes `IMAGE` (batches and video frames work) and outputs:

- **cutout**: RGBA `IMAGE`. Feed it to *Save Image* to get a transparent PNG.
- **foreground**: RGB `IMAGE`. These are the refined colors when `refine_foreground` is on.
- **mask**: the alpha matte as a `MASK`.

Options:

- **refine_foreground**: unmixes leftover background color from soft edges to reduce halos, using a radius of 90. Use the separate refinement node if you need a different radius.
- **prompt**: SAM3 only, e.g. `the dog`. Leave it empty to use the model's default "main subject" prompt.

Batches and video frames are processed one image at a time to limit memory use.
The model moves to ComfyUI's offload device after inference.

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

## Upgrading to 0.1.2

The loader now shows only model selection. Background removal keeps only the
foreground-refinement toggle and optional SAM3 text prompt; hardware overrides,
batch-size tuning, download location, refinement radius, model-retention, boxes,
and SAM3 aggregation/threshold widgets have been removed.

Node names and output slots are unchanged, but old widget values may not map
correctly to the trimmed inputs. Re-create the loader and removal nodes in older
workflows (or start from the updated examples), reconnect them, and check the
refinement toggle and prompt before running. Removed API inputs no longer configure
execution through ComfyUI.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests, and pull-request guidelines.

## License

Apache-2.0. Model weights have their own licenses: SAM3 uses Meta's SAM License.
