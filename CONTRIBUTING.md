# Contributing to ComfyUI-NoBG

Bug reports, documentation improvements, example workflows, and code contributions
are welcome. Keep changes focused and include tests for fixes or new behavior.

## Development setup

Use Python 3.11 or newer. From a clone of this repository, create a virtual
environment:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS/Linux, or
`.venv\Scripts\Activate.ps1` in PowerShell. Then install the lightweight test
dependencies and CPU-only PyTorch builds:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy pillow pytest ruff
```

Unit tests do not require ComfyUI, nobg, model weights, or credentials. They stub
ComfyUI's APIs and nobg in `tests/conftest.py`. For development inside an existing
ComfyUI environment, keep its hardware-appropriate PyTorch installation instead
of replacing it with CPU builds.

## Run checks

Run these commands from the repository root:

```bash
ruff check .
pytest
```

The pytest configuration keeps collection inside `tests/` because the repository
name contains a hyphen. Package-loading tests separately exercise ComfyUI's
importlib loading mechanism; do not change the root `__init__.py` to use an
absolute `import nodes` just to satisfy pytest.

Backwards-compatibility CI checks for removed node registrations and changed or
removed output slots. To compare two local checkouts:

```bash
python tests/check_compatibility.py --base /path/to/base-checkout --current .
```

This is a static contract check, not a complete check of input compatibility or
runtime behavior. Keep registered node names and output-slot order stable.
Keep the node UI minimal: prefer automatic hardware/model settings and avoid
exposing specialized tuning controls in the everyday workflow. Input changes
need explicit upgrade notes and updated UI/API examples. Version 0.1.2 deliberately
trims the loader and removal inputs; see the README's upgrade instructions.

## Real ComfyUI smoke test

Use a separate ComfyUI installation and an environment containing both
repositories' dependencies. Install compatible `torch` and `torchvision` builds
first, then:

```bash
python -m pip install -r /path/to/ComfyUI/requirements.txt
python -m pip install -r requirements.txt
```

Place or link this repository at `ComfyUI/custom_nodes/ComfyUI-NoBG`. The folder
name matters because the test runner whitelists it. From this repository, run:

```bash
python tests/run_workflow.py --comfyui /path/to/ComfyUI --output artifacts
```

The runner starts its own localhost-only CPU server, runs the API-format FeyNobg
workflow, validates the saved RGBA PNG, and shuts the server down. The first run
downloads public model weights into the Hugging Face cache. Logs, execution
history, validation results, and output images are written to the ignored
`artifacts/` directory.

GitHub Actions runs the same smoke test on a pinned ComfyUI revision. No GPU
runner, registry token, or cloud-storage credentials are needed. This tests
integration and transparency, not matting quality. Gated SAM3 inference is not
part of default CI; use the optional checks below when changing SAM3 behavior.

### Extended photo and SAM3 checks

Use your own dog photo with foreground and background visible. Keep private
inputs and generated results under ignored `artifacts/`. With CUDA-enabled
PyTorch installed, run:

```bash
python tests/run_workflow.py --comfyui /path/to/ComfyUI --image artifacts/dog.png --device cuda --extended --sam3 --output artifacts/photo-checks
```

Omit `--sam3` to test only FeyNobg. Use `--device cpu` for CPU-only testing.
For SAM3, accept Meta's model license yourself and authenticate with
`hf auth login` before downloading gated weights. Never use the registry API
key as a Hugging Face token.

The extended checks compare single and batched FeyNobg results using automatic
per-image processing and offloading. They also compare integrated and standalone
foreground refinement and check that alpha is preserved. SAM3 checks cover dog versus grass text
prompts, the default prompt, and batching. Legacy box/aggregation/threshold
behavior is covered by direct Python unit tests, not the trimmed ComfyUI interface.
Each case writes its own outputs, validation JSON, and execution history.
These checks need more memory and time than default CI
and do not replace visual review across varied photos.

## Code and examples

- Follow the surrounding Python style and keep changes scoped to the task.
- Add regression tests in `tests/` for behavioral changes.
- Use ComfyUI's interruption and device-management APIs for long-running work.
- Preserve ComfyUI image/mask tensor shapes and test batch behavior.
- Keep `requirements.txt` and `[project].dependencies` in `pyproject.toml` aligned.
- Update the README when changing node inputs, outputs, or installation steps.
- Put UI-format examples in `example_workflows/`; use same-stem `.jpg` thumbnails
  for the template browser. Keep the CI workflow in API format under
  `tests/workflows/`.
- Attribute bundled images and avoid including model weights or large generated
  files in the repository. Label illustrative cards separately from actual
  inference results.

## Issues and pull requests

For a bug report, include:

- Operating system, Python, ComfyUI, PyTorch, and nobg versions.
- ComfyUI device/launch settings, selected model, and relevant node options.
- Minimal reproduction steps or a small workflow.
- The traceback and relevant log output, with credentials and personal paths
  removed where appropriate.

For a PR, summarize the change, explain how you tested it, and note any model or
hardware requirements. Include before/after examples for output-quality changes
when practical. Run lint and unit tests before requesting review.

Never commit `.env`, access tokens, private inputs, or cached model files. Do not
paste secrets into issues, workflow JSON, screenshots, or logs. Publishing uses
the repository's `REGISTRY_ACCESS_TOKEN` GitHub Actions secret; contributors do
not need it to run tests.

## Releases

Maintainers publish under the `chonk-lain` registry publisher. The GitHub
repository remains `feyninc/ComfyUI-NoBG`. Publishing is manual-only through
**Publish to Comfy registry → Run workflow** in GitHub Actions, and requires unit
and end-to-end checks to pass. Pushes and PRs never publish. Each release needs
a new semantic version. Coordinate release/version
changes with maintainers rather than including them in unrelated PRs.

Registry versions cannot be overwritten. If publishing reports
`The node version already exists`, choose a new, unused version in
`pyproject.toml`. Put that change on the release branch, then start a new workflow
run. Re-running an old run uses its original commit and version.

Release checklist:

1. Bump `[project].version` in `pyproject.toml`; keep the
   dependency list aligned with `requirements.txt`.
2. Update README upgrade notes and bundled workflows for any input changes.
   Existing widget arrays are positional, so removing controls can require users
   to re-create affected nodes even when names and output slots stay unchanged.
3. Run lint, unit tests, compatibility checks, and the real ComfyUI smoke test.
4. After checks pass on the release commit, run the manual publishing workflow.
5. Verify the version and its status on the [registry page](https://registry.comfy.org/publishers/chonk-lain/nodes/comfyui-nobg).
   The [registry version API](https://api.comfy.org/nodes/comfyui-nobg/versions)
   also reports each version's status. Resolve a flagged release with the registry
   before recommending it to users.
6. Test a fresh installation in a separate ComfyUI environment with
   `comfy node install comfyui-nobg`. Separately test an upgrade from the previous
   release with `comfy node update comfyui-nobg`. Check the selected installation
   with `comfy which`, or pass `--workspace /path/to/ComfyUI` before `node`.
7. Restart the test ComfyUI backend to apply any queued update, then run a
   background-removal workflow. Check
   that the saved PNG has transparency and that older workflows follow the
   README's upgrade instructions.

Keep development checkouts separate from registry installations. Link your clone
into a test installation for unreleased changes. Do not run the registry updater
over that link or install a second copy of this pack beside it.

## License

Contributions are provided under this repository's Apache-2.0 license. Model
weights and third-party assets may have their own licenses; preserve their
attributions and restrictions.
