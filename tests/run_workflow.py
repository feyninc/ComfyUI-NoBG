"""Run the real API workflow against an isolated ComfyUI server (CPU by default).

Install this repository in ComfyUI/custom_nodes before running this script.
The default smoke test needs no credentials. Optional SAM3 checks use authorized
Hugging Face access or existing local cached weights.
"""

import argparse
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import uuid

from PIL import Image

HERE = Path(__file__).resolve().parent
NODE_TYPES = {"NoBGModelLoader", "NoBGRemoveBackground", "NoBGRefineForeground"}


def request_json(base_url, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(base_url + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=15) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"ComfyUI {path}: HTTP {error.code}: {error.read().decode()}") from error


def validate_png(data, expected_size):
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGBA" or image.size != expected_size:
            raise RuntimeError(f"Expected RGBA PNG {expected_size}; got {image.format} {image.mode} {image.size}")
        low, high = image.getchannel("A").getextrema()
        if low >= 250 or high <= 5 or high - low < 32:
            raise RuntimeError(f"Expected foreground and background in the alpha channel; got range {low}..{high}")
        return {"width": image.width, "height": image.height, "mode": image.mode, "alpha_range": [low, high]}


def run(comfyui, output, timeout, image_path=None, device="cpu", extended=False, sam3=False):
    output.mkdir(parents=True, exist_ok=True)
    # Use a fresh input name and prompt ID to avoid accepting stale results.
    image_name = f"nobg-ci-{uuid.uuid4().hex}.png"
    input_path = comfyui / "input" / image_name
    input_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path or HERE / "inputs" / "example.png") as image:
        image.convert("RGB").save(input_path)
    with Image.open(input_path) as image:
        expected_size = image.size
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    command = [
        sys.executable, str(comfyui / "main.py"), "--listen", "127.0.0.1", "--port", str(port),
        "--disable-auto-launch", "--disable-all-custom-nodes", "--whitelist-custom-nodes", "ComfyUI-NoBG",
        "--offline", "--output-directory", str(output),
    ]
    if device == "cpu":
        command.append("--cpu")
    env = dict(os.environ, PYTHONIOENCODING="utf8")
    process = None
    try:
        with (output / "comfyui.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=comfyui, stdout=log, stderr=subprocess.STDOUT, env=env)
            deadline = time.monotonic() + min(timeout, 180)
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"ComfyUI exited with {process.returncode}; see comfyui.log")
                try:
                    info = request_json(base_url, "/object_info")
                    break
                except (URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("ComfyUI did not start within 180 seconds")
                    time.sleep(1)
            missing = NODE_TYPES - info.keys()
            if missing:
                raise RuntimeError(f"Custom nodes did not load: {sorted(missing)}; see comfyui.log")

            workflow = json.loads((HERE / "workflows" / "NoBG_remove_background_api.json").read_text())
            workflow["1"]["inputs"]["image"] = image_name
            workflow["2"]["inputs"].update(device=device, dtype="fp32")
            workflow["4"]["inputs"]["filename_prefix"] = "nobg-ci"

            def execute_workflow(graph, label, expected_count=1):
                case_dir = output if label == "baseline" else output / label
                case_dir.mkdir(exist_ok=True)
                graph["4"]["inputs"]["filename_prefix"] = f"{label}/nobg"
                queued = request_json(base_url, "/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})
                if queued.get("node_errors"):
                    raise RuntimeError(f"Workflow validation failed: {queued['node_errors']}")
                prompt_id = queued["prompt_id"]
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"ComfyUI exited with {process.returncode} during inference")
                    history = request_json(base_url, f"/history/{prompt_id}").get(prompt_id)
                    if history is not None:
                        (case_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
                        status = history.get("status", {})
                        if status.get("status_str") == "error":
                            raise RuntimeError(f"Workflow execution failed: {status.get('messages')}")
                        if status.get("completed"):
                            break
                    time.sleep(2)
                else:
                    raise TimeoutError(f"Workflow did not complete within {timeout} seconds")

                images = history.get("outputs", {}).get("4", {}).get("images", [])
                if len(images) != expected_count:
                    raise RuntimeError(f"Expected {expected_count} output images, got {len(images)}")
                reports, results = [], []
                for index, saved in enumerate(images):
                    if saved.get("type") != "output":
                        raise RuntimeError(f"Unexpected output type: {saved}")
                    with urlopen(base_url + "/view?" + urlencode(saved), timeout=15) as response:
                        data = response.read()
                    reports.append(validate_png(data, expected_size))
                    (case_dir / f"cutout-{index}.png").write_bytes(data)
                    with Image.open(io.BytesIO(data)) as image:
                        results.append(image.copy())
                (case_dir / "validation.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
                print(f"{label} passed: {json.dumps(reports)}", flush=True)
                return results

            baseline = execute_workflow(workflow, "baseline")[0]
            if extended or sam3:
                run_additional_checks(workflow, execute_workflow, baseline, extended, sam3, expected_size, output)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        input_path.unlink(missing_ok=True)


def run_additional_checks(workflow, execute_workflow, baseline, extended, sam3, size, output):
    import copy

    import numpy as np

    checks = {}
    if extended:
        batch = copy.deepcopy(workflow)
        batch["5"] = {"class_type": "ImageBatch", "inputs": {"image1": ["1", 0], "image2": ["1", 0]}}
        batch["3"]["inputs"].update(image=["5", 0], batch_size=2, keep_model_loaded=False)
        images = execute_workflow(batch, "birefnet-batch-offload", expected_count=2)
        delta = max(int(np.abs(np.asarray(image).astype(int) - np.asarray(baseline).astype(int)).max()) for image in images)
        if delta > 2:
            raise RuntimeError(f"Batched and single-image output differ by {delta}/255")
        checks["batch_max_pixel_difference"] = delta

        refined = copy.deepcopy(workflow)
        refined["3"]["inputs"].update(refine_foreground=True, refine_radius=90)
        integrated = execute_workflow(refined, "birefnet-refined")[0]
        standalone = copy.deepcopy(workflow)
        standalone["5"] = {
            "class_type": "NoBGRefineForeground", "inputs": {"image": ["1", 0], "mask": ["3", 2], "radius": 90},
        }
        standalone["4"]["inputs"]["images"] = ["5", 1]
        separate = execute_workflow(standalone, "standalone-refined")[0]
        if integrated.getchannel("A").tobytes() != baseline.getchannel("A").tobytes():
            raise RuntimeError("Foreground refinement changed the alpha matte")
        delta = int(np.abs(np.asarray(integrated).astype(int) - np.asarray(separate).astype(int)).max())
        if delta > 1:
            raise RuntimeError(f"Standalone and integrated refinement differ by {delta}/255")
        checks["refinement_max_pixel_difference"] = delta
        checks["refinement_changes_rgb"] = bool(np.any(np.asarray(integrated)[..., :3] != np.asarray(baseline)[..., :3]))
        if not checks["refinement_changes_rgb"]:
            raise RuntimeError("Refinement did not change foreground RGB on this test image")

    if sam3:
        graph = copy.deepcopy(workflow)
        graph["2"]["inputs"].update(model="facebook/sam3", sam3_aggregate="semantic")
        graph["3"]["inputs"].update(prompt="the dog", keep_model_loaded=False)
        dog = execute_workflow(graph, "sam3-dog")[0]
        graph["3"]["inputs"]["prompt"] = "the grass"
        grass = execute_workflow(graph, "sam3-grass")[0]
        difference = float(np.abs(np.asarray(dog)[..., 3].astype(float) - np.asarray(grass)[..., 3]).mean())
        if difference < 5:
            raise RuntimeError(f"Dog and grass prompts produced nearly identical masks: mean delta {difference}")
        checks["sam3_prompt_mean_alpha_difference"] = difference
        graph["3"]["inputs"].update(prompt="the dog", boxes=f"{size[0]*0.05},{size[1]*0.02},{size[0]*0.95},{size[1]*0.98}")
        execute_workflow(graph, "sam3-text-box")
        graph["3"]["inputs"]["prompt"] = ""
        execute_workflow(graph, "sam3-box-only")
        graph["3"]["inputs"].update(prompt="", boxes="")
        execute_workflow(graph, "sam3-default-prompt")
        graph["3"]["inputs"].update(prompt="the dog", batch_size=2, image=["5", 0])
        graph["5"] = {"class_type": "ImageBatch", "inputs": {"image1": ["1", 0], "image2": ["1", 0]}}
        execute_workflow(graph, "sam3-batch", expected_count=2)
        graph["3"]["inputs"].update(image=["1", 0], batch_size=1, score_threshold=0.5)
        graph["2"]["inputs"]["sam3_aggregate"] = "max"
        execute_workflow(graph, "sam3-instance-threshold")

    (output / "extended-validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(f"Additional checks passed: {json.dumps(checks)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--image", type=Path, help="Use a real photo instead of the bundled logo")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--extended", action="store_true", help="Check batching, offloading, and both refinement paths")
    parser.add_argument("--sam3", action="store_true", help="Also test gated SAM3 on a dog photo (requires authorized HF login)")
    args = parser.parse_args()
    if args.sam3 and args.image is None:
        parser.error("--sam3 requires --image pointing to a real dog photo")
    run(args.comfyui.resolve(), args.output.resolve(), args.timeout, args.image, args.device, args.extended, args.sam3)


if __name__ == "__main__":
    main()
