#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


MODEL_ID = "openvla/openvla-7b"
UNNORM_KEY = "bridge_orig"

CAMERA_SOURCE = "/dev/video4"
DETECTION_SCRIPT = Path("measure_detection_stability.py")

DEFAULT_INSTRUCTION = "move the can to the top right"

SAMPLES = 20
DETECTION_TIMEOUT_SEC = 180


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "OpenVLA instruction check -> existing verified "
            "RT-DETR + D455 detection pipeline DRY RUN"
        )
    )

    parser.add_argument(
        "--camera",
        default=CAMERA_SOURCE,
    )

    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=SAMPLES,
    )

    return parser.parse_args()


def open_camera(source: str):
    if source.isdigit():
        camera_source = int(source)
    else:
        camera_source = source

    cap = cv2.VideoCapture(
        camera_source,
        cv2.CAP_V4L2,
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Camera could not be opened: {source}"
        )

    for _ in range(30):
        cap.read()
        time.sleep(0.03)

    return cap


def capture_frame(camera_source: str):
    cap = open_camera(camera_source)

    try:
        ok, frame = cap.read()

        if not ok or frame is None:
            raise RuntimeError(
                "Failed to capture RGB frame."
            )

        return frame

    finally:
        cap.release()


def run_openvla(
    frame_bgr: np.ndarray,
    instruction: str,
):
    print()
    print("[OPENVLA] Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    print("[OPENVLA] Loading model...")

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0")

    model.eval()

    rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB,
    )

    image = Image.fromarray(rgb)

    clean_instruction = (
        instruction.strip().rstrip(".")
    )

    prompt = (
        "In: What action should the robot take to "
        f"{clean_instruction}?\n"
        "Out:"
    )

    inputs = processor(
        prompt,
        image,
        return_tensors="pt",
    ).to(
        "cuda:0",
        dtype=torch.bfloat16,
    )

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        action = model.predict_action(
            **inputs,
            unnorm_key=UNNORM_KEY,
            do_sample=False,
        )

    torch.cuda.synchronize()
    latency = time.perf_counter() - start

    action = np.asarray(
        action,
        dtype=np.float64,
    ).reshape(-1)

    if action.shape != (7,):
        raise RuntimeError(
            f"Expected OpenVLA 7D action, got {action.shape}"
        )

    if not np.all(np.isfinite(action)):
        raise RuntimeError(
            f"OpenVLA action contains NaN or Inf: {action}"
        )

    return action, latency


def run_existing_detection(samples: int):
    if not DETECTION_SCRIPT.exists():
        raise FileNotFoundError(
            DETECTION_SCRIPT
        )

    print()
    print("=" * 84)
    print("EXISTING VERIFIED DETECTION PIPELINE")
    print("RT-DETR + D455 DEPTH + 3D BASE COORDINATE")
    print("NO ROBOT MOTION")
    print("=" * 84)

    command = [
        sys.executable,
        str(DETECTION_SCRIPT),
        "--samples",
        str(samples),
        "--timeout",
        str(DETECTION_TIMEOUT_SEC),
    ]

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            "Existing detection pipeline failed: "
            f"returncode={result.returncode}"
        )

    if (
        "FINAL STABILITY RESULT: PASS"
        not in result.stdout
    ):
        raise RuntimeError(
            "NO-GO: existing detection stability "
            "check did not pass"
        )

    mean_match = re.search(
        r"평균 object_base_xyz_mm\s*=\s*"
        r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*"
        r"([-+]?\d+(?:\.\d+)?)\s*,\s*"
        r"([-+]?\d+(?:\.\d+)?)\s*\)",
        result.stdout,
    )

    if mean_match is None:
        raise RuntimeError(
            "NO-GO: failed to parse stable "
            "object_base_xyz_mm"
        )

    base_xyz_mm = np.array(
        [
            float(mean_match.group(1)),
            float(mean_match.group(2)),
            float(mean_match.group(3)),
        ],
        dtype=np.float64,
    )

    return base_xyz_mm


def main():
    args = parse_args()

    print("=" * 84)
    print("OPENVLA -> EXISTING PICK PIPELINE DRY RUN")
    print("=" * 84)
    print("Camera            :", args.camera)
    print("Instruction       :", args.instruction)
    print("Detection samples :", args.samples)
    print("Robot motion      : DISABLED")
    print("=" * 84)

    frame = capture_frame(args.camera)

    output_dir = Path(
        "outputs/openvla_existing_pipeline"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_path = output_dir / "openvla_input.jpg"

    if not cv2.imwrite(
        str(image_path),
        frame,
    ):
        raise RuntimeError(
            f"Failed to save image: {image_path}"
        )

    print()
    print("[CAMERA]")
    print("Frame shape :", frame.shape)
    print("Saved image :", image_path.resolve())

    action, latency = run_openvla(
        frame,
        args.instruction,
    )

    raw_xyz_mm = action[:3] * 1000.0
    raw_rpy_deg = np.degrees(action[3:6])

    print()
    print("=" * 84)
    print("OPENVLA RESULT")
    print("=" * 84)
    print("Raw action :", action.tolist())
    print("XYZ [mm]   :", raw_xyz_mm.tolist())
    print("RPY [deg]  :", raw_rpy_deg.tolist())
    print("Gripper    :", float(action[6]))
    print("Latency    :", f"{latency:.3f} sec")

    print()
    print(
        "[IMPORTANT] OpenVLA XYZ is not used "
        "as the final object position."
    )
    print(
        "[IMPORTANT] Existing D455 Depth + "
        "Camera-to-Base pipeline determines "
        "the actual target position."
    )

    stable_base_xyz_mm = run_existing_detection(
        samples=args.samples,
    )

    # 현재 검증된 Pick & Place 설정과 동일한 값
    grasp_xy_offset_mm = np.array(
        [0.002, -17.0],
        dtype=np.float64,
    )

    final_grasp_z_mm = 168.0
    transfer_dx_mm = 150.0
    transfer_dy_mm = 50.0
    place_z_mm = 171.0

    grasp_x_mm = float(
        stable_base_xyz_mm[0]
        + grasp_xy_offset_mm[0]
    )

    grasp_y_mm = float(
        stable_base_xyz_mm[1]
        + grasp_xy_offset_mm[1]
    )

    place_x_mm = float(
        grasp_x_mm + transfer_dx_mm
    )

    place_y_mm = float(
        grasp_y_mm + transfer_dy_mm
    )

    workspace_file = Path(
        "can_workspace_calibration.json"
    )

    if not workspace_file.exists():
        raise RuntimeError(
            f"Missing workspace file: {workspace_file}"
        )

    workspace = json.loads(
        workspace_file.read_text(
            encoding="utf-8"
        )
    )

    bounds = workspace[
        "verified_object_bounds_mm"
    ]

    detected_ok = (
        float(bounds["x_min"])
        <= float(stable_base_xyz_mm[0])
        <= float(bounds["x_max"])
        and float(bounds["y_min"])
        <= float(stable_base_xyz_mm[1])
        <= float(bounds["y_max"])
        and float(bounds["z_min"])
        <= float(stable_base_xyz_mm[2])
        <= float(bounds["z_max"])
    )

    grasp_x_min = (
        float(bounds["x_min"])
        + float(grasp_xy_offset_mm[0])
    )

    grasp_x_max = (
        float(bounds["x_max"])
        + float(grasp_xy_offset_mm[0])
    )

    grasp_y_min = (
        float(bounds["y_min"])
        + float(grasp_xy_offset_mm[1])
    )

    grasp_y_max = (
        float(bounds["y_max"])
        + float(grasp_xy_offset_mm[1])
    )

    grasp_ok = (
        grasp_x_min <= grasp_x_mm <= grasp_x_max
        and grasp_y_min <= grasp_y_mm <= grasp_y_max
    )

    place_ok = (
        200.0 <= place_x_mm <= 400.0
        and grasp_y_min <= place_y_mm <= grasp_y_max
    )

    all_targets_finite = bool(
        np.all(
            np.isfinite(
                [
                    *stable_base_xyz_mm.tolist(),
                    grasp_x_mm,
                    grasp_y_mm,
                    final_grasp_z_mm,
                    place_x_mm,
                    place_y_mm,
                    place_z_mm,
                ]
            )
        )
    )

    runtime_go = (
        detected_ok
        and grasp_ok
        and place_ok
        and all_targets_finite
    )

    runtime_data = {
        "instruction": args.instruction,
        "openvla_raw_action": action.tolist(),
        "openvla_raw_xyz_mm": raw_xyz_mm.tolist(),
        "openvla_raw_rpy_deg": (
            raw_rpy_deg.tolist()
        ),
        "openvla_gripper": float(action[6]),
        "openvla_latency_sec": float(latency),
        "detected_reference_xyz_mm": (
            stable_base_xyz_mm.tolist()
        ),
        "grasp_xy_offset_mm": (
            grasp_xy_offset_mm.tolist()
        ),
        "grasp_target_tcp_mm": [
            grasp_x_mm,
            grasp_y_mm,
            final_grasp_z_mm,
        ],
        "place_target_tcp_mm": [
            place_x_mm,
            place_y_mm,
            place_z_mm,
        ],
        "transfer_dx_mm": transfer_dx_mm,
        "transfer_dy_mm": transfer_dy_mm,
        "workspace_checks": {
            "detected_inside_bounds": detected_ok,
            "grasp_inside_bounds": grasp_ok,
            "place_inside_bounds": place_ok,
            "all_values_finite": all_targets_finite,
        },
        "final_runtime_result": (
            "GO" if runtime_go else "NO-GO"
        ),
        "robot_motion": False,
        "gripper_command": False,
    }

    runtime_path = Path(
        "openvla_pick_runtime.json"
    )

    runtime_path.write_text(
        json.dumps(
            runtime_data,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 84)
    print("OPENVLA PICK RUNTIME TARGET — DRY RUN")
    print("=" * 84)
    print(
        "Detected XYZ :",
        stable_base_xyz_mm.tolist(),
    )
    print(
        "Grasp target :",
        [
            grasp_x_mm,
            grasp_y_mm,
            final_grasp_z_mm,
        ],
    )
    print(
        "Place target :",
        [
            place_x_mm,
            place_y_mm,
            place_z_mm,
        ],
    )
    print(
        "Detected guard:",
        "PASS" if detected_ok else "FAIL",
    )
    print(
        "Grasp guard  :",
        "PASS" if grasp_ok else "FAIL",
    )
    print(
        "Place guard  :",
        "PASS" if place_ok else "FAIL",
    )
    print(
        "Finite guard :",
        "PASS" if all_targets_finite else "FAIL",
    )
    print("Saved runtime :", runtime_path.resolve())
    print(
        "FINAL RUNTIME TARGET RESULT:",
        "GO" if runtime_go else "NO-GO",
    )
    print("NO ROBOT OR GRIPPER COMMAND SENT")
    print("=" * 84)

    if not runtime_go:
        raise RuntimeError(
            "NO-GO: runtime target validation failed"
        )

    print()
    print("=" * 84)
    print("FINAL INTEGRATION DRY-RUN RESULT")
    print("=" * 84)
    print("Instruction:")
    print(args.instruction)

    print()
    print("OpenVLA raw XYZ [mm]:")
    print(raw_xyz_mm.tolist())

    print()
    print("Verified object Base XYZ [mm]:")
    print(stable_base_xyz_mm.tolist())

    print()
    print("PASS:")
    print(
        "- OpenVLA received the live RGB frame."
    )
    print(
        "- OpenVLA generated a valid 7D action."
    )
    print(
        "- Existing RT-DETR + D455 Depth "
        "stability pipeline passed."
    )
    print(
        "- Stable Base-frame object position "
        "was obtained."
    )

    print()
    print("BLOCKED:")
    print(
        "- OpenVLA raw XYZ was not transmitted."
    )
    print(
        "- Pick & Place live script was not called."
    )
    print(
        "- No FAIRINO motion or gripper command "
        "was sent."
    )
    print("=" * 84)


if __name__ == "__main__":
    main()
