#!/usr/bin/env python3

import json
import math
import time
from pathlib import Path

import numpy as np

from fairino import Robot
from dh_gripper import DHGripper


ROBOT_IP = "192.168.58.2"

RUNTIME_FILE = Path("openvla_pick_runtime.json")
OBSERVATION_FILE = Path("observation_pose.json")
WORKSPACE_FILE = Path("can_workspace_calibration.json")

EXPECTED_INSTRUCTION = "move the can to the top right"

EXPECTED_TOOL = 1
EXPECTED_USER = 0

DH_PORT = "/dev/ttyUSB4"
DH_SLAVE_ID = 1
DH_BAUDRATE = 115200

GRIPPER_OPEN = 900

OBSERVATION_TOLERANCE_MM = 10.0
GRIPPER_TOLERANCE = 10

MAX_RUNTIME_AGE_SEC = 600.0

EXPECTED_FINAL_GRASP_Z = 168.0
EXPECTED_PLACE_Z = 171.0
EXPECTED_TRANSFER_DX = 150.0
EXPECTED_TRANSFER_DY = 50.0


def unpack(result, name):
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError(
            f"{name}: unexpected result={result}"
        )

    code = int(result[0])
    value = result[1]

    if code != 0:
        raise RuntimeError(
            f"{name} failed: code={code}, result={result}"
        )

    return value


def finite(values):
    array = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    return bool(
        array.size > 0
        and np.all(np.isfinite(array))
    )


def xyz_distance(first, second):
    return math.sqrt(
        sum(
            (
                float(first[index])
                - float(second[index])
            ) ** 2
            for index in range(3)
        )
    )


def add_check(checks, name, passed, detail):
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "detail": str(detail),
        }
    )


def main():
    print("=" * 92)
    print("OPENVLA + FAIRINO PICK & PLACE — FINAL READ-ONLY PREFLIGHT")
    print("=" * 92)
    print("Robot movement  : DISABLED")
    print("Gripper command : DISABLED")
    print("Purpose         : Validate LIVE inputs and current hardware state")
    print("=" * 92)

    checks = []

    for required_file in (
        RUNTIME_FILE,
        OBSERVATION_FILE,
        WORKSPACE_FILE,
    ):
        add_check(
            checks,
            f"File exists: {required_file}",
            required_file.is_file(),
            required_file.resolve(),
        )

    if not all(
        path.is_file()
        for path in (
            RUNTIME_FILE,
            OBSERVATION_FILE,
            WORKSPACE_FILE,
        )
    ):
        runtime = {}
        observation_data = {}
        workspace = {}
    else:
        runtime = json.loads(
            RUNTIME_FILE.read_text(
                encoding="utf-8"
            )
        )

        observation_data = json.loads(
            OBSERVATION_FILE.read_text(
                encoding="utf-8"
            )
        )

        workspace = json.loads(
            WORKSPACE_FILE.read_text(
                encoding="utf-8"
            )
        )

    runtime_age = (
        time.time()
        - RUNTIME_FILE.stat().st_mtime
        if RUNTIME_FILE.exists()
        else float("inf")
    )

    add_check(
        checks,
        "Runtime file age",
        runtime_age <= MAX_RUNTIME_AGE_SEC,
        (
            f"{runtime_age:.1f} sec "
            f"(limit={MAX_RUNTIME_AGE_SEC:.1f})"
        ),
    )

    instruction = str(
        runtime.get("instruction", "")
    ).strip()

    add_check(
        checks,
        "Instruction",
        instruction == EXPECTED_INSTRUCTION,
        (
            f"actual={instruction!r}, "
            f"expected={EXPECTED_INSTRUCTION!r}"
        ),
    )

    add_check(
        checks,
        "Runtime result",
        runtime.get("final_runtime_result") == "GO",
        runtime.get("final_runtime_result"),
    )

    workspace_checks = runtime.get(
        "workspace_checks",
        {}
    )

    required_runtime_checks = (
        "detected_inside_bounds",
        "grasp_inside_bounds",
        "place_inside_bounds",
        "all_values_finite",
    )

    for key in required_runtime_checks:
        add_check(
            checks,
            f"Runtime guard: {key}",
            workspace_checks.get(key) is True,
            workspace_checks.get(key),
        )

    add_check(
        checks,
        "Runtime robot_motion flag",
        runtime.get("robot_motion") is False,
        runtime.get("robot_motion"),
    )

    add_check(
        checks,
        "Runtime gripper_command flag",
        runtime.get("gripper_command") is False,
        runtime.get("gripper_command"),
    )

    action = runtime.get(
        "openvla_raw_action",
        [],
    )

    add_check(
        checks,
        "OpenVLA action shape",
        isinstance(action, list) and len(action) == 7,
        f"length={len(action) if isinstance(action, list) else 'invalid'}",
    )

    add_check(
        checks,
        "OpenVLA action finite",
        finite(action),
        action,
    )

    detected = runtime.get(
        "detected_reference_xyz_mm",
        [],
    )

    grasp = runtime.get(
        "grasp_target_tcp_mm",
        [],
    )

    place = runtime.get(
        "place_target_tcp_mm",
        [],
    )

    add_check(
        checks,
        "Detected XYZ valid",
        isinstance(detected, list)
        and len(detected) == 3
        and finite(detected),
        detected,
    )

    add_check(
        checks,
        "Grasp target valid",
        isinstance(grasp, list)
        and len(grasp) == 3
        and finite(grasp),
        grasp,
    )

    add_check(
        checks,
        "Place target valid",
        isinstance(place, list)
        and len(place) == 3
        and finite(place),
        place,
    )

    if len(grasp) == 3:
        add_check(
            checks,
            "Final grasp Z",
            abs(
                float(grasp[2])
                - EXPECTED_FINAL_GRASP_Z
            ) <= 0.001,
            f"{float(grasp[2]):.3f} mm",
        )

    if len(place) == 3:
        add_check(
            checks,
            "Place Z",
            abs(
                float(place[2])
                - EXPECTED_PLACE_Z
            ) <= 0.001,
            f"{float(place[2]):.3f} mm",
        )

    if len(grasp) == 3 and len(place) == 3:
        transfer_dx = (
            float(place[0])
            - float(grasp[0])
        )

        transfer_dy = (
            float(place[1])
            - float(grasp[1])
        )

        add_check(
            checks,
            "Transfer +X distance",
            abs(
                transfer_dx
                - EXPECTED_TRANSFER_DX
            ) <= 0.001,
            f"{transfer_dx:.3f} mm",
        )

        add_check(
            checks,
            "Transfer +Y distance",
            abs(
                transfer_dy
                - EXPECTED_TRANSFER_DY
            ) <= 0.001,
            f"{transfer_dy:.3f} mm",
        )

    bounds = workspace.get(
        "verified_object_bounds_mm",
        {}
    )

    required_bound_keys = (
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "z_min",
        "z_max",
    )

    bounds_valid = all(
        key in bounds
        and math.isfinite(float(bounds[key]))
        for key in required_bound_keys
    )

    add_check(
        checks,
        "Workspace bounds valid",
        bounds_valid,
        bounds,
    )

    if (
        bounds_valid
        and isinstance(detected, list)
        and len(detected) == 3
    ):
        detected_inside = (
            float(bounds["x_min"])
            <= float(detected[0])
            <= float(bounds["x_max"])
            and float(bounds["y_min"])
            <= float(detected[1])
            <= float(bounds["y_max"])
            and float(bounds["z_min"])
            <= float(detected[2])
            <= float(bounds["z_max"])
        )

        add_check(
            checks,
            "Detected XYZ inside calibrated workspace",
            detected_inside,
            detected,
        )

    observation = observation_data.get(
        "tcp_pose",
        [],
    )

    add_check(
        checks,
        "Observation pose valid",
        isinstance(observation, list)
        and len(observation) == 6
        and finite(observation),
        observation,
    )

    print()
    print("[CONNECTING FAIRINO — READ ONLY]")

    robot = Robot.RPC(ROBOT_IP)
    time.sleep(1.0)

    emergency = int(unpack(
        robot.GetRobotEmergencyStopState(),
        "Emergency",
    ))

    safety_stop = list(unpack(
        robot.GetSafetyStopState(),
        "SafetyStop",
    ))

    robot_error = list(unpack(
        robot.GetRobotErrorCode(),
        "RobotError",
    ))

    safety_code = int(
        robot.GetSafetyCode()
    )

    tool = int(unpack(
        robot.GetActualTCPNum(),
        "Tool",
    ))

    user = int(unpack(
        robot.GetActualWObjNum(),
        "User",
    ))

    current_tcp = list(map(
        float,
        unpack(
            robot.GetActualTCPPose(),
            "Current TCP",
        )[:6],
    ))

    add_check(
        checks,
        "Emergency stop",
        emergency == 0,
        emergency,
    )

    add_check(
        checks,
        "Safety stop",
        safety_stop == [0, 0],
        safety_stop,
    )

    add_check(
        checks,
        "Robot error",
        robot_error == [0, 0],
        robot_error,
    )

    add_check(
        checks,
        "Safety code",
        safety_code == 0,
        safety_code,
    )

    add_check(
        checks,
        "Tool frame",
        tool == EXPECTED_TOOL,
        tool,
    )

    add_check(
        checks,
        "User frame",
        user == EXPECTED_USER,
        user,
    )

    add_check(
        checks,
        "Current TCP finite",
        len(current_tcp) == 6
        and finite(current_tcp),
        current_tcp,
    )

    if (
        isinstance(observation, list)
        and len(observation) == 6
        and len(current_tcp) == 6
    ):
        observation_error = xyz_distance(
            current_tcp,
            observation,
        )

        add_check(
            checks,
            "Robot near Observation pose",
            (
                observation_error
                <= OBSERVATION_TOLERANCE_MM
            ),
            (
                f"XYZ error={observation_error:.3f} mm, "
                f"limit={OBSERVATION_TOLERANCE_MM:.3f}"
            ),
        )

    print()
    print("[CONNECTING GRIPPER — READ ONLY]")

    gripper = DHGripper(
        port=DH_PORT,
        slave_id=DH_SLAVE_ID,
        baudrate=DH_BAUDRATE,
        timeout=0.2,
    )

    try:
        gripper_init = int(
            gripper.get_init_state()
        )
        gripper_ready = int(
            gripper.get_state()
        )
        gripper_position = int(
            gripper.get_position()
        )

        add_check(
            checks,
            "Gripper initialized",
            gripper_init == 1,
            gripper_init,
        )

        add_check(
            checks,
            "Gripper ready",
            gripper_ready == 1,
            gripper_ready,
        )

        add_check(
            checks,
            "Gripper open near 900",
            abs(
                gripper_position
                - GRIPPER_OPEN
            ) <= GRIPPER_TOLERANCE,
            (
                f"position={gripper_position}, "
                f"expected={GRIPPER_OPEN}"
            ),
        )

    finally:
        if hasattr(gripper, "close_port"):
            gripper.close_port()

    print()
    print("=" * 92)
    print("FINAL PREFLIGHT REPORT")
    print("=" * 92)

    for check in checks:
        state = (
            "PASS"
            if check["passed"]
            else "FAIL"
        )

        print(
            f"[{state}] "
            f"{check['name']:<42} : "
            f"{check['detail']}"
        )

    passed_count = sum(
        1
        for check in checks
        if check["passed"]
    )

    failed_checks = [
        check
        for check in checks
        if not check["passed"]
    ]

    print("-" * 92)
    print(
        f"Checks passed : "
        f"{passed_count}/{len(checks)}"
    )

    if failed_checks:
        print()
        print("FINAL OPENVLA PICK & PLACE PREFLIGHT: NO-GO")
        print("실제 MoveL / ServoCart / Gripper 명령 전송 금지")

        for check in failed_checks:
            print(
                "  -",
                check["name"],
                ":",
                check["detail"],
            )
    else:
        print()
        print("FINAL OPENVLA PICK & PLACE PREFLIGHT: GO")
        print("LIVE 입력과 현재 하드웨어 상태가 일치합니다.")
        print(
            "아직 MoveL / ServoCart / Gripper 명령은 "
            "전송되지 않았습니다."
        )

    print("=" * 92)


if __name__ == "__main__":
    main()
