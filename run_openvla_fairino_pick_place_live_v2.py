#!/usr/bin/env python3

import json
import math
import time
from pathlib import Path

from fairino import Robot
from dh_gripper import DHGripper


ROBOT_IP = "192.168.58.2"

OBSERVATION_FILE = Path("observation_pose.json")
RUNTIME_FILE = Path("openvla_pick_runtime.json")
WORKSPACE_FILE = Path("can_workspace_calibration.json")

DH_PORT = "/dev/ttyUSB4"
DH_SLAVE_ID = 1
DH_BAUDRATE = 115200

EXPECTED_TOOL = 1
EXPECTED_USER = 0

GRIPPER_OPEN = 900
GRIPPER_GRASP = 560

APPROACH_Z = 255.0
PREGRASP_Z = 210.0
NEAR_GRASP_Z = 180.0
FINAL_GRASP_Z = 168.0
TRANSFER_Z = 255.0

TEST_LIFT_MM = 3.0
MAIN_LIFT_MM = 10.0

TRANSFER_DX = 150.0
TRANSFER_DY = 50.0
S_AMPLITUDE = 12.0

PLACE_Z = 171.0
RETREAT_Z = 191.0

START_TOLERANCE_MM = 10.0

METRICS_FILE = Path(
    "openvla_action_execution_metrics.json"
)

MOTION_METRICS = []


def unpack(result, name):
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError(
            f"{name}: unexpected result={result}"
        )

    code, value = result[0], result[1]

    if int(code) != 0:
        raise RuntimeError(
            f"{name} failed: code={code}"
        )

    return value


def finite(values):
    return all(math.isfinite(float(v)) for v in values)


def distance_xyz(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    )


def require_exact(message, expected):
    # Full-auto mode: no intermediate keyboard confirmation.
    # Safety state is still rechecked immediately before every MoveL.
    print()
    print("[AUTO CONTINUE]", expected)
    print(message)
    time.sleep(0.3)


def read_robot_status(robot):
    return {
        "emergency": int(
            unpack(
                robot.GetRobotEmergencyStopState(),
                "Emergency",
            )
        ),
        "safety_stop": list(
            unpack(
                robot.GetSafetyStopState(),
                "Safety stop",
            )
        ),
        "robot_error": list(
            unpack(
                robot.GetRobotErrorCode(),
                "Robot error",
            )
        ),
        "safety_code": int(
            robot.GetSafetyCode()
        ),
    }


def status_ok(status):
    return (
        status["emergency"] == 0
        and status["safety_stop"] == [0, 0]
        and status["robot_error"] == [0, 0]
        and status["safety_code"] == 0
    )


def assert_robot_safe(robot, stage):
    status = read_robot_status(robot)

    if not status_ok(status):
        raise RuntimeError(
            f"NO-GO at {stage}: {status}"
        )


def read_tcp(robot):
    return list(map(
        float,
        unpack(
            robot.GetActualTCPPose(),
            "GetActualTCPPose",
        )[:6],
    ))


def wait_until_target(
    robot,
    target,
    stage,
    timeout_sec=30.0,
    xyz_tolerance_mm=1.5,
):
    """
    GetRobotMotionDone()==1이고 최종 TCP가 목표 근처에
    연속 3회 들어올 때까지 기다린다.
    """

    deadline = time.monotonic() + float(timeout_sec)
    stable_count = 0
    last_tcp = None
    last_done = None

    # 명령이 컨트롤러에 반영될 시간을 조금 준다.
    time.sleep(0.05)

    while time.monotonic() < deadline:
        assert_robot_safe(
            robot,
            f"{stage} MOTION WAIT",
        )

        result = robot.GetRobotMotionDone()

        if (
            not isinstance(result, tuple)
            or len(result) < 2
        ):
            raise RuntimeError(
                f"{stage}: invalid GetRobotMotionDone "
                f"result={result}"
            )

        return_code = int(result[0])
        motion_done = int(result[1])

        if return_code != 0:
            raise RuntimeError(
                f"{stage}: GetRobotMotionDone failed "
                f"code={return_code}"
            )

        current = read_tcp(robot)
        xyz_error = distance_xyz(
            current,
            target,
        )

        last_tcp = current
        last_done = motion_done

        if (
            motion_done == 1
            and xyz_error <= xyz_tolerance_mm
        ):
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= 3:
            print(
                f"{stage} motion complete | "
                f"XYZ error={xyz_error:.3f} mm"
            )
            return current

        time.sleep(0.05)

    raise RuntimeError(
        f"NO-GO: {stage} motion timeout | "
        f"motion_done={last_done} | "
        f"last_tcp={last_tcp} | "
        f"target={target}"
    )


def move_l(
    robot,
    target,
    tool,
    user,
    velocity,
    override,
    stage,
    blend_radius=-1.0,
):
    assert_robot_safe(robot, stage)

    if len(target) != 6 or not finite(target):
        raise RuntimeError(
            f"NO-GO: invalid target at {stage}: {target}"
        )

    before_tcp = read_tcp(robot)

    print()
    print("=" * 78)
    print(stage)
    print("=" * 78)
    print("Current TCP :", before_tcp)
    print("Target TCP  :", target)
    print("Velocity    :", velocity)
    print("Override    :", override)

    ret = robot.MoveL(
        desc_pos=target,
        tool=tool,
        user=user,
        vel=float(velocity),
        acc=0.0,
        ovl=float(override),
        blendR=float(blend_radius),
    )

    print("MoveL return:", ret)

    if int(ret) != 0:
        raise RuntimeError(
            f"{stage} failed: MoveL={ret}"
        )

    actual = wait_until_target(
        robot=robot,
        target=target,
        stage=stage,
        timeout_sec=30.0,
        xyz_tolerance_mm=1.5,
    )

    print("Actual TCP  :", actual)

    command_delta = [
        float(target[i]) - float(before_tcp[i])
        for i in range(6)
    ]

    measured_delta = [
        float(actual[i]) - float(before_tcp[i])
        for i in range(6)
    ]

    target_error = [
        float(actual[i]) - float(target[i])
        for i in range(6)
    ]

    command_xyz_norm = math.sqrt(
        sum(
            float(command_delta[i]) ** 2
            for i in range(3)
        )
    )

    measured_xyz_norm = math.sqrt(
        sum(
            float(measured_delta[i]) ** 2
            for i in range(3)
        )
    )

    tracking_xyz_error = math.sqrt(
        sum(
            float(target_error[i]) ** 2
            for i in range(3)
        )
    )

    metric = {
        "stage": str(stage),
        "velocity": float(velocity),
        "override": float(override),
        "before_tcp": list(map(float, before_tcp)),
        "target_tcp": list(map(float, target)),
        "actual_tcp": list(map(float, actual)),
        "command_delta_6d": command_delta,
        "measured_delta_6d": measured_delta,
        "target_error_6d": target_error,
        "command_xyz_norm_mm": command_xyz_norm,
        "measured_xyz_norm_mm": measured_xyz_norm,
        "tracking_xyz_error_mm": tracking_xyz_error,
    }

    MOTION_METRICS.append(metric)

    print()
    print("[ACTION EXECUTION METRIC]")
    print("Command delta 6D :", command_delta)
    print("Measured delta 6D:", measured_delta)
    print(
        "Tracking XYZ err :",
        f"{tracking_xyz_error:.3f} mm",
    )

    return actual


def move_gripper(
    gripper,
    target,
    expected_min,
    expected_max,
    stage,
):
    current = int(gripper.get_position())

    print()
    print("=" * 78)
    print(stage)
    print("=" * 78)
    print("Current gripper:", current)
    print("Target gripper :", target)

    if not (
        expected_min <= current <= expected_max
    ):
        raise RuntimeError(
            f"NO-GO: unexpected gripper position "
            f"{current} at {stage}"
        )

    ok = gripper.move(
        position=int(target),
        wait=True,
        timeout=5.0,
        tolerance=5,
    )

    measured = int(gripper.get_position())

    print("Success :", ok)
    print("Measured:", measured)

    if not ok or abs(measured - target) > 8:
        raise RuntimeError(
            f"Gripper move failed at {stage}"
        )

    return measured


def make_pose(reference, x=None, y=None, z=None):
    pose = list(map(float, reference))

    if x is not None:
        pose[0] = float(x)

    if y is not None:
        pose[1] = float(y)

    if z is not None:
        pose[2] = float(z)

    return pose


def make_s_curve(
    start_pose,
    dx_total,
    dy_total,
):
    """
    시작점에서 최종 (+dx_total, +dy_total)까지
    5개 waypoint로 부드럽게 이동한다.

    곡선의 수직 편차는 최종 Y 이동에
    S_AMPLITUDE를 추가해 만든다.
    """

    fractions = [
        0.20,
        0.40,
        0.60,
        0.80,
        1.00,
    ]

    curve_offsets = [
        +7.0,
        +12.0,
        +12.0,
        +7.0,
        0.0,
    ]

    waypoints = []

    for fraction, curve_offset in zip(
        fractions,
        curve_offsets,
    ):
        pose = start_pose.copy()

        pose[0] += (
            float(dx_total)
            * float(fraction)
        )

        pose[1] += (
            float(dy_total)
            * float(fraction)
            + float(curve_offset)
        )

        waypoints.append(pose)

    return waypoints

def execute_waypoints(
    robot,
    waypoints,
    tool,
    user,
    velocity,
    override,
    name,
):
    """
    검증된 최종 목표까지 단일 MoveL로 이동한다.
    NewSpline과 중간 waypoint 실행은 사용하지 않는다.
    """

    if not waypoints:
        raise RuntimeError(
            f"NO-GO: empty path: {name}"
        )

    for index, target in enumerate(
        waypoints,
        start=1,
    ):
        if len(target) != 6 or not finite(target):
            raise RuntimeError(
                f"NO-GO: invalid path point "
                f"{index}: {target}"
            )

    final_target = list(map(
        float,
        waypoints[-1],
    ))

    # 단일 MoveL을 보내기 전에 최종 목표 IK를 확인한다.
    joint_result = robot.GetActualJointPosDegree()

    if (
        not isinstance(joint_result, tuple)
        or len(joint_result) < 2
        or int(joint_result[0]) != 0
    ):
        raise RuntimeError(
            f"NO-GO: failed to read joints: "
            f"{joint_result}"
        )

    current_joints = list(map(
        float,
        joint_result[1],
    ))

    ik_result = robot.GetInverseKinHasSolution(
        type=0,
        desc_pos=final_target,
        joint_pos_ref=current_joints,
    )

    print()
    print("=" * 78)
    print(name + " — SINGLE MoveL")
    print("=" * 78)
    print("Current TCP  :", read_tcp(robot))
    print("Final target :", final_target)
    print("IK result    :", ik_result)
    print("Velocity     :", velocity)
    print("Override     :", override)

    if ik_result != (0, True):
        raise RuntimeError(
            f"NO-GO: no IK solution at {name}: "
            f"{ik_result}"
        )

    return move_l(
        robot=robot,
        target=final_target,
        tool=tool,
        user=user,
        velocity=velocity,
        override=override,
        stage=name + " — SINGLE MoveL",
        blend_radius=-1.0,
    )


def main():
    if not OBSERVATION_FILE.exists():
        raise SystemExit(
            f"Missing: {OBSERVATION_FILE}"
        )

    if not RUNTIME_FILE.exists():
        raise SystemExit(
            f"FINAL GO/NO-GO: NO-GO — missing {RUNTIME_FILE}"
        )

    observation_data = json.loads(
        OBSERVATION_FILE.read_text(
            encoding="utf-8"
        )
    )

    runtime = json.loads(
        RUNTIME_FILE.read_text(
            encoding="utf-8"
        )
    )

    observation = list(map(
        float,
        observation_data["tcp_pose"],
    ))

    expected_instruction = (
        "move the can to the top right"
    )

    runtime_instruction = str(
        runtime.get("instruction", "")
    ).strip()

    if runtime_instruction != expected_instruction:
        raise SystemExit(
            "FINAL GO/NO-GO: NO-GO — "
            "runtime instruction mismatch | "
            f"actual={runtime_instruction!r} | "
            f"expected={expected_instruction!r}"
        )

    if runtime.get("final_runtime_result") != "GO":
        raise SystemExit(
            "FINAL GO/NO-GO: NO-GO — "
            "runtime result is not GO"
        )

    runtime_checks = runtime.get(
        "workspace_checks",
        {}
    )

    required_checks = (
        "detected_inside_bounds",
        "grasp_inside_bounds",
        "place_inside_bounds",
        "all_values_finite",
    )

    failed_runtime_checks = [
        name
        for name in required_checks
        if runtime_checks.get(name) is not True
    ]

    if failed_runtime_checks:
        raise SystemExit(
            "FINAL GO/NO-GO: NO-GO — "
            "runtime guards failed: "
            + ", ".join(failed_runtime_checks)
        )

    detected = list(map(
        float,
        runtime[
            "detected_reference_xyz_mm"
        ],
    ))

    offset = list(map(
        float,
        runtime[
            "grasp_xy_offset_mm"
        ],
    ))

    runtime_grasp = list(map(
        float,
        runtime[
            "grasp_target_tcp_mm"
        ],
    ))

    runtime_place = list(map(
        float,
        runtime[
            "place_target_tcp_mm"
        ],
    ))

    if (
        len(detected) != 3
        or len(offset) != 2
        or len(runtime_grasp) != 3
        or len(runtime_place) != 3
    ):
        raise SystemExit(
            "FINAL GO/NO-GO: NO-GO — "
            "invalid runtime target dimensions"
        )

    runtime_age_sec = (
        time.time()
        - RUNTIME_FILE.stat().st_mtime
    )

    MAX_RUNTIME_AGE_SEC = 600.0

    if runtime_age_sec > MAX_RUNTIME_AGE_SEC:
        raise SystemExit(
            "FINAL GO/NO-GO: NO-GO — "
            f"runtime target expired: "
            f"{runtime_age_sec:.1f} sec"
        )

    grasp_x = detected[0] + offset[0]
    grasp_y = detected[1] + offset[1]

    if (
        abs(grasp_x - runtime_grasp[0]) > 0.001
        or abs(grasp_y - runtime_grasp[1]) > 0.001
        or abs(FINAL_GRASP_Z - runtime_grasp[2]) > 0.001
    ):
        raise SystemExit(
            "FINAL GO/NO-GO: NO-GO — "
            "runtime grasp target mismatch"
        )

    # FINAL_CALIBRATED_WORKSPACE_GUARD_V2
    if not WORKSPACE_FILE.exists():
        raise SystemExit(
            f"FINAL GO/NO-GO: NO-GO — missing {WORKSPACE_FILE}"
        )

    workspace_data = json.loads(
        WORKSPACE_FILE.read_text(encoding="utf-8")
    )

    bounds = workspace_data[
        "verified_object_bounds_mm"
    ]

    detected_x = float(detected[0])
    detected_y = float(detected[1])
    detected_z = float(detected[2])

    detected_ok = (
        float(bounds["x_min"])
        <= detected_x
        <= float(bounds["x_max"])
        and float(bounds["y_min"])
        <= detected_y
        <= float(bounds["y_max"])
        and float(bounds["z_min"])
        <= detected_z
        <= float(bounds["z_max"])
    )

    if not detected_ok:
        raise SystemExit(
            "FINAL GO/NO-GO: NO-GO — detection outside "
            "calibrated workspace "
            f"XYZ=({detected_x:.3f}, "
            f"{detected_y:.3f}, "
            f"{detected_z:.3f})"
        )

    grasp_x_min = (
        float(bounds["x_min"]) + float(offset[0])
    )
    grasp_x_max = (
        float(bounds["x_max"]) + float(offset[0])
    )
    grasp_y_min = (
        float(bounds["y_min"]) + float(offset[1])
    )
    grasp_y_max = (
        float(bounds["y_max"]) + float(offset[1])
    )

    grasp_ok = (
        grasp_x_min <= grasp_x <= grasp_x_max
        and grasp_y_min <= grasp_y <= grasp_y_max
    )

    if not grasp_ok:
        raise SystemExit(
            "FINAL GO/NO-GO: NO-GO — invalid grasp target "
            f"XY=({grasp_x:.3f}, {grasp_y:.3f})"
        )

    place_x = grasp_x + TRANSFER_DX
    place_y = grasp_y + TRANSFER_DY

    if (
        abs(place_x - runtime_place[0]) > 0.001
        or abs(place_y - runtime_place[1]) > 0.001
        or abs(PLACE_Z - runtime_place[2]) > 0.001
    ):
        raise SystemExit(
            "FINAL GO/NO-GO: NO-GO — "
            "runtime place target mismatch"
        )

    if not (200.0 <= place_x <= 400.0):
        raise SystemExit(
            "FINAL GO/NO-GO: NO-GO — invalid place target "
            f"X={place_x:.3f}"
        )

    print(
        "FINAL CALIBRATED WORKSPACE GUARD: PASS | "
        f"Detected=({detected_x:.3f}, "
        f"{detected_y:.3f}, "
        f"{detected_z:.3f}) | "
        f"Grasp=({grasp_x:.3f}, "
        f"{grasp_y:.3f}) | "
        f"Place=({place_x:.3f}, "
        f"{place_y:.3f})"
    )

    robot = Robot.RPC(ROBOT_IP)
    time.sleep(1.0)

    # FAIRINO controller global speed override
    global_speed_ret = robot.SetSpeed(60)
    print("SetSpeed(60) return:", global_speed_ret)

    if int(global_speed_ret) != 0:
        raise RuntimeError(
            f"NO-GO: SetSpeed(60) failed: {global_speed_ret}"
        )

    tool = int(unpack(
        robot.GetActualTCPNum(),
        "Tool",
    ))

    user = int(unpack(
        robot.GetActualWObjNum(),
        "User",
    ))

    current = read_tcp(robot)
    status = read_robot_status(robot)

    print("=" * 84)
    print("OPENVLA + FAIRINO PICK & PLACE — LIVE V2 SAFE")
    print("=" * 84)
    print("Status      :", status)
    print("Tool / User :", tool, "/", user)
    print("Current TCP :", current)
    print("Observation :", observation)
    print("Grasp XY    :", grasp_x, grasp_y)
    print("Final Z     :", FINAL_GRASP_Z)
    print("Gripper     :", GRIPPER_OPEN, "/", GRIPPER_GRASP)

    if not status_ok(status):
        raise SystemExit(
            "FINAL GO/NO-GO: NO-GO — robot status"
        )

    if tool != EXPECTED_TOOL:
        raise SystemExit(
            f"FINAL GO/NO-GO: NO-GO — tool={tool}"
        )

    if user != EXPECTED_USER:
        raise SystemExit(
            f"FINAL GO/NO-GO: NO-GO — user={user}"
        )

    if distance_xyz(
        current,
        observation,
    ) > START_TOLERANCE_MM:
        raise SystemExit(
            "FINAL GO/NO-GO: NO-GO — "
            "robot is not near observation pose"
        )

    approach = make_pose(
        observation,
        x=grasp_x,
        y=grasp_y,
        z=APPROACH_Z,
    )

    pregrasp = make_pose(
        approach,
        z=PREGRASP_Z,
    )

    near_grasp = make_pose(
        approach,
        z=NEAR_GRASP_Z,
    )

    grasp = make_pose(
        approach,
        z=FINAL_GRASP_Z,
    )

    lift_test = make_pose(
        grasp,
        z=FINAL_GRASP_Z + TEST_LIFT_MM,
    )

    lift = make_pose(
        grasp,
        z=TRANSFER_Z,
    )

    print()
    print("FINAL GO/NO-GO: GO")
    print()
    print("현장 최종 확인:")
    print("  1. 비상정지 버튼을 즉시 누를 수 있음")
    print("  2. 캔과 그리퍼 중심이 맞음")
    print("  3. 전체 S자 경로가 비어 있음")
    print(
        "  4. +X 150 mm, +Y 50 mm "
        "Top-right Place 위치가 비어 있음"
    )
    print("  5. 카메라와 케이블이 걸리지 않음")
    print("  6. 사람은 작업영역 밖에 있음")

    require_exact(
        "OpenVLA runtime 좌표로 전체 Pick & Place를 "
        "시작하려면 RUN OPENVLA PICK PLACE LIVE 입력:",
        "RUN OPENVLA PICK PLACE LIVE",
    )

    gripper = DHGripper(
        port=DH_PORT,
        slave_id=DH_SLAVE_ID,
        baudrate=DH_BAUDRATE,
        timeout=0.2,
    )

    try:
        if gripper.get_init_state() != 1:
            raise RuntimeError(
                "NO-GO: gripper not initialized"
            )

        if gripper.get_state() != 1:
            raise RuntimeError(
                "NO-GO: gripper not ready"
            )

        # 시작 시 그리퍼 900 확인
        current_gripper = int(
            gripper.get_position()
        )

        if abs(current_gripper - GRIPPER_OPEN) > 8:
            raise RuntimeError(
                f"NO-GO: starting gripper="
                f"{current_gripper}, expected≈900"
            )

        # 높은 위치에서 XY 정렬
        high_xy = make_pose(
            observation,
            x=grasp_x,
            y=grasp_y,
            z=observation[2],
        )

        move_l(
            robot,
            high_xy,
            tool,
            user,
            20.0,
            40.0,
            "STEP 1 — HIGH-Z XY ALIGNMENT",
        )

        # PICK_CONTINUOUS_DESCENT_V1
        # 높은 위치에서 XY 정렬을 마친 뒤,
        # X/Y와 자세를 고정하고 최종 파지 Z까지 한 번에 수직 하강한다.
        move_l(
            robot,
            grasp,
            tool,
            user,
            15.0,
            30.0,
            "STEP 2 — CONTINUOUS DESCENT TO GRASP Z168",
        )

        move_gripper(
            gripper,
            GRIPPER_GRASP,
            890,
            910,
            "STEP 6 — GRASP CAN",
        )

        # PICK_CONTINUOUS_LIFT_V1
        # 파지 후 Z168에서 Z181까지 한 번에 상승한다.
        move_l(
            robot,
            lift,
            tool,
            user,
            20.0,
            40.0,
            "STEP 4 — CONTINUOUS PICK LIFT TO Z255",
        )

        forward_s = make_s_curve(
            lift,
            TRANSFER_DX,
            TRANSFER_DY,
        )

        execute_waypoints(
            robot,
            forward_s,
            tool,
            user,
            20.0,
            40.0,
            "STEP 9 — FORWARD TOP-RIGHT TRANSFER",
        )

        place_high = forward_s[-1]
        place = make_pose(
            place_high,
            x=place_x,
            y=place_y,
            z=PLACE_Z,
        )

        require_exact(
            "Place 위치 아래가 비어 있고 "
            "하강해도 충돌하지 않습니다.\n"
            "내리려면 LOWER TO PLACE 입력:",
            "LOWER TO PLACE",
        )

        move_l(
            robot,
            place,
            tool,
            user,
            15.0,
            30.0,
            "STEP 10 — LOWER TO PLACE",
        )

        require_exact(
            "캔 바닥이 작업대에 닿고 안정적으로 "
            "설 수 있는 것을 확인했습니다.\n"
            "놓으려면 RELEASE CAN 900 입력:",
            "RELEASE CAN 900",
        )

        move_gripper(
            gripper,
            GRIPPER_OPEN,
            550,
            570,
            "STEP 11 — RELEASE CAN",
        )

        # PLACE_RETREAT_RELATIVE_30MM_V1
        # 실제 Place 도착 TCP를 기준으로 정확히 +30 mm 수직 후퇴한다.
        actual_place_tcp = read_tcp(robot)

        retreat = make_pose(
            actual_place_tcp,
            z=TRANSFER_Z,
        )

        print(
            "Place retreat target: "
            f"Z {actual_place_tcp[2]:.3f} -> "
            f"{retreat[2]:.3f} mm"
        )

        require_exact(
            "캔이 그대로 서 있고 그리퍼에서 "
            "분리된 것을 확인했습니다.\n"
            "Z255까지 상승하려면 RETREAT TO Z255 입력:",
            "RETREAT TO Z255",
        )

        move_l(
            robot,
            retreat,
            tool,
            user,
            20.0,
            40.0,
            "STEP 12 — VERTICAL RETREAT TO Z255",
        )

        reverse_s = make_s_curve(
            retreat,
            -TRANSFER_DX,
            -TRANSFER_DY,
        )

        require_exact(
            "역방향 S자 경로가 비어 있습니다.\n"
            "복귀하려면 RETURN S 150 입력:",
            "RETURN S 150",
        )

        execute_waypoints(
            robot,
            reverse_s,
            tool,
            user,
            20.0,
            40.0,
            "STEP 13 — SINGLE MoveL RETURN",
        )

        return_low = reverse_s[-1]

        return_high = make_pose(
            return_low,
            z=observation[2],
        )

        move_l(
            robot,
            return_high,
            tool,
            user,
            20.0,
            40.0,
            "STEP 14 — RAISE TO OBSERVATION HEIGHT",
        )

        move_l(
            robot,
            observation,
            tool,
            user,
            20.0,
            40.0,
            "STEP 15 — EXACT OBSERVATION RETURN",
        )

        final_tcp = read_tcp(robot)

        openvla_action = list(map(
            float,
            runtime.get(
                "openvla_raw_action",
                [],
            ),
        ))

        openvla_xyz_mm = list(map(
            float,
            runtime.get(
                "openvla_raw_xyz_mm",
                [],
            ),
        ))

        openvla_rpy_deg = list(map(
            float,
            runtime.get(
                "openvla_raw_rpy_deg",
                [],
            ),
        ))

        grasp_target = list(map(
            float,
            runtime["grasp_target_tcp_mm"],
        ))

        place_target = list(map(
            float,
            runtime["place_target_tcp_mm"],
        ))

        planner_transfer_xyz = [
            place_target[i] - grasp_target[i]
            for i in range(3)
        ]

        prediction_to_plan_xyz = None
        prediction_to_plan_norm = None

        if len(openvla_xyz_mm) == 3:
            prediction_to_plan_xyz = [
                planner_transfer_xyz[i]
                - openvla_xyz_mm[i]
                for i in range(3)
            ]

            prediction_to_plan_norm = math.sqrt(
                sum(
                    value ** 2
                    for value in prediction_to_plan_xyz
                )
            )

        report = {
            "instruction": runtime.get(
                "instruction"
            ),
            "openvla": {
                "raw_action_7d": openvla_action,
                "predicted_xyz_mm": openvla_xyz_mm,
                "predicted_rpy_deg": openvla_rpy_deg,
                "predicted_gripper": runtime.get(
                    "openvla_gripper"
                ),
                "latency_sec": runtime.get(
                    "openvla_latency_sec"
                ),
            },
            "planner": {
                "detected_reference_xyz_mm":
                    runtime.get(
                        "detected_reference_xyz_mm"
                    ),
                "grasp_target_tcp_mm":
                    grasp_target,
                "place_target_tcp_mm":
                    place_target,
                "planned_transfer_xyz_mm":
                    planner_transfer_xyz,
            },
            "prediction_to_plan": {
                "xyz_difference_mm":
                    prediction_to_plan_xyz,
                "xyz_difference_norm_mm":
                    prediction_to_plan_norm,
                "interpretation":
                    (
                        "Diagnostic only. OpenVLA output "
                        "is one short action step, while "
                        "the planner transfer is a full "
                        "task-level displacement."
                    ),
            },
            "robot_execution": {
                "motions": MOTION_METRICS,
                "final_tcp": list(
                    map(float, final_tcp)
                ),
                "final_observation_error_mm":
                    distance_xyz(
                        final_tcp,
                        observation,
                    ),
                "final_gripper_position":
                    int(
                        gripper.get_position()
                    ),
            },
            "limitations": {
                "openvla_raw_action_transmitted":
                    False,
                "closed_loop_openvla_rollout":
                    False,
                "actual_can_final_position_measured":
                    False,
                "gripper_token_semantics_validated":
                    False,
            },
        }

        METRICS_FILE.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 84)
        print("OPENVLA / PLANNER / ROBOT COMPARISON")
        print("=" * 84)
        print(
            "OpenVLA predicted XYZ [mm]:",
            openvla_xyz_mm,
        )
        print(
            "Planner transfer XYZ [mm] :",
            planner_transfer_xyz,
        )
        print(
            "Prediction-plan difference:",
            prediction_to_plan_xyz,
        )
        print(
            "Prediction-plan norm [mm] :",
            prediction_to_plan_norm,
        )
        print(
            "Recorded robot motions     :",
            len(MOTION_METRICS),
        )
        print(
            "Metrics saved               :",
            METRICS_FILE.resolve(),
        )
        print("=" * 84)

        print()
        print("=" * 84)
        print("AUTO PICK & PLACE COMPLETED")
        print("=" * 84)
        print("Final TCP     :", final_tcp)
        print(
            "Observation error:",
            distance_xyz(
                final_tcp,
                observation,
            ),
            "mm",
        )
        print("Gripper       :", int(
            gripper.get_position()
        ))
        print("FINAL RESULT  : COMPLETE")
        print("=" * 84)

    finally:
        gripper.close_port()


if __name__ == "__main__":
    main()
