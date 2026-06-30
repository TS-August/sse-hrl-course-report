import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SSE"))

from SSE.main import get_args
from rl.launcher import launch


def build_args(config_path, output_name):
    with open(config_path, "r") as f:
        config = json.load(f)

    saved_argv = sys.argv[:]
    sys.argv = ["render_antmaze_success_video"]
    args = get_args()
    sys.argv = saved_argv

    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    args.log_mode = "disabled"
    args.save_dir = str(ROOT / "visualizations" / "_tmp_eval")
    args.ckpt_name = output_name + "_tmp"
    args.buffer_size = 1000
    args.batch_size = 8
    args.n_test_rollouts = 1
    args.eval_render = False
    args.save_video = False
    return args


def path_points(algo, ag):
    points = [np.asarray(ag[:2], dtype=float)]
    gp = algo.graphplanner
    if getattr(gp, "waypoint_vec", None) is not None and getattr(gp, "landmarks", None) is not None:
        for idx in gp.waypoint_vec[gp.waypoint_idx:]:
            if isinstance(idx, (int, np.integer)) and 0 <= idx < len(gp.landmarks):
                points.append(np.asarray(gp.landmarks[idx][:2], dtype=float))
    if algo.curr_subgoal is not None:
        points.append(np.asarray(algo.curr_subgoal[:2], dtype=float))
    return np.asarray(points)


def render_frame(algo, records, current, high_goals, frame_size=(960, 720)):
    fig, ax = plt.subplots(figsize=(frame_size[0] / 120, frame_size[1] / 120), dpi=120)

    map_size, wall_x, wall_y = algo.get_map_info()
    ax.plot(wall_x, wall_y, color="black", linewidth=2.0, label="maze wall")
    ax.set_xlim(map_size[0])
    ax.set_ylim(map_size[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    goal = current["goal"]
    ax.scatter([goal[0]], [goal[1]], c="#d62728", marker="*", s=180, label="final goal", zorder=5)

    if records:
        path = np.asarray([r["ag"] for r in records])
        ax.plot(path[:, 0], path[:, 1], color="#2ca02c", linewidth=2.3, label="agent trajectory", zorder=3)
        ax.scatter(path[-1:, 0], path[-1:, 1], color="#1b7f1b", s=55, zorder=6)

    if high_goals:
        hg = np.asarray(high_goals)
        ax.scatter(hg[:, 0], hg[:, 1], c="#1f77b4", s=28, alpha=0.28, label="past high subgoals", zorder=2)
        for i, xy in enumerate(high_goals[-10:], start=max(0, len(high_goals) - 10)):
            ax.text(xy[0] + 0.15, xy[1] + 0.15, str(i + 1), fontsize=7, color="#1f77b4")

    planned = current["planned_path"]
    if len(planned) >= 2:
        ax.plot(planned[:, 0], planned[:, 1], color="#ff7f0e", linestyle="--", linewidth=2.2,
                label="current planned path", zorder=4)
        ax.scatter(planned[1:-1, 0], planned[1:-1, 1], c="#ff7f0e", s=38, zorder=5)

    subgoal = current["subgoal"]
    waypoint = current["waypoint"]
    if subgoal is not None:
        ax.scatter([subgoal[0]], [subgoal[1]], c="#1f77b4", marker="X", s=115,
                   label="current high subgoal", zorder=7)
    if waypoint is not None:
        ax.scatter([waypoint[0]], [waypoint[1]], c="#9467bd", marker="D", s=80,
                   label="current low waypoint", zorder=7)

    cmd_note = "new high-level command" if current["new_command"] else "following planned waypoint"
    ax.set_title(
        f"SSE AntMaze success rollout | t={current['t']} | command #{len(high_goals)} | {cmd_note}",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)
    return rgb


def write_video(frames, output_path, fps):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def run_rollout(algo, max_attempts, outcome):
    env = algo.test_env
    for attempt in range(max_attempts):
        np.random.seed(algo.args.seed + attempt)
        env.seed(algo.args.seed + attempt)

        algo.curr_subgoal = None
        algo.way_to_subgoal = 0
        algo.prev_ag = None
        algo.stay_count = 0
        algo.do_next_high = True
        algo.new_command_count = 0
        algo.inside_count = 0
        algo.graphplanner.deleted_node = []

        observation = env.reset()
        ob = observation["observation"]
        bg = observation["desired_goal"]
        ag = observation["achieved_goal"]
        algo.prev_ag = ag.copy()

        records = []
        high_goals = []
        frames = []
        first = True
        done = False
        truncated = False
        info = {}

        for timestep in range(algo.env_params["max_timesteps"]):
            if algo.curr_subgoal is not None:
                temp_dist = env.goal_distance(ag, algo.curr_subgoal)
                if temp_dist <= algo.args.lambda1_inside:
                    algo.inside_count += 1
                else:
                    algo.inside_count = 0
                truncated = algo.calc_move(ag, timestep)
                algo.do_next_high = (
                    (temp_dist <= algo.args.lambda1)
                    or ((temp_dist <= algo.args.lambda1_inside) and algo.inside_count >= algo.args.inside_count)
                    or (algo.stay_count > 300 and algo.new_command_count > 100)
                )

            new_command = algo.curr_subgoal is None or algo.do_next_high
            act = algo.eval_get_actions(ob, ag, bg, first=first, timestep=timestep)
            algo.new_command_count += 1
            if new_command and algo.curr_subgoal is not None:
                high_goals.append(np.asarray(algo.curr_subgoal[:2], dtype=float).copy())
            if algo.do_next_high:
                algo.do_next_high = False
                algo.inside_count = 0

            current = {
                "t": timestep,
                "ag": np.asarray(ag[:2], dtype=float).copy(),
                "goal": np.asarray(bg[:2], dtype=float).copy(),
                "subgoal": None if algo.curr_subgoal is None else np.asarray(algo.curr_subgoal[:2], dtype=float).copy(),
                "waypoint": None if algo.waypoint_subgoal is None else np.asarray(algo.waypoint_subgoal[:2], dtype=float).copy(),
                "planned_path": path_points(algo, ag),
                "new_command": new_command,
            }
            records.append(current)
            frames.append(render_frame(algo, records, current, high_goals))

            first = False
            observation, _, _, info = env.step(act)
            ob = observation["observation"]
            ag = observation["achieved_goal"]

            train_dist = env.goal_distance(ag, bg)
            if train_dist <= env.distance_threshold_high:
                done = True

            if done or truncated:
                final = current.copy()
                final["t"] = timestep + 1
                final["ag"] = np.asarray(ag[:2], dtype=float).copy()
                final["planned_path"] = path_points(algo, ag)
                records.append(final)
                frames.append(render_frame(algo, records, final, high_goals))
                break

        if (outcome == "success" and done) or (outcome == "failure" and not done):
            return attempt, records, frames, info

    raise RuntimeError(f"No {outcome} rollout found in {max_attempts} attempts")


def write_trace(records, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t", "agent_x", "agent_y", "goal_x", "goal_y",
            "subgoal_x", "subgoal_y", "waypoint_x", "waypoint_y", "new_high_command",
        ])
        for r in records:
            subgoal = r["subgoal"]
            waypoint = r["waypoint"]
            writer.writerow([
                r["t"],
                r["ag"][0], r["ag"][1],
                r["goal"][0], r["goal"][1],
                "" if subgoal is None else subgoal[0],
                "" if subgoal is None else subgoal[1],
                "" if waypoint is None else waypoint[0],
                "" if waypoint is None else waypoint[1],
                int(r["new_command"]),
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="umaze_1m_seed1")
    parser.add_argument("--output-name", default="antmaze_umaze_success_with_subgoals")
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--policy", choices=["checkpoint", "initial"], default="checkpoint")
    parser.add_argument("--outcome", choices=["success", "failure"], default="success")
    args_cli = parser.parse_args()

    run_dir = ROOT / "exp" / "AntMaze" / args_cli.run
    state_dir = run_dir / "state"
    config_path = run_dir / "config.json"
    if not state_dir.exists():
        raise SystemExit(f"Missing checkpoint state directory: {state_dir}")

    args = build_args(config_path, args_cli.output_name)
    algo = launch(args)
    if args_cli.policy == "checkpoint":
        algo.low_agent.load(str(state_dir))
        algo.high_agent.load(str(state_dir))
        algo.graphplanner.load(str(state_dir))
    algo.low_agent.eval()
    algo.high_agent.actor.eval()

    attempt, records, frames, _ = run_rollout(algo, args_cli.max_attempts, args_cli.outcome)

    out_dir = ROOT / "visualizations" / args_cli.output_name
    video_path = out_dir / f"{args_cli.output_name}.mp4"
    trace_path = out_dir / f"{args_cli.output_name}_trace.csv"
    write_video(frames, video_path, args_cli.fps)
    write_trace(records, trace_path)

    print(f"success_attempt={attempt}")
    print(f"frames={len(frames)}")
    print(f"video={video_path}")
    print(f"trace={trace_path}")


if __name__ == "__main__":
    main()
