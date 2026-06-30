import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SSE"))

from SSE.main import get_args
from rl.launcher import launch


def build_args(config_path, output_name, cuda=None):
    with open(config_path, "r") as f:
        config = json.load(f)

    saved_argv = sys.argv[:]
    sys.argv = ["export_eval_trace"]
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
    if cuda is not None:
        args.cuda = cuda
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


def reset_eval_state(algo):
    algo.curr_subgoal = None
    algo.way_to_subgoal = 0
    algo.prev_ag = None
    algo.stay_count = 0
    algo.do_next_high = True
    algo.new_command_count = 0
    algo.inside_count = 0
    algo.graphplanner.deleted_node = []


def run_rollout_trace(algo, max_attempts, outcome):
    env = algo.test_env
    for attempt in range(max_attempts):
        np.random.seed(algo.args.seed + attempt)
        env.seed(algo.args.seed + attempt)
        reset_eval_state(algo)

        observation = env.reset()
        ob = observation["observation"]
        bg = observation["desired_goal"]
        ag = observation["achieved_goal"]
        algo.prev_ag = ag.copy()

        records = []
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
            if algo.do_next_high:
                algo.do_next_high = False
                algo.inside_count = 0

            record = {
                "t": timestep,
                "ag": np.asarray(ag[:2], dtype=float).copy(),
                "goal": np.asarray(bg[:2], dtype=float).copy(),
                "subgoal": None if algo.curr_subgoal is None else np.asarray(algo.curr_subgoal[:2], dtype=float).copy(),
                "waypoint": None if algo.waypoint_subgoal is None else np.asarray(algo.waypoint_subgoal[:2], dtype=float).copy(),
                "planned_path": path_points(algo, ag),
                "new_command": new_command,
            }
            records.append(record)

            first = False
            observation, _, _, info = env.step(act)
            ob = observation["observation"]
            ag = observation["achieved_goal"]

            train_dist = env.goal_distance(ag, bg)
            if train_dist <= env.distance_threshold_high:
                if algo.args.env_name == "AntMazeKeyChest":
                    done = bool(getattr(env, "has_key", False))
                elif algo.args.env_name == "AntMazeDoubleKeyChest":
                    done = bool(getattr(env, "has_key1", False) and getattr(env, "has_key2", False))
                else:
                    done = True

            truncated = (timestep == algo.env_params["max_timesteps"] - 1) or truncated
            if done or truncated:
                final = record.copy()
                final["t"] = timestep + 1
                final["ag"] = np.asarray(ag[:2], dtype=float).copy()
                final["planned_path"] = path_points(algo, ag)
                records.append(final)
                break

        if (outcome == "success" and done) or (outcome == "failure" and not done):
            return attempt, records, info, done

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
    parser.add_argument("--exp-group", default="AntMaze")
    parser.add_argument("--run", default="umaze_1m_seed1")
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--outcome", choices=["success", "failure"], default="success")
    parser.add_argument("--cpu", action="store_true")
    args_cli = parser.parse_args()

    run_dir = ROOT / "exp" / args_cli.exp_group / args_cli.run
    state_dir = run_dir / "state"
    config_path = run_dir / "config.json"
    if not state_dir.exists():
        raise SystemExit(f"Missing checkpoint state directory: {state_dir}")

    args = build_args(config_path, args_cli.output_name, cuda=False if args_cli.cpu else None)
    algo = launch(args)
    algo.low_agent.load(str(state_dir))
    algo.high_agent.load(str(state_dir))
    algo.graphplanner.load(str(state_dir))
    algo.low_agent.eval()
    algo.high_agent.actor.eval()

    attempt, records, _, done = run_rollout_trace(algo, args_cli.max_attempts, args_cli.outcome)

    out_dir = ROOT / "visualizations" / args_cli.output_name
    trace_path = out_dir / f"{args_cli.output_name}_trace.csv"
    write_trace(records, trace_path)

    print(f"attempt={attempt}")
    print(f"success={int(done)}")
    print(f"steps={len(records)}")
    print(f"trace={trace_path}")


if __name__ == "__main__":
    main()
