"""
Visualize and record a trained RL agent on a MiniGrid environment,
while keeping the exact same wrappers you used at training time.

Usage:
    python visualize.py --env ENV_ID --model MODEL_NAME [options]

Example:
    python visualize.py \
      --env MiniGrid-DoorKey-5x5-v0 \
      --model DoorKey \
      --seed 42 \
      --memory \
      --episodes 5
"""
import argparse
import os
import time
from datetime import datetime
import json
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
import utils
from utils import device, CustomRedBlueDoorEnv

def parse_args():
    p = argparse.ArgumentParser("Visualize & record a trained RL agent")
    p.add_argument("--model", required=True, help="Name of model directory")
    p.add_argument("--modtype", required=True, help="Name of model type: lstm, transformer, mamba")
    p.add_argument("--seed",  type=int, default=7090, help="Random seed")
    p.add_argument("--shift", type=int, default=0, help="No-op resets before start")
    p.add_argument("--argmax", action="store_true", help="Use argmax action")
    p.add_argument("--memory", action="store_true", help="Use LSTM memory")
    p.add_argument("--text", action="store_true", help="Use GRU text encoder")
    p.add_argument("--pause", type=float, default=0.1, help="Pause between steps (s)")
    p.add_argument("--episodes",type=int, default=100, help="Number of episodes")
    p.add_argument("--env_size",type=int, default=10, help="Size of the environment")
    return p.parse_args()

def make_recording_env(env_id, seed, video_folder, env_size):
    # Re-use the exact same wrappers you trained with, just switch to rgb_array
    env = CustomRedBlueDoorEnv(
        size=env_size,
        render_mode="rgb_array",
    )
    # record statistics and videos
    env = RecordEpisodeStatistics(env, buffer_length=100)
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix="episode",
        episode_trigger=lambda _: True
    )
    return env

def save_run_summary(args, agent, env, filepath):
    """
    Dump a JSON file at filepath containing:
        - args: the fields passed on the command line
        - model_parameter_shapes: dict of {name: shape} for each nn.Parameter
        - episode_stats: lists of returns, lengths, durations from the env
    """
    # 1) Hyper-parameters
    hyperparams = vars(args)

    # 2) Model parameter shapes
    param_shapes = {
        name: tuple(param.shape)
        for name, param in agent.acmodel.named_parameters()
    }

    # Combine and dump
    summary = {
        "hyperparameters":       hyperparams,
        "model_parameter_shapes": param_shapes,
    }

    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved run summary to {filepath}")

def main():
    args = parse_args()
    utils.seed(args.seed)
    print(f"Device: {device}\n")

    # 1) Load the “human” env to build agent on the exact same spaces
    env = CustomRedBlueDoorEnv(
        size=args.env_size,
        render_mode="rgb_array",
    )

    for _ in range(args.shift):
        env.reset()
    print("Base (human) env loaded\n")

    # 2) Instantiate agent
    model_dir = utils.get_model_dir(args.model)
    agent = utils.Agent(
        env.observation_space,
        env.action_space,
        model_dir,
        argmax=args.argmax,
        use_memory=args.memory,
        use_text=args.text,
        model_type=args.modtype,
    )
    print("Agent loaded from", model_dir, "\n")

    # 3) Prepare the recording env with the *same* wrappers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir   = os.path.join(os.getcwd(), f"{model_dir}_{timestamp}_visuals")
    os.makedirs(out_dir, exist_ok=True)

    rec_env = make_recording_env(env, args.seed, out_dir, args.env_size)
    print(f"Recording every episode to {out_dir}\n")

    # 4) Run & record
    for ep in range(args.episodes):
        obs, info = rec_env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = rec_env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
            agent.analyze_feedback(reward, done)
            if args.pause > 0:
                time.sleep(args.pause)

        ep_info = info.get("episode", {}).copy()
        ep_info["total_steps"] = step_count
        rec_env.close()
        print(f"Episode {ep:3d} → reward {total_reward:.2f} steps {step_count} ")

    # 5) Save a summary of the model parameters and episode stats
    summary_path = os.path.join(out_dir, "run_summary.json")
    save_run_summary(args, agent, rec_env, summary_path)


if __name__ == "__main__":
    main()
