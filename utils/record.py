import os
import datetime
import torch
import logging

from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
import utils  # Assuming this is a custom module with make_env and get_obss_preprocessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_recording_env(env_id, seed, video_folder):
    env = utils.make_env(env_id, seed, render_mode="rgb_array")
    env = RecordEpisodeStatistics(env, buffer_length=100)
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix="episode",
        episode_trigger=lambda _: True
    )
    env.episode_id = 0
    return env

def record(acmodel, args, model_dir, episode):
    # 1) Build human‐playable env and preprocessor
    human_env = utils.make_env(args.env, args.seed, render_mode="human")
    _, preprocess_obss = utils.get_obss_preprocessor(human_env.observation_space)

    # 2) Eval mode & optional memory init
    acmodel.eval()
    if args.mem:
        memories = torch.zeros(1, acmodel.memory_size, device=device)
    else:
        memories = None

    # 3) Prepare output dir + dedicated logger
    out_dir = os.path.join(model_dir, f"batch_{episode}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")

    # Create a dedicated logger so basicConfig elsewhere doesn't interfere
    logger = logging.getLogger(f"recording_logger_{episode}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logger.propagate = False  # don't pass through to root handlers
    logger.info(f"Recording every episode to {out_dir}")

    # 4) Run & record 10 episodes
    rec_env = make_recording_env(args.env, args.seed, out_dir)
    for ep in range(1, 11):
        obs, info = rec_env.reset(seed=args.seed + ep)
        total_reward = 0.0
        done = False

        if args.mem:
            memories.zero_()

        while not done:
            batch = preprocess_obss([obs], device=device)
            with torch.no_grad():
                if args.mem:
                    dist, _, memories = acmodel(batch, memories)
                else:
                    dist, _ = acmodel(batch)
            action = dist.sample().item()

            obs, reward, terminated, truncated, info = rec_env.step(action)
            total_reward += reward
            done = terminated or truncated

        logger.info(f"Episode {episode}, Attempt {ep:3d} -> reward {total_reward:.2f}")

    rec_env.close()