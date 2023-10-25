import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument("--domain_name", default="carla")
    parser.add_argument("--task_name", default="drive")
    parser.add_argument("--frame_stack", default=3, type=int)
    parser.add_argument("--action_repeat", default=5, type=int)
    parser.add_argument("--episode_length", default=1000, type=int)
    parser.add_argument("--eval_mode", default="color_easy", type=str)
    parser.add_argument("--capacity", default=40_000, type=str)
    parser.add_argument("--n_episodes", default=2_000, type=int)
    parser.add_argument("--lower_limit_return_", default=-3, type=float)

    # agent
    parser.add_argument("--algorithm", default="sac", type=str)
    parser.add_argument(
        "--train_steps",
        default=501_000,
        type=str,
        help="the number of steps that the agent will train",
    )
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument(
        "--init_steps",
        default=1_000,
        type=int,
        help="the number of initial steps to take before the agent will be update after each frame",
    )
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--hidden_dim", default=1024, type=int)

    # actor
    parser.add_argument("--actor_lr", default=1e-4, type=float)
    parser.add_argument("--actor_beta", default=0.9, type=float)
    parser.add_argument("--actor_log_std_min", default=-10, type=float)
    parser.add_argument("--actor_log_std_max", default=2, type=float)
    parser.add_argument("--actor_update_freq", default=2, type=int)

    # critic
    parser.add_argument("--critic_lr", default=1e-4, type=float)
    parser.add_argument("--critic_beta", default=0.9, type=float)
    parser.add_argument("--critic_tau", default=0.01, type=float)
    parser.add_argument("--critic_target_update_freq", default=2, type=int)
    parser.add_argument("--critic_weight_decay", default=0, type=float)

    # architecture
    parser.add_argument("--num_shared_layers", default=11, type=int)
    parser.add_argument("--num_head_layers", default=0, type=int)
    parser.add_argument("--num_filters", default=32, type=int)
    parser.add_argument("--projection_dim", default=100, type=int)
    parser.add_argument("--encoder_tau", default=0.05, type=float)

    # entropy maximization
    parser.add_argument("--init_temperature", default=0.3, type=float)
    parser.add_argument("--alpha_lr", default=1e-4, type=float)
    parser.add_argument("--alpha_beta", default=0.5, type=float)

    # auxiliary tasks
    parser.add_argument("--aux_lr", default=3e-4, type=float)
    parser.add_argument("--aux_beta", default=0.9, type=float)
    parser.add_argument("--aux_update_freq", default=2, type=int)

    # soda
    parser.add_argument("--soda_batch_size", default=256, type=int)
    parser.add_argument("--soda_tau", default=0.005, type=float)

    # svea
    parser.add_argument("--svea_alpha", default=0.5, type=float)
    parser.add_argument("--svea_beta", default=0.5, type=float)
    parser.add_argument("--sgqn_quantile", default=0.5, type=float)
    parser.add_argument("--svea_contrastive_coeff", default=0.1, type=float)
    parser.add_argument("--svea_norm_coeff", default=0.1, type=float)
    parser.add_argument("--attrib_coeff", default=0.25, type=float)
    parser.add_argument("--consistency", default=1, type=int)

    # eval
    parser.add_argument("--save_freq", default="50", type=str)
    parser.add_argument("--eval_freq", default="50", type=str)
    parser.add_argument("--eval_episodes", default=3, type=int)
    parser.add_argument("--distracting_cs_intensity", default=0.0, type=float)

    # misc
    parser.add_argument("--seed", default=10143, type=int)
    parser.add_argument("--log_dir", default="logs", type=str)
    parser.add_argument("--save_video", default=False, action="store_true")

    parser.add_argument(
        "--alpha_blending",
        default=0.2,
        type=float,
        help="the percentage of blending between the true image captured from the camera and a corruption image taken form the dataset",
    )

    args = parser.parse_args()

    assert args.algorithm in {
        "sac",
        "rad",
        "curl",
        "pad",
        "soda",
        "drq",
        "svea",
        "saca",
        "sacfa",
        "sgsac",
    }, f'specified algorithm "{args.algorithm}" is not supported'

    assert args.eval_mode in {
        "train",
        "color_easy",
        "color_hard",
        "video_easy",
        "video_hard",
        "distracting_cs",
        "all",
        "none",
    }, f'specified mode "{args.eval_mode}" is not supported'
    assert args.seed is not None, "must provide seed for experiment"
    assert args.log_dir is not None, "must provide a log directory for experiment"

    intensities = {0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}
    assert (
        args.distracting_cs_intensity in intensities
    ), f"distracting_cs has only been implemented for intensities: {intensities}"

    if type(args.train_steps) is str:
        args.train_steps = int(args.train_steps.replace("k", "000"))

    args.save_freq = int(args.save_freq.replace("k", "000"))
    args.eval_freq = int(args.eval_freq.replace("k", "000"))

    if args.eval_mode == "none":
        args.eval_mode = None

    if args.algorithm in {"rad", "curl", "pad", "soda"}:
        args.image_size = 100
        args.image_crop_size = 84
    else:
        args.image_size = 84
        args.image_crop_size = 84

    return args
