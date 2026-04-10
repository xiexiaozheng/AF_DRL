from __future__ import annotations

import argparse
import json

import torch

from evaluation import evaluate_phase1_model, rollout_actor_on_txt, save_rollout_traces, summarise_rollouts
from models import AutofocusActorCritic



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a phase-1 checkpoint.")
    parser.add_argument("--txt_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--pe_dim", type=int, default=16)
    parser.add_argument("--raw_suffix", type=str, default=".npy")
    parser.add_argument("--rollout_max_steps", type=int, default=4)
    parser.add_argument("--rollout_json", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imagenet_pretrained", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = AutofocusActorCritic(pe_dim=args.pe_dim, imagenet_pretrained=args.imagenet_pretrained).to(device)
    model.load_phase1_weights(args.checkpoint)
    model.eval()

    one_step_metrics = evaluate_phase1_model(
        model,
        txt_path=args.txt_path,
        data_root=args.data_root,
        device=device,
        patch_size=args.patch_size,
        pe_dim=args.pe_dim,
        raw_suffix=args.raw_suffix,
    )
    print("phase1_single_step", json.dumps(one_step_metrics, indent=2))

    rollout_traces = rollout_actor_on_txt(
        model,
        txt_path=args.txt_path,
        data_root=args.data_root,
        device=device,
        max_steps=args.rollout_max_steps,
        patch_size=args.patch_size,
        pe_dim=args.pe_dim,
        raw_suffix=args.raw_suffix,
        deterministic=True,
    )
    rollout_metrics = summarise_rollouts(rollout_traces, max_steps=args.rollout_max_steps)
    print("phase1_multi_step", json.dumps(rollout_metrics, indent=2))
    if args.rollout_json:
        save_rollout_traces(rollout_traces, args.rollout_json)


if __name__ == "__main__":
    main()
