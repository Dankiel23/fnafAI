# FNAF Gymnasium

`fnaf_gymnasium` is a Gymnasium environment for training reinforcement learning agents to play Five Nights at Freddy's 1.

The repository is centered on a pure Python simulation of the game's mechanics, along with environment variants, reward-shaping tools, training scripts, evaluation tooling, and tests. It is meant for training AI agents in simulation rather than interacting with the live game directly.

## What This Project Includes

- A pure Python FNAF 1 simulation engine
- Gymnasium-compatible environments for RL training
- Stable-Baselines3-friendly training and evaluation scripts
- Fully observable and CV-style partially observable environment variants
- Replay logging and episode analysis utilities
- Tests for game logic, wrappers, custom envs, and edge cases

## Environment Variants

- `FnafNight-v0`
  Standard environment with a 17-action discrete action space and 77-dimensional observation.
- `FnafCVReady-v0`
  Partial-observability environment with memory features and CV-style signals for eventual screen-capture deployment.
- `FnafCustomNight-v0`
  Custom AI levels for each animatronic.
- `FnafRandomDifficulty-v0`
  Random AI levels every reset.
- `FnafMultiNight-v0`
  Multiple nights chained into one episode.

## Core Python Files

- `fnaf_gymnasium/envs/game_logic.py`
  Core mechanics: animatronic behavior, power drain, timing, and game-over rules.
- `fnaf_gymnasium/envs/constants.py`
  Tuning constants derived from the reverse-engineering notes.
- `fnaf_gymnasium/envs/fnaf_env.py`
  Standard training environment.
- `fnaf_gymnasium/envs/cv_env.py`
  CV-oriented environment for partial observability and memory-based inference.
- `fnaf_gymnasium/wrappers.py`
  Reward shaping, sparse reward, time penalty, and action masking wrappers.
- `fnaf_gymnasium/callbacks.py`
  Stable-Baselines3 metrics and curriculum callbacks.

## Install

Base package:

```bash
pip install -e .
```

Training extras:

```bash
pip install -e ".[train]"
```

Or:

```bash
pip install -r requirements.txt
```

## Quick Start

Train a baseline PPO agent on Night 1:

```bash
python quickstart.py
```

Train manually:

```bash
python train.py train --algo ppo --night 1 --total-timesteps 500000
```

Train the CV-oriented variant:

```bash
python train_cv.py --steps 2000000 --target-night 3
```

Train from config:

```bash
python train_from_config.py configs/ppo_night1.json
```

Watch a trained agent:

```bash
python watch_agent.py --model fnaf_quickstart_model.zip --night 1
```

Evaluate a model across nights:

```bash
python evaluate_all_nights.py --model fnaf_quickstart_model.zip --algo ppo
```

## Repository Layout

```text
fnaf_gymnasium/   Python package and Gymnasium environments
tests/            Pytest suite
configs/          JSON training configurations
research/         Reverse-engineering notes for FNAF mechanics
legacy_simulator/ Archived browser simulator and web assets
*.py              Training, evaluation, replay, and benchmark scripts
```

## Research Notes

The mechanics in the Python simulator are based on the reverse-engineering notes in:

- `research/how-the-game-works.md`
- `research/how-sound-works.md`

The first is the most relevant reference for AI training and simulator correctness.

## Credit

This project builds on the research, reverse-engineering, and original simulator groundwork by [CeriW / fnaf1-ai-simulator](https://github.com/CeriW/fnaf1-ai-simulator).

Credit goes to that project for the heavy lifting around documenting FNAF 1 logic, validating game behavior, and establishing much of the simulator-side understanding this RL environment depends on.

## Legacy Browser Simulator

The original browser-based simulator and its non-AI web assets were moved into `legacy_simulator/` so the repository root stays focused on the AI/RL project.

That folder contains:

- the old TypeScript frontend source
- compiled browser output
- frontend tooling files
- images, icons, and audio assets for the simulator website

It is still useful as reference material when comparing the Python port against the earlier simulator, but it is not required for Python training or testing.

## Testing

Run the test suite with:

```bash
pytest
```
