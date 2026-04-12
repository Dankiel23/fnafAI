# FNAF Gymnasium

`fnaf_gymnasium` is a Gymnasium environment for training reinforcement learning agents to play Five Nights at Freddy's 1.

The repository is centered on a pure Python simulation of the game's mechanics, plus environment variants, reward shaping tools, training scripts, evaluation utilities, replay tooling, and tests. The main workflow is:

- train in the standard fully observable environment
- progress night by night with fine-tuning and checkpoint evaluation
- optionally train a separate CV-oriented policy for real-game deployment

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
  Partial-observability environment with memory features and CV-style signals for eventual screen-capture deployment. Observation size is 87.
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
- `train.py`
  Main standard-env train/eval CLI with checkpointing, fine-tuning, and reward-shaping flags.
- `train_cv.py`
  Separate CV-env curriculum training script.

## Install

Base package:

```bash
pip install -e .
```

Training extras:

```bash
pip install -e ".[train]"
```

Development extras:

```bash
pip install -e ".[dev]"
```

Or:

```bash
pip install -r requirements.txt
```

## Standard Training Workflow

Train a baseline PPO agent on Night 1:

```bash
python quickstart.py
```

Train manually:

```bash
python train.py train --algo ppo --night 1 --total-timesteps 500000
```

Continue training from an earlier checkpoint:

```bash
python train.py train --algo ppo --night 2 --total-timesteps 1500000 --load-model .\training_logs\night2_v3\night2_ppo\final_model.zip --reward-shaping --log-dir ./training_logs/night3_v1
```

Train from config:

```bash
python train_from_config.py configs/ppo_night1.json
```

Watch a trained agent:

```bash
python watch_agent.py --model fnaf_quickstart_model.zip --night 1
```

Evaluate a model on a specific night:

```bash
python train.py eval --model-path .\training_logs\night2_v3\night2_ppo\final_model.zip --algo ppo --night 2 --eval-episodes 500
```

Evaluate across multiple nights:

```bash
python evaluate_all_nights.py --model fnaf_quickstart_model.zip --algo ppo
```

## Checkpoints And Outputs

`train.py` writes several useful artifacts during training:

- periodic checkpoints under `training_logs/.../checkpoints/`
- an eval-selected best model under `training_logs/.../best_model/best_model.zip`
- a final model under `training_logs/.../final_model.zip`
- TensorBoard logs under `training_logs/.../tensorboard/`

For this project, direct eval win rate is usually more important than the callback's cumulative training `win_rate`. It is common for a periodic checkpoint to outperform `best_model` if `best_model` was selected by reward rather than raw win rate.

## CV Training Workflow

Train the CV-oriented variant:

```bash
python train_cv.py --steps 2000000 --target-night 3
```

The CV environment is a separate training track intended for real-game deployment. It uses a different observation space from the standard env, so a strong standard-env model is evidence that the policy ideas work, but it is not a direct drop-in checkpoint for `FnafCVReady-v0`.

## Real-Game Deployment Note

Finishing CV training does not by itself let the model directly play the Steam version. A deployment pipeline is still required:

- game window capture
- CV/OCR/template matching to build observations
- stateful memory tracking between frames
- policy inference
- keyboard/mouse action execution

The CV env was designed so this bridge is possible, not so it is automatic.

## Repository Layout

```text
fnaf_gymnasium/   Python package and Gymnasium environments
tests/            Pytest suite
configs/          JSON training configurations
research/         Reverse-engineering notes for FNAF mechanics
legacy_simulator/ Archived browser simulator and web assets
training_logs/    Generated checkpoints, best models, final models, and TensorBoard data
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
python -m pytest
```
