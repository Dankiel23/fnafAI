"""
Episode replay and logging utilities.

Records full episodes for analysis and debugging of trained agents.
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional

from .game_logic import FNAFGame, CAMERAS


@dataclass
class StepRecord:
    """Record of a single step in an episode."""
    step: int
    second: float
    action: int
    action_name: str
    reward: float
    cumulative_reward: float
    power: float
    power_usage: int
    cameras_on: bool
    current_camera: str
    left_door: bool
    right_door: bool
    left_light: bool
    right_light: bool
    freddy_pos: str
    bonnie_pos: str
    chica_pos: str
    foxy_pos: str
    foxy_sub: int
    freddy_ai: int
    bonnie_ai: int
    chica_ai: int
    foxy_ai: int
    left_doorway_threat: bool
    right_doorway_threat: bool
    power_out: bool
    game_over: bool
    game_over_reason: str


ACTION_NAMES = [
    'NOOP', 'Toggle Cameras', 'Toggle Left Door', 'Toggle Right Door',
    'Toggle Left Light', 'Toggle Right Light',
] + [f'Switch to Cam {cam}' for cam in CAMERAS]


class EpisodeRecorder:
    """Records a full episode for replay and analysis."""

    def __init__(self):
        self.steps: list[StepRecord] = []
        self.cumulative_reward = 0.0

    def record_step(self, step_num: int, action: int, obs, reward: float,
                    info: dict, game: FNAFGame):
        """Record one step of the episode."""
        self.cumulative_reward += reward

        record = StepRecord(
            step=step_num,
            second=info.get('current_second', 0),
            action=action,
            action_name=ACTION_NAMES[action] if action < len(ACTION_NAMES) else f'Unknown({action})',
            reward=reward,
            cumulative_reward=self.cumulative_reward,
            power=info.get('power', 0),
            power_usage=info.get('power_usage', 1),
            cameras_on=game.player.cameras_on,
            current_camera=game.player.current_camera,
            left_door=game.player.left_door_closed,
            right_door=game.player.right_door_closed,
            left_light=game.player.left_light_on,
            right_light=game.player.right_light_on,
            freddy_pos=game.freddy.current_position,
            bonnie_pos=game.bonnie.current_position,
            chica_pos=game.chica.current_position,
            foxy_pos=game.foxy.current_position,
            foxy_sub=game.foxy.sub_position,
            freddy_ai=game.freddy.current_ai_level,
            bonnie_ai=game.bonnie.current_ai_level,
            chica_ai=game.chica.current_ai_level,
            foxy_ai=game.foxy.current_ai_level,
            left_doorway_threat=game.is_animatronic_in_doorway('left'),
            right_doorway_threat=game.is_animatronic_in_doorway('right'),
            power_out=game.power_out,
            game_over=game.game_over,
            game_over_reason=game.game_over_reason.name if game.game_over else 'NONE',
        )
        self.steps.append(record)

    def save(self, filepath: str):
        """Save episode to JSON file."""
        data = {
            'total_steps': len(self.steps),
            'total_reward': self.cumulative_reward,
            'outcome': self.steps[-1].game_over_reason if self.steps else 'UNKNOWN',
            'final_power': self.steps[-1].power if self.steps else 0,
            'steps': [asdict(s) for s in self.steps],
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def summary(self) -> str:
        """Generate a text summary of the episode."""
        if not self.steps:
            return "Empty episode"

        last = self.steps[-1]
        lines = [
            f"Episode Summary ({last.step} steps)",
            f"  Outcome: {last.game_over_reason}",
            f"  Final power: {last.power:.1f}%",
            f"  Total reward: {self.cumulative_reward:.2f}",
            f"  Duration: {last.second:.0f}s",
            "",
            "Action counts:",
        ]

        action_counts = {}
        for s in self.steps:
            action_counts[s.action_name] = action_counts.get(s.action_name, 0) + 1

        for name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {count}")

        # Key events
        threats = [s for s in self.steps if s.left_doorway_threat or s.right_doorway_threat]
        if threats:
            lines.append(f"\nDoorway threats detected: {len(threats)} steps")

        power_out_step = next((s for s in self.steps if s.power_out), None)
        if power_out_step:
            lines.append(f"Power ran out at step {power_out_step.step} ({power_out_step.second:.0f}s)")

        return "\n".join(lines)
