"""
Custom Stable-Baselines3 callbacks for FNAF training.
"""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class FnafMetricsCallback(BaseCallback):
    """
    Tracks FNAF-specific metrics during training:
    - Win rate (survival rate)
    - Average power remaining at end of episode
    - Average survival time
    - Death causes breakdown
    """

    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = 0
        self.deaths = 0
        self.death_causes = {}
        self.final_powers = []
        self.total_episodes = 0

    def _on_step(self) -> bool:
        # Check for episode completion in infos
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if 'game_over_reason' in info and info['game_over_reason'] != 'NONE':
                    self.total_episodes += 1
                    reason = info['game_over_reason']

                    if reason == 'SURVIVED':
                        self.wins += 1
                    else:
                        self.deaths += 1
                        self.death_causes[reason] = self.death_causes.get(reason, 0) + 1

                    self.final_powers.append(info.get('power', 0))

        if self.num_timesteps % self.log_freq == 0 and self.total_episodes > 0:
            win_rate = self.wins / self.total_episodes
            avg_power = np.mean(self.final_powers[-100:]) if self.final_powers else 0

            self.logger.record('fnaf/win_rate', win_rate)
            self.logger.record('fnaf/total_episodes', self.total_episodes)
            self.logger.record('fnaf/avg_final_power', avg_power)
            self.logger.record('fnaf/total_wins', self.wins)

            for cause, count in self.death_causes.items():
                self.logger.record(f'fnaf/deaths_{cause.lower()}', count)

            if self.verbose > 0:
                print(f"  [FNAF] Episodes: {self.total_episodes} | "
                      f"Win rate: {win_rate*100:.1f}% | "
                      f"Avg power: {avg_power:.1f}%")

        return True

    def _on_training_end(self):
        if self.total_episodes > 0:
            print(f"\n{'='*50}")
            print(f"FNAF Training Summary")
            print(f"{'='*50}")
            print(f"Total episodes: {self.total_episodes}")
            print(f"Wins: {self.wins} ({self.wins/self.total_episodes*100:.1f}%)")
            print(f"Deaths: {self.deaths}")
            if self.death_causes:
                print(f"Death breakdown:")
                for cause, count in sorted(self.death_causes.items(),
                                          key=lambda x: x[1], reverse=True):
                    print(f"  {cause}: {count} ({count/self.total_episodes*100:.1f}%)")
            if self.final_powers:
                print(f"Avg final power: {np.mean(self.final_powers):.1f}%")


class FnafCurriculumCallback(BaseCallback):
    """
    Automatically increases night difficulty when win rate exceeds threshold.
    """

    def __init__(self, win_threshold=0.7, check_freq=5000,
                 min_episodes=50, max_night=7, verbose=1):
        super().__init__(verbose)
        self.win_threshold = win_threshold
        self.check_freq = check_freq
        self.min_episodes = min_episodes
        self.max_night = max_night
        self.current_night = 1
        self.episode_results = []  # True=win, False=loss

    def _on_step(self) -> bool:
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if 'game_over_reason' in info and info['game_over_reason'] != 'NONE':
                    self.episode_results.append(info['game_over_reason'] == 'SURVIVED')

        if self.num_timesteps % self.check_freq == 0:
            recent = self.episode_results[-self.min_episodes:]
            if len(recent) >= self.min_episodes:
                win_rate = sum(recent) / len(recent)

                if win_rate >= self.win_threshold and self.current_night < self.max_night:
                    self.current_night += 1
                    if self.verbose > 0:
                        print(f"\n[Curriculum] Win rate {win_rate*100:.1f}% exceeds "
                              f"{self.win_threshold*100:.0f}% threshold. "
                              f"Advancing to Night {self.current_night}!")

                    # Update environment night via options on next reset
                    for env in self.training_env.envs:
                        if hasattr(env, 'unwrapped'):
                            env.unwrapped.night = self.current_night

                    self.episode_results.clear()

        return True
