"""Small headless runner for repeated Project 2 Q-learning episodes."""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from team01.eval.run_trials import _PROJECT1, _map_game
from Bomberman.events import Event
from Bomberman.sensed_world import SensedWorld
from team01.agent.q_learning import QAgent
from team01.agent.evaluation import q_feature_names
from team01.agent.world_model import WorldModel


@dataclass
class EpisodeResult:
    episode: int
    outcome: str
    ticks: int
    total_reward: float
    bombs_placed: int
    non_finite_q_fallbacks: int
    non_finite_q_values: int
    mean_abs_td_error: float
    td_update_count: int
    max_abs_q: float
    feature_activation_counts: dict[str, int]
    feature_vector_count: int = 0
    bias_activation_count: int = 0


def _verify_bias_feature_vectors(agent: QAgent) -> None:
    agent.feature_vector_count = 0
    agent.bias_activation_count = 0
    original_features = agent.features

    def checked_features(sensed_world, action):
        features = original_features(sensed_world, action)
        agent.feature_vector_count += 1
        if features.get("bias") != 1.0:
            raise RuntimeError("Q-learning feature vector bias must equal 1.0")
        agent.bias_activation_count += 1
        return features

    agent.features = checked_features


def _agent_is_registered(world, agent: QAgent) -> bool:
    return any(agent in characters for characters in world.characters.values())


def run_episode(
    world,
    agent: QAgent,
    episode: int = 1,
    progress_label: Optional[str] = None,
    heartbeat_interval: int = 100,
    on_tick: Optional[Callable[[int], None]] = None,
) -> EpisodeResult:
    """Run one world to a terminal event or timeout."""
    ticks = 0
    evaluation_reward = 0.0
    training = agent.training
    if getattr(agent, "q_contributions_diagnostic", False):
        agent.diagnostic_tick = 0
    world.next_decisions()

    while world.time > 0 and world.characters:
        world.next()
        ticks += 1
        if on_tick is not None:
            on_tick(ticks)
        if (
            progress_label is not None
            and heartbeat_interval > 0
            and ticks % heartbeat_interval == 0
        ):
            print(f"[{progress_label}] still running... tick={ticks}", flush=True)
        if agent.non_finite_q_values or agent.non_finite_q_fallbacks:
            outcome = "NON_FINITE"
            break
        event_types = {event.tpe for event in world.events}

        if Event.CHARACTER_FOUND_EXIT in event_types:
            if not training:
                evaluation_reward += agent.R_WIN
            outcome = "WON"
            break
        if (
            Event.CHARACTER_KILLED_BY_MONSTER in event_types
            or Event.BOMB_HIT_CHARACTER in event_types
        ):
            if not training:
                evaluation_reward += agent.R_LOSE
            outcome = "LOST"
            break
        if world.time <= 0:
            if training:
                # Temporary timeout penalty; tune with terminal rewards later.
                agent.finish_episode(agent.R_LOSE)
            else:
                evaluation_reward += agent.R_LOSE
            outcome = "TIMEOUT"
            break

        if not training and agent.prev_model is not None and agent.prev_action is not None:
            current_model = WorldModel.from_sensed_world(SensedWorld.from_world(world))
            evaluation_reward += agent.calc_reward(
                agent.prev_model,
                current_model,
                agent.prev_action,
            )
        if getattr(agent, "q_contributions_diagnostic", False):
            agent.diagnostic_tick = ticks
        world.next_decisions()
    else:
        if training:
            agent.finish_episode(agent.R_LOSE)
        else:
            evaluation_reward += agent.R_LOSE
        outcome = "LOST" if not world.characters else "TIMEOUT"

    return EpisodeResult(
        episode=episode,
        outcome=outcome,
        ticks=ticks,
        total_reward=agent.episode_reward if training else evaluation_reward,
        bombs_placed=agent.bombs_placed,
        non_finite_q_fallbacks=agent.non_finite_q_fallbacks,
        non_finite_q_values=agent.non_finite_q_values,
        mean_abs_td_error=(
            agent.total_abs_td_error / agent.td_update_count
            if agent.td_update_count
            else 0.0
        ),
        td_update_count=agent.td_update_count,
        max_abs_q=agent.max_abs_q,
        feature_activation_counts=dict(agent.feature_activation_counts),
        feature_vector_count=getattr(agent, "feature_vector_count", 0),
        bias_activation_count=getattr(agent, "bias_activation_count", 0),
    )


def evaluate_policy(
    weights: dict[str, float],
    episodes: int = 10,
    seed: int = 0,
    map_path: Optional[Path] = None,
    checkpoint: int = 0,
) -> list[EpisodeResult]:
    """Evaluate frozen weights on repeatable seeds without touching training RNG."""
    if episodes < 1:
        raise ValueError("evaluation episodes must be at least 1")
    if map_path is None:
        map_path = _PROJECT1 / "map.txt"

    saved_random_state = random.getstate()
    frozen_weights = dict(weights)
    results = []
    try:
        for episode in range(episodes):
            random.seed(seed + episode)
            game = _map_game(map_path)
            agent = QAgent("me", "C", 0, 0)
            agent.weights = dict(weights)
            agent.sync_weights(dict.fromkeys(q_feature_names(), 0.0))
            agent.training = False
            agent.epsilon = 0.0
            _verify_bias_feature_vectors(agent)
            original_weights = dict(agent.weights)
            game.add_character(agent)
            if not _agent_is_registered(game.world, agent):
                raise RuntimeError("evaluation agent was not registered in its world")
            if agent.training or agent.epsilon != 0.0:
                raise RuntimeError("evaluation agent is not using a frozen policy")

            result = run_episode(
                game.world,
                agent,
                episode + 1,
                progress_label=f"EVAL @ {checkpoint} | {episode + 1}/{episodes}",
            )
            if agent.weights != original_weights:
                raise RuntimeError("frozen evaluation modified QAgent weights")
            results.append(result)
            print(
                f"[EVAL @ {checkpoint} | {episode + 1}/{episodes}] "
                f"outcome={result.outcome:<7} ticks={result.ticks} "
                f"reward={result.total_reward:.0f} bombs={result.bombs_placed}",
                flush=True,
            )
            if (
                result.outcome == "NON_FINITE"
                or result.non_finite_q_values
                or result.non_finite_q_fallbacks
            ):
                break
        if weights != frozen_weights:
            raise RuntimeError("frozen evaluation modified its input weights")
    finally:
        random.setstate(saved_random_state)
    return results


def print_evaluation(checkpoint: int, results: list[EpisodeResult]):
    episodes = len(results)
    wins = sum(result.outcome == "WON" for result in results)
    losses = sum(result.outcome == "LOST" for result in results)
    timeouts = sum(result.outcome == "TIMEOUT" for result in results)
    average_ticks = sum(result.ticks for result in results) / episodes
    average_reward = sum(result.total_reward for result in results) / episodes
    bombs = sum(result.bombs_placed for result in results)
    print(
        f"EVAL @ {checkpoint} | wins={wins}/{episodes} losses={losses} "
        f"timeouts={timeouts} win_rate={100.0 * wins / episodes:.1f}% "
        f"avg_ticks={average_ticks:.1f} avg_reward={average_reward:.1f} bombs={bombs}",
        flush=True,
    )


def run_training(
    episodes: int = 10,
    seed: Optional[int] = 0,
    map_path: Optional[Path] = None,
    debug: bool = False,
    weights_in: Optional[Path] = None,
    weights_out: Optional[Path] = None,
    eval_every: int = 0,
    eval_episodes: int = 10,
    eval_seed: int = 0,
    episode_offset: int = 0,
    total_episodes: Optional[int] = None,
):
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if eval_every < 0:
        raise ValueError("eval_every cannot be negative")
    if episode_offset < 0:
        raise ValueError("episode_offset cannot be negative")
    display_total = total_episodes if total_episodes is not None else episode_offset + episodes
    if display_total < episode_offset + episodes:
        raise ValueError("total_episodes cannot be less than the requested episode range")

    if map_path is None:
        map_path = _PROJECT1 / "map.txt"

    weights: dict[str, float] = {}
    if weights_in is not None:
        loader = QAgent("me", "C", 0, 0)
        loader.load_weights(str(weights_in))
        weights = loader.weights

    results = []
    last_agent = None
    eval_seeds = [eval_seed + index for index in range(eval_episodes)]
    print(
        f"Experiment | map={map_path} episodes={episode_offset + 1}-"
        f"{episode_offset + episodes}/{display_total} seed={seed} "
        f"alpha=0.1 gamma=0.9 epsilon=0.1 "
        f"eval_seeds={eval_seeds}",
        flush=True,
    )

    def evaluate_checkpoint(checkpoint: int) -> None:
        print(
            f"--- Evaluation checkpoint after {checkpoint} training episodes ---",
            flush=True,
        )
        weights_before = dict(weights)
        evaluation_results = evaluate_policy(
            weights,
            eval_episodes,
            eval_seed,
            map_path,
            checkpoint=checkpoint,
        )
        if weights != weights_before:
            raise RuntimeError("evaluation modified training weights")
        print_evaluation(checkpoint, evaluation_results)
        if any(
            result.outcome == "NON_FINITE"
            or result.non_finite_q_values
            or result.non_finite_q_fallbacks
            for result in evaluation_results
        ):
            failed = evaluation_results[-1]
            print(
                f"STOP | non-finite value detected during evaluation @ {checkpoint}; "
                f"trial={failed.episode} mean_abs_td={failed.mean_abs_td_error:.6g} "
                f"max_abs_q={failed.max_abs_q:.6g}",
                flush=True,
            )
            return False
        return True

    stop_training = False
    if eval_every and episode_offset == 0:
        stop_training = not evaluate_checkpoint(0)
    for local_episode in range(1, episodes + 1):
        if stop_training:
            break
        episode = episode_offset + local_episode
        if seed is not None:
            random.seed(seed + episode - 1)

        game = _map_game(map_path)
        agent = QAgent("me", "C", 0, 0)
        agent.weights = weights
        agent.debug = debug
        _verify_bias_feature_vectors(agent)
        game.add_character(agent)

        result = run_episode(
            game.world,
            agent,
            episode,
            progress_label=f"TRAIN {episode}/{display_total}",
        )
        results.append(result)
        weights = agent.weights
        last_agent = agent
        if weights_out is not None:
            agent.save_weights(str(weights_out))
        print(
            f"[TRAIN {episode}/{display_total}] outcome={result.outcome:<7} "
            f"ticks={result.ticks} reward={result.total_reward:.0f} "
            f"bombs={result.bombs_placed} mean_abs_td={result.mean_abs_td_error:.3f} "
            f"| max_abs_q={result.max_abs_q:.3f} "
            f"| q_nonfinite={result.non_finite_q_values} "
            f"fallbacks={result.non_finite_q_fallbacks}",
            flush=True,
        )
        if (
            result.outcome == "NON_FINITE"
            or result.non_finite_q_values
            or result.non_finite_q_fallbacks
            or not all(math.isfinite(value) for value in weights.values())
        ):
            print(
                f"STOP | non-finite value detected in training episode {episode}; "
                f"mean_abs_td={result.mean_abs_td_error:.6g} "
                f"max_abs_q={result.max_abs_q:.6g}",
                flush=True,
            )
            stop_training = True
            break
        if eval_every and episode % eval_every == 0:
            if not evaluate_checkpoint(episode):
                stop_training = True
                break

    completed_episodes = len(results)
    wins = sum(result.outcome == "WON" for result in results)
    losses = sum(result.outcome == "LOST" for result in results)
    timeouts = sum(result.outcome == "TIMEOUT" for result in results)
    average_ticks = (
        sum(result.ticks for result in results) / completed_episodes
        if completed_episodes
        else 0.0
    )
    average_reward = (
        sum(result.total_reward for result in results) / completed_episodes
        if completed_episodes
        else 0.0
    )
    bomb_count = sum(result.bombs_placed for result in results)
    non_finite_q_values = sum(result.non_finite_q_values for result in results)
    non_finite_q_fallbacks = sum(result.non_finite_q_fallbacks for result in results)
    all_weights_finite = all(math.isfinite(value) for value in weights.values())
    max_abs_weight = max((abs(value) for value in weights.values() if math.isfinite(value)), default=0.0)
    print(
        f"Summary | wins={wins} losses={losses} timeouts={timeouts} "
        f"avg_ticks={average_ticks:.1f} avg_reward={average_reward:.1f} "
        f"bombs={bomb_count} q_nonfinite={non_finite_q_values} "
        f"fallbacks={non_finite_q_fallbacks}"
    )
    if stop_training:
        print(f"Experiment stopped after {completed_episodes} completed training episodes.", flush=True)
    print(f"Weights: {weights}")
    print(f"Weight stability | all_finite={all_weights_finite} max_abs={max_abs_weight:.6g}")

    total_updates = sum(result.td_update_count for result in results)
    total_abs_td_error = sum(result.mean_abs_td_error * result.td_update_count for result in results)
    print(
        f"Training diagnostics | mean_abs_td="
        f"{total_abs_td_error / total_updates if total_updates else 0.0:.3f} "
        f"max_abs_q={max((result.max_abs_q for result in results), default=0.0):.3f}"
    )
    for name in q_feature_names():
        activation_count = sum(
            result.feature_activation_counts.get(name, 0)
            for result in results
        )
        activation_percentage = 100.0 * activation_count / total_updates if total_updates else 0.0
        print(
            f"Feature activation | {name}={activation_count}/{total_updates} "
            f"({activation_percentage:.1f}%)"
        )

    feature_vectors = sum(result.feature_vector_count for result in results)
    bias_activations = sum(result.bias_activation_count for result in results)
    bias_percentage = 100.0 * bias_activations / feature_vectors if feature_vectors else 0.0
    print(
        f"Bias verification | activation={bias_activations}/{feature_vectors} "
        f"({bias_percentage:.1f}%) learned_weight={weights.get('bias', 0.0):.6g}",
        flush=True,
    )

    block_size = eval_every or episodes
    for start in range(0, len(results), block_size):
        block = results[start : start + block_size]
        block_updates = sum(result.td_update_count for result in block)
        block_td_total = sum(
            result.mean_abs_td_error * result.td_update_count for result in block
        )
        print(
            f"Training block {episode_offset + start + 1}-"
            f"{episode_offset + start + len(block)} | "
            f"avg_reward={sum(item.total_reward for item in block) / len(block):.1f} "
            f"avg_ticks={sum(item.ticks for item in block) / len(block):.1f} "
            f"mean_abs_td={block_td_total / block_updates if block_updates else 0.0:.3f} "
            f"max_abs_q={max((item.max_abs_q for item in block), default=0.0):.3f}",
            flush=True,
        )

    if weights_out is not None and last_agent is not None:
        last_agent.save_weights(str(weights_out))

    return results, weights


def main():
    parser = argparse.ArgumentParser(description="Train QAgent for a small number of episodes.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map", type=Path, default=_PROJECT1 / "map.txt")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--weights-in", type=Path)
    parser.add_argument("--weights-out", type=Path)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--total-episodes", type=int)
    args = parser.parse_args()
    run_training(
        episodes=args.episodes,
        seed=args.seed,
        map_path=args.map,
        debug=args.debug,
        weights_in=args.weights_in,
        weights_out=args.weights_out,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        episode_offset=args.episode_offset,
        total_episodes=args.total_episodes,
    )


if __name__ == "__main__":
    main()