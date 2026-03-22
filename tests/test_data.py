"""Tests for data modules (episode collection and logging)."""

from policywerk.data.episode import collect_episode, collect_episodes
from policywerk.data.logging import MetricLog, TrainingLog
from policywerk.world.random_walk import RandomWalk
from policywerk.primitives.random import create_rng


class TestEpisode:
    def test_collect_episode(self):
        rng = create_rng(42)
        env = RandomWalk()

        def random_policy(state):
            return 0 if rng.random() < 0.5 else 1

        episode = collect_episode(env, random_policy)
        assert len(episode.transitions) > 0
        assert episode.transitions[-1].done is True

    def test_collect_episodes(self):
        rng = create_rng(42)
        env = RandomWalk()

        def random_policy(state):
            return 0 if rng.random() < 0.5 else 1

        episodes = collect_episodes(env, random_policy, num_episodes=3)
        assert len(episodes) == 3
        for ep in episodes:
            assert len(ep.transitions) > 0


class TestMetricLog:
    def test_record(self):
        log = MetricLog(name="loss")
        log.record(1.0)
        log.record(2.0)
        log.record(3.0)
        assert log.last == 3.0
        assert abs(log.mean - 2.0) < 1e-10
        assert abs(log.recent_mean(2) - 2.5) < 1e-10


class TestTrainingLog:
    def test_record_and_summary(self):
        tlog = TrainingLog()
        tlog.record("loss", 0.5)
        tlog.record("loss", 0.3)
        tlog.record("reward", 1.0)
        summary = tlog.summary()
        assert "loss" in summary
        assert "reward" in summary
