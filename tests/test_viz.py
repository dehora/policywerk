"""Tests for visualization infrastructure."""

import os
import tempfile

from policywerk.viz.animate import (
    create_lesson_figure, FrameSnapshot, FrameRecorder,
    save_animation, save_poster, save_figure,
)
from policywerk.viz.traces import plot_training_traces, update_trace_axes
from policywerk.viz.values import (
    draw_value_heatmap, draw_policy_arrows, draw_grid_overlay,
    draw_value_bars, draw_q_bars,
)
from policywerk.viz.trajectories import (
    draw_trajectory, draw_agent, draw_target, draw_pixel_env,
    draw_policy_gaussian, draw_real_vs_imagined, draw_cliff_grid,
)


class TestAnimateSkeleton:
    def test_create_lesson_figure(self):
        fig, axes = create_lesson_figure("Test Lesson", "Paper 2024")
        assert "env" in axes
        assert "algo" in axes
        assert "trace" in axes
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_frame_recorder(self):
        rec = FrameRecorder(record_interval=5)
        assert rec.should_record(0)
        assert not rec.should_record(1)
        assert rec.should_record(5)
        assert rec.should_record(10)

        rec.record(FrameSnapshot(episode=0, total_reward=1.0))
        rec.record(FrameSnapshot(episode=5, total_reward=2.0))
        assert rec.frame_count == 2

    def test_save_animation_gif(self):
        fig, axes = create_lesson_figure("Test")
        def update(frame):
            axes["trace"].clear()
            axes["trace"].plot([0, frame], [0, frame])
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            path = f.name
        try:
            save_animation(fig, update, frame_count=3, path=path, fps=2)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_save_poster(self):
        fig, axes = create_lesson_figure("Test")
        def update(frame):
            axes["trace"].clear()
            axes["trace"].plot([0, 1], [0, 1])
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            save_poster(fig, update, frame_index=0, path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)
            os.unlink(path)

    def test_save_figure(self):
        fig, axes = create_lesson_figure("Test")
        axes["trace"].plot([1, 2, 3], [1, 4, 9])
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            save_figure(fig, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)


class TestTraces:
    def test_plot_training_traces(self):
        import matplotlib.pyplot as plt
        fig = plot_training_traces(
            {"reward": [1.0, 2.0, 3.0], "loss": [0.5, 0.3, 0.1]},
            title="Test",
        )
        assert fig is not None
        plt.close(fig)

    def test_update_trace_axes(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        update_trace_axes(ax, [1.0, 2.0, 3.0], label="reward")
        plt.close(fig)


class TestValues:
    def test_draw_value_heatmap(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        draw_value_heatmap(ax, [[0.0, 0.5], [1.0, -0.5]])
        plt.close(fig)

    def test_draw_value_heatmap_skip_cells(self):
        import matplotlib.pyplot as plt
        values = [[0.0, 0.5], [1.0, -0.5]]

        # Without skip_cells
        fig1, ax1 = plt.subplots()
        draw_value_heatmap(ax1, values)
        texts_all = [c for c in ax1.get_children()
                     if isinstance(c, plt.Text) and c.get_text().strip()]
        plt.close(fig1)

        # With skip_cells={(0, 0)}
        fig2, ax2 = plt.subplots()
        draw_value_heatmap(ax2, values, skip_cells={(0, 0)})
        texts_skip = [c for c in ax2.get_children()
                      if isinstance(c, plt.Text) and c.get_text().strip()]
        plt.close(fig2)

        assert len(texts_skip) < len(texts_all)

    def test_draw_policy_arrows(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        draw_policy_arrows(ax, {"0,0": 1, "0,1": 2}, rows=2, cols=2)
        plt.close(fig)

    def test_draw_grid_overlay(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        draw_grid_overlay(ax, 3, 3, walls=[(1, 1)], goals=[(0, 2)], pits=[(2, 0)])
        plt.close(fig)

    def test_draw_value_bars(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        draw_value_bars(ax, [0.1, 0.3, 0.5], [0.17, 0.33, 0.5], ["A", "B", "C"])
        plt.close(fig)

    def test_draw_q_bars(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        draw_q_bars(ax, [1.0, 2.0, 0.5, 1.5], ["N", "E", "S", "W"])
        plt.close(fig)


class TestTrajectories:
    def test_draw_trajectory(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        draw_trajectory(ax, [(0, 0), (1, 1), (2, 0)])
        plt.close(fig)

    def test_draw_agent_and_target(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        draw_agent(ax, (1.0, 2.0))
        draw_target(ax, (3.0, 4.0))
        plt.close(fig)

    def test_draw_pixel_env(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        frame = [[0.0] * 16 for _ in range(16)]
        frame[8][8] = 1.0
        draw_pixel_env(ax, frame)
        plt.close(fig)

    def test_draw_policy_gaussian(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        draw_policy_gaussian(ax, mean=0.0, std=1.0)
        plt.close(fig)

    def test_draw_real_vs_imagined(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        real = [[0.0] * 8 for _ in range(8)]
        imagined = [[0.5] * 8 for _ in range(8)]
        draw_real_vs_imagined(ax, real, imagined)
        plt.close(fig)

    def test_draw_cliff_grid(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        cliff = [(3, c) for c in range(1, 11)]
        draw_cliff_grid(ax, 4, 12, cliff, (3, 0), (3, 11))
        plt.close(fig)

    def test_draw_cliff_grid_with_agent(self):
        import matplotlib.pyplot as plt
        # Without agent_pos: count collections
        fig1, ax1 = plt.subplots()
        cliff = [(3, c) for c in range(1, 11)]
        draw_cliff_grid(ax1, 4, 12, cliff, (3, 0), (3, 11))
        collections_without = len(ax1.collections)
        plt.close(fig1)

        # With agent_pos: should have one more collection (the scatter marker)
        fig2, ax2 = plt.subplots()
        draw_cliff_grid(ax2, 4, 12, cliff, (3, 0), (3, 11),
                        agent_pos=(2, 1),
                        caption="test caption")
        collections_with = len(ax2.collections)
        plt.close(fig2)

        assert collections_with > collections_without, \
            "agent_pos should add a scatter collection for the triangle marker"
