import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from oversight_study.analysis import diagnostics as d
from oversight_study.analysis import plots
from oversight_study.analysis.compare import build_table, wording_robustness
from oversight_study.analysis.metrics import late_shift, protocol_accuracy, shift
from oversight_study.analysis.synthetic import make_synthetic_records


class PlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.study = make_synthetic_records(effect=0.2, n_episodes=40, seed=2)
        late = cls.study.late
        cls.t_accuracy = build_table(protocol_accuracy(late), 1.5)
        cls.t_shift = build_table(late_shift(late, 1.5), 1.5)
        cls.n = len(cls.t_accuracy)

    def assert_written(self, paths) -> None:
        self.assertEqual([p.suffix for p in paths], [".svg", ".png"])
        for path in paths:
            self.assertGreater(path.stat().st_size, 1000, path)

    def test_all_seven_figures(self) -> None:
        late = self.study.late
        with TemporaryDirectory() as temp:
            out = Path(temp)
            self.assert_written(plots.plot_frame_difference(self.t_accuracy, self.t_shift, out,
                                                            n_boot=200))
            dist = d.response_distribution(late.assign(k=late.k))
            self.assert_written(plots.plot_response_distribution(dist, out, self.n))
            curve = d.learning_curve(late, 1.5)
            self.assert_written(plots.plot_learning_curve(curve, out, self.n))
            rates = pd.DataFrame(
                {"description": [0.1, 0.6, 0.9, 0.1, 0.4, 0.9],
                 "application": [0.2, 0.5, 0.8, 0.2, 0.3, 0.8]},
                index=pd.MultiIndex.from_product([["human", "neutral"], ["c", "b", "f"]],
                                                 names=["frame", "letter"]))
            self.assert_written(plots.plot_rule_acquisition(rates, out, self.n))
            s_values = late.assign(s=[shift(y, p, 1.5) for y, p in zip(late.y, late.r_star,
                                                                         strict=True)])
            self.assert_written(plots.plot_shift_histogram(s_values, out, self.n))
            points = pd.DataFrame({"frame": ["human", "human", "neutral"],
                                   "s_episode": [1.0, 0.5, 0.0], "s_end": [0.0, 1.0, None],
                                   "invalid_end": [False, False, True]})
            self.assert_written(plots.plot_episode_vs_end(points, out, self.n))
            two = pd.concat([self.study.late, self.study.late.assign(wording=2)])
            estimate = wording_robustness(build_table(protocol_accuracy(two), 1.5), 0.15, n=200)
            self.assert_written(plots.plot_wording_robustness({"accuracy": estimate}, 0.15, out))
            self.assertEqual(len(list(out.iterdir())), 14)

    def test_missing_series_are_skipped(self) -> None:
        with TemporaryDirectory() as temp:
            paths = plots.plot_frame_difference(self.t_accuracy, self.t_shift, Path(temp),
                                                n_boot=50)
            self.assertTrue(all(p.exists() for p in paths))


if __name__ == "__main__":
    unittest.main()
