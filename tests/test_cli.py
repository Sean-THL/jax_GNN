import contextlib
import io
import sys
import unittest
from unittest import mock

from gnn import cli


class CliTests(unittest.TestCase):
    def test_help_lists_the_small_public_surface(self):
        output = io.StringIO()
        with mock.patch.object(sys, "argv", ["gnn"]), contextlib.redirect_stdout(output):
            cli.main()
        rendered = output.getvalue()
        self.assertIn("python -m gnn train", rendered)
        self.assertIn("integrated", rendered)
        self.assertIn("historical", rendered)

    def test_integrated_selection_forwards_remaining_arguments(self):
        forwarded = []
        with (
            mock.patch.object(
                sys, "argv", ["gnn", "train", "--model", "gat", "--epochs", "3"]
            ),
            mock.patch("gnn.variants.train_main") as train,
        ):
            train.side_effect = lambda *_: forwarded.extend(sys.argv[1:])
            cli.main()
            train.assert_called_once_with("new", "gat")
        self.assertEqual(forwarded, ["--epochs", "3"])

    def test_historical_selection(self):
        with (
            mock.patch.object(
                sys, "argv", ["gnn", "test", "--pipeline", "historical", "--model", "gatv2"]
            ),
            mock.patch("legacy.historical_baseline.test_main") as test,
        ):
            cli.main()
        test.assert_called_once_with("gatv2")


if __name__ == "__main__":
    unittest.main()
