"""Suite selection never starts an implicit model run or repeats a check."""
from contextlib import redirect_stderr
from io import StringIO
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('cosluma_test_runner', Path(__file__).with_name('run.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_suites_partition_the_existing_offline_regressions(self):
        self.assertFalse(set(runner.CORE) & set(runner.INTEGRATION))
        full = runner.selected_checks(['full'])
        self.assertEqual(len(full), len(set(full)))
        self.assertFalse(set(full) & runner.MODEL_CHECKS)
        self.assertIn('photo_shaping', runner.INTEGRATION)
        self.assertIn('manual_target_budget', runner.CORE)
        for name in full:
            self.assertTrue(Path(__file__).with_name(name+'_smoke.py').is_file(), name)

    def test_repeated_names_execute_once_and_implicit_all_is_rejected(self):
        self.assertEqual(runner.selected_checks(['photo_editing', 'photo_editing_smoke.py']), ['photo_editing'])
        for names in (['all'], ['core', 'photo_editing'], ['../outside'], ['sub\\outside']):
            with self.assertRaises(ValueError):
                runner.selected_checks(names)

    def test_list_does_not_start_children_and_model_runtime_is_explicit(self):
        with patch.object(runner.subprocess, 'run') as child, patch('builtins.print'):
            self.assertEqual(runner.main(['full', '--list']), 0)
            child.assert_not_called()
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                runner.main(['sapiens_current'])
            self.assertEqual(error.exception.code, 2)
            child.assert_not_called()

    def test_failure_stops_before_remaining_checks(self):
        with patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=7)) as child, \
                patch('builtins.print'):
            self.assertEqual(runner.main(['photo_editing', 'asset_identity']), 7)
            self.assertEqual(child.call_count, 1)


if __name__ == '__main__':
    unittest.main()
