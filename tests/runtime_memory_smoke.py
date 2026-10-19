"""CPU-only stage budget contracts; no models or live service calls."""
import unittest
from unittest.mock import patch

import runtime_memory as memory


class StageMemoryBudgetContracts(unittest.TestCase):
    def budget(self, snapshot, *, commit=23*memory.GIB):
        with patch.object(memory, 'memory_status', return_value=snapshot):
            return memory.memory_budget(32*memory.GIB, next_stage_commit_bytes=commit)

    def test_independent_demands_fit_at_their_respective_thresholds(self):
        result = self.budget({'available_bytes':36*memory.GIB, 'available_commit_bytes':27*memory.GIB})
        self.assertTrue(result['fits'])
        self.assertEqual(result['schema'], 'cpu_stage_memory_budget.v2')
        self.assertEqual(result['required_available_physical_bytes'], 32*memory.GIB+memory.CPU_RESERVE_BYTES)
        self.assertEqual(result['required_available_commit_bytes'], 23*memory.GIB+memory.CPU_RESERVE_BYTES)
        self.assertNotIn('required_available_bytes', result)

    def test_physical_shortage_is_not_hidden_by_commit_headroom(self):
        result = self.budget({'available_bytes':32*memory.GIB+memory.CPU_RESERVE_BYTES-1, 'available_commit_bytes':100*memory.GIB})
        self.assertFalse(result['fits'])

    def test_commit_shortage_is_not_hidden_by_physical_headroom(self):
        result = self.budget({'available_bytes':100*memory.GIB, 'available_commit_bytes':23*memory.GIB+memory.CPU_RESERVE_BYTES-1})
        self.assertFalse(result['fits'])

    def test_ordinary_allocation_defaults_to_matching_commit_demand(self):
        result = self.budget({'available_bytes':36*memory.GIB, 'available_commit_bytes':36*memory.GIB}, commit=None)
        self.assertTrue(result['fits'])
        self.assertEqual(result['next_stage_commit_bytes'], 32*memory.GIB)
        self.assertEqual(result['required_available_commit_bytes'], result['required_available_physical_bytes'])

    def test_zero_commit_still_requires_reserve(self):
        result = self.budget({'available_bytes':36*memory.GIB, 'available_commit_bytes':memory.CPU_RESERVE_BYTES-1}, commit=0)
        self.assertFalse(result['fits'])
        self.assertEqual(result['required_available_commit_bytes'], memory.CPU_RESERVE_BYTES)

    def test_unknown_memory_is_refused(self):
        result = self.budget(None)
        self.assertFalse(result['fits'])
        self.assertEqual(result['reason'], 'memory_unknown')

    def test_invalid_commit_demands_are_refused_before_reading_memory(self):
        with patch.object(memory, 'memory_status') as snapshot:
            for value in (-1, 1.0, True, '1'):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    memory.memory_budget(32*memory.GIB, next_stage_commit_bytes=value)
            snapshot.assert_not_called()


if __name__ == '__main__':
    unittest.main()
