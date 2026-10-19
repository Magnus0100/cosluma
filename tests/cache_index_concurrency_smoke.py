"""Real process contention for content-cache insertion, lookup, and eviction."""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import tempfile
import time

from cache_store import ContentCache


class ContendedCache(ContentCache):
    def _load_index(self):
        value = super()._load_index()
        # Widen the read/modify window so an unlocked implementation reliably
        # loses another process's insert or access update.
        time.sleep(.003)
        return value


def mutate(root, worker, limit, barrier):
    cache = ContendedCache(root, max_bytes=limit)
    barrier.wait(timeout=10)
    for index in range(30):
        key = cache.make_key('shared-source', 'fixture', params={'worker': worker, 'index': index})
        expected = bytes([worker, index])*32
        cache.put_bytes(key, 'proxy.bin', expected)
        path = cache.get(key, 'proxy.bin')
        # With a small budget the other process may evict this entry between
        # put and get; every returned, still-existing file must remain intact.
        if path is not None:
            try:
                assert path.read_bytes() == expected
            except FileNotFoundError:
                assert limit < 1_000_000
        elif limit >= 1_000_000:
            raise AssertionError('Concurrent index update lost an inserted entry')
        cache.evict()


def main():
    context = multiprocessing.get_context('spawn')
    with tempfile.TemporaryDirectory(prefix='content-cache-index-') as temporary:
        for name, budget in [('retained', 1_000_000), ('evicted', 512)]:
            root = Path(temporary)/name
            barrier = context.Barrier(2)
            children = [context.Process(target=mutate, args=(str(root), worker, budget, barrier)) for worker in (1, 2)]
            try:
                for child in children: child.start()
                for child in children:
                    child.join(30)
                    assert child.exitcode == 0, child.exitcode
                cache = ContentCache(root, max_bytes=budget)
                index = json.loads(cache.index_path.read_text(encoding='utf8'))
                if name == 'retained':
                    assert len(index['entries']) == 60, len(index['entries'])
                else:
                    assert len(index['entries']) <= 8
                for key, item in index['entries'].items():
                    paths = [cache.entry_dir(key)/artifact['path'] for artifact in item['artifacts'].values()]
                    assert all(path.is_file() for path in paths)
                    assert item['bytes'] == sum(path.stat().st_size for path in paths)
                assert cache.total_bytes() <= budget
            finally:
                for child in children:
                    if child.is_alive(): child.terminate(); child.join(5)
    print(json.dumps({'ok': True, 'processes_per_phase': 2,
        'all_60_concurrent_inserts_retained': True, 'parallel_get_and_eviction_consistent': True}))


if __name__ == '__main__':
    main()
