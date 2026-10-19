"""Request-owned GPU sampling invariants, with desktop/small CPU coverage."""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import native_cuda_fields as fields
import native_cuda_sampling as sampling
import render_reference_colour_composite as renderer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    assert 'torch' not in sys.modules and 'cupy' not in sys.modules
    source = np.full((24, 31, 4), .02, np.float32)
    owners = np.full(source.shape[:2], 4, np.int32)
    original = source.copy()
    @sampling.request_sampling_lease(('source', 'owners'))
    def small(source, owners):
        return renderer.sample_maps(source, (3, 5, 49, 31), (31, 24), (60, 42), owners=owners)
    first = small(source, owners)
    with fields.cuda_field_scope():
        second = small(source, owners)
        report = fields.execution()['sampling_requests'][-1]
        assert report['actual_backend'] == 'numpy_cpu' and not report['gpu_calls']
        assert report['cpu_calls'] == {'S': 1} and report['lease_released']
        assert report['host_snapshot_bytes'] == report['resident_H2D_bytes'] == 0
    assert first.tobytes() == second.tobytes() and np.array_equal(source, original)
    assert sampling._CURRENT.get() is None and 'torch' not in sys.modules and 'cupy' not in sys.modules
    # A generator's lease is alive during iteration and closes on cancellation.
    @sampling.request_sampling_lease(('source', 'owners'))
    def streamed(source, owners):
        yield sampling._CURRENT.get()
        raise AssertionError('Cancelled generator resumed')
    with fields.cuda_field_scope():
        generator = streamed(source, owners)
        assert sampling._CURRENT.get() is None
        state = next(generator)
        assert state is sampling._CURRENT.get()
        generator.close()
        assert sampling._CURRENT.get() is None and not state.entries
        assert fields.execution()['sampling_requests'][-1]['lease_released']
    @sampling.request_sampling_lease(('source', 'owners'))
    def failed(source, owners):
        raise RuntimeError('Deliberate caller cancellation')
    with fields.cuda_field_scope():
        try:
            failed(source, owners)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Matching failure was concealed')
        report = fields.execution()['sampling_requests'][-1]
        assert report['failures'] and report['lease_released'] and sampling._CURRENT.get() is None
    # Memory/report diagnostics cannot retain a request or replace its error.
    calls = []
    def synchronize(device):
        assert device == 'cuda:3'; calls.append(device)
    def unavailable(device):
        assert device == 'cuda:3'; raise RuntimeError('Deliberate diagnostic failure')
    fake_cuda = SimpleNamespace(is_initialized=lambda: False, synchronize=synchronize,
        mem_get_info=unavailable, memory_allocated=unavailable,
        max_memory_allocated=lambda device: 0, max_memory_reserved=lambda device: 0)
    fake_torch = SimpleNamespace(__version__='diagnostics_mock_no_GPU', cuda=fake_cuda,
        float32='float32', int32='int32', tensor=lambda *args, **kwargs: object())
    @sampling.request_sampling_lease(('source', 'owners'))
    def capture(source, owners):
        return sampling._CURRENT.get()
    class FailingDiagnostics(list):
        def append(self, value):
            raise RuntimeError('Deliberate report append failure')
    with (patch.dict(sys.modules, {'torch': fake_torch}), patch.object(sampling, 'MIN_PIXELS', 1),
          patch.object(fields, '_device', return_value='cuda:3')):
        with fields.cuda_field_scope():
            diagnostic_state = capture(source, owners)
            assert sampling._CURRENT.get() is None and not diagnostic_state.entries
            # Admission is lazy: an unused reserved bank must not initialize
            # Torch/CUDA or run memory diagnostics merely to close the lease.
            assert diagnostic_state.report['lease_released']
            assert diagnostic_state.report['resident_H2D_bytes'] == 0
            assert diagnostic_state.report['banks_uploaded'] == 0
            assert 'diagnostic_failures' not in diagnostic_state.report
        with fields.cuda_field_scope():
            fields._STATE.get()['sampling_requests'] = FailingDiagnostics()
            diagnostic_state = capture(source, owners)
            assert sampling._CURRENT.get() is None and not diagnostic_state.entries
            assert diagnostic_state.report['diagnostic_failures'][-1]['metric'] == 'field_scope_append'
        with fields.cuda_field_scope():
            try:
                failed(source, owners)
            except RuntimeError as error:
                assert str(error) == 'Deliberate caller cancellation'
            else:
                raise AssertionError('Metric failure concealed the original caller failure')
            assert sampling._CURRENT.get() is None and fields.execution()['sampling_requests'][-1]['lease_released']
        target, proxy, canvas = (3, 5, 49, 31), (31, 24), (60, 42)
        grid = renderer._sample_grid(target, proxy, canvas)
        plan = renderer._sample_plan(grid, source.shape[:2])
        malformed = [
            {'owner_indices': plan['owner_indices'].astype(np.float32)},
            {'owner_indices': np.full(grid[0].shape, 1 << 40, np.int64)},
            {'owner_indices': np.full(grid[0].shape, -1, np.int32)},
            {'owner_indices': np.full(grid[0].shape, source.shape[0]*source.shape[1], np.int32)},
            {'indices': (plan['indices'][0][:, :-1], *plan['indices'][1:])},
            {'weights': (np.full(grid[0].shape, np.nan), *plan['weights'][1:])},
            {'weights': (plan['weights'][0].astype(np.complex128), *plan['weights'][1:])},
            {'yy': plan['yy'][:, :-1]},
        ]
        @sampling.request_sampling_lease(('source', 'owners'))
        def invalid_plan(source, owners, candidate):
            return renderer.sample_maps(source, target, proxy, canvas, owners=owners, grid=grid, sample_plan=candidate)
        for change in malformed:
            with fields.cuda_field_scope():
                try:
                    invalid_plan(source, owners, {**plan, **change})
                except ValueError:
                    pass
                else:
                    raise AssertionError('Malformed plan reached a GPU cast/gather')
                assert sampling._CURRENT.get() is None and fields.execution()['sampling_requests'][-1]['lease_released']
        @sampling.request_sampling_lease(('source', 'owners'))
        def compound(source, owners):
            original_source[0, 0, 0] += .001
            raise InterruptedError('Original matching cancellation')
        original_source = source.copy()
        with fields.cuda_field_scope():
            try:
                compound(original_source, owners)
            except InterruptedError as error:
                assert str(error) == 'Original matching cancellation'
            else:
                raise AssertionError('Hash closure concealed the primary cancellation')
            report = fields.execution()['sampling_requests'][-1]
            assert not report['source_and_snapshot_hashes_closed']
            assert any(e['phase'] == 'source_hash_closure' for e in report['failures'])
            assert report['lease_released'] and sampling._CURRENT.get() is None
    # No caller in this block reached a valid GPU sample, so lazy admission
    # never queried the diagnostic CUDA methods or imported Torch.
    assert not calls and 'torch' not in sys.modules

    # The first eligible consumer initializes one reserved bank under the lease
    # lock; concurrent initializers must observe the same pair.  A failed owner
    # upload must leave no resident bytes or live entries.
    upload_source = np.full((64, 64, 4), .02, np.float32)
    upload_owners = np.ones(upload_source.shape[:2], np.int32)
    tensor_calls = []
    fake_upload_cuda = SimpleNamespace(
        is_initialized=lambda: False, synchronize=lambda device: None,
        mem_get_info=lambda device: (10000, 20000), memory_allocated=lambda device: 0,
        max_memory_allocated=lambda device: 0, max_memory_reserved=lambda device: 0)
    fake_upload_torch = SimpleNamespace(__version__='lazy_upload_mock', cuda=fake_upload_cuda,
        float32='float32', int32='int32',
        tensor=lambda value, **kwargs: (tensor_calls.append((value.nbytes, kwargs['device'])), object())[1])
    @sampling.request_sampling_lease(('source', 'owners'))
    def concurrent_upload(source, owners):
        state = sampling._CURRENT.get()
        entry = next(iter(state.entries.values()))
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _item: sampling._ensure_entry_device(state, entry, fields), (0, 1)))
        return state, results
    with (patch.dict(sys.modules, {'torch': fake_upload_torch}),
          patch.object(sampling, 'MIN_PIXELS', 1), patch.object(fields, '_device', return_value='cuda:3')):
        with fields.cuda_field_scope():
            upload_state, upload_results = concurrent_upload(upload_source, upload_owners)
            upload_report = fields.execution()['sampling_requests'][-1]
            assert upload_results[0] == upload_results[1]
            assert len(tensor_calls) == 2 and upload_report['banks_uploaded'] == 1
            assert upload_report['resident_H2D_bytes'] == upload_report['reserved_bytes']
            assert upload_report['resident_H2D_bytes'] == upload_source.nbytes + upload_owners.nbytes
            assert upload_report['lease_released'] and not upload_state.entries
    failed_calls = []
    def failing_tensor(value, **kwargs):
        failed_calls.append(value.nbytes)
        if len(failed_calls) == 2:
            raise RuntimeError('Deliberate owner upload failure')
        return object()
    failing_torch = SimpleNamespace(__version__='lazy_upload_failure_mock', cuda=fake_upload_cuda,
        float32='float32', int32='int32', tensor=failing_tensor)
    @sampling.request_sampling_lease(('source', 'owners'))
    def failed_upload(source, owners):
        state = sampling._CURRENT.get(); entry = next(iter(state.entries.values()))
        sampling._ensure_entry_device(state, entry, fields)
    with (patch.dict(sys.modules, {'torch': failing_torch}),
          patch.object(sampling, 'MIN_PIXELS', 1), patch.object(fields, '_device', return_value='cuda:3')):
        with fields.cuda_field_scope():
            try:
                failed_upload(upload_source, upload_owners)
            except RuntimeError as error:
                assert str(error) == 'Deliberate owner upload failure'
            else:
                raise AssertionError('Partial lazy bank upload was concealed')
            failure_report = fields.execution()['sampling_requests'][-1]
            assert failure_report['lease_released'] and failure_report['resident_H2D_bytes'] == 0
            assert failure_report['failures'] and sampling._CURRENT.get() is None
    # A large identity-grid caller stays on the proven CPU path without
    # copying/hashing/uploading banks or initializing Torch at admission.
    same_grid = np.full((513, 401, 4), .02, np.float32)
    same_owners = np.full(same_grid.shape[:2], 4, np.int32)
    @sampling.request_sampling_lease(('source', 'owners'))
    def identity(source, owners, native):
        assert source is same_grid and owners is same_owners
        return renderer.sample_maps(source, (0, 0, native[0], native[1]), native, native, owners=owners)
    with (fields.cuda_field_scope(), patch.object(sampling, '_sha', side_effect=AssertionError('Identity banks hashed')),
          patch.dict(sys.modules, {'torch': None})):
        result = identity(same_grid, same_owners, (401, 513))
        report = fields.execution()['sampling_requests'][-1]
        assert report['resident_H2D_bytes'] == report['host_snapshot_bytes'] == 0 and not report['admission']
        assert report['admission_skips'][0]['reason'] == 'native_identity_grid' and report['actual_backend'] == 'numpy_cpu'
        assert result.tobytes() == same_grid.tobytes() and sampling._CURRENT.get() is None
    assert 'torch' not in sys.modules
    if args.cuda:
        rng = np.random.default_rng(20261014)
        bank = rng.uniform(-.02, .04, (613, 431, 4)).astype(np.float32)
        domain = np.full(bank.shape[:2], 23, np.int32)
        domain[:, :211] = 4; domain[151:250, 137:198] = 1004
        box, proxy, native = (71, 193, 811, 707), (431, 613), (1024, 1001)
        grid = renderer._sample_grid(box, proxy, native)
        plan = renderer._sample_plan(grid, bank.shape[:2])
        support = rng.uniform(.1, .9, grid[0].shape).astype(np.float32)
        expected = renderer.sample_maps(bank, box, proxy, native, owners=domain, grid=grid,
            sample_plan=plan, hair_face_support=support)
        bank_hash, owner_hash = sampling._sha(bank), sampling._sha(domain)
        @sampling.request_sampling_lease(('source', 'owners'))
        def large(source, owners):
            state = sampling._CURRENT.get()
            assert source.flags.owndata and not source.flags.writeable and source.base is None
            assert owners.flags.owndata and not owners.flags.writeable and owners.base is None
            return renderer.sample_maps(source, box, proxy, native, owners=owners, grid=grid,
                sample_plan=plan, hair_face_support=support), state
        with fields.cuda_field_scope():
            actual, state = large(bank[:, ::-1][:, ::-1], domain[:, ::-1][:, ::-1])
            report = fields.execution()['sampling_requests'][-1]
            assert report['gpu_calls'] == {'S': 1} and not report['failures']
            assert report['source_and_snapshot_hashes_closed'] and report['lease_released']
            assert report['live_entries_after_release'] == 0 and not state.entries and sampling._CURRENT.get() is None
            assert report['owner_dtype'] == 'int32' and report['resident_H2D_bytes'] == bank.nbytes+domain.nbytes
            assert report['plan_support_H2D_bytes'] == grid[0].size*56
            assert report['result_D2H_bytes'] == actual.nbytes and report['torch_version']
            assert report['cuda_free_total_after_release_bytes'][0] > 0
        assert np.isfinite(actual).all() and actual.dtype == np.float32 and actual.shape == expected.shape
        assert np.max(np.abs(actual-expected)) < 2e-6
        assert sampling._sha(bank) == bank_hash and sampling._sha(domain) == owner_hash
        # A second request admits/uploads again; no tensor is reused across it.
        with fields.cuda_field_scope():
            again, state2 = large(bank, domain)
            assert state2 is not state and not state2.entries and np.max(np.abs(again-expected)) < 2e-6
        @sampling.request_sampling_lease(('source', 'owners'))
        def mutated(source, owners):
            bank[0, 0, 0] += .001
        with fields.cuda_field_scope():
            try:
                mutated(bank, domain)
            except ValueError as error:
                assert 'source/snapshot changed' in str(error)
            else:
                raise AssertionError('Changed original source was allowed to publish')
            report = fields.execution()['sampling_requests'][-1]
            assert report['failures'] and report['lease_released'] and sampling._CURRENT.get() is None
        @sampling.request_sampling_lease(('source', 'owners'))
        def GPU_failed(source, owners):
            with patch.object(state2.torch, 'index_select', side_effect=RuntimeError('Deliberate CUDA failure')):
                return renderer.sample_maps(source, box, proxy, native, owners=owners)
        with fields.cuda_field_scope():
            try:
                GPU_failed(bank, domain)
            except RuntimeError as error:
                assert 'CPU retry is disabled' in str(error)
            else:
                raise AssertionError('Selected GPU gather fell back to CPU')
            assert sampling._CURRENT.get() is None and fields.execution()['sampling_requests'][-1]['lease_released']
    print('native sampling smoke pass: desktop laziness, small CPU, generator/cancel/failure cleanup, failing diagnostics cleanup and selected-device queries'+
          (', source/readonly backing, original owner/global coordinates/face support, dtype/finite, transfer counters, source mutation rejection, explicit GPU failure, no cross-request residency' if args.cuda else ''))


if __name__ == '__main__':
    main()
