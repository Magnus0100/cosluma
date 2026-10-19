"""Interrupted and concurrently observed writes never publish partial JSON."""
import concurrent.futures
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import single_source_workflow as workflow


def main():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'manifest.json'
        workflow.write_json(path, {'old': True})
        real_replace = os.replace
        calls = []

        def transient(source, destination):
            calls.append(source)
            if len(calls) < 3:
                raise PermissionError('transient Windows reader lock')
            return real_replace(source, destination)

        with patch.object(workflow.os, 'replace', side_effect=transient):
            workflow.write_json(path, {'published': True})
        assert json.loads(path.read_text()) == {'published': True}
        assert len(calls) == 3 and not list(Path(folder).glob('*.tmp'))
        with patch.object(workflow.os, 'replace', side_effect=PermissionError('persistent lock')):
            try:
                workflow.write_json(path, {'must_not_publish': True})
            except PermissionError:
                pass
            else:
                raise AssertionError('Persistent locks must remain errors')
        assert json.loads(path.read_text()) == {'published': True}
        assert not list(Path(folder).glob('*.tmp'))
        def write(index):
            workflow.write_json(path, {'sequence': index, 'payload': 'x'*10000})
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(write, range(30)))
        result = json.loads(path.read_text())
        assert result['sequence'] in range(30) and len(result['payload']) == 10000
        assert not list(Path(folder).glob('*.tmp'))
    print('OK: transient lock retry, persistent failure preserves prior snapshot, concurrent complete publication')


if __name__ == '__main__':
    main()
