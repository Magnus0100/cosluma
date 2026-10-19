"""Installation pins and stat changes fail before image submission."""
import hashlib,json,os,tempfile
from pathlib import Path
from unittest.mock import patch
import beautygrpo_nunchaku as backend


def main():
    with tempfile.TemporaryDirectory(prefix='beauty-quantized-installation-') as folder:
        root=Path(folder);directory=root/'nunchaku';directory.mkdir()
        model=directory/backend.MODEL_FILE;model.write_bytes(b'pin fixture, not a model')
        digest=hashlib.sha256(model.read_bytes()).hexdigest()
        receipt=model.with_suffix('.safetensors.download.json')
        receipt.write_text(json.dumps({'status':'verified','sha256':digest,'expected_bytes':model.stat().st_size}))
        config={'config_path':str(root/'installation.json')}
        with patch.object(backend,'MODEL_SHA',digest),patch.object(backend,'MODEL_BYTES',model.stat().st_size):
            try:backend.installed_model(config)
            except ValueError:pass
            else:raise AssertionError('Unaudited weight admitted')
            path,binding=backend.installed_model(config,installation=True)
            assert path==model and binding['sha256']==digest
            with patch('asset_identity.sha256_file',side_effect=AssertionError('Daily model rehash')):
                assert backend.installed_model(config)[1]==binding
            stat=model.stat();os.utime(model,ns=(stat.st_atime_ns,stat.st_mtime_ns+1_000_000))
            try:backend.installed_model(config)
            except ValueError:pass
            else:raise AssertionError('Changed model stat admitted')
            model.write_bytes(b'wrong model bytes!!!!!!')
            try:backend.installed_model(config,installation=True)
            except ValueError:pass
            else:raise AssertionError('Changed SHA admitted')
    print(json.dumps({'ok':True,'installation_pin_verified':True,'daily_no_model_rehash':True,
        'missing_or_changed_audit_rejected':True,'paid_requests':0,'CUDA_equality_gate':False}))


if __name__=='__main__':main()
