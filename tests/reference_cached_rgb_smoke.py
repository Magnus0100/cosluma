"""Cached RGB equivalence permits only ICC dates, and workers use caller roots."""
from pathlib import Path
import json
import tempfile
from PIL import Image, ImageCms
from raw_support import RGB_DECODE_VERSION
from reference_local_correspondence import _cached_rgb, file_sha


def main():
    with tempfile.TemporaryDirectory(prefix='cached-rgb-') as directory:
        root = Path(directory)
        record = {'sha256': 'a'*64, 'oriented_size': [8, 12]}
        base = root/'outputs/cache/local-analysis'/record['sha256'][:16]
        profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes())
        paths = []
        for index in (1, 2):
            item = base/str(index); item.mkdir(parents=True)
            (item/'analysis.json').write_text(json.dumps({'source': record,
                'analysis_identity': {'decode_version': RGB_DECODE_VERSION,
                    'input_proxy': 'aspect_preserving_2048_lanczos_srgb'}}), encoding='utf8')
            profile[24:36] = bytes([index])*12
            path = item/'source-srgb.png'; paths.append(path)
            Image.new('RGB', (8, 12), '#123456').save(path, icc_profile=bytes(profile))
        assert file_sha(paths[0]) != file_sha(paths[1])
        selected = _cached_rgb(root, record)
        assert selected[0] == paths[0] and selected[2] == file_sha(paths[0])
        def rejected():
            try:
                _cached_rgb(root, record)
            except ValueError:
                return
            raise AssertionError('Conflicting RGB observation was accepted')
        Image.new('RGB', (8, 12), '#123457').save(paths[1], icc_profile=bytes(profile))
        rejected()
        profile[-1] ^= 1
        Image.new('RGB', (8, 12), '#123456').save(paths[1], icc_profile=bytes(profile))
        rejected()
        profile[-1] ^= 1
        Image.new('RGB', (9, 12), '#123456').save(paths[1], icc_profile=bytes(profile))
        rejected()
    # These are real subprocess entry points: their caches must belong to the
    # caller's project rather than the source/frozen installation directory.
    import ast
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    for name in ('evaluate_reference_colour', 'reference_colour_blend', 'reference_scene_balance'):
        source = (scripts/(name+'.py')).read_text(encoding='utf8')
        ast.parse(source)
        assert "parser.add_argument('--root', type=Path, default=Path.cwd())" in source
        assert 'root = args.root.resolve()' in source
    print(json.dumps({'ok': True, 'equal_rgb_icc_date_only': True,
        'changed_rgb_icc_or_canvas_rejected': True, 'selected_file_bytes_bound': True,
        'worker_root_explicit': True}))


if __name__ == '__main__':
    main()
