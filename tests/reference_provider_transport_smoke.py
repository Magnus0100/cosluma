"""Current image transport counts submissions and keeps download credentials local."""
from __future__ import annotations

import base64
from email.message import Message
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from PIL import Image
import numpy as np

import reference_provider as provider
import workflow_timing


class Response(io.BytesIO):
    def __init__(self, body: bytes, content_type: str):
        super().__init__(body)
        self.status = 200
        self.headers = Message()
        self.headers['Content-Type'] = content_type


def preparation_checks(root):
    from raw_support import open_rgb,deterministic_srgb_profile
    from raw_color_smoke import linear_srgb_profile
    for index,size in enumerate(((127,191),(191,127))):
        source=root/('native-'+str(index)+'.png')
        pixels=np.random.default_rng(index).integers(0,256,(size[1],size[0],3),np.uint8)
        Image.fromarray(pixels).save(source)
        original=source.read_bytes()
        first,record=provider.prepare_api_input(source,root/('first-'+str(index)),long_edge=64,resize_raster=True)
        second,other=provider.prepare_api_input(source,root/('second-'+str(index)),long_edge=64,resize_raster=True)
        assert source.read_bytes()==original and first!=source
        assert record['derived'] and record['source']['sha256']==provider.sha256(source)
        assert record['source']['size']==list(size) and max(record['size'])==64
        assert record['sha256']==other['sha256'] and first.read_bytes()==second.read_bytes()
        with Image.open(first) as actual:assert actual.info['icc_profile']==deterministic_srgb_profile()
        # The shared default still owns exact Qwen raster bytes, even above its bound.
        unchanged,local=provider.prepare_api_input(source,root/('local-'+str(index)),long_edge=64)
        assert unchanged==source and not local['derived'] and local['sha256']==provider.sha256(source)
    small=root/'small.png';Image.new('RGBA',(31,47),(30,80,120,128)).save(small)
    uploaded,record=provider.prepare_api_input(small,root/'small-api',long_edge=64,resize_raster=True)
    assert uploaded==small and not record['derived'] and not (root/'small-api').exists()
    profile=linear_srgb_profile().tobytes()
    tagged=root/'tagged-rgb.png';exif=Image.Exif();exif[274]=6
    Image.new('RGB',(160,96),(64,128,192)).save(tagged,icc_profile=profile,exif=exif)
    expected=open_rgb(tagged);expected.thumbnail((64,64),Image.Resampling.LANCZOS)
    target,record=provider.prepare_api_input(tagged,root/'tagged-api',long_edge=64,resize_raster=True)
    with Image.open(target) as actual:
        assert actual.tobytes()==expected.tobytes() and actual.size==expected.size
        assert actual.getexif().get(274) is None and actual.info['icc_profile']==deterministic_srgb_profile()
    assert record['source']['size']==[160,96] and record['decode']['oriented_size']==[96,160]
    for mode in ('RGBA','P','LA'):
        opacity=np.zeros((96,160),np.uint8);opacity[:,:80]=128
        if mode=='RGBA':
            pixels=np.full((96,160,4),[255,0,255,0],np.uint8);pixels[:,:80]=[64,128,192,128]
            picture=Image.fromarray(pixels)
        elif mode=='P':
            picture=Image.fromarray((opacity>0).astype(np.uint8),'P')
            picture.putpalette([255,0,255,64,128,192]+[0,0,0]*254)
            picture.info['transparency']=bytes([0,128])
        else:picture=Image.fromarray(np.stack((np.full_like(opacity,77),opacity),axis=-1),'LA')
        exif=Image.Exif();exif[274]=6
        source=root/('oriented-'+mode+'.png')
        picture.save(source,exif=exif,**({'icc_profile':profile} if mode!='LA' else {}))
        before=source.read_bytes();colour=open_rgb(source);expected=colour.getpixel((0,0))
        target,record=provider.prepare_api_input(source,root/(mode+'-api'),long_edge=64,resize_raster=True)
        with Image.open(target) as actual:
            assert actual.mode=='RGBA' and actual.size==(38,64)
            assert actual.getexif().get(274) is None and actual.info['icc_profile']==deterministic_srgb_profile()
            pixels=np.asarray(actual)
        expected_alpha=np.rot90(opacity,k=3).astype(np.float32)/255
        expected_alpha=np.asarray(Image.fromarray(expected_alpha).resize((38,64),Image.Resampling.BILINEAR))
        np.testing.assert_array_equal(pixels[...,3],np.rint(expected_alpha*255).astype(np.uint8))
        # Constant visible colour remains constant through fractional-alpha edges.
        np.testing.assert_array_equal(pixels[expected_alpha>0,:3],np.tile(expected,(np.count_nonzero(expected_alpha),1)))
        assert record['decode']['alpha_preserved'] and record['decode']['orientation_applied']
        assert record['decode']['oriented_size']==[96,160] and source.read_bytes()==before
        if mode!='LA':assert record['decode']['color_management']=='embedded_icc_to_sRGB'
    # Local ordered conditions continue to share their declared native spaces,
    # including binary L masks and an exact source-identity crop.
    scene=root/'condition-scene.png';identity=root/'condition-identity.png'
    mask=root/'condition-mask.png';crop=root/'condition-crop.png'
    Image.new('RGB',(96,2120),(30,70,100)).save(scene)
    parent=Image.fromarray(np.random.default_rng(12).integers(0,256,(2200,112,3),np.uint8));parent.save(identity)
    box=[12,16,90,160];parent.crop(tuple(box)).save(crop)
    allowed=Image.new('L',(96,2120),0);allowed.paste(255,(8,16,80,160));allowed.save(mask)
    rows=[{'role':role,'path':str(path),'sha256':provider.sha256(path),'coordinate_space':space}
        for role,path,space in (('scene_reference',scene,'scene'),('source_identity',identity,'source'),
            ('edit_mask',mask,'scene'),('identity_crop',crop,'source'))]
    rows[-1].update(box=box,parent_sha256=provider.sha256(identity))
    manifest=root/'conditions.json';manifest.write_text(json.dumps(rows),encoding='utf8')
    prepared=provider.prepare_condition_inputs(manifest,scene,root,root/'conditions-api')
    for before,after in zip(rows,prepared):
        assert after['api_input']['path']==before['path'] and after['api_input']['sha256']==before['sha256']
        assert after['input_sha256']==before['sha256'] and not after['api_input']['derived']
    assert prepared[-1]['box']==box and prepared[-1]['parent_sha256']==rows[1]['sha256']
    assert not (root/'conditions-api').exists()


def main():
    with tempfile.TemporaryDirectory(prefix='cosplay-provider-transport-') as directory:
        preparation_checks(Path(directory))
        source = Path(directory)/'source.png'
        Image.new('RGB', (96, 64), 'steelblue').save(source)
        image_bytes = source.read_bytes()
        for mode in ('url', 'b64_json'):
            annotations, calls = {}, []
            data = ({'url': 'https://signed.example/returned.png'} if mode == 'url'
                    else {'b64_json': base64.b64encode(image_bytes).decode('ascii')})

            def fake_open(request, **kwargs):
                calls.append(request)
                if request.get_method() == 'POST':
                    assert request.get_header('Authorization') == 'Bearer test-key'
                    return Response(json.dumps({'data': [data]}).encode(), 'application/json')
                assert request.get_method() == 'GET'
                assert request.get_header('Authorization') is None
                return Response(image_bytes, 'image/png')

            with (patch.object(provider.urllib.request, 'urlopen', side_effect=fake_open),
                  patch.object(workflow_timing, 'annotate',
                               side_effect=lambda **fields: annotations.update(fields))):
                _, actual, response_mode = provider.request_edit(
                    'https://provider.example/v1/images/edits', 'test-key', source,
                    'configured-current-model', 'exact brief', None, 1)
            assert actual == image_bytes and response_mode == mode
            assert annotations['model_submissions'] == 1
            assert annotations['http_submissions'] == len(calls) == (2 if mode == 'url' else 1)
    print(json.dumps({'ok': True, 'presigned_get_counted': True,
                      'download_without_provider_credentials': True, 'real_model_requests': 0}))


if __name__ == '__main__':
    main()
