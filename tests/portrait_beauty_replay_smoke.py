"""Sparse face replay preserves other people, untouched pixels and frozen identity."""
from pathlib import Path
import hashlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from portrait_beauty import apply_result


def main():
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        source=np.full((30,40,3),.3,np.float32)
        faces=[]
        # Context boxes overlap, but the two actual face supports do not.
        # Writing an entire ROI would erase the first person's treatment.
        for index,(box,active) in enumerate((( [0,0,25,25], (slice(3,9),slice(3,9)) ),
                                            ( [5,0,35,25], (slice(12,18),slice(16,22)) ))):
            left,top,right,bottom=box
            support=np.zeros((bottom-top,right-left),np.float32)
            support[active]=1
            path=directory/f'{index}.npz'
            np.savez_compressed(path,skin_support=support,
                blend_map=np.full((*support.shape,3),.7,np.float32),
                repair_delta_srgb=np.zeros((*support.shape,3),np.float32),
                pre_gated_repair_delta=np.asarray(True))
            faces.append({'native_box':box,'field_npz':str(path),
                'field_npz_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        result={'schema':'portrait_beauty_result.v1','working_domain':'display_linear_sRGB',
                'native_size':[40,30],'faces':faces}
        after=apply_result(source,result)
        assert np.any(after[3:9,3:9]!=source[3:9,3:9])
        assert np.any(after[12:18,21:27]!=source[12:18,21:27])
        untouched=np.ones(source.shape[:2],bool)
        untouched[3:9,3:9]=False; untouched[12:18,21:27]=False
        assert np.array_equal(after[untouched],source[untouched])
        assert np.array_equal(apply_result(source,result,skin_strength=0,blemish_strength=0),source)
        path=Path(faces[0]['field_npz'])
        with path.open('ab') as stream: stream.write(b'changed')
        try: apply_result(source,result)
        except ValueError: pass
        else: raise AssertionError('Changed frozen face map was accepted')
    print('portrait_beauty_replay_smoke: passed')


if __name__=='__main__':
    main()
