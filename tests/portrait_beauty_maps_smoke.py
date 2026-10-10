"""Native blend replay retains precision, permissions and independent layers."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from portrait_beauty_worker import decode_srgb, encode_srgb, replay_maps

def main():
    rng = np.random.default_rng(23)
    encoded = rng.uniform(.08,.85,(35,43,3)).astype(np.float32)
    source = decode_srgb(encoded)
    support = np.zeros(source.shape[:2],np.float32)
    support[5:29,6:37] = 1
    support[14:18,16:24] = 0  # Deliberately protected eye/lip-like island.
    blend = np.full(source.shape,.61,np.float32)
    repair = np.zeros(source.shape,np.float32)
    repair[20:23,25:28] = np.asarray([.018,-.011,.004],np.float32)
    maps = {'skin_support':support,'blend_map':blend,'repair_delta_srgb':repair}
    after = replay_maps(source,maps)
    assert np.array_equal(after[support == 0],source[support == 0])
    assert np.array_equal(replay_maps(source,maps,skin_strength=0,blemish_strength=0),source)
    neutral = {**maps,'blend_map':np.full(source.shape,.5,np.float32),'repair_delta_srgb':np.zeros_like(source)}
    assert np.array_equal(replay_maps(source,neutral),source)
    repaired = replay_maps(source,maps,skin_strength=0)
    assert np.array_equal(repaired[np.all(repair == 0,axis=-1)],source[np.all(repair == 0,axis=-1)])
    expected_encoded = np.clip(encoded+repair,0,1)
    expected = decode_srgb((1-2*.61)*expected_encoded**2+2*.61*expected_encoded)
    assert np.allclose(after[support > 0],expected[support > 0],atol=2e-7)
    # The learned map is a curve acting on the full float carrier, not a
    # quantized target paste: sub-8-bit differences remain observable.
    a = np.full((2,2,3),.40001,np.float32)
    b = np.full((2,2,3),.40003,np.float32)
    local = {'skin_support':np.ones((2,2),np.float32),'blend_map':np.full((2,2,3),.64,np.float32),
             'repair_delta_srgb':np.zeros((2,2,3),np.float32)}
    assert not np.array_equal(replay_maps(decode_srgb(a),local),replay_maps(decode_srgb(b),local))
    assert np.array_equal(np.uint8(a*255+.5),np.uint8(b*255+.5))
    try:
        replay_maps(source,maps,skin_strength=2)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid beauty dose accepted')
    print('portrait_beauty_maps_smoke: passed')

if __name__ == '__main__':
    main()
