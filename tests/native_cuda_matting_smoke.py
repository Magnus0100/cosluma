"""Real CUDA physical F/B constraints; no CPU/CUDA equality gate."""
import json
import numpy as np
from native_cuda_fields import cuda_field_scope
from native_cuda_matting import estimate_foreground_background
from source_foreground import estimate_source_background


def main():
    height, width = 1024, 1024
    alpha = np.broadcast_to(np.clip((.75-np.linspace(0, 1, width, dtype=np.float32))/.5, 0, 1),
                            (height, width)).copy()
    foreground = np.array([.8, .2, .1], np.float32)
    background = np.array([.08, .25, .5], np.float32)
    source = (alpha[..., None]*foreground+(1-alpha)[..., None]*background)*2.5
    original_alpha=alpha.copy()
    trimap=np.full(alpha.shape,128,np.uint8);trimap[alpha==0]=0;trimap[alpha==1]=255
    with cuda_field_scope():
        old, fp, support, solves, scale, diagnostics=estimate_source_background(source,alpha,trimap)
    assert solves==1 and scale>1
    assert np.array_equal(alpha,original_alpha)
    assert np.isfinite(old).all() and np.isfinite(fp).all() and np.isfinite(support).all()
    assert (fp>=0).all() and (support>=0).all() and (support<=1).all()
    assert np.array_equal(fp[alpha==0],np.zeros_like(fp[alpha==0]))
    assert np.array_equal(fp[alpha==1],source[alpha==1])
    error=float(np.max(np.abs(fp+(1-alpha)[...,None]*old-source)))
    assert error < 5e-7
    assert len(diagnostics['fit_error'])==int((trimap==128).sum())
    assert float(diagnostics['fit_error'].mean()) < .02
    assert np.array_equal(diagnostics['overlap_weight'],np.ones_like(diagnostics['overlap_weight']))
    # A selected invalid CUDA input must fail, never silently rerun on CPU.
    bad=source.copy();bad[0,0,0]=np.nan
    try: estimate_foreground_background(bad,alpha)
    except ValueError: pass
    else: raise AssertionError('Nonfinite F/B input accepted')
    print(json.dumps({'ok':True,'alpha_preserved':True,'finite_HDR_reconstruction':True,
        'transparent_Fp_zero':True,'reconstruction_max_error':error,'CUDA_equality_gate':False}))


if __name__=='__main__':main()
