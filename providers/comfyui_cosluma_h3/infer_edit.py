"""One MiniMax-H3-Image edit as three short-lived ComfyUI processes.

Text/reference conditioning, DiT sampling and VAE decoding each run in their own
process, load only the weights that stage needs, and end with the process so the
weights are reclaimed by the OS. The parent never imports torch, so the 15.7 GB
text encoder and the diffusion model are never resident at the same time and no
retired model is moved back to CPU by ComfyUI's unload path.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import traceback

# The official H3-Image template samples the checkpoint's distilled PDD grid with these two.
SAMPLER = 'euler'
SCHEDULER = 'simple'
CANVAS_MULTIPLE = 32
STAGES = ('condition','sample','decode')


class StageError(RuntimeError):
    def __init__(self,stage,detail):
        super().__init__(stage+' 阶段失败：'+str(detail))
        self.stage = stage


def checkpoint_header(path):
    # Storage types and model config read from the file header, without loading any tensor.
    with open(path,'rb') as stream:
        header = json.loads(stream.read(struct.unpack('<Q',stream.read(8))[0]))
    counts = {}
    for name,tensor in header.items():
        if name != '__metadata__':
            dtype = tensor['dtype']
            counts[dtype] = counts.get(dtype,0)+1
    metadata = header.get('__metadata__') or {}
    transformer = (json.loads(metadata['config']) if 'config' in metadata else {}).get('transformer') or {}
    return {'file_bytes':Path(path).stat().st_size,'tensor_storage_types':counts,
        'pdd_steps':transformer.get('pdd_steps'),'pdd_sigmas':transformer.get('pdd_sigmas')}


def canvas_size(width, height, resolution):
    # image_1's aspect at resolution**2 pixels of area, on the model's 32 pixel grid.
    scale = math.sqrt((resolution*resolution)/(width*height))
    return (max(CANVAS_MULTIPLE,round(width*scale/CANVAS_MULTIPLE)*CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE,round(height*scale/CANVAS_MULTIPLE)*CANVAS_MULTIPLE))


def _read_rgb(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        return np.array(image.convert('RGB'),dtype=np.float32)/255


def _pre_resize_canvas(pixels, width, height):
    """The picture to edit, resampled onto the canvas that will be generated.

    The official image-edit graph pre-resizes image_1 and then takes the conditioning
    width and height from that image, so its reference sits on the lattice being
    generated instead of a coarse anchor for a large canvas. LANCZOS only resamples
    the photograph that is already there; it adds no detail.
    """
    import numpy as np
    from PIL import Image
    if (int(pixels.shape[1]),int(pixels.shape[0])) == (width,height):
        return pixels
    image = Image.fromarray(np.uint8(np.clip(pixels,0,1)*255+.5))
    return np.array(image.resize((width,height),Image.Resampling.LANCZOS),dtype=np.float32)/255


WORKER_MEMORY_FIELDS = ('peak_working_set_bytes','rss_bytes','private_commit_bytes',
    'pagefile_commit_bytes','peak_pagefile_commit_bytes')


def worker_memory():
    """Current and peak OS counters of this process, in bytes.

    Windows exposes PROCESS_MEMORY_COUNTERS_EX through psutil, so each counter keeps
    the name of the counter it comes from: peak_wset is the peak and rss the current
    working set, pagefile is the commit charge and peak_pagefile its peak, and private
    is PrivateUsage. The working set is the resident pages of this process, private,
    shared and file-mapped ones included, so part of it can still be reclaimed; the
    commit counters are the OS commit charge, which is neither RSS nor bytes that sit
    in the pagefile. The peaks belong to one process each, so stage peaks are not
    additive and these fields are measurements, not a memory budget.
    """
    import psutil
    info = psutil.Process().memory_info()
    return {'peak_working_set_bytes':getattr(info,'peak_wset',None),'rss_bytes':int(info.rss),
        'private_commit_bytes':getattr(info,'private',None),
        'pagefile_commit_bytes':getattr(info,'pagefile',None),
        'peak_pagefile_commit_bytes':getattr(info,'peak_pagefile',None)}


def write_json(path,record):
    Path(path).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')


def finish(code):
    """End this process while the calling stage frame still owns its models.

    Returning instead would drop the clip/model/vae locals and let refcounting run
    ComfyUI's unpatch, which copies the packed weights back to CPU after the peak
    was recorded. The OS reclaims everything when the process ends.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _comfy_flags(comfy_root, *, dynamic=True):
    sys.path.insert(0,comfy_root)
    # One layout, selected before loading; checkpoint quantization is explicit.
    sys.argv = [sys.argv[0],'--enable-dynamic-vram' if dynamic else '--disable-dynamic-vram','--disable-cuda-malloc',
        '--disable-pinned-memory','--cuda-device','0']
    import comfy.options
    comfy.options.enable_args_parsing()
    if not dynamic:
        print('VAE decode uses resident CUDA weights',flush=True)
        return
    import comfy_aimdo.control
    comfy_aimdo.control.init(nvml_pressure=True)
    import comfy.model_management as memory
    import comfy.memory_management
    import comfy.model_patcher
    if not comfy_aimdo.control.init_devices([(memory.get_torch_device().index,0)]):
        raise RuntimeError('H3 DynamicVRAM 初始化失败')
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True
    print('ComfyUI DynamicVRAM initialized; model storage policy follows the actual disk',flush=True)


def stage_condition(args, settings):
    """Prompt and reference pictures -> conditioning tensors and the starting latent."""
    _comfy_flags(args.comfy_root)
    import torch
    import comfy.model_management as memory
    import comfy.sd
    import comfy.utils
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning

    directory = Path(args.directory)
    first = _read_rgb(settings['images'][0])
    width,height = canvas_size(first.shape[1],first.shape[0],settings['resolution'])
    images,references = {},[]
    for index,path in enumerate(settings['images'],1):
        pixels = first if index == 1 else _read_rgb(path)
        original = [int(pixels.shape[1]),int(pixels.shape[0])]
        record = {'slot':'image_'+str(index),'original_pixels':original,
            'role':'picture to edit' if index == 1 else 'reference picture'}
        if index == 1:
            # The official graph pre-resizes image_1 and then takes the conditioning width and height from
            # that image, so its reference sits on the lattice being generated instead of acting as a coarse
            # anchor for a large canvas. Only image_1 is pre-scaled; other references keep the native policy.
            resampled = (original[0],original[1]) != (width,height)
            pixels = _pre_resize_canvas(pixels,width,height)
            record['pre_resize'] = {'method':'lanczos' if resampled else 'none: already the canvas',
                'from_pixels':original,'to_canvas':[width,height],'adds_detail':False}
        record['pixels'] = [int(pixels.shape[1]),int(pixels.shape[0])]
        images['image_'+str(index)] = torch.from_numpy(pixels)[None]
        references.append(record)

    def plain(value):
        # Conditioning metadata as plain tensors/lists/dicts so the sampling process can reload it.
        if isinstance(value,torch.Tensor):
            return value.detach().cpu()
        if isinstance(value,(str,int,float,bool)) or value is None:
            return value
        if isinstance(value,list):
            return [plain(item) for item in value]
        if isinstance(value,tuple):
            return tuple(plain(item) for item in value)
        if isinstance(value,dict):
            return {key:plain(item) for key,item in value.items()}
        raise TypeError('conditioning metadata holds '+type(value).__name__+', which cannot cross to the sampling process')

    started = time.perf_counter()
    vae_state,metadata = comfy.utils.load_torch_file(settings['vae'],return_metadata=True)
    vae = comfy.sd.VAE(sd=vae_state,metadata=metadata)
    del vae_state
    vae_dtype = str(vae.vae_dtype)
    clip = comfy.sd.load_clip(ckpt_paths=[settings['clip']],
        clip_type=comfy.sd.CLIPType.MINIMAX,
        model_options={'dtype':torch.bfloat16,'load_device':memory.get_torch_device(),
            'offload_device':torch.device('cpu')})
    with torch.no_grad():
        print('Qwen3-VL-32B CUDA with CPU offload: encode references and prompt; selected checkpoint '
            +settings['clip'],flush=True)
        positive,latent = MiniMaxH3ImageConditioning.execute(clip=clip,prompt=settings['prompt'],
            width=width,height=height,vae=vae,images=images).result
        # The official image-edit path: an empty latent, every picture as a reference, no mask and no
        # partial grid anywhere. The selection never reaches the model; only the writeback uses it.
        torch.save({'entries':[{'cond':cond.detach().cpu(),'meta':plain(meta)} for cond,meta in positive],
            'latent':{key:(value.detach().cpu() if torch.is_tensor(value) else value)
                for key,value in latent.items()}},directory/'condition.pt')
    # The records separate the photograph's own pixels, the canvas the model generates and the pixels
    # the conditioner actually received, so no entry describes a size that was not passed to the model.
    write_json(directory/'stage-condition.json',{'seconds':time.perf_counter()-started,
        'requested_canvas':{'area_side':settings['resolution'],'width':width,'height':height},
        'image_1_pre_resize':references[0]['pre_resize'],'width':width,'height':height,
        'references':references,'vae_compute_dtype':vae_dtype,
        'text_encoder_compute_dtype':'bfloat16',
        'storage_policy':{'dynamic_vram':True,'text_encoder_fast_disk':clip.patcher.fast_disk,
            'vae_fast_disk':vae.patcher.fast_disk},'worker_memory':worker_memory()})
    finish(0)


def stage_sample(args, settings):
    """Conditioning tensors + starting latent -> the sampled still latent."""
    _comfy_flags(args.comfy_root)
    import torch
    import comfy.samplers
    import comfy.sd
    import nodes

    directory = Path(args.directory)
    started = time.perf_counter()
    saved = torch.load(directory/'condition.pt',weights_only=True)
    positive = [[entry['cond'],entry['meta']] for entry in saved['entries']]
    # An empty latent with no noise_mask: the whole grid is executed from sigma 1.
    latent = dict(saved['latent'])
    model = comfy.sd.load_diffusion_model(settings['model'])
    transformer_dtype = str(model.model_dtype())
    # The sampler's own schedule, built by the same KSampler the native call builds, so the recorded
    # values are the ones actually used - the whole distilled grid, with no step skipped.
    schedule = comfy.samplers.KSampler(model,steps=settings['steps'],device=model.load_device,
        sampler=SAMPLER,scheduler=SCHEDULER,denoise=1.,model_options=model.model_options).sigmas
    with torch.no_grad():
        print('H3-Image transformer loaded; '+str(settings['steps'])+' interval distilled PDD grid: '
            +str(len(schedule)-1)+' denoise step(s) from sigma '+str(round(float(schedule[0]),6)),flush=True)
        negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
        sampled = nodes.common_ksampler(model,settings['seed'],settings['steps'],1.,
            SAMPLER,SCHEDULER,positive,negative,latent)[0]
        torch.save({'samples':sampled['samples'].detach().cpu()},directory/'latent.pt')
    write_json(directory/'stage-sample.json',{'seconds':time.perf_counter()-started,
        'transformer_model_dtype':transformer_dtype,
        'full_grid_sigmas':[float(value) for value in schedule],
        'starting_sigma':float(schedule[0]),'effective_denoise_steps':max(len(schedule)-1,0),
        'sampler_mask':'noise_mask' in latent,
        'storage_policy':{'dynamic_vram':True,'transformer_fast_disk':model.fast_disk},
        'worker_memory':worker_memory()})
    finish(0)


def stage_decode(args, settings):
    """Sampled latent -> the raw model PNG at its native size."""
    # This small VAE fits on CUDA. Keep its weights resident across tiled rows.
    _comfy_flags(args.comfy_root,dynamic=False)
    import numpy as np
    from PIL import Image
    import torch
    import comfy.sd
    import comfy.utils
    import nodes

    directory = Path(args.directory)
    started = time.perf_counter()
    vae_state,metadata = comfy.utils.load_torch_file(settings['vae'],return_metadata=True)
    vae = comfy.sd.VAE(sd=vae_state,metadata=metadata)
    del vae_state
    vae_dtype = str(vae.vae_dtype)
    latent = torch.load(directory/'latent.pt',weights_only=True)
    with torch.no_grad():
        print('VAE loaded; decode the still latent',flush=True)
        # The published node, because VAE.decode returns BTHWC (or BHWC) and flattens the still frame.
        decoded = nodes.VAEDecode().decode(vae,latent)[0]
        pixels = np.uint8(np.clip(decoded[0].cpu().numpy(),0,1)*255+.5)
        Image.fromarray(pixels).save(directory/'model-output.png')
    write_json(directory/'stage-decode.json',{'seconds':time.perf_counter()-started,
        'output_width':int(pixels.shape[1]),'output_height':int(pixels.shape[0]),
        'vae_compute_dtype':vae_dtype,
        'storage_policy':{'dynamic_vram':False,'vae_weight_layout':'resident CUDA',
            'vae_load_device':str(vae.device)},
        'worker_memory':worker_memory()})
    finish(0)


STAGE_FUNCTIONS = {'condition':stage_condition,'sample':stage_sample,'decode':stage_decode}


def run_stage(args):
    directory = Path(args.directory)
    settings = json.loads((directory/'request.json').read_text(encoding='utf-8'))
    try:
        STAGE_FUNCTIONS[args.stage](args,settings)
    except Exception as error:
        write_json(directory/('stage-'+args.stage+'-error.json'),
            {'error':type(error).__name__+': '+str(error),'worker_memory':worker_memory()})
        traceback.print_exc()
        finish(1)
    # Every stage ends from inside its own frame; reaching this line means it returned instead.
    finish(1)


def orchestrate(args):
    """Run the three stages one after another; this process never imports torch."""
    directory = Path(args.directory)
    stages = {}
    started = time.perf_counter()
    try:
        settings = json.loads((directory/'request.json').read_text(encoding='utf-8'))
        storage = {key:checkpoint_header(settings[key]) for key in ('model','clip','vae')}
        declared,grid = storage['model']['pdd_steps'],storage['model']['pdd_sigmas']
        # The heads are distilled on one sigma grid; the step count is fixed by the checkpoint, not tuned.
        if not grid:
            raise RuntimeError('MiniMax-H3-Image checkpoint has no transformer.pdd_sigmas grid: '+settings['model'])
        if settings['steps'] != len(grid)-1 or (declared is not None and declared != settings['steps']):
            raise RuntimeError('MiniMax-H3-Image samples its distilled PDD grid: checkpoint declares '
                f'{declared} steps / {len(grid)} grid points, requested {settings["steps"]}')
        environment = {**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
        flags = subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        for stage in STAGES:
            command = [sys.executable,'-B','-X','utf8','-X','faulthandler','-u',str(Path(__file__).resolve()),
                '--comfy-root',args.comfy_root,'--directory',str(directory),'--stage',stage]
            log_path = directory/('stage-'+stage+'.log')
            stage_started = time.perf_counter()
            with log_path.open('w',encoding='utf-8') as log:
                result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=environment,
                    creationflags=flags,check=False)
            record_path = directory/('stage-'+stage+'.json')
            record = json.loads(record_path.read_text(encoding='utf-8')) if record_path.is_file() else {}
            elapsed = round(time.perf_counter()-stage_started,1)
            stages[stage] = {'seconds':elapsed,'wall_seconds':elapsed,'returncode':result.returncode,
                **record.pop('worker_memory',{}),**record}
            print('stage '+stage+': '+str(stages[stage]['seconds'])+' s, peak working set '
                +str(stages[stage].get('peak_working_set_bytes'))+' bytes, peak commit '
                +str(stages[stage].get('peak_pagefile_commit_bytes'))+' bytes',flush=True)
            if result.returncode:
                error_path = directory/('stage-'+stage+'-error.json')
                detail = json.loads(error_path.read_text(encoding='utf-8'))['error'] if error_path.is_file() else (
                    log_path.read_text(encoding='utf-8')[-1800:])
                raise StageError(stage,'退出码 '+str(result.returncode)+'；'+detail)
        condition,sample,decode = stages['condition'],stages['sample'],stages['decode']
        record = {'engine':'MiniMax-H3-Image',
            'implementation':'installed ComfyUI MiniMaxH3ImageConditioning and KSampler in three short-lived processes',
            'model':settings['model'],'clip':settings['clip'],'vae':settings['vae'],
            'checkpoint_storage':storage,'pdd_steps':declared,'pdd_sigmas':grid,
            'steps':settings['steps'],'executed_denoise_steps':sample['effective_denoise_steps'],
            'sampling_schedule':{'full_grid_sigmas':sample['full_grid_sigmas'],
                'starting_sigma':sample['starting_sigma'],
                'effective_denoise_steps':sample['effective_denoise_steps'],
                'sampler_mask':sample['sampler_mask']},
            'sampler':SAMPLER,'scheduler':SCHEDULER,'cfg':1.,'denoise':1.,
            'resolution_area_side':settings['resolution'],'width':condition['width'],'height':condition['height'],
            'requested_canvas':condition['requested_canvas'],'image_1_pre_resize':condition['image_1_pre_resize'],
            'sampling':'empty latent: official full-context image edit','mask':None,
            'seed':settings['seed'],'prompt':settings['prompt'],'references':condition['references'],
            'text_encoder_device':'cuda:0','text_encoder_offload_device':'cpu',
            'text_encoder_compute_dtype':condition['text_encoder_compute_dtype'],
            'transformer_model_dtype':sample['transformer_model_dtype'],
            'vae_compute_dtype':decode['vae_compute_dtype'],'device':'cuda:0',
            'quantized_operations':'ComfyUI operations selected by checkpoint quantization metadata',
            'prompt_enhancer':'not used; the prompt is encoded as given',
            'model_input_bit_depth':8,'model_output_bit_depth':8,
            'output_width':decode['output_width'],'output_height':decode['output_height'],
            'stages':stages,
            'worker_memory':{'stages':{name:{key:stage[key] for key in WORKER_MEMORY_FIELDS if key in stage}
                    for name,stage in stages.items()},
                'orchestrator':worker_memory(),
                'note':'each stage entry holds the OS counters of that process; one short-lived process per stage, so '
                    'the stages never run at the same time and their peaks are not summed; working set is that '
                    'process resident pages (private, shared, file-mapped), commit counters are OS commit charge, '
                    'so these are per-process measurements and not a memory budget'},
            'timings':{'conditioning_seconds':condition['seconds'],'sampling_seconds':sample['seconds'],
                'decode_seconds':decode['seconds'],'total_seconds':time.perf_counter()-started},
            'memory':'ComfyUI DynamicVRAM file-backed text encoder and transformer with per-model storage detection; '
                'VAE decode uses resident CUDA weights; '
                'each stage process ends with its own weights; no retired model is moved back to CPU'}
        write_json(directory/'model.json',record)
        return 0
    except Exception as error:
        write_json(directory/'error.json',{'error':str(error),'stage':getattr(error,'stage',None),'stages':stages,
            'worker_memory':{'orchestrator':worker_memory(),
                'note':'per-stage OS counters are in stages; the stages never run at the same time'}})
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root',required=True)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--stage',choices=STAGES)
    args = parser.parse_args()
    if args.stage is None:
        # The parent owns no Torch state, so it can return normally; it still leaves without
        # interpreter shutdown so no stage-related finalizer can run in this process.
        code = orchestrate(args)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    # A stage never returns: it ends from inside its own frame, while its models are still owned.
    run_stage(args)


if __name__=='__main__':
    main()
