"""One current model configuration for every production entry point."""
from current_model_registry import load_registry

CURRENT = 'current'


def options_for_choice(choice, root):
    if choice != CURRENT:
        raise ValueError('Only the current model configuration is supported')
    path = load_registry(root)['registry_path']
    return {'algorithm_profile': 'dinov3_vitl16', 'model_registry': str(path),
            'material_finish_profile': 'owner_adaptive_multiscale_skin_v2'}


def choice_for_options(options):
    if options.get('algorithm_profile', 'dinov3_vitl16') != 'dinov3_vitl16':
        raise ValueError('Saved model configuration is unsupported; create a current run')
    return CURRENT
