import torch
from functools import reduce
import numpy as np

def set_uniform_mask(model, mask_size):
    """Uniformly chooses `mask_size` parameters from the model and generates a boolean mask for every component.
    Uniformly sample `mask_size` parameters from the entire model parameters, and in fine-tuning process only them
    will be fine-tuned.
    Args:
        mask_size (int): number of non-masked parameters.
    """

    total_params = 0
    masks, params_per_component = dict(), dict()
    for name, param in model.named_parameters():
        masks[name] = torch.zeros(param.size(), dtype=torch.bool)
        component_params = reduce(lambda x, y: x * y, param.shape)
        params_per_component[name] = component_params
        if "bias" in name:
            total_params += component_params

    tunable_params_per_component = {k: round((v * mask_size) / total_params) for k, v in
                                    params_per_component.items() if "bias" in k}
    tunable_params = reduce(lambda x, y: x + y, tunable_params_per_component.values())
    print(f'Non-Masked params amount: {tunable_params}. '
                f'Total params: {total_params}')
    count = 0
    extra = tunable_params-mask_size
    total = 0
    for name, param in model.named_parameters():
        if "bias" in name:
            component_mask_size = tunable_params_per_component[name]
            if count < extra:
                component_mask_size -= 1
                count += 1
            if component_mask_size > 0:
                total += component_mask_size
                component_params = params_per_component[name]
                indices = np.random.randint(0, component_params, component_mask_size)
                mask = masks[name]
                for index in indices:
                    if len(param.shape) == 1:
                        mask[index] = True
                    else:
                        mask[int(index / param.shape[1]), index % param.shape[1]] = True
    print(f"after truncated, there is {total} tuned params")
    return masks