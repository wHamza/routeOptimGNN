import threading

import torch
from torch._utils import ExceptionWrapper
from torch.cuda._utils import _get_device_index
from torch.cuda.amp import autocast


def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, (list, tuple)):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def parallel_apply(modules, inputs, step, num_depots, greedy=False, temperature=1.0, kwargs_tup=None, devices=None):
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)

    devices = [_get_device_index(x, True) if x is not None else None for x in devices]
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()
    autocast_enabled = torch.is_autocast_enabled()

    def _worker(i, module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            tensor = get_a_var(input)
            device = tensor.get_device() if tensor is not None and tensor.is_cuda else None

        try:
            if device is not None:
                with torch.cuda.device(device), autocast(enabled=autocast_enabled):
                    if not isinstance(input, (list, tuple)):
                        input = (input,)
                    output = module(*input, step, num_depots, greedy, temperature, **kwargs)
            else:
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module(*input, step, num_depots, greedy, temperature, **kwargs)

            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [
            threading.Thread(target=_worker, args=(i, module, input, kwargs, device))
            for i, (module, input, kwargs, device) in enumerate(zip(modules, inputs, kwargs_tup, devices))
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs
