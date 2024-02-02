import torch


def auto_get_device():
    if torch.cuda.is_available():
        # get the gpu's name
        gpu_name = torch.cuda.get_device_name(0)
        # if device contains 1030, use cpu
        if '1030' in gpu_name:
            return torch.device('cpu')
        else:
            return torch.device('cuda')
    else:
        return torch.device('cpu')
