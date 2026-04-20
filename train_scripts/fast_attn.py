import importlib
from contextlib import nullcontext

import torch

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    SDPBackend = None
    sdpa_kernel = None

_MPS_REPLACE_SDPA = None
_MPS_SDPA_PATCHED = False
_MPS_SDPA_ERROR = None
_FLASH_DTYPES = {torch.float16, torch.bfloat16}


def _device_type(device):
    return device.type if isinstance(device, torch.device) else str(device)


def configure_fast_attn(device, attention_dtype):
    global _MPS_REPLACE_SDPA, _MPS_SDPA_PATCHED, _MPS_SDPA_ERROR

    device_type = _device_type(device)
    if device_type != "mps":
        return "sdpa_flash" if device_type == "cuda" else "sdpa"
    if attention_dtype not in _FLASH_DTYPES:
        return f"sdpa (MPS flash needs fp16/bf16; got {attention_dtype})"
    if _MPS_SDPA_ERROR is not None:
        return f"sdpa ({_MPS_SDPA_ERROR})"
    if _MPS_REPLACE_SDPA is None:
        try:
            _MPS_REPLACE_SDPA = importlib.import_module("mps_flash_attn").replace_sdpa
        except ModuleNotFoundError:
            return "sdpa (install mps-flash-attn manually, often with --no-build-isolation)"
        except Exception as exc:
            _MPS_SDPA_ERROR = f"mps_flash_attn import failed: {exc}"
            return f"sdpa ({_MPS_SDPA_ERROR})"
    if not _MPS_SDPA_PATCHED:
        try:
            _MPS_REPLACE_SDPA()
        except Exception as exc:
            _MPS_SDPA_ERROR = f"mps_flash_attn patch failed: {exc}"
            return f"sdpa ({_MPS_SDPA_ERROR})"
        _MPS_SDPA_PATCHED = True
    return "mps_flash_attn"


def fast_attn_context(device):
    device_type = _device_type(device)
    if device_type != "cuda" or sdpa_kernel is None or SDPBackend is None:
        return nullcontext()
    return sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
        set_priority=True,
    )
