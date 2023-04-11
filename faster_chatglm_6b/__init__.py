import os
os.environ["ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT"] = "1"
os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"

revision = "aa51e62ddc9c9f334858b0af44cf59b05c70148a"

import oneflow as flow
from oneflow import nn

flow.mock_torch.enable(lazy=True)

def new_skip_init(module_cls, *args, **kwargs):
    return module_cls(*args, **kwargs)

nn.utils.skip_init = new_skip_init

from transformers import dynamic_module_utils
orig_get_class_from_dynamic_module = dynamic_module_utils.get_class_from_dynamic_module

def hook_get_class_from_dynamic_module(*args, **kwargs):
    _, _, cls_name = args
    if cls_name == "ChatGLMModel":
        from .oneflow_modules import ChatGLMModel
        return ChatGLMModel
    else:
        return orig_get_class_from_dynamic_module(*args, **kwargs)

dynamic_module_utils.get_class_from_dynamic_module = hook_get_class_from_dynamic_module
