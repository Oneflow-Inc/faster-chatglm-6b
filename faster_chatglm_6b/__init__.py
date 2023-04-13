import os

os.environ["ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT"] = "1"
os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"

revision = "096f3de6b4959ce38bef7bb05f3129c931a3084e"

import oneflow as flow
from oneflow import nn

flow.mock_torch.enable(lazy=True)


def new_skip_init(module_cls, *args, **kwargs):
    return module_cls(*args, **kwargs)


if not hasattr(nn.utils, "skip_init"):
    nn.utils.skip_init = new_skip_init


from transformers import dynamic_module_utils

orig_get_class_from_dynamic_module = dynamic_module_utils.get_class_from_dynamic_module


def hook_get_class_from_dynamic_module(*args, **kwargs):
    _, _, cls_name = args
    if cls_name == "ChatGLMForConditionalGeneration":
        from .modeling_chatglm import ChatGLMForConditionalGeneration

        return ChatGLMForConditionalGeneration
    else:
        return orig_get_class_from_dynamic_module(*args, **kwargs)


dynamic_module_utils.get_class_from_dynamic_module = hook_get_class_from_dynamic_module
