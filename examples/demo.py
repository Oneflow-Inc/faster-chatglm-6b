# This code is base on https://github.com/THUDM/ChatGLM-6B project

import faster_chatglm_6b

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = (
    AutoModel.from_pretrained(
        "THUDM/chatglm-6b", trust_remote_code=True, revision=faster_chatglm_6b.revision
    )
    .half()
    .cuda()
)
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
