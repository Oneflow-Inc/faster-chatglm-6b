# faster-chatglm-6b

`faster-chatglm-6b`是一个使用OneFlow深度学习框架为后端加速[THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)运行的项目。

## 软件依赖

请参考[Install OneFlow](https://github.com/Oneflow-Inc/oneflow#install-oneflow)安装OneFlow，或者直接使用下面命令安装OneFlow最新安装包：

```shell
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu117
```

## 安装方式

下载或clone本项目后，在项目目录中运行：

```shell
python3 -m pip install -e .
```

## 代码调用

```ipython
>>> import faster_chatglm_6b
>>> from transformers import AutoTokenizer, AutoModel
>>> tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
>>> model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
>>> response, history = model.chat(tokenizer, "你好", history=[])
>>> print(response)
你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。
```

注意：这里和[THUDM/chatglm-6b 官方示例](https://huggingface.co/THUDM/chatglm-6b#%E4%BB%A3%E7%A0%81%E8%B0%83%E7%94%A8)不同的是，在官方示例的最前面增加了一行`import faster_chatglm_6b`，这样就把后台切换成了OneFlow。

下面解释一下，这一行代码主要发生了什么:

1. 设置`OneFlow`的一些环境变量，用于控制 OneFlow 框架的行为。
2. 使用`OneFlow`的`mock_torch`方法把所有的`PyTorch`模块替换成对应的`OneFlow`模块。
3. 利用`transformers`模块的动态模块工具，把找到的原`ChatGLM-6B`中的`ChatGLMForConditionalGeneration`模块替换成经过`OneFlow`优化的`ChatGLMForConditionalGeneration`模块

这一行的详细行为，请参考`faster_chatglm_6b/__init__.py`。

## 更多演示

我们模仿https://github.com/THUDM/ChatGLM-6B 项目也提供了命令行和网页版的演示，请参考`examples`目录下的文件。

```shell
examples/
├── demo.py     # 单轮对话演示
├── cli_demo.py # 命令行多轮对话演示
└── web_demo.py # 网页版对话演示
```

## TODOs

1. OneFlow支持skip_init之后，`faster_chatglm_6b/__init__.py`中移除`new_skip_init`。
2. 移除`cli_demo.py`和`web_demo.py`中对`torch.no_grad`的依赖。


