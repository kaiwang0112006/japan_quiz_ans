# 日语拍题解题

调用 **Qwen2.5-VL-7B-Instruct** 和 **Deepseek-R1** 两个大模型，前者实现图片OCR，获取图片中的日语题目文本，后者
进行答题解题。

# 环境

```shell
export MODELSCOPE_API_KEY="YOU MODELSCOPE API KEY"
pip install openai
```

# 示例

- [modelscope空间](https://modelscope.cn/studios/milowang2009/japan_QA)
- [huggingface space](https://huggingface.co/spaces/milowang2009/japan_quiz_ans)

![avatar](https://github.com/kaiwang0112006/japan_quiz_ans/blob/main/images/demo1_ans.png)

![avatar](https://github.com/kaiwang0112006/japan_quiz_ans/blob/main/images/demo2_ans.png)

    

# 文件说明

- **app.py** ：gradio 应用，通过python app.py 启动, 需修改添加自己在[魔塔社区](https://modelscope.cn/)的api_key
- **others/app_gpu.py** ：本地化加载模型文件，只测试了7B的模型
- **others/app_gpu_stream.py** ：本地化加载模型文件，并流式返回，只测试了7B的模型

