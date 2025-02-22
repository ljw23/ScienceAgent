# ScienceAgent

## 1. 安装依赖

python3.11

`pip install -r requirements.txt`

注册aliyun账号，获取api_key，参考

https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key?spm=a2c4g.11186623.0.0.74b04823NsBef2

执行

```
export DASHSCOPE_API_KEY="XXXXXX"
```

（把XXX换成你的api_key）

## 2. 运行

Step1. 

```
$git clone https://github.com/ljw23/ScienceAgent.git
$cd ScienceAgent
```

Step2. 将需要总结的pdf放到docs🀄️

Step3. 执行

`python src/scienceagent/examples/simple_pdf_summary.py ` 

执行完成后会在summaries目录生成各个文档的总结，并生成所有汇总的总结summary.txt。

