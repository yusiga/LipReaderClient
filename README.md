# LipreadingClient
## 1、安装依赖

```shell
conda create -n lipreading python=3.8 -y
conda activate lipreading
```

```shell
conda install pip openssl certifi -y
```

```shell
pip install PyQt-Fluent-Widgets -i https://pypi.org/simple/
```

```shell
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

```shell
pip install face_alignment icecream matplotlib numpy opencv_python pandas
```

## 2、运行程序

```shell
python main.py
```

## 3、项目架构
* 前端：app/view
* 后端：app/lipreader
* 前后端联调：app/lipreader/api/api.py
* 模型 checkpoint 存放路径：app/lipreader/checkpoint/
* 模型 checkpoint 路径配置：app/lipreader/config/config.py