# 1. 基础镜像 (使用官方 Python 环境，或者带 PyTorch 的镜像)
FROM python:3.9-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 复制项目文件到容器里
# 把当前目录(deployment)的上级目录(项目根目录)下的所有东西，复制到容器的 /app.py 下
COPY . /app

# 4. 安装依赖
# 使用清华源加速
RUN pip install --no-cache-dir -r deployment/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 暴露端口
EXPOSE 8000

# 6. 启动命令
# 注意路径：因为 WORKDIR 是 /app.py，所以运行 deployment/app.py
CMD ["python", "deployment/server.py"]