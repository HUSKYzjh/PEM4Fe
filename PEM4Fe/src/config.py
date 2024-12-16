import os
import json
import yaml
import logging

def load_config(config_file):
    """
    加载配置文件（支持 JSON 和 YAML）。

    参数:
    - config_file (str): 配置文件的路径。

    返回:
    - dict: 包含配置参数的字典。
    """
    _, ext = os.path.splitext(config_file)
    try:
        with open(config_file, 'r') as file:
            if ext == '.json':
                return json.load(file)
            elif ext in ('.yaml', '.yml'):
                return yaml.safe_load(file)
            else:
                raise ValueError("不支持的配置文件格式，请使用 JSON 或 YAML。")
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件 {config_file} 不存在，请检查路径。")
    except (json.JSONDecodeError, yaml.YAMLError):
        raise ValueError(f"配置文件 {config_file} 格式错误，无法解析。")

def setup_logging(output_dir="./logs", log_file="app.log", level=logging.INFO):
    """
    设置日志记录。
    
    参数:
    - output_dir (str): 日志文件保存的目录，默认是 "./logs"。
    - log_file (str): 日志文件名，默认保存为 "app.log"。
    - level (int): 日志级别，默认是 INFO。
    
    返回:
    - None
    """
    # 确保日志目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 完整日志文件路径
    log_file_path = os.path.join(output_dir, log_file)

    # 清理旧的日志配置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 配置日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 配置日志记录
    logging.basicConfig(
        level=level,  # 设置日志级别
        format=log_format,  # 设置日志格式
        handlers=[
            logging.FileHandler(log_file_path, encoding="utf-8"),  # 将日志写入文件
            logging.StreamHandler()             # 同时输出到控制台
        ]
    )
    # 将 matplotlib 的日志级别设置为 WARNING
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # 测试日志
    logging.info(f"日志文件保存路径: {log_file_path}")
