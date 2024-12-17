import unittest
import os
import logging
import tempfile
import json
import yaml
from PEM4Fe.config import load_config, setup_logging

class TestConfig(unittest.TestCase):
    """测试 config.py 中的函数"""

    def setUp(self):
        """初始化临时配置文件和环境"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.json_file = os.path.join(self.temp_dir.name, "test_config.json")
        self.yaml_file = os.path.join(self.temp_dir.name, "test_config.yaml")

        # 创建测试 JSON 文件
        with open(self.json_file, 'w') as f:
            json.dump({"param1": "value1", "param2": 42}, f)

        # 创建测试 YAML 文件
        with open(self.yaml_file, 'w') as f:
            yaml.dump({"param1": "value1", "param2": 42}, f)

    def tearDown(self):
        """清理临时文件"""
        self.temp_dir.cleanup()

    def test_load_config_json(self):
        """测试加载 JSON 配置文件"""
        config = load_config(self.json_file)
        self.assertEqual(config['param1'], "value1")
        self.assertEqual(config['param2'], 42)

    def test_load_config_yaml(self):
        """测试加载 YAML 配置文件"""
        config = load_config(self.yaml_file)
        self.assertEqual(config['param1'], "value1")
        self.assertEqual(config['param2'], 42)

    def test_load_config_file_not_found(self):
        """测试文件不存在时的异常"""
        with self.assertRaises(FileNotFoundError):
            load_config("non_existent_file.json")

    def test_load_config_invalid_format(self):
        """测试不支持的文件格式"""
        invalid_file = os.path.join(self.temp_dir.name, "invalid.txt")
        with open(invalid_file, 'w') as f:
            f.write("invalid content")
        with self.assertRaises(ValueError):
            load_config(invalid_file)

    def test_setup_logging(self):
        """测试 setup_logging 是否正确配置日志"""
        log_dir = os.path.join(self.temp_dir.name, "logs")
        log_file = "test.log"
        setup_logging(output_dir=log_dir, log_file=log_file)

        logger = logging.getLogger()
        logger.info("测试日志记录")

        log_path = os.path.join(log_dir, log_file)
        self.assertTrue(os.path.exists(log_path))  # 检查日志文件是否生成
        with open(log_path, 'r') as f:
            content = f.read()
            self.assertIn("测试日志记录", content)
        
        # 关闭日志处理器，确保文件句柄释放
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    unittest.main()
