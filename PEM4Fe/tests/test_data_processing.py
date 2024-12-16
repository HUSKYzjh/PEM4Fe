import unittest
import os
import pandas as pd
import numpy as np
from src.data_processing import collect_data_from_files, clean_data, save_to_csv

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        """初始化测试环境"""
        self.test_dir = "./tests/temp_data"
        os.makedirs(self.test_dir, exist_ok=True)

        # 创建测试数据文件
        with open(os.path.join(self.test_dir, "pem-Nc-123.txt"), "w") as f:
            f.write("0 10\n1 20\n2 30\n")

    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, file))
            os.rmdir(self.test_dir)

    def test_collect_data_from_files(self):
        """测试数据文件收集功能"""
        df = collect_data_from_files(self.test_dir)
        self.assertFalse(df.empty)
        self.assertEqual(df.shape[1], 2)  # 确保两列数据

    def test_clean_data(self):
        """测试数据清洗功能"""
        t = pd.Series([0, 1, 2, 3])
        N = pd.Series([10, 20, None, np.inf])
        t_clean, N_clean = clean_data(t, N)
        self.assertEqual(len(N_clean), 2)  # 清理后的数据点数量

    def test_save_to_csv(self):
        """测试数据保存功能"""
        df = pd.DataFrame({'t': [0, 1, 2], 'N': [10, 20, 30]})
        output_path = os.path.join(self.test_dir, "test_output.csv")
        save_to_csv(df, output_path)
        self.assertTrue(os.path.exists(output_path))

if __name__ == "__main__":
    unittest.main()
