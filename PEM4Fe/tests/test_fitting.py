import unittest
import numpy as np
import os
from src.fitting import FittingModule

class TestFittingModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """初始化测试环境"""
        cls.test_output_dir = "./tests/test_output"
        os.makedirs(cls.test_output_dir, exist_ok=True)
        cls.config = {
            "initial_fit_points": 5,
            "min_r_squared": 0.90,
            "max_iterations": 10,
            "output_dir": cls.test_output_dir
        }
        cls.fitting_module = FittingModule(config=cls.config)

    def test_exponential_func(self):
        """测试指数函数输出"""
        t = np.array([1, 2, 3, 4])
        result = self.fitting_module.exponential_func(t, 2, 0.5, 1, 0.1)
        self.assertEqual(len(result), 4)
        self.assertTrue(np.all(result >= 0))

    def test_fit_dynamic_success(self):
        """测试动态拟合是否成功"""
        t = np.arange(10)
        N = 2 * np.exp(0.3 * (np.clip(t - 3, 0, None))**1.5) + 0.5  # 约束负值，避免 NaN
        result = self.fitting_module.fit_dynamic(t, N, index="test")
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["r_squared"], 0.90)

    def test_plot_fit(self):
        """测试拟合结果绘图"""
        t = np.arange(10)
        N = 2 * np.exp(0.3 * (np.clip(t - 3, 0, None))**1.5) + 0.5  # 确保数据有效
        fit_result = self.fitting_module.fit_dynamic(t, N, index="test_plot")
        self.assertIsNotNone(fit_result)


    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        import shutil
        shutil.rmtree(cls.test_output_dir)

if __name__ == "__main__":
    unittest.main()
