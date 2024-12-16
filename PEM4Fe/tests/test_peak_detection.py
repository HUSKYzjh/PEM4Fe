import unittest
import numpy as np
import os
import matplotlib.pyplot as plt
from src.peak_detection import PeakDetectionModule

class TestPeakDetectionModule(unittest.TestCase):

    def setUp(self):
        # 配置和模块初始化
        self.config = {
            "Nsc": 1,
            "width_threshold": 2,
            "output_dir": "./test_output"
        }
        self.module = PeakDetectionModule(config=self.config)
        os.makedirs(self.config['output_dir'], exist_ok=True)

    def test_detect_peaks(self):
        """测试峰值检测"""
        x = np.linspace(0, 2*np.pi, 100)
        N = np.sin(x) * x + np.random.normal(0, 0.01, 100)
        peaks_info = self.module.detect_peaks(N, "test_column")
        self.assertEqual(len(peaks_info), 1)  # 应检测到两个峰
        self.assertGreater(peaks_info[0]['mean'], self.config['Nsc'])

    def test_find_nearest_interval(self):
        """测试平台检测"""
        t = np.linspace(0, 1000, 1001)
        N = np.random.normal(500, 1, 1001)
        index = np.random.randint(0, 1000) 
        N[index:index+100] = 1200
        fig, ax = plt.subplots()  # 创建绘图对象
        _, _, mean = self.module.find_nearest_interval(ax, t, N, N_forecast=1200)
        self.assertAlmostEqual(mean, 1200, delta=50)
        plt.close(fig)  # 关闭图形对象

    def tearDown(self):
        # 清理 Matplotlib 图像缓存
        plt.close('all')
        
        # 清理测试输出文件
        if os.path.exists(self.config['output_dir']):
            for root, dirs, files in os.walk(self.config['output_dir'], topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))


if __name__ == '__main__':
    unittest.main()
