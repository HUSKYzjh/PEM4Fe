import unittest
from unittest.mock import MagicMock, patch
import os
import numpy as np
import pandas as pd
from src.output import MainModule
from src.peak_detection import PeakDetectionModule
from src.fitting import FittingModule
import yaml

class TestMainModule(unittest.TestCase):

    def setUp(self):
        """初始化测试环境"""
        self.config = {
            "input_dir": "./test_input",
            "output_dir": "./test_output",
            "Nsc": 1000,
            "N0": 0,
        }
        self.module = MainModule(config=self.config)
        os.makedirs(self.config['output_dir'], exist_ok=True)

        # 创建测试用配置文件
        self.test_config_path = "./test_config.yaml"
        with open(self.test_config_path, "w") as f:
            yaml.dump(self.config, f)

        # 模拟数据
        self.test_data = pd.DataFrame({
            't': np.linspace(0, 10, 11),
            'N_1': np.linspace(0, 1000, 11),
            'N_2': np.linspace(100, 1100, 11)
        })

    def tearDown(self):
        """清理测试输出"""
        import shutil
        import logging

        # 移除日志处理器
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # 删除测试目录和配置文件
        shutil.rmtree(self.config['output_dir'], ignore_errors=True)
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)



    @patch('src.peak_detection.PeakDetectionModule')
    @patch('src.fitting.FittingModule')
    def test_All_PTM(self, MockFittingModule, MockPeakDetectionModule):
        """测试 All_PTM 方法的完整流程"""

        # Mock 模块实例
        mock_peak = MockPeakDetectionModule.return_value
        mock_peak.detect_peaks.return_value = [{'mean': 100, 'column': 'N_1'}]
        mock_peak.plot_analysis_figure.return_value = [{'mean': 120, 'left': 1, 'right': 10}]

        mock_fit = MockFittingModule.return_value
        mock_fit.fit_dynamic.return_value = {'r_squared': 0.99, 'params': [1, 2, 3, 4, 5, 6]}

        # 加载测试数据
        with patch('src.output.load_clean_save_data', return_value=(self.test_data, 1000)):
            self.module.All_PTM(show_output=False)

        # 检查是否调用了峰值检测
        mock_peak.detect_peaks.assert_called()
        mock_peak.plot_analysis_figure.assert_called()

        # 检查是否调用了拟合模块
        mock_fit.fit_dynamic.assert_called()

        # 检查 CSV 文件是否生成
        csv_path = os.path.join(self.config['output_dir'], "3_overall_analysis_plots", "N_star_info.csv")
        self.assertTrue(os.path.exists(csv_path))

        # 检查总体分析图是否生成
        plot_path = os.path.join(self.config['output_dir'], "3_overall_analysis_plots", "Overall_analysis.png")
        self.assertTrue(os.path.exists(plot_path))

if __name__ == "__main__":
    unittest.main()
