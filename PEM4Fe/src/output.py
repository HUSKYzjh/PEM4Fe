import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from src.peak_detection import PeakDetectionModule
from src.fitting import FittingModule
from src.data_processing import load_clean_save_data, clean_data
from src.config import load_config, setup_logging


class MainModule:
    """主体模块"""
    def __init__(self, config=None):
        """
        初始化主体模块。
        
        参数:
        - config (dict): 配置字典，包含所需参数。
        """
        self.config = config or {}
        self.Nsc = self.config.get('Nsc', 1000)
        self.N0 = self.config.get('N0', 0)
        self.input_dir = self.config.get('input_dir', './input')
        self.output_dir = self.config.get('output_dir', './output')
        self.overall_path = os.path.join(self.output_dir, "3_overall_analysis_plots")
        os.makedirs(self.overall_path, exist_ok=True)

        # 调用日志设置函数
        log_file = "application.log"
        setup_logging(output_dir=self.output_dir, log_file=log_file, level=logging.DEBUG)
        
        # 为当前类实例创建特定的日志记录器
        self.logger = logging.getLogger(self.__class__.__name__)


    def save_results_to_csv(self, Peak, Plateaus):
        """保存分析结果到 CSV 文件"""
        
        directory_path = self.overall_path

        # 构建 DataFrame 并保存为 CSV 文件
        data = []
        for peaks_info, plateaus_info in zip(Peak, Plateaus):
            # 处理峰值信息
            for peak_info in peaks_info:
                data.append({
                    'Column': peak_info['column'],
                    'Peak Index': peak_info['peak_index'],
                    'Left': peak_info['left'],
                    'Right': peak_info['right'],
                    'Mean': peak_info['mean'],
                    'Width': peak_info['width'],
                    'Choice': False,  
                    'PTM': False       
                })
            
            # 处理平台信息
            for plateau_info in plateaus_info:
                data.append({
                    'Column': plateau_info['column'],
                    'Peak Index': plateau_info['density_index'],
                    'Left': plateau_info['left'],
                    'Right': plateau_info['right'],
                    'Mean': plateau_info['mean'],
                    'Width': plateau_info['width'],
                    'Choice': False,  
                    'PTM': True       
                })
        
        # 创建 DataFrame
        df = pd.DataFrame(data)

        # 保存到 CSV 文件
        output_file = os.path.join(directory_path, 'N_star_info.csv')
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"分析结果已保存到 {output_file}")

    def plot_overall_analysis(self, Peak, Plateaus):
        """
        绘制总体分析图

        参数:
        - peaks_info (list): 峰值信息列表。
        - plateaus_info (list): 平台信息列表。
        - N_star (float): 总体均值。
        """ 
        
        directory_path = self.overall_path
        fig, ax = plt.subplots(figsize=(15, 6))

        values_all = np.array([])
        columns = []
        colors = []            

        # 绘制散点图

        for i in range(len(Peak)):
            peaks_info = Peak[i]
            peek_values = [info['mean'] for info in peaks_info]
            plateaus_info = Plateaus[i]
            plateaus_values = [info['mean'] for info in plateaus_info]
    
            if len(peaks_info) != 0:
                info=peaks_info[0]
            elif len(plateaus_info) != 0:
                info=plateaus_info[0]
            else:
                continue
            column_name = info['column']
            columns.append(column_name)

            values_all = np.concatenate((values_all, peek_values, plateaus_values))
            scatter = ax.scatter(i*np.ones_like(peek_values), peek_values, marker='D',s=80)
            
            colors.append(scatter.get_facecolor()[0])  # 记录颜色
            ax.scatter(i*np.ones_like(plateaus_values), plateaus_values, facecolors='none', edgecolors=colors[i], marker='o', s=80)

            # 风琴图
            values= np.concatenate((np.array(peek_values), np.array(plateaus_values)))
            ax.violinplot(values, positions=[i], showmedians=True, showextrema=True)

        N_star = np.mean(values_all)
        ax.axhline(y=N_star, color='k', linestyle='--', label=f'N* = {N_star:.2f}')

        ax.set_xticks(range(len(Peak)))
        ax.set_xticklabels(columns, rotation=90)

        # 添加图例和标题
        ax.set_xlabel('Index')
        ax.set_ylabel('Mean Value')
        ax.set_title(f'Overall Analysis for {len(Peak)} Columns (N*={N_star}, Nsc={self.Nsc}, N0={self.N0})')
        ax.legend()

        # 保存图像
        plot_file = os.path.join(directory_path, 'Overall_analysis.png')
        plt.savefig(plot_file)
        plt.close()

        self.logger.info(f"总体分析图已保存到 {plot_file}")

    def All_PTM(self,show_output=False):
        
        self.overall_path = os.path.join(self.output_dir, "3_overall_analysis_plots")
        os.makedirs(self.overall_path, exist_ok=True)

        collected_data,nc = load_clean_save_data(self.input_dir,self.output_dir)

        # 初始化 FittingModule 实例
        peak_detection_module = PeakDetectionModule(config=self.config)
        logging.info("PeakDetectionModule 实例已初始化。")

        # 加载数据
        t = collected_data['t']
        Peak = []
        Plateaus = []

        fitting_module = FittingModule(config=self.config)
        logging.info("FittingModule 实例已初始化。")

        for col in collected_data.columns[1:]:  # 从第2列到结束
            N = collected_data[col]  # 确保 'N' 列名正确
            index = str(col)
            logging.info(f"\n开始执行动态指数拟合：{col}...")
            # 去除无效值
            t_clean, N_clean = clean_data(t, N)
            fit_result = fitting_module.fit_dynamic(t_clean, N_clean, index)
            
            if fit_result:
                # print(fit_result)
                params = fit_result['params']
                r_squared = fit_result['r_squared']
                logging.info(f"拟合成功。R² = {r_squared:.4f}")
                logging.info(f"拟合参数: a = {params[0]:.4f}, b = {params[1]:.4f}, c = {params[2]:.4f}, d = {params[3]:.4f}, e = {params[4]:.4f}, f = {params[5]:.4f}")
            else:
                logging.error("拟合失败。")
            logging.info("--------------------------------------------------")

        # 对每一列数据执行峰值检测
        for col in collected_data.columns[1:]:  # 从第2列到结束
            N = collected_data[col]  # 确保 'N' 列名正确
            index = str(col)
            logging.info(f"\n开始执行峰值检测：{col}...")

            # 去除无效值
            t_clean, N_clean = clean_data(t, N)
            
            # 执行峰值检测
            peaks_info = peak_detection_module.detect_peaks(N_clean,index)
            
            # 打印检测结果
            if peaks_info:
                logging.info(f"检测到 {len(peaks_info)} 个峰值。")
            else:
                logging.warning(f"{col} 没有检测到峰值。")
            
            # 绘制并保存峰值检测结果
            plateaus_info = peak_detection_module.plot_analysis_figure(t_clean, N_clean, peaks_info, index)

            Peak.append(peaks_info)
            Plateaus.append(plateaus_info)
            logging.info("--------------------------------------------------")
        
        # 保存结果到 CSV 文件
        self.save_results_to_csv(Peak, Plateaus)
        self.plot_overall_analysis(Peak, Plateaus)
        