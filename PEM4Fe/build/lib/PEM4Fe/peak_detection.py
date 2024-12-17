import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import pandas as pd

class PeakDetectionModule:
    """峰值检测模块"""
    
    def __init__(self, config=None, logger=None):
        """
        初始化拟合模块。
        
        参数:
        - config (dict): 配置字典，包含拟合参数。
        - logger (logging.Logger): 可选的日志记录器。如果未提供，将使用根日志记录器。
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        # 提取配置中的参数
        self.Nsc = self.config.get('Nsc', 1000)
        self.N0 = self.config.get('N0', 0)

        self.width_threshold = self.config.get('width_threshold', 5)
        self.plateau_size = self.config.get('plateau_size', 1)
        self.num_density = self.config.get('num_density', -1)

        self.window_size = self.config.get('window_size', 5)
        self.tolerance_ratio = self.config.get('tolerance_ratio', 0.1)

        self.output_dir = self.config.get('output_dir', './output')
        
        # 创建输出目录的子文件夹
        self.plot_output_dir = os.path.join(self.output_dir, "2_peak_detection_plots")
        os.makedirs(self.plot_output_dir, exist_ok=True)
        
        self.logger.info("峰值检测模块初始化完成, output_dir: %s", self.output_dir)

    def detect_peaks(self, N, column_name, Nsc=None, width_threshold=None):
        """
        检测数据中的峰值。
        
        参数:
        - N (array-like): 数据序列。
        - Nsc (float, optional): 峰值的阈值。若未提供，则使用 config 中的 Nsc。
        - width_threshold (int, optional): 峰的最小宽度（数据点数量）。若未提供，则使用 config 中的值。
        
        返回:
        - list: 包含每个峰的信息的列表，每个元素是一个字典，包含峰的范围、宽度和均值。
        """
        Nsc = Nsc or self.Nsc
        width_threshold = width_threshold or self.width_threshold
        plateau_size=self.plateau_size

        self.logger.info("开始检测峰值...")

        # 寻找峰值
        peaks, properties = find_peaks(N, height=self.Nsc, width=width_threshold, plateau_size=plateau_size)
        peaks_info = []

        for i, peak in enumerate(peaks):
            width = int(properties['widths'][i])
            if width >= width_threshold:
                left_ips = int(properties['left_ips'][i])
                right_ips = int(properties['right_ips'][i])
                peak_range = range(left_ips, right_ips + 1)
                N_peak = N[peak_range]
                peak_mean = np.mean(N_peak)
                self.logger.debug(f"峰值 {peak}: left={left_ips}, right={right_ips}, width={width}, mean={peak_mean:.4f}")
                if peak_mean > Nsc:
                    peaks_info.append({
                        'column': column_name,
                        'peak_index': peak,
                        'left': left_ips,
                        'right': right_ips,
                        'width': width,
                        'mean': peak_mean,
                        'indices': peak_range
                    })

        self.logger.info(f"检测到 {len(peaks_info)} 个峰值。")
        return peaks_info

    def plot_peaks(self, ax, t, N, peaks_info, column_name):
        """
        绘制峰值检测结果，并保存图像。
        
        参数:
        - ax (matplotlib.axes._axes.Axes): 子图对象，用于绘制数据。
        - t (array-like): 时间序列。
        - N (array-like): 数据序列。
        - peaks_info (list): 包含峰值信息的列表。
        - column_name (str): 数据列名。
        """
        self.logger.info(f"绘制峰值检测图 for {column_name}...")
        
        ax.plot(t, N, label="Original Data", color="blue")
        
        # 绘制峰值
        for peak_info in peaks_info:
            peak_t = t[peak_info['indices']]
            peak_N = N[peak_info['indices']]
            ax.plot(peak_t, peak_N, color="red")
            peak_left = peak_info['left']
            peak_right = peak_info['right']
            ax.fill_between(t[peak_left:peak_right + 1], N[peak_left:peak_right + 1], self.N0, alpha=0.5)
            ax.hlines(peak_info['mean'], t[peak_left], t[peak_right], color="r", linestyle="--")
        

    def plot_density_analysis(self, ax, ax_twin, N, peaks_info, column_name, num_density=-1 ,fit_result=None):
        """
        绘制右侧子图，显示峰值密集度分析和直方图。
        
        参数:
        - ax (matplotlib.axes._axes.Axes): 子图对象，用于绘制密集区域分析。
        - ax_twin (matplotlib.axes._axes.Axes): 共享 y 轴的子图对象，用于绘制直方图和密度曲线。
        - N (pd.Series): 数据序列。
        - Nsc (float): 峰值的阈值。
        - N0 (float): 数据的基准值。
        - column_name (str): 数据列名。
        - directory_path (str): 保存结果的目录路径。
        - num_density (int): 检测的密集区域峰值数量，默认 -1 表示检测所有峰值。
        - fit_result (dict): 增长拟合结果字典。
        
        返回:
        - list: 包含密集区域峰值信息的列表。
        """
        self.logger.info(f"开始绘制密集区域分析图 for {column_name}...")
        widths = [info['width'] for info in peaks_info]
        means = [info['mean'] for info in peaks_info]
        Nsc, N0 = self.Nsc, self.N0
        
        if means:
            N_cut = np.max(means)
        else:
            N_cut = Nsc
        ymax = N.max()
        
        if widths and means:
            ax.scatter(widths, means, color='blue')

        # 设置 ax 的 y 轴范围
        ax.set_ylim(Nsc, ymax)
        ax.set_title(f"{column_name} Density Analysis")
        
        # 绘制原始直方图
        if fit_result != None:
            t_fit = fit_result["t_fit"]
            end_index = fit_result["end_idx"]
            # end_index = np.argmin(np.abs(t_fit))
            N_smooth = N[:end_index+1]
            # N_smooth = N[N<=2*self.N0]
        else:
            N_smooth = N[(N >= N_cut) & (N <= ymax)]
            
        ax_twin.hist(N_smooth, bins=2000, orientation='horizontal', alpha=0.6, color='gray', label='(N | N >= Nsc) Distribution')

        # 计算直方图数据
        counts, bin_edges = np.histogram(N_smooth[N_smooth>=N_cut], bins=500)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # 对直方图数据进行高斯平滑
        sigma = 0.5  # 控制平滑强度
        smoothed_counts = gaussian_filter1d(counts[:np.nonzero(counts)[0][-1] + 1], sigma=sigma)

        # 绘制高斯平滑曲线
        ax_twin.plot(smoothed_counts, bin_centers, color='red', linestyle='-', linewidth=0.5, label='Smoothed Density', alpha=0.8)
        
        # 峰值检测
        peak_indices, properties = find_peaks(smoothed_counts, height=0, plateau_size=0, width=0)

        density_info = []

        if peak_indices.size != 0:
            up = np.array(properties["plateau_sizes"])
            down = np.array(properties["width_heights"])
            high = np.array(properties["peak_heights"])
            sorted_peak_indices = peak_indices[np.argsort((up + down) * 0.5 * high)][::-1]

            # 确保 num_density 不超过 sorted_peak_indices 的长度
            num_density = min(num_density, len(sorted_peak_indices))

            density_place = bin_centers[sorted_peak_indices[:num_density]]
            ax_twin.scatter(smoothed_counts[sorted_peak_indices[:num_density]], bin_centers[sorted_peak_indices[:num_density]], 
                            color='green', label='Density Peaks', marker='x')

            for i, N_forecast in enumerate(density_place):
                density_info.append({
                    'peak_index': i,
                    'forecast': N_forecast,
                    'smoothed_count': smoothed_counts[sorted_peak_indices[i]],
                })
        
        # 添加图例
        ax_twin.legend(fontsize=8, loc='upper right')
        
        return density_info
    
    def find_nearest_interval(self, ax, t, N, N_forecast, window_size=None, tolerance_ratio=None):
        """
        寻找 N 中接近 N_forecast 的所有区段，返回多个区段的起止点。

        参数：
        - ax (matplotlib.axes._axes.Axes): 绘图的子图，用于标记结果。
        - t (array-like): 时间序列。
        - N (array-like): 实际数据数组。
        - N_forecast (float): 目标值，寻找接近该值的多个区段。
        - window_size (int): 平滑窗口大小，默认为 5。
        - tolerance_ratio (float): 容差范围比例，默认为 0.01。

        返回：
        - list of dict: 每个区段的信息，包括起止点、均值等。
        """
        window_size = self.window_size or window_size
        tolerance_ratio = self.tolerance_ratio or tolerance_ratio

        # Step 0: 归一化
        N = np.array(N)
        min_N = N.min()
        max_N = N.max()
        Nsc = (self.Nsc - N.min()) / (N.max() - N.min())  # 归一化 Nsc
        N_forecast = (N_forecast - N.min()) / (N.max() - N.min())  # 归一化 N_forecast
        N = (N - N.min()) / (N.max() - N.min())

        tolerance = tolerance_ratio * N_forecast  # 容差范围

        # Step 1: 平滑数据
        N_smooth = pd.Series(N).rolling(window=window_size, center=True).mean().fillna(0).to_numpy()

        # Step 2: 过滤小于 Nsc 的值
        valid_indices = N > Nsc
        N_smooth[~valid_indices] = 0

        # Step 3: 找到所有满足条件的索引
        valid_mask = np.abs(N_smooth - N_forecast) <= tolerance
        valid_indices = np.where(valid_mask)[0]  # 获取满足条件的索引
        self.logger.debug(f"目标 {N_forecast*(max_N-min_N)+min_N}，找到 {len(valid_indices)} 个满足条件的索引")

        if len(valid_indices) == 0:
            self.logger.warning(f"未找到满足条件的区段")
            return None, None, None
        
        # Step 4: 合并连续的有效索引为区段
        intervals = []
        if len(valid_indices) > 0:
            start_idx = valid_indices[0]
            for i in range(1, len(valid_indices)):
                # 判断是否连续
                if valid_indices[i] != valid_indices[i - 1] + 1:
                    # 新区段开始
                    end_idx = valid_indices[i - 1]
                    intervals.append((start_idx, end_idx))
                    start_idx = valid_indices[i]
            # 添加最后一个区段
            intervals.append((start_idx, valid_indices[-1]))

        intervals.sort(key=lambda x: x[0])
        merged_intervals = []
        for current in intervals:
            # 如果 merged_intervals 为空，或者当前区间与最后一个区间不重叠，则加入结果
            if not merged_intervals or merged_intervals[-1][1] < current[0]:
                merged_intervals.append(current)
            else:
                # 否则，合并当前区间和最后一个区间
                merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], current[1]))

        # Step 5: 计算每个区段的均值，并返回信息
        interval_info = []
        for start_idx, end_idx in merged_intervals:
            start_idx = start_idx - window_size // 2
            end_idx = end_idx + window_size // 2
            interval_mean = np.mean(N[start_idx:end_idx + 1]*(max_N - min_N) + min_N)
            interval_info.append({
                'start_index': start_idx,
                'end_index': end_idx,
                'mean': interval_mean,
                'std': np.std(N[start_idx:end_idx + 1]*(max_N - min_N) + min_N)
            })
            self.logger.debug(f"区段 [{start_idx}-{end_idx}] 的均值为 {interval_mean:.4f}, 标准差为 {np.std(N[start_idx:end_idx + 1]):.4f}")
            # 可视化标记区段
            ax.fill_between(t[start_idx:end_idx + 1], N[start_idx:end_idx + 1], alpha=0.3, color='green')
        
        return start_idx, end_idx, interval_mean


    def plot_plateaus(self, ax, t, N, density_info, column_name, fit_result=None):
        '''
        绘制平台检测结果，并保存图像。

        参数:
        - ax (matplotlib.axes._axes.Axes): 子图对象，用于绘制数据。
        - t (array-like): 时间序列。
        - N (array-like): 数据序列。
        - peaks_info (list): 包含峰值信息的列表。
        - column_name (str): 数据列名。

        返回:
        - list: 包含平台信息的列表。
        '''
        self.logger.info(f"绘制平台检测图 for {column_name}...")

        plateau_info = []
        for i, info in enumerate(density_info):
            N_forecast = info['forecast']
            start_index, end_index, mean= self.find_nearest_interval(ax, t, N, N_forecast)
            if start_index is None or end_index is None:
                continue
            plateau_t = t[start_index:end_index + 1]
            plateau_N = N[start_index:end_index + 1]
            ax.plot(plateau_t, plateau_N, color="green")
            ax.fill_between(plateau_t, plateau_N, np.max(N), alpha=0.3, color='black')
            plateau_info.append({
                "column": column_name,
                "density_index": i,
                "left": start_index,
                "right": end_index,
                "width": end_index - start_index + 1,
                "mean": N_forecast,
                "indices": range(start_index, end_index + 1)
            })
       

        if fit_result != None:
            t_fit = fit_result['t_fit']
            N_fit = fit_result['N_fit']
            r_squared = fit_result['r_squared']
            ax.plot(t_fit, N_fit, label=f"Fitted Curve ({r_squared})", color="orange",alpha=0.5)
            popt = fit_result['params']
            line_idx = fit_result['line_idx']
            ax.vlines(popt[-1], np.min(N_fit), np.max(N_fit), color='orange', linestyle='dashed', label='Cut Line',alpha=0.5,linewidth=1)

        ax.set_title(f"Peak Detection for {column_name}")
        ax.set_ylim(self.N0*0.9, np.max(N)*1.1)
        ax.set_xlabel("Time(ps)")
        ax.set_ylabel("N")
        ax.legend()
        ax.grid(True)

        return plateau_info


    def plot_analysis_figure(self, t, N, peaks_info, column_name, fit_result=None, num_density=None):
        """
        绘制分析图，有两个子图。

        参数:
        - t (pd.Series): 时间序列。
        - N (pd.Series): 数据序列。
        - peaks_info (list): 峰值信息列表。
        - Nsc (float): Nsc 值。
        - N0 (float): 数据的基准值。
        - column_name (str): 数据列名。
        - directory_path (str): 保存结果的目录路径。
        - num_density (int): 检测的密集区域峰值数量，默认 -1 表示检测所有峰值。
        
        返回:
        - tuple: (density_info, N_star)
        """
        N0 = self.N0 or N.min()
        Nsc = self.Nsc or N.mean()
        num_density = self.num_density or -1

        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 1]}, figsize=(14, 6), dpi=300)
        ax2_twin = ax2.twiny()  # 为右侧子图设置共享轴

        # 绘制左侧子图 ax1
        self.plot_peaks(ax1, t, N, peaks_info, column_name)

        # 绘制右侧子图 ax2
        density_info = self.plot_density_analysis(ax2, ax2_twin, N, peaks_info, column_name, num_density, fit_result)
        plateaus_info = self.plot_plateaus(ax1, t, N, density_info, column_name, fit_result)

        # 设置整体标题
        plt.suptitle(f'{column_name} Analysis')
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plot_file = os.path.join(self.plot_output_dir, f"{column_name}_peaks.png")
        plt.savefig(plot_file)
        plt.close()
        self.logger.info(f"峰值图已保存到 {plot_file}")

        return plateaus_info
