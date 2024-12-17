import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize

class FittingModule:
    """指数拟合模块"""
    
    def __init__(self, logger=None, config=None):
        """
        初始化拟合模块。
        
        参数:
        - logger (logging.Logger): 可选的日志记录器。如果未提供，将使用根日志记录器。
        - config (dict): 配置字典，包含拟合参数。
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        self.initial_fit_points = self.config.get('initial_fit_points', 10)
        self.min_r_squared = self.config.get('min_r_squared', 0.95)
        self.max_iterations = self.config.get('max_iterations', 10)
        self.output_dir = self.config.get('output_dir', './output')
        
        logging.info("拟合模块初始化完成, output_dir: %s", self.output_dir)
        
        # 创建输出目录的子文件夹
        self.init_output_dir = os.path.join(self.output_dir, "0_initial")
        os.makedirs(self.init_output_dir, exist_ok=True)
        self.fit_output_dir = os.path.join(self.output_dir, "1_exponential_fit")
        os.makedirs(self.fit_output_dir, exist_ok=True)
    
    def exponential_func(self, t, a, b, c, d):
        """
        定义拟合函数：a * exp(b * (t - c)) + d * t + e
        
        参数:
        - t (array-like): 自变量数据。
        - a, b, c, d, e (float): 拟合参数。
        
        返回:
        - array-like: 函数值。
        """
        return a * np.exp(b * (t - c)**(3/2)) + d
    
    def line_exponential_func(self, t, a, b ,c, d, e, f):
        '''
        定义分段拟合函数：a * exp(b * (t - c)) + d * t
        
        参数:
        - t (array-like): 自变量数据。
        - a, b, c, d, e (float): 拟合参数。
        
        返回:
        - array-like: 函数值。
        '''
        # 找到nb.abs(t-c)最小值的位置
        index = np.argmin(np.abs(t-f))
        N1 = e*np.ones_like(t[:index])
        N2 = a * np.exp(b * (t[index:] - c)**(3/2)) + d
        return np.concatenate((N1,N2))
    

    def find_line(self, new_popt, t_fit, N_fit, new_start_idx, new_end_idx):
        """
        优化拟合参数c和e，使得拟合的R²尽可能大。
        
        参数:
        - new_popt (list): 当前拟合的参数，包含 a, b, c, d, e。
        - new_r_squared (float): 当前拟合的 R² 值。
        - new_start_idx (int): 拟合数据的起始索引。
        - new_end_idx (int): 拟合数据的结束索引。
        
        返回:
        - final_popt (list): 优化后的拟合参数，包含 a, b, c, d, e。
        - final_r_squared (float): 优化后的 R² 值。
        """
        
        # 提取当前拟合参数
        a, b, c, d= new_popt
        e = 0.9999*d
        f = 1.0001*c
        # 优化目标函数，最大化 R²
        def objective(params):
            # 固定 a, b, c, d, 只优化 e 和 f
            e, f = params
            
            # 使用 line_exponential_func 进行拟合
            fitted_N = self.line_exponential_func(t_fit, a, b, c, d, e, f)
            index = np.argmin(np.abs(t_fit-f))
            gap = np.abs(a * np.exp(b * (t_fit[index] - c)**(3/2)) + d - e)

            # 计算拟合残差
            residuals = fitted_N - N_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((N_fit - np.mean(N_fit))**2)
            
            # 计算 R² 值
            r_squared = 1 - (ss_res / ss_tot)
            
            self.logger.info(f"优化中：参数={params}, R²={r_squared:.4f}")
            
            # 目标是最大化 R²，最小化负的 R²
            return -r_squared*(f/t_fit[-1])*(1-gap/(np.max(N_fit)-np.min(N_fit))) if r_squared >= self.min_r_squared else 0

        # 使用 minimize 优化 c 和 e
        result = minimize(objective, x0=[e, f], bounds=[(0, np.inf), (np.min(t_fit)-0.001, np.max(t_fit)+0.001)], method='L-BFGS-B')
        
        # 获取优化后的参数
        optimized_e, optimized_f = result.x
        final_popt = [a, b, c, d, optimized_e, optimized_f]

        # 计算最终的 R²
        fitted_N = self.line_exponential_func(t_fit, *final_popt)

        # 计算拟合残差
        residuals = fitted_N - N_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((N_fit - np.mean(N_fit))**2)
        final_r_squared = 1 - (ss_res / ss_tot)

        self.logger.debug(f"最后拟合结果：参数={final_popt}, R²={final_r_squared:.4f}")
        return final_popt, final_r_squared, np.argmin(np.abs(t_fit-optimized_f))+new_start_idx, new_end_idx


    def fit_dynamic(self, t, N, index, initial_fit_points=None, min_r_squared=None, max_iterations=None):
        """
        动态进行指数拟合，逐步扩大拟合范围直到满足置信度要求或达到最大迭代次数。
        
        参数:
        - t (pd.Series or np.ndarray): 时间序列。
        - N (pd.Series or np.ndarray): 数据序列。
        - initial_fit_points (int, optional): 初始拟合的数据点数。
        - min_r_squared (float, optional): 最小的 R² 值要求。
        - max_iterations (int, optional): 最大拟合尝试次数，防止无限循环。
        
        返回:
        - dict or None: 包含拟合参数和 R² 值的字典，若拟合失败则返回 None。
        """
        # 使用传入的参数或配置中的参数
        initial_fit_points = initial_fit_points or self.initial_fit_points
        min_r_squared = min_r_squared or self.min_r_squared
        max_iterations = max_iterations or self.max_iterations
        
        self.logger.info("开始动态指数拟合...\n - 初始拟合点数: %d  - 最小 R²: %.2f  - 最大迭代次数: %d", initial_fit_points, min_r_squared, max_iterations)
        
        # 确保输入为 numpy 数组
        t = np.array(t)
        N = np.array(N)
        self.plot_source(t, N, index)

        total_points = len(t)
        if total_points < initial_fit_points:
            self.logger.error(f"数据点不足，无法进行拟合。需要至少 {initial_fit_points} 个点，当前有 {total_points} 个点。")
            return None
        
        # 初始拟合范围：最后 initial_fit_points 个点
        start_idx = total_points - initial_fit_points
        end_idx = total_points
        new_start_idx = start_idx
        new_end_idx = end_idx

        iteration = 0
        new_popt = None
        new_r_squared = -np.inf
        new_t_fit=t

        
        while iteration < max_iterations and start_idx >= 0:
            self.logger.debug(f"拟合迭代 {iteration + 1}: 使用数据点 {start_idx} 到 {end_idx}（共 {end_idx - start_idx} 个点）")
            t_fit = t[start_idx:end_idx]
            N_fit = N[start_idx:end_idx]
            
            try:
                # 初始参数猜测
                initial_params = [1.0, 0.1, t_fit[0], np.min(N_fit)]
                # 参数边界
                bounds = ([0, 0, -0.001, 0], [np.inf, np.inf, np.min(t_fit), np.max(N_fit)])
                
                popt, pcov = curve_fit(
                    self.exponential_func, 
                    t_fit, 
                    N_fit, 
                    p0=initial_params, 
                    bounds=bounds, 
                    maxfev=10000
                )
                
                # 计算拟合结果
                residuals = N_fit - self.exponential_func(t_fit, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((N_fit - np.mean(N_fit))**2)
                r_squared = 1 - (ss_res / ss_tot)
                
                self.logger.debug(f"拟合结果：参数={popt}, R²={r_squared:.4f}")
                
                # 检查 R² 是否达到要求
                if r_squared < min_r_squared and r_squared > 0:
                    self.logger.info(f"拟合成功，R²={new_r_squared:.4f}，使用数据点 {new_start_idx} 到 {new_end_idx}")
                    final_popt, final_r_squared, final_start_idx, final_end_idx= self.find_line(new_popt, t_fit, N_fit, new_start_idx, new_end_idx)
                    # 绘制并保存拟合结果
                    self.plot_fit(t_fit, N_fit, new_t_fit, index, final_popt, final_r_squared, final_start_idx, final_end_idx)
                    fitted_N = self.line_exponential_func(new_t_fit, *final_popt)
                    
                    return {
                        "index": index,
                        'params': final_popt,
                        'r_squared': final_r_squared,
                        'line_idx': final_start_idx,
                        'start_idx': new_start_idx,
                        'end_idx': final_end_idx,
                        "t_fit": new_t_fit,
                        "N_fit": fitted_N
                    }
                else:
                    # 保存最佳拟合结果
                    new_popt = popt
                    new_r_squared = r_squared
                    new_start_idx = start_idx
                    new_end_idx = end_idx
                    new_t_fit = t_fit
                
                # 扩大拟合范围
                start_idx = max(0, start_idx - initial_fit_points)
                iteration += 1
                
            except RuntimeError as e:
                self.logger.warning(f"拟合失败在数据点 {start_idx} 到 {end_idx}: {e}")
                # 扩大拟合范围继续尝试
                start_idx = max(0, start_idx - initial_fit_points)
                iteration += 1
            except Exception as e:
                self.logger.error(f"拟合过程中发生未预料的错误: {e}")
                return None
        
        # 如果未达到要求，返回最佳结果并绘图
        if new_r_squared >= min_r_squared:
            self.logger.info(f"拟合成功，R²={r_squared:.4f}，使用数据点 {start_idx} 到 {end_idx}")
            final_popt, final_r_squared, final_start_idx, final_end_idx= self.find_line(new_popt, t_fit, N_fit, new_start_idx, new_end_idx)
            # 绘制并保存拟合结果
            self.plot_fit(t_fit, N_fit, new_t_fit, index, final_popt, final_r_squared, final_start_idx, final_end_idx)
            fitted_N = self.line_exponential_func(new_t_fit, *final_popt)
            return {
                "index": index,
                'params': final_popt,
                'r_squared': final_r_squared,
                'line_idx': new_start_idx,
                'start_idx': final_start_idx,
                'end_idx': final_end_idx,
                "t_fit": new_t_fit,
                "N_fit": fitted_N
            }
        elif iteration >= max_iterations:
            self.logger.error("超过限制次数，所有拟合尝试均失败。")
            return None
    
    def plot_fit(self, t_fit, N_fit, new_t_fit, index, popt, r_squared, start_idx, end_idx):
        """
        绘制拟合曲线并保存图像。
        
        参数:
        - t_fit (np.ndarray): 拟合的时间序列。
        - N_fit (np.ndarray): 拟合的数据序列。
        - popt (list): 拟合参数。
        - r_squared (float): 拟合的 R² 值。
        - start_idx (int): 拟合的起始索引。
        - end_idx (int): 拟合的结束索引。
        
        返回:
        - None
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(t_fit, N_fit, label='Data', color='blue')
        fitted_N = self.line_exponential_func(new_t_fit, *popt)
        plt.plot(new_t_fit, fitted_N, label=f'Fit (R²={r_squared:.4f})', color='red')
        plt.vlines(popt[-1], np.min(N_fit), np.max(N_fit), colors='green', linestyles='dashed', label='Increase Parse')
        plt.title(f'N{index} Exponential Fit: Data points {start_idx} to {end_idx}')
        plt.xlabel('Time(ps)')
        plt.ylabel('N')
        plt.legend()
        plt.grid(True)
        
        # 修复路径
        plot_filename = os.path.join("1_exponential_fit", f"{index}-{start_idx}_{end_idx}.png")
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"拟合图已保存到 {plot_path}")

    def plot_source(self, t, N, index):
        """
        绘制原始数据并保存图像。
        
        参数:
        - t (np.ndarray): 时间序列。
        - N (np.ndarray): 数据序列。
        - index (int): 数据索引。
        
        返回:
        - None
        """
        plt.figure(figsize=(10, 3))
        plt.plot(t, N, "-o", label='Data', color='blue')
        plt.title(f'N{index} Source Data')
        plt.xlabel('Time(ps)')
        plt.ylabel('N')
        plt.legend()
        plt.grid(True)
        
        # 修复路径
        plot_filename = os.path.join("0_initial", f"{index}-source.png")
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"原始数据图已保存到 {plot_path}")
