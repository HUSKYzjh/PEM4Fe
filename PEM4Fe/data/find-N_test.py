import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pwlf
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import warnings
from joblib import Parallel, delayed
import tempfile
from scipy.ndimage import gaussian_filter1d
import seaborn as sb

# -------------------------- Step 1: 设置临时目录 -------------------------- #

# 设置临时目录路径
temp_dir = '/home/jhzhai/Nucleation/N/temp'

# 如果临时目录不存在，则创建
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# 设置 Python 的临时目录
tempfile.tempdir = temp_dir

# -------------------------- Step 2: 数据收集与预处理 -------------------------- #

def collect_data_from_files(directory):
    """
    从指定目录中收集所有符合命名规则的文本文件，并合并为一个 DataFrame。
    
    参数:
    - directory (str): 数据文件所在目录路径。
    
    返回:
    - pd.DataFrame: 合并后的数据框，包含时间列 't' 和多个 'N_' 列。
    """
    combined_df = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.startswith('pem-Nc-') and filename.endswith('.txt'):
            suffix = filename.split('-')[-1].split('.')[0]
            filepath = os.path.join(directory, filename)
            try:
                df = pd.read_csv(filepath, sep='\s+', names=['t', f'N_{suffix}'])
                if combined_df.empty:
                    combined_df = df
                else:
                    combined_df = pd.merge(combined_df, df, on='t', how='outer')
            except Exception as e:
                print(f"读取文件 {filepath} 失败: {e}")
                continue
    return combined_df

def save_to_csv(dataframe, output_file):
    """
    将 DataFrame 保存为 CSV 文件。
    
    参数:
    - dataframe (pd.DataFrame): 要保存的数据框。
    - output_file (str): 输出 CSV 文件路径。
    """
    dataframe.to_csv(output_file, index=False)

def clean_data(t, N):
    """
    清洗数据，去除 NaN 和无穷值。
    
    参数:
    - t (pd.Series): 时间序列。
    - N (pd.Series): 数据序列。
    
    返回:
    - pd.Series: 清洗后的时间序列。
    - pd.Series: 清洗后的数据序列。
    """
    clean_mask = np.isfinite(N)
    t_clean = t[clean_mask]
    N_clean = N[clean_mask]
    return t_clean.reset_index(drop=True), N_clean.reset_index(drop=True)

def load_and_clean_data(directory_path, output_csv_path='pem_N.csv'):
    """
    加载、清洗并保存数据。
    
    参数:
    - directory_path (str): 数据文件所在目录路径。
    - output_csv_path (str): 保存合并后数据的 CSV 文件名。
    
    返回:
    - pd.DataFrame: 清洗后的数据框。
    - float: 从目录名中提取的 nc 值。
    """
    # 提取目录名中的 nc 值
    base_dir = os.path.basename(directory_path)
    try:
        nc_str = [part for part in base_dir.split('/') if 'nc' in part][0]
        nc = float(nc_str.replace('nc', ''))
    except (IndexError, ValueError):
        print("无法从目录名中提取 'nc' 值，请确保目录名包含 'nc' 后跟数值，例如 'nc1000'")
        nc = None
    
    collected_data = collect_data_from_files(directory_path)
    save_to_csv(collected_data, os.path.join(directory_path, output_csv_path))
    print(f"数据已保存到 {os.path.join(directory_path, output_csv_path)}")
    df = pd.read_csv(os.path.join(directory_path, output_csv_path))
    return df, nc

# -------------------------- Step 3: 分段拟合（指数拟合） -------------------------- #

def exponential_func(t, a, b, c, d ,e):
    # return a * (t - b) + c * (t - b) ** 5 + d
    return a*np.exp(b*(t-c))+d*t+e
    # return a*np.tan(b*(t-c))+d

def exponential_fit_with_confidence(t, N, popt=None):
    """
    对给定数据进行指数拟合，返回拟合参数和置信度（R²）。

    参数:
    - t (pd.Series): 时间序列。
    - N (pd.Series): 数据序列。

    返回:
    - tuple: (拟合参数 popt, 置信度 r_squared)
    """
    try:
        # Define bounds to ensure the first and third parameters are positive
        # bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
        bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.max(t), np.inf, np.max(N)])
        if popt is None:
            initial_params = [1,0.1,min(t),0,min(N)]
        else:
            initial_params = popt
        popt, pcov = curve_fit(exponential_func, t, N,p0=initial_params, bounds=bounds, maxfev=10000)
        # popt, pcov = curve_fit(exponential_func, t, N, maxfev=10000)
        residuals = N - exponential_func(t, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((N - np.mean(N))**2)
        r_squared = 1 - (ss_res / ss_tot)
        return popt, r_squared
    except RuntimeError as e:
        print(f"指数拟合失败: {e}")
        return None, None

def exponential_fit_dynamic(t, N):
    """
    尝试对后 5 个数据点进行指数拟合，如果置信度过低则放弃。
    动态扩大指数拟合范围，直到置信度下降为初始置信度的 80%。

    参数:
    - t (pd.Series): 时间序列。
    - N (pd.Series): 数据序列。

    返回:
    - dict or None: 包含指数拟合结果的字典，或者在失败时返回 None。
    """
    min_points = 10
    if len(t) < min_points:
        print("数据点不足，无法进行指数拟合。")
        return None

    indices = list(range(len(t) - min_points, len(t)))
    t_fit = t.iloc[indices]
    N_fit = N.iloc[indices]

    # 初始拟合
    popt, r_squared = exponential_fit_with_confidence(t_fit, N_fit)
    if r_squared is None or r_squared < 0.95:
        print(f"初始指数拟合置信度{r_squared}过低，放弃指数拟合。")
        return None

    initial_r_squared = r_squared

    # 动态扩展
    while True:
        # 检查是否已经到达数据开头
        if indices[0] == 0:
            break  # 已经到达数据开头，无法继续扩展

        # 增加一个点时，确保不会超出边界
        new_start_index = max(0, indices[0] - 10)  # 防止负索引
        indices = list(range(new_start_index, indices[-1] + 1))

        # 重新获取拟合数据
        t_fit = t.iloc[indices]
        N_fit = N.iloc[indices]

        # 进行拟合
        popt_new, r_squared_new = exponential_fit_with_confidence(t_fit, N_fit,popt)

        # 检查置信度是否下降过多
        if r_squared_new is None or r_squared_new < initial_r_squared * 0.999:
            indices = indices[10:]  # 移除最后添加的点
            t_fit = t.iloc[indices]
            N_fit = N.iloc[indices]
            # residuals = N_fit - exponential_func(t_fit, *popt)
            # ss_res = residuals**2
            # ss_res = np.array(ss_res)
            # #选择拟合效果好，ss_res小于1的数据的index
            # indices_opt = [indices[i] if ss_res[i] < 0.5*np.mean(ss_res) else None for i in range(len(indices))]
            # indices_opt = [i for i in indices_opt if i is not None]
            # print(indices_opt)
            # print(indices)
            # print(N_fit)
            # print(len(indices_opt), len(indices), len(N_fit))
            # indices = list(range(indices_opt[0], indices[-1] + 1))
            # print(indices)
            break
        else:
            popt = popt_new
            r_squared = r_squared_new

    # 返回结果
    return {
        't_fit': t_fit,
        'N_fit': N_fit,
        'params': popt,
        'r_squared': r_squared,
        'indices': indices
    }

# -------------------------- Step 4: 峰值检测 -------------------------- #

def detect_peaks(N, Nsc, width_threshold=5, indices=None):
    """
    对整条数据高于 Nsc 的部分进行寻峰，峰的宽度要超过指定的数据点数量。

    参数:
    - N (pd.Series): 数据序列。
    - Nsc (float): 峰值的阈值。
    - width_threshold (int): 峰的最小宽度（数据点数量）。
    - indices (list): 要分区检测峰的数据点索引列表。

    返回:
    - list: 包含每个峰的信息的列表，每个元素是一个字典，包含峰的范围、宽度和均值。
    """
    peaks, properties = find_peaks(N, height=Nsc, width=width_threshold)
    peaks_info = []
    for i, peak in enumerate(peaks):
        width = int(properties['widths'][i])
        if width >= width_threshold:
            left_ips = int(properties['left_ips'][i])
            right_ips = int(properties['right_ips'][i])
            peak_range = range(left_ips, right_ips + 1)
            N_peak = N.iloc[peak_range]
            peak_mean = np.mean(N_peak)
            if peak_mean > Nsc:
                peaks_info.append({
                    'peak_index': peak,
                    'left': left_ips,
                    'right': right_ips,
                    'width': width,
                    'mean': peak_mean,
                    'indices': peak_range
                })
    # 分区检测峰 
    if indices is not None:
        peaks, properties = find_peaks(N[indices], height=Nsc, plateau_size=int(width_threshold/2))
        for i, peak in enumerate(peaks):
            width = int(properties['widths'][i])
            if width >= width_threshold:
                left_ips = int(properties['left_ips'][i])+indices[0]
                right_ips = int(properties['right_ips'][i])+indices[0]
                peak_range = range(left_ips, right_ips + 1)
                N_peak = N.iloc[peak_range]
                peak_mean = np.mean(N_peak)
                if peak_mean > Nsc:
                    peaks_info.append({
                        'peak_index': peak,
                        'left': left_ips,
                        'right': right_ips,
                        'width': width,
                        'mean': peak_mean,
                        'indices': peak_range
                    })
    return peaks_info

def find_nearest_interval(N, N_forecast,Nsc):
    """
    寻找N中在N_forecast左右的区间段，输出起止点。

    参数：
    N (array-like): 实际数据数组。
    N_forecast (float): 预期值。
    tolerance (float): 容差范围，默认为0.1。

    返回：
    tuple: 区间段的起止点 (start_index, end_index)。
    """
    N = np.array(N)
    # 找到与N_forecast最接近的值及其索引
    window_size=10
    N_smooth = pd.Series(N).rolling(window=window_size, center=True).mean().to_numpy()
    # 替换nan值为0
    N_smooth[np.isnan(N_smooth)] =0
    N[N<=Nsc]=0
    closest_index = np.argmin(np.abs(N_smooth - N_forecast))
    
    # 初始化区间段的起止点
    start_index = closest_index-5
    end_index = closest_index+5

    # tolerance定义为100
    tolerance = 0.01 * N_forecast
    
    # 向前扩展区间段
    while start_index > 0 and np.abs(np.mean(N[start_index - 1:end_index]) - N_forecast) <= tolerance and N[start_index - 1]>Nsc:
        start_index -= 1
    
    # 向后扩展区间段
    while end_index < len(N) - 1 and np.abs(np.mean(N[start_index:end_index+1]) - N_forecast) <= tolerance and N[end_index + 1]>Nsc:
        end_index += 1
    
    return start_index, end_index

# -------------------------- Step 5: 绘制分析图 -------------------------- #

def plot_analysis_figure(t, N, exp_fit_result, peaks_info, Nsc, N0, column_name, directory_path, num_density = -1):
    """
    绘制分析图，有两个子图。

    参数:
    - t (pd.Series): 时间序列。
    - N (pd.Series): 数据序列。
    - exp_fit_result (dict or None): 指数拟合结果。
    - peaks_info (list): 峰值信息列表。
    - Nsc (float): Nsc 值。
    - N0 (float): N0 值。
    - N_star (float): 峰均值的平均值。
    - column_name (str): 数据列名。
    - directory_path (str): 保存结果的目录路径。
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 1]}, figsize=(14, 6))

    # ------------------------------------左侧子图 1
    ax1.plot(t, N, label='Original Data', color='blue')

    # 绘制指数拟合结果和分界线
    if exp_fit_result is not None:
        t_fit = exp_fit_result['t_fit']
        N_fit = exponential_func(t_fit, *exp_fit_result['params'])
        ax1.plot(t_fit, N_fit, label='Exponential Fit', color='red',alpha=0.5)
        # 分界线
        division_time = t_fit.iloc[0]
        ax1.axvline(x=division_time, color='green', linestyle='--', label='Exponential Fit Start')
    else:
        N_fit = [N.iloc[-1]] if not N.empty else []

    # 对每个峰范围对应的数据到 x 轴围成的区域着色，并绘制水平线峰值 n
    for peak_info in peaks_info:
        indices = peak_info['indices']
        t_peak = t.iloc[indices]
        N_peak = N.iloc[indices]
        ax1.fill_between(t_peak, N_peak, N0, where=(N_peak >= N0), alpha=0.6)
        # 峰值 n 的水平线
        ax1.hlines(peak_info['mean'], t_peak.iloc[0], t_peak.iloc[-1], colors='r', linestyles='--')

    # 绘制两条水平虚线：Nsc、N0
    ax1.axhline(y=Nsc, color='purple', linestyle='--', label=f'Nsc={Nsc}')
    ax1.axhline(y=N0, color='brown', linestyle='--', label=f'N0={N0}')

    ax1.set_xlabel('Time(ps)')
    ax1.set_ylabel(column_name)
    ax1.set_title(f'{column_name} Data Show')
    # x轴时间刻度细化
    ax1.xaxis.set_major_locator(plt.MultipleLocator(100))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(10))
    ax1.grid(True)

    # --------------------------------------右侧子图 1
    widths = [info['width'] for info in peaks_info]
    means = [info['mean'] for info in peaks_info]
    if widths and means:
        ax2.scatter(widths, means, color='blue')


    ax2.set_xlabel('Peak Width (Data Points)')
    ax2.set_ylabel('Peak Mean Value n')

    if len(means) > 0:
        ymax=min(np.max(means)*2,np.max(N))
        N_cut=min(np.max(means)-10,np.max(N)*4/5)
    else:
        ymax=3*N0
        N_cut=Nsc*1.5

    ax2.set_ylim(Nsc,ymax)
    ax2.set_title(f'{column_name} Analysis (Right)')
    ax2_twin = ax2.twiny()
    ax2_twin.set_ylim(ax2.get_ylim()) 

    # 绘制原始直方图
    ax2_twin.hist(N[N >= Nsc], bins=2000, orientation='horizontal', alpha=0.6, color='gray', label='(N | N >= Nsc) Distribution')
 
    # 计算直方图数据
    N_smooth=N[(N >= N_cut)&(N <=ymax)]
    counts, bin_edges = np.histogram(N_smooth, bins=1000)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # 对直方图数据进行高斯平滑
    sigma = 0.5  # 控制平滑的强度，值越大越平滑
    smoothed_counts = gaussian_filter1d(counts[:np.nonzero(counts)[0][-1] + 1], sigma=sigma)
    # 在右侧子图上绘制高斯平滑曲线
    ax2_twin.plot(smoothed_counts, bin_centers, color='red', linestyle='-', linewidth=0.5, label='Smoothed Density', alpha=0.8)
    peak_indices, properties = find_peaks(smoothed_counts, height=0, plateau_size=0, width=0)

    density_info = []
    if peak_indices.size != 0:
        up=np.array(properties["plateau_sizes"])
        down=np.array(properties["width_heights"])
        high=np.array(properties["peak_heights"])
        sorted_peak_indices = peak_indices[np.argsort((up+down)*0.5*high)][::-1]

        # 确保 num_density 不超过 sorted_peak_indices 的长度
        num_density = min(num_density, len(sorted_peak_indices))

        if num_density < 0:
            density_place = bin_centers[sorted_peak_indices]
            ax2_twin.scatter(smoothed_counts[sorted_peak_indices], bin_centers[sorted_peak_indices], color='green', label='Density Peaks',marker='x')
        else:
            density_place = bin_centers[sorted_peak_indices[:num_density]]
            ax2_twin.scatter(smoothed_counts[sorted_peak_indices[:num_density]], bin_centers[sorted_peak_indices[:num_density]], color='green', label='Density Peaks',marker='x')

        if len(density_place) == 0:
            print(f"未找到合适聚集点，{sorted_peak_indices}")
        else:
            print(f"找到{len(density_place)}个聚集点，{density_place}")

        for i,N_forecast in enumerate(density_place):
            start_index, end_index=find_nearest_interval(N, N_forecast,Nsc)

            indices=list(range(start_index, end_index+1))

            t_peak = t.iloc[indices]
            N_peak = N.iloc[indices]
            
            ax1.plot(t_peak, N_peak, label=f'Peak {i+1}', color='green', alpha=0.9,linewidth=2)
            ax1.fill_between(t_peak, N_peak, Nsc, where=(N_peak >= Nsc), alpha=0.4, color='black')
            ax1.hlines(np.mean(N_peak), t_peak.iloc[0], t_peak.iloc[-1], colors='k', linestyles='--')

            density_info.append({
                'peak_index': i,
                'left': start_index,
                'right': end_index,
                'indices': indices,
                'mean': np.mean(N_peak),
                'width': len(indices)
            })
    else:
        print(f"未找到density_place")
        density_info=peaks_info
        if peaks_info is not None:
            density_place=[peak_info['mean'] for peak_info in peaks_info]


    if density_info is not None:
        N_star=np.mean(density_place)
        ax1.axhline(y=N_star, color='gray', linestyle='--', label=f'N*={N_star:.2f}')
        ax2.axhline(y=N_star, color='gray', linestyle='--', label=f'N*={N_star:.2f}')

    ax2.legend()
    ax1.legend()
    ax2_twin.legend(fontsize=8)


    # ----------------------------------整体标题
    plt.suptitle(f'{column_name} Analysis\nN* = {N_star:.2f}')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # 调整标题位置

    # 保存图像
    plt.savefig(os.path.join(directory_path, f'{column_name}_analysis.png'))
    plt.close()
    
    
    return density_info,N_star

# -------------------------- Step 6: 处理单个数据列 -------------------------- #
def process_column(column, t_clean, N_clean, Nsc, N0, directory_path,width_threshold,num_density):
    """
    处理单个数据列，包括指数拟合、峰值检测和可视化。

    参数:
    - column (str): 数据列名。
    - t_clean (pd.Series): 清洗后的时间序列。
    - N_clean (pd.Series): 清洗后的数据序列。
    - Nsc (float): 传入的 Nsc 值。
    - N0 (float): 传入的 N0 值。
    - directory_path (str): 保存结果的目录路径。

    返回:
    - dict: 包含峰值信息和 N* 值的结果字典。
    """
    print(f"\n\n正在处理 {column} 列...")

    # Step 1: 指数拟合
    exp_fit_result = exponential_fit_dynamic(t_clean, N_clean)

    # 如果指数拟合成功，获取指数拟合部分的索引范围
    if exp_fit_result is not None:
        exp_fit_indices = exp_fit_result['indices']
        exp_fit_params = exp_fit_result['params']
    else:
        exp_fit_indices = []
        exp_fit_params = None

    # Step 2: 确认剩余部分的均值在 Nsc 和 N0 之间
    remaining_indices = [i for i in range(len(N_clean)) if i not in exp_fit_indices]
    if remaining_indices:
        remaining_N = N_clean.iloc[remaining_indices]
        remaining_mean = np.mean(remaining_N)
        if remaining_mean < min(1.5*Nsc, N0) or remaining_mean > max(1.5*Nsc, N0):
            print(f"剩余部分的均值 {remaining_mean} 不在 1.5倍Nsc ({1.5*Nsc}) 和 N0 ({N0}) 之间，分界有误")
    else:
        print("没有剩余数据用于均值计算，该列临界晶核不显著。")

    # Step 3: 峰值检测
    peaks_info = detect_peaks(N_clean, Nsc, width_threshold,exp_fit_indices)
    
    # Step 4: 绘制分析图，获取密集位置
    density_info,N_star=plot_analysis_figure(t_clean, N_clean, exp_fit_result, peaks_info, Nsc, N0, column, directory_path, num_density)

    # 返回结果
    return {
        'peaks_info': peaks_info,
        'N_star': N_star,
        "dense_info":density_info
    }

# -------------------------- Step 6: 绘制总体分析图 -------------------------- #

def plot_overall_analysis(results, directory_path):
    """
    处理完所有数据后，绘制风琴图和散点图，x 轴为不同组数据，计算所有峰均值的均值。

    参数:
    - results (dict): 包含所有列结果的字典。
    - directory_path (str): 保存结果的目录路径。
    """
    all_N_stars = []
    data_labels = []
    peak_means_list = []
    dense_place_list = []
    
    for column, result in results.items():
        N_star = result['N_star']
        peaks_info = result['peaks_info']
        peak_means = [info['mean'] for info in peaks_info]
        dense_info = result["dense_info"]
        dense_place = [info['mean'] for info in dense_info]
        
        all_N_stars.append(N_star)
        data_labels.append(column)
        peak_means_list.append(peak_means)
        dense_place_list.append(dense_place)


    # peak_means_list和dense_place_list的数据按序号拼接
    peak_means_list = np.array(peak_means_list, dtype=object)
    dense_place_list = np.array(dense_place_list, dtype=object)
    print(f"peak_means_list: {peak_means_list}")
    print(f"dense_place_list: {dense_place_list}")
    combined_list=[]

    for i in range(max(len(peak_means_list),len(dense_place_list))):
        combined_list.append(list(np.hstack((peak_means_list[i],dense_place_list[i]))))

    dense_place_list=list(dense_place_list)
    overall_mean = np.mean([np.mean(sublist) for sublist in combined_list if sublist]) if combined_list else 0
    print(f"所有峰均值的总体均值: {overall_mean}")

    # 绘制风琴图
    plt.figure(figsize=(10, 6))
    # 过滤掉 combined_list 为空的项，同时跳过相应的 data_labels
    filtered_combined_list = [lst for lst in combined_list if lst]
    filtered_data_labels = [data_labels[i] for i in range(len(combined_list)) if combined_list[i]]

    plt.violinplot(filtered_combined_list, showmeans=True)
    plt.xticks(range(1, len(filtered_data_labels) + 1), filtered_data_labels, rotation=45)

    # 绘制散点图
    colors = []

    # 绘制 peak_means 并记录颜色
    for i, peak_means in enumerate(peak_means_list):
        x = np.full(len(peak_means), i + 1)
        if i == 0:
            scatter = plt.scatter(x, peak_means, label='Peak Mean Values')
        else:    
            scatter = plt.scatter(x, peak_means)
        colors.append(scatter.get_facecolor()[0])  # 记录颜色


    # 绘制 dense_place 并使用相同的颜色
    for i, dense_place in enumerate(dense_place_list):
        x = np.full(len(dense_place), i + 1)
        if i == 0:
            plt.scatter(x, dense_place, facecolors='none', edgecolors=colors[i], marker='D', label='Dense Place')
        else:
            plt.scatter(x, dense_place, facecolors='none', edgecolors=colors[i], marker='D')


    plt.xticks(range(1, len(data_labels) + 1), data_labels, rotation=45)
    plt.axhline(y=overall_mean, color='red', linestyle='--', label=f'Overall Mean N*={overall_mean:.2f}')
    plt.xlabel('Data Series')
    plt.ylabel('Peak Mean Values n')
    plt.title('Scatter and Violin Plot of Peak Mean Values')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(directory_path, 'overall_plot.png'))
    plt.close()

    # 保存总体均值到文件
    with open(os.path.join(directory_path, 'overall_mean.txt'), 'w') as f:
        f.write(f'所有峰均值的总体均值: {overall_mean}\n')

    with open(os.path.join(directory_path, 'overall_mean.txt'), 'a') as f:
        for column, result in results.items():
            f.write(f"Column: {column}\n")
            f.write(f"  N_star: {result['N_star']:.2f}\n\n")
            f.write("  Peaks Info:\n")
            for peak_info in result['peaks_info']:
                f.write(f"    Peak Index: {peak_info['peak_index']}\n")
                f.write(f"    Left: {peak_info['left']}\n")
                f.write(f"    Right: {peak_info['right']}\n")
                f.write(f"    Width: {peak_info['width']}\n")
                f.write(f"    Mean: {peak_info['mean']:.2f}\n\n")
            f.write("  Dense Info:\n")
            for dense_info in result['dense_info']:
                f.write(f"    Peak Index: {dense_info['peak_index']}\n")
                f.write(f"    Left: {dense_info['left']}\n")
                f.write(f"    Right: {dense_info['right']}\n")
                f.write(f"    Mean: {dense_info['mean']:.2f}\n")
                f.write(f"    Width: {dense_info['width']}\n\n")
            f.write("\n\n\n")

    # 保存result['dense_info']到csv
    # 如果没有N_star_info.csv
    if not os.path.exists(os.path.join(directory_path, 'N_star_info.csv')):
        dense_info_df = pd.DataFrame()
        for column, result in results.items():
            for dense_info in result['dense_info']:
                dense_info_df = pd.concat([dense_info_df, pd.DataFrame({
                    'Column': [column],
                    'Peak Index': [dense_info['peak_index']],
                    'Left': [dense_info['left']],
                    'Right': [dense_info['right']],
                    'Mean': [dense_info['mean']],
                    'Width': [dense_info['width']],
                    'Choice': [False],
                    'PTM': [False]
                })], ignore_index=True)
        # 排序
        dense_info_df = dense_info_df.sort_values(by=['Column'], ascending=False)
        dense_info_df.to_csv(os.path.join(directory_path, 'N_star_info.csv'), index=False)
   

# -------------------------- Step 7: 主函数 -------------------------- #

def main():
    directory_path = '/home/jhzhai/Nucleation/peek-find'  # 您提供的数据目录路径!!!
    output_csv_path = 'pem_N.csv'
    # N_star_info.csv

    # 加载并清洗数据
    df, nc = load_and_clean_data(directory_path, output_csv_path)
    T = df['t']
    print(f"数据加载完成，共有 {len(df.columns) - 1} 列数据。")

    try:
        base_dir_name = os.path.basename(directory_path)
        if 'nc' in base_dir_name:
            N0 = float(base_dir_name.replace('nc', ''))
        else:
            print("无法从目录名中提取 N0 值，请确保目录名包含 'nc' 后跟数值，例如 'nc1000'")
    except ValueError:
        print("无法解析 N0 值，请检查目录名格式")

    Nsc = None
    for filename in os.listdir(directory_path):
        if filename.startswith('pem-Nc-') and filename.endswith('.txt'):
            try:
                # 提取 Nc 值
                Nc_part = filename.split('-')[2]  # '2550.00'
                Nsc = float(Nc_part)
                break
            except (IndexError, ValueError):
                continue
    if Nsc is None:
        print("无法从文件名中提取 Nsc 值，请确保文件名包含 'Nc' 后跟数值，例如 'pem-Nc-2550.00-2433.txt'")

    print(f"提取到的 Nsc 值: {Nsc}, N0 值: {N0}")

    # 存储结果以保存到文本文件
    results = {}

    # 获取所有以 'N_' 开头的列
    columns_to_process = [col for col in df.columns if col.startswith('N_')]

    # 处理所有列
    for col in columns_to_process:
        N = df[col]
        t_clean, N_clean = clean_data(T, N)
        result = process_column(col, t_clean, N_clean, Nsc, N0, directory_path,width_threshold=5,num_density=6)#！！！！！
        results[col] = result
    
    # 绘制总体分析图
    plot_overall_analysis(results, directory_path)
    print(f"分析完成，结果已保存至 {directory_path}")

# 执行主函数
if __name__ == "__main__":
    print("开始处理...")
    main()