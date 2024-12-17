import os
import logging
import pandas as pd
import numpy as np


def collect_data_from_files(directory):
    """
    从指定目录中收集所有符合命名规则的文本文件，并合并为一个 DataFrame。
    
    参数:
    - directory (str): 数据文件所在目录路径。
    
    返回:
    - pd.DataFrame: 合并后的数据框，包含时间列 't' 和多个 'N_' 列。
    """
    logging.info(f"开始从目录 {directory} 中收集数据文件...")
    combined_df = pd.DataFrame()
    file_count = 0

    if not os.path.exists(directory):
        logging.error(f"指定的目录 {directory} 不存在。")
        return combined_df

    for filename in os.listdir(directory):
        if filename.startswith('pem-Nc-') and filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            try:
                suffix = filename.split('-')[-1].split('.')[0]
                # 使用原始字符串避免转义警告
                df = pd.read_csv(filepath, sep=r'\s+', names=['t', f'N_{suffix}'])
                if combined_df.empty:
                    combined_df = df
                else:
                    combined_df = pd.merge(combined_df, df, on='t', how='outer')
                file_count += 1
                logging.info(f"成功读取文件: {filepath}")
            except Exception as e:
                logging.warning(f"读取文件 {filepath} 失败: {e}")

    if file_count == 0:
        logging.warning(f"目录 {directory} 中没有找到符合命名规则的文件。")
    else:
        logging.info(f"共处理了 {file_count} 个文件。")

    return combined_df

def clean_data(t, N):
    """
    清洗数据，去除无效值（NaN 和无穷值）。
    
    参数:
    - t (pd.Series): 时间序列。
    - N (pd.Series): 数据序列。
    
    返回:
    - pd.Series: 清洗后的时间序列。
    - pd.Series: 清洗后的数据序列。
    """
    logging.info("开始清洗数据...")
    initial_length = len(N)

    # 去除无效值
    clean_mask = np.isfinite(N)
    t_clean = t[clean_mask]
    N_clean = N[clean_mask]

    # 重置索引
    t_clean = t_clean.reset_index(drop=True)
    N_clean = N_clean.reset_index(drop=True)

    cleaned_length = len(N_clean)
    logging.info(f"数据清洗完成：原始数据点 {initial_length} 个，清洗后数据点 {cleaned_length} 个。")
    
    return t_clean, N_clean

def load_clean_save_data(directory_path, output_csv_path):
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
        logging.error("无法从目录名中提取 'nc' 值，请确保目录名包含 'nc' 后跟数值，例如 'nc1000'")
        nc = None
    
    collected_data = collect_data_from_files(directory_path)

    output_csv_path = os.path.join(output_csv_path, 'pem_N.csv')
    save_to_csv(collected_data, os.path.join(directory_path, output_csv_path))
    logging.info(f"nc={nc}已处理完毕")
    df = pd.read_csv(os.path.join(directory_path, output_csv_path))
    return df, nc

def save_to_csv(dataframe, output_file):
    """
    将 DataFrame 保存为 CSV 文件。

    参数:
    - dataframe (pd.DataFrame): 要保存的数据框。
    - output_file (str): 输出 CSV 文件路径。
    """
    if dataframe.empty:
        logging.warning("数据框为空，未保存任何数据。")
    else:
        dataframe.to_csv(output_file, index=False)
        logging.info(f"数据成功保存到 {output_file}")
