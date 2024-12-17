import argparse
import os
import logging
from PEM4Fe.config import load_config, setup_logging
from PEM4Fe.data_processing import clean_data, load_clean_save_data
from PEM4Fe.fitting import FittingModule
from PEM4Fe.peak_detection import PeakDetectionModule
from PEM4Fe.output import MainModule

def main():

    # --------------------- 解析命令行参数 --------------------- #
    parser = argparse.ArgumentParser(description="PEM4Fe: Data Analysis Tool")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (YAML or JSON).",
    )
    args = parser.parse_args()

    # --------------------- 加载配置文件 --------------------- #
    try:
        config = load_config(args.config)
        logging.info(f"成功加载配置文件：{args.config}")
    except Exception as e:
        logging.error(f"加载配置文件失败：{e}")
        return

    # --------------------- 设置日志 --------------------- #
    log_file = "application.log"
    setup_logging(output_dir=config.get("output_dir", "./output"), log_file=log_file)
    logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.StreamHandler(),           # 同时输出到控制台
                    logging.FileHandler("application.log", encoding="utf-8")
                ]
            )
    # 获取 Logger 对象
    logger = logging.getLogger("Main")
    logger.info("日志记录初始化完成。")


    # --------------------- 加载数据 --------------------- #
    input_dir = config.get("input_dir", "./data")
    output_dir = config.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)

    try:
        collected_data, nc = load_clean_save_data(input_dir, output_dir)
        logger.info("数据加载成功。")
    except Exception as e:
        logger.error(f"数据加载失败：{e}")
        return

    t = collected_data['t']

    # --------------------- 初始化模块 --------------------- #
    fitting_module = FittingModule(config=config, logger=logger)
    peak_detection_module = PeakDetectionModule(config=config, logger=logger)
    main_module = MainModule(config=config)

    logger.info("模块初始化完成。")

    # --------------------- 执行核心流程 --------------------- #
    peaks_results = []
    plateaus_results = []

    for col in collected_data.columns[1:]:  # 跳过时间列
        logger.info(f"开始处理列：{col}")
        try:
            # 数据清洗
            t_clean, N_clean = clean_data(t, collected_data[col])

            # 动态拟合
            fit_result = fitting_module.fit_dynamic(t_clean, N_clean, index=col)
            if fit_result:
                logger.info(f"拟合成功：R²={fit_result['r_squared']:.4f}")
            else:
                logger.warning(f"{col} 拟合失败，跳过增长检测。")
            # 峰值检测
            peaks_info = peak_detection_module.detect_peaks(N_clean, col)
            if peaks_info:
                logger.info(f"{col} 检测到 {len(peaks_info)} 个峰值。")
            else:
                logger.warning(f"{col} 未检测到峰值。")

            # 分析和可视化
            plateaus_info = peak_detection_module.plot_analysis_figure(
                t_clean, N_clean, peaks_info, col, fit_result
            )
            if plateaus_info:
                logger.info(f"{col} 检测到 {len(plateaus_info)} 个平台。")
            else:
                logger.warning(f"{col} 未检测到平台。")
            
            peaks_results.append(peaks_info)
            plateaus_results.append(plateaus_info)

        except Exception as e:
            logger.error(f"处理列 {col} 时发生错误：{e}")

    # --------------------- 保存结果 --------------------- #
    try:
        main_module.save_results_to_csv(peaks_results, plateaus_results)
        main_module.plot_overall_analysis(peaks_results, plateaus_results)
        logger.info("结果保存成功。")
    except Exception as e:
        logger.error(f"保存结果失败：{e}")

    logger.info("分析流程完成。")

if __name__ == "__main__":
    main()
