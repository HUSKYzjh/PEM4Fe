```
PEM4Fe/
├── src/                  # 核心代码
│   ├── config.py
│   ├── fitting.py
│   ├── peak_detection.py
│   └── output.py
├── tests/                # 测试文件
│   ├── test_config.py
│   ├── test_fitting.py
│   ├── test_peak_detection.py
│   └── test_output.py
├── data/                 # 数据目录（原始数据）
├── logs/                 # 日志文件目录
├── output/               # 程序输出目录
├── requirements.txt      # 依赖库文件
├── README.md             # 项目说明文件
└── PEM4Fe.py               # 主程序入口

```

**config.yaml 示例**

```
# PEM4Fe --config *.yaml 

# pressure: 压力的数值设置(GPa)
pressure: 1000.0
# N0: 起始值，通常代表初始形核大小
N0: 700.0
# Nsc: 受力形核大小
Nsc: 1050.0
# input_dir: 输入数据的目录路径
input_dir: /home/cgao/jhzhai/Nucleation/N/p10000000/cool1700/nc700
# output_dir: 输出结果的目录路径
output_dir: /home/cgao/jhzhai/Nucleation/N/p10000000/cool1700/output-1700

# max_iterations: 最大迭代次数
max_iterations: 100
# initial_fit_points: 初始拟合点的数量
initial_fit_points: 3
# min_r_squared: 最小拟合优度（R平方值），用于验证拟合的质量
min_r_squared: 0.95

# width_threshold: 寻峰宽度阈值，用于某种筛选或计算
width_threshold: 8
# plateau_size: 寻峰平台大小，可能与数据处理或分析相关
plateau_size: 0

# num_density: 平台密度参数的数量或设置
num_density: 6
# window_size: 窗口大小，用于移动平均算法
window_size: 3
# tolerance_ratio: 容忍度比率，根据平台挑选可能区段
tolerance_ratio: 0.05

```
