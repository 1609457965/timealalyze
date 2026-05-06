# timealalyze

这是一个基于 Python 的多国货币汇率预测项目，主要用于获取汇率时间序列数据、训练预测模型，并通过桌面 GUI 展示预测结果。

当前内容聚焦于汇率预测系统本身。

## 功能概览

- 从 FRED 获取美元相关货币对的历史汇率数据
- 对汇率序列进行缺失值处理、平稳性检验和差分处理
- 使用 ARIMA、Prophet、LSTM 三类模型进行训练与预测
- 保存和加载本地模型文件
- 使用 Tkinter 构建桌面端图形界面
- 展示历史数据、预测曲线、预测明细和误差指标

## 项目结构

```text
TimeAnalyse/
├── currency_forecast.py              # 汇率预测系统主程序
├── currency_forecast.spec            # PyInstaller 打包配置
├── exchange_rate_data/               # 汇率数据缓存
├── exchange_rate_data_fred/          # FRED 汇率数据
├── saved_models/                     # 已训练模型
├── dist/                             # 打包后的可执行程序
├── GUI.png                           # GUI 界面截图
├── prediction_plot.html              # 预测结果可视化
├── *.csv                             # 汇率相关数据
├── *.pkl / *.h5                      # 模型与归一化器文件
└── README.md                         # 项目说明
```

## 核心文件

| 文件或目录 | 说明 |
| --- | --- |
| `currency_forecast.py` | 主程序，包含数据获取、预处理、模型训练、预测评估和 Tkinter GUI |
| `currency_forecast.spec` | PyInstaller 打包配置 |
| `exchange_rate_data/` | 本地汇率数据缓存 |
| `exchange_rate_data_fred/` | 从 FRED 获取的汇率数据 |
| `saved_models/` | 已保存的 ARIMA、Prophet、LSTM 模型 |
| `dist/汇率预测系统.exe` | 已打包的 Windows 可执行程序 |
| `GUI.png` | 程序界面截图 |
| `prediction_plot.html` | 预测结果图表 |

## 程序模块

`currency_forecast.py` 中的主要类：

| 类 | 作用 |
| --- | --- |
| `ExchangeRateDataLoader` | 货币代码映射、FRED 序列 ID 生成、汇率数据下载和本地缓存 |
| `DataPreprocessor` | 缺失值处理、ADF 平稳性检验、差分和反差分 |
| `BaseForecastModel` | 模型保存、加载路径等公共逻辑 |
| `ARIMAForecaster` | ARIMA 模型训练、加载和预测 |
| `ProphetForecaster` | Prophet 模型训练、加载和预测 |
| `LSTMForecaster` | LSTM 数据集构造、训练、加载和多步预测 |
| `CurrencyForecastSystem` | 组织数据加载、模型训练、模型加载、预测和评估 |
| `DataCache` | 简单数据缓存工具 |
| `ExchangeRateForecastApp` | Tkinter 图形界面 |

## 环境依赖

建议使用 Python 3.8 及以上版本。

```bash
pip install pandas numpy matplotlib statsmodels prophet tensorflow scikit-learn joblib yfinance pandas_datareader requests pyinstaller
```

Tkinter 通常随 Python 一起安装。如果运行 GUI 时提示缺少 Tkinter，需要安装带 Tcl/Tk 支持的 Python。

## 运行方式

启动 GUI：

```bash
python currency_forecast.py
```

使用 PyInstaller 打包：

```bash
pyinstaller currency_forecast.spec
```

注意：`currency_forecast.spec` 中配置了 `icon='icon.ico'`。如果本地没有该图标文件，打包时可补充图标，或移除 spec 中的图标配置。

## 数据与模型

当前仓库保留的是汇率预测项目相关数据和模型：

- `DEXCHUS.csv`、`exchange_rate.csv`：人民币汇率相关时间序列数据
- `exchange_rate_data/*.csv`：本地汇率数据缓存
- `exchange_rate_data_fred/*.csv`：FRED 数据源导出的汇率数据
- `saved_models/*.pkl`、`saved_models/*.h5`：已训练模型
- 根目录中的 `arima_model.pkl`、`prophet_model.pkl`、`lstm_model.h5`、`lstm_scaler.pkl`：模型文件和归一化器


