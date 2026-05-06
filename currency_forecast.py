# -*- coding: utf-8 -*-
"""
多国货币汇率预测系统 - 可执行版本
环境要求：Python 3.8+
打包前需要安装：
pip install pandas numpy matplotlib statsmodels prophet tensorflow scikit-learn joblib yfinance pyinstaller
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import joblib
import os
import warnings
import yfinance as yf
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sys
import time
from functools import wraps
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import json
import pandas_datareader.data as web
warnings.filterwarnings('ignore')

# 设置matplotlib后端，避免打包后的显示问题
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用跨平台字体
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 数据获取 (已修改为FRED) ====================
class ExchangeRateDataLoader:
    def __init__(self):
        """初始化加载器，创建数据存储目录"""
        self.data_dir = "exchange_rate_data_fred"
        os.makedirs(self.data_dir, exist_ok=True)
        # 原始货币代码 -> 标准3位代码 的映射
        self.currency_map = {
            'US': 'USD', 'CN': 'CNY', 'EU': 'EUR', 'JP': 'JPY',
            'GB': 'GBP', 'AU': 'AUD', 'CA': 'CAD', 'CH': 'CHF',
            'HK': 'HKD', 'NZ': 'NZD', 'SE': 'SEK', 'KR': 'KRW',
            'SG': 'SGD', 'NO': 'NOK', 'MX': 'MXN', 'IN': 'INR',
            'RU': 'RUB', 'ZA': 'ZAR', 'TR': 'TRY', 'BR': 'BRL',
            'UK': 'GBP' # 添加英国的映射
        }
        # 标准货币代码 -> FRED 国家/地区代码 的映射
        # 注意: FRED的命名并不总是遵循统一标准
        self.fred_country_map = {
            'CNY': 'CH', 'EUR': 'EU', 'JPY': 'JP', 'GBP': 'UK',
            'AUD': 'AU', 'CAD': 'CA', 'CHF': 'CH', 'HKD': 'HK',
            'NZD': 'NZ', 'SEK': 'SE', 'KRW': 'KO', 'SGD': 'SG',
            'NOK': 'NO', 'MXN': 'MX', 'INR': 'IN', 'RUB': 'RU',
            'ZAR': 'SF', 'TRY': 'TU', 'BRL': 'BZ'
        }

    def get_fred_symbol(self, from_currency, to_currency):
        """
        获取FRED的货币对序列ID。
        FRED主要提供相对于美元的汇率。
        返回 (序列ID, 是否需要取倒数)
        """
        from_code = self.currency_map.get(from_currency.upper(), from_currency.upper())
        to_code = self.currency_map.get(to_currency.upper(), to_currency.upper())

        # 情况1: 查询 USD -> XXX (例如 USD/CNY)
        # FRED提供的是 XXX/USD (DEXCHUS), 这正是我们需要的汇率值
        if from_code == 'USD' and to_code != 'USD':
            country_code = self.fred_country_map.get(to_code)
            if country_code:
                # 例如 DEXCHUS 代表 人民币 / 美元
                return f"DEX{country_code}US", False 
            else:
                print(f"警告: 无法找到 {to_code} 在FRED中的对应代码。")
                return None, False

        # 情况2: 查询 XXX -> USD (例如 EUR/USD)
        # FRED提供的是 USD/XXX (DEXUSEU), 我们需要取其倒数
        if to_code == 'USD' and from_code != 'USD':
            # FRED中部分此类汇率以 DEXUS<country_code> 形式存在
            country_code = self.fred_country_map.get(from_code)
            symbol = f"DEXUS{country_code}"
            # 这是一个特殊情况，EUR是反过来的
            if from_code == 'EUR':
                symbol = 'DEXUSEU'
            # 对于大多数货币，FRED只提供DEX<COUNTRY_CODE>US
            # 我们获取它然后取倒数
            if country_code:
                # 例如获取 DEXCHUS, 然后取倒数得到 USD/CNY
                return f"DEX{country_code}US", True
            else:
                print(f"警告: 无法找到 {from_code} 在FRED中的对应代码。")
                return None, False
        
        # 情况3: 不涉及USD的货币对 (暂不支持)
        print(f"警告: 当前实现仅支持与美元(USD)相关的货币对。无法处理 {from_code}/{to_code}。")
        return None, False

    def fetch_exchange_rate(self, from_currency="USD", to_currency="CNY", start_date='2010-01-01'):
        """从FRED (Federal Reserve Economic Data) 获取汇率数据"""
        symbol, invert_rate = self.get_fred_symbol(from_currency, to_currency)
        if not symbol:
            return None

        file_path = os.path.join(self.data_dir, f"{from_currency}_{to_currency}.csv")
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            print(f"正在从FRED下载 {from_currency}/{to_currency} (序列: {symbol}) 汇率数据...")
            
            # 使用pandas_datareader从FRED下载数据
            df = web.DataReader(symbol, 'fred', start_date, end_date)
            
            if df.empty:
                print(f"无法从FRED获取 {symbol} 的数据")
                return None

            # 如果需要，对汇率取倒数
            if invert_rate:
                print(f"注意: 对获取的序列 {symbol} 取倒数以匹配 {from_currency}/{to_currency} 报价。")
                df[symbol] = 1 / df[symbol]

            # 数据预处理
            df = df.rename(columns={symbol: 'exchange_rate'})
            df.index.name = 'Date'
            
            # 保存数据
            df.to_csv(file_path)
            print(f"数据已保存到 {file_path}")
            
            # FRED数据在周末和节假日有缺失值，需要填充以确保连续性
            # 使用'ffill' (forward-fill) 填充周末和节假日
            df = df.resample('D').ffill()
            df = df.dropna()
            
            return df
            
        except Exception as e:
            print(f"从FRED下载数据时出错: {e}")
            
            # 尝试从本地加载
            if os.path.exists(file_path):
                print(f"从本地加载 {from_currency}/{to_currency} 汇率数据")
                df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                return df
            
            return None

# ==================== 数据预处理 ====================
class DataPreprocessor:
    def __init__(self):
        self.diff_needed = False
        self.last_value = None
        
    def preprocess(self, df):
        """预处理汇率数据"""
        # 处理缺失值
        df = df.interpolate(method='time')
        
        # 平稳性检验
        def is_stationary(series):
            if len(series) < 20:  # 数据太少，跳过检验
                return True
            result = adfuller(series)
            return result[1] < 0.05
        
        if not is_stationary(df['exchange_rate']):
            self.diff_needed = True
            self.last_value = df['exchange_rate'].iloc[-1]
            df['diff'] = df['exchange_rate'].diff().dropna()
            print("数据非平稳，已进行差分处理")
        else:
            self.diff_needed = False
            print("数据是平稳的")
            
        return df
    
    def inverse_transform(self, diff_predictions):
        """将差分预测转换回原始值"""
        if not self.diff_needed:
            return diff_predictions
            
        predictions = []
        current = self.last_value
        for delta in diff_predictions:
            current += delta
            predictions.append(current)
        return np.array(predictions)

# ==================== 模型基类 ====================
class BaseForecastModel:
    def __init__(self, currency_pair):
        self.currency_pair = currency_pair
        self.model_dir = "saved_models"
        os.makedirs(self.model_dir, exist_ok=True)
        
    def get_model_path(self, model_type):
        return os.path.join(self.model_dir, f"{self.currency_pair}_{model_type}.pkl")
    
    def save_model(self, model, model_type):
        path = self.get_model_path(model_type)
        joblib.dump(model, path)
        print(f"模型已保存到 {path}")
        
    def load_model(self, model_type):
        path = self.get_model_path(model_type)
        if os.path.exists(path):
            return joblib.load(path)
        return None

# ==================== ARIMA模型 ====================
class ARIMAForecaster(BaseForecastModel):
    def __init__(self, currency_pair):
        super().__init__(currency_pair)
        self.model = None
        self.d = 0
        
    def train(self, df, diff_needed=False):
        """训练ARIMA模型"""
        target_col = 'diff' if diff_needed and 'diff' in df.columns else 'exchange_rate'
        data = df[target_col].dropna()
        
        if len(data) < 10:
            raise ValueError("数据量不足，无法训练ARIMA模型")
        
        self.d = 1 if diff_needed else 0
        
        try:
            model = ARIMA(data, order=(2, self.d, 2))
            self.model = model.fit()
            self.save_model(self.model, "arima")
        except Exception as e:
            print(f"ARIMA训练失败，使用简单模型: {e}")
            model = ARIMA(data, order=(1, self.d, 1))
            self.model = model.fit()
            self.save_model(self.model, "arima")
        
    def load(self):
        """加载ARIMA模型"""
        self.model = self.load_model("arima")
        return self.model is not None
        
    def predict(self, steps=30, diff_needed=False):
        """使用ARIMA预测"""
        if not self.model and not self.load():
            raise Exception("ARIMA模型未训练且未找到保存的模型")
            
        forecast = self.model.get_forecast(steps=steps)
        pred = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        return pred, conf_int

# ==================== Prophet模型 ====================
class ProphetForecaster(BaseForecastModel):
    def __init__(self, currency_pair):
        super().__init__(currency_pair)
        self.model = None
        
    def train(self, df):
        """训练Prophet模型"""
        df_reset = df.reset_index()
        date_col = df_reset.columns[0]
        df_prophet = df_reset[[date_col, 'exchange_rate']]
        df_prophet.columns = ['ds', 'y']
        
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        self.model.fit(df_prophet)
        
        self.save_model(self.model, "prophet")
        
    def load(self):
        """加载Prophet模型"""
        self.model = self.load_model("prophet")
        return self.model is not None
        
    def predict(self, steps=30):
        """使用Prophet预测"""
        if not self.model and not self.load():
            raise Exception("Prophet模型未训练且未找到保存的模型")
            
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        
        pred = forecast[['ds', 'yhat']].set_index('ds').iloc[-steps:]
        return pred['yhat'], forecast

# ==================== LSTM模型 ====================
class LSTMForecaster(BaseForecastModel):
    def __init__(self, currency_pair):
        super().__init__(currency_pair)
        self.model = None
        self.scaler = MinMaxScaler()
        self.look_back = 30  # 减少look_back以适应更少的数据
        
    def create_dataset(self, data, look_back=30):
        """创建监督学习数据集"""
        X, y = [], []
        for i in range(len(data)-look_back-1):
            X.append(data[i:(i+look_back), 0])
            y.append(data[i+look_back, 0])
        return np.array(X), np.array(y)
        
    def train(self, df):
        """训练LSTM模型"""
        if len(df) < self.look_back + 10:
            raise ValueError("数据量不足，无法训练LSTM模型")
            
        data = self.scaler.fit_transform(df[['exchange_rate']])
        X, y = self.create_dataset(data, self.look_back)
        
        if len(X) == 0:
            raise ValueError("无法创建训练数据")
        
        # 构建模型
        self.model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(self.look_back, 1)),
            LSTM(16),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
        # 训练
        X = X.reshape(X.shape[0], X.shape[1], 1)
        self.model.fit(X, y, epochs=30, batch_size=16, verbose=0)
        
        # 保存模型和scaler
        model_path = self.get_model_path("lstm").replace(".pkl", ".h5")
        self.model.save(model_path)
        self.save_model(self.scaler, "lstm_scaler")
        
    def load(self):
        """加载LSTM模型"""
        model_path = self.get_model_path("lstm").replace(".pkl", ".h5")
        scaler_path = self.get_model_path("lstm_scaler")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            return True
        return False
        
    def predict(self, df, steps=30):
        """使用LSTM预测"""
        if not self.model and not self.load():
            raise Exception("LSTM模型未训练且未找到保存的模型")
            
        # 准备输入数据
        data = self.scaler.transform(df[['exchange_rate']])
        
        # 使用最后look_back个数据点进行预测
        last_sequence = data[-self.look_back:]
        predictions = []
        
        for _ in range(steps):
            X = last_sequence.reshape(1, self.look_back, 1)
            pred = self.model.predict(X, verbose=0)
            predictions.append(pred[0, 0])
            
            # 更新序列
            last_sequence = np.append(last_sequence[1:], pred)
        
        # 反向转换
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()

# ==================== 汇率预测系统 ====================
class CurrencyForecastSystem:
    def __init__(self, from_currency="USD", to_currency="CNY"):
        self.from_currency = from_currency
        self.to_currency = to_currency
        self.currency_pair = f"{from_currency}_{to_currency}"
        
        self.data_loader = ExchangeRateDataLoader()
        self.preprocessor = DataPreprocessor()
        
        self.arima = ARIMAForecaster(self.currency_pair)
        self.prophet = ProphetForecaster(self.currency_pair)
        self.lstm = LSTMForecaster(self.currency_pair)
        
        self.df = None
        self.processed_df = None
        
    def load_data(self, start_date='2010-01-01'):
        """加载或下载汇率数据"""
        self.df = self.data_loader.fetch_exchange_rate(
            self.from_currency, self.to_currency, start_date)
        
        if self.df is not None and len(self.df) > 0:
            self.processed_df = self.preprocessor.preprocess(self.df)
            return True
        return False
        
    def train_all_models(self):
        """训练所有模型"""
        if self.processed_df is None:
            print("请先加载数据")
            return False
            
        print(f"训练 {self.currency_pair} 汇率预测模型...")
        
        success_count = 0
        
        # 训练ARIMA模型
        try:
            print("训练ARIMA模型...")
            self.arima.train(self.processed_df, self.preprocessor.diff_needed)
            success_count += 1
        except Exception as e:
            print(f"ARIMA模型训练失败: {e}")
        
        # 训练Prophet模型
        try:
            print("训练Prophet模型...")
            self.prophet.train(self.processed_df)
            success_count += 1
        except Exception as e:
            print(f"Prophet模型训练失败: {e}")
        
        # 训练LSTM模型
        try:
            print("训练LSTM模型...")
            self.lstm.train(self.processed_df)
            success_count += 1
        except Exception as e:
            print(f"LSTM模型训练失败: {e}")
        
        print(f"成功训练 {success_count}/3 个模型")
        return success_count > 0
        
    def load_all_models(self):
        """尝试加载所有模型"""
        loaded = 0
        
        if self.arima.load():
            loaded += 1
        if self.prophet.load():
            loaded += 1
        if self.lstm.load():
            loaded += 1
            
        print(f"已加载 {loaded}/3 个模型")
        return loaded > 0
        
    def predict_all(self, steps=30):
        """使用所有模型进行预测"""
        results = {}
        
        # ARIMA预测
        try:
            arima_pred, arima_ci = self.arima.predict(steps, self.preprocessor.diff_needed)
            if self.preprocessor.diff_needed:
                arima_pred = self.preprocessor.inverse_transform(arima_pred)
                arima_ci_df = arima_ci.copy()
                for col in arima_ci_df.columns:
                    arima_ci_df[col] = self.preprocessor.inverse_transform(arima_ci_df[col].values)
                arima_ci = arima_ci_df
            results['ARIMA'] = {
                'prediction': arima_pred,
                'confidence_interval': arima_ci
            }
        except Exception as e:
            print(f"ARIMA预测失败: {e}")
        
        # Prophet预测
        try:
            prophet_pred, prophet_forecast = self.prophet.predict(steps)
            results['Prophet'] = {
                'prediction': prophet_pred,
                'forecast': prophet_forecast
            }
        except Exception as e:
            print(f"Prophet预测失败: {e}")
        
        # LSTM预测
        try:
            lstm_pred = self.lstm.predict(self.processed_df, steps)
            results['LSTM'] = {
                'prediction': lstm_pred
            }
        except Exception as e:
            print(f"LSTM预测失败: {e}")
        
        return results
        
    def evaluate_models(self, test_period=30):
        """评估模型性能"""
        if self.processed_df is None or len(self.processed_df) < test_period * 2:
            print("数据量不足，无法评估")
            return None
            
        # 划分训练集和测试集
        train_df = self.processed_df.iloc[:-test_period]
        test_df = self.processed_df.iloc[-test_period:]
        
        # 临时预处理器
        temp_processor = DataPreprocessor()
        temp_train = temp_processor.preprocess(train_df)
        
        # 临时训练模型
        print("临时训练模型进行评估...")
        temp_results = {}
        
        # ARIMA
        try:
            temp_arima = ARIMAForecaster(self.currency_pair)
            temp_arima.train(temp_train, temp_processor.diff_needed)
            arima_pred, _ = temp_arima.predict(test_period, temp_processor.diff_needed)
            if temp_processor.diff_needed:
                arima_pred = temp_processor.inverse_transform(arima_pred)
            temp_results['ARIMA'] = {'prediction': arima_pred}
        except Exception as e:
            print(f"ARIMA评估失败: {e}")
        
        # Prophet
        try:
            temp_prophet = ProphetForecaster(self.currency_pair)
            temp_prophet.train(temp_train)
            prophet_pred, _ = temp_prophet.predict(test_period)
            temp_results['Prophet'] = {'prediction': prophet_pred}
        except Exception as e:
            print(f"Prophet评估失败: {e}")
        
        # LSTM
        try:
            temp_lstm = LSTMForecaster(self.currency_pair)
            temp_lstm.train(temp_train)
            lstm_pred = temp_lstm.predict(temp_train, test_period)
            temp_results['LSTM'] = {'prediction': lstm_pred}
        except Exception as e:
            print(f"LSTM评估失败: {e}")
        
        # 计算指标
        actual = test_df['exchange_rate'].values
        metrics = {}
        
        for model_name, result in temp_results.items():
            try:
                pred = result['prediction'][:len(actual)]
                if hasattr(pred, 'values'):
                    pred = pred.values
                mae = mean_absolute_error(actual, pred)
                rmse = np.sqrt(mean_squared_error(actual, pred))
                metrics[model_name] = {'MAE': mae, 'RMSE': rmse}
            except Exception as e:
                print(f"{model_name} 指标计算失败: {e}")
            
        return metrics

# ==================== 数据缓存 ====================
class DataCache:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, from_currency, to_currency, date):
        return f"{from_currency}_{to_currency}_{date}.json"
    
    def save_to_cache(self, data, from_currency, to_currency, date):
        cache_file = os.path.join(self.cache_dir, 
                                 self.get_cache_key(from_currency, to_currency, date))
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    
    def get_from_cache(self, from_currency, to_currency, date):
        cache_file = os.path.join(self.cache_dir, 
                                 self.get_cache_key(from_currency, to_currency, date))
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

# ==================== GUI界面 ====================
class ExchangeRateForecastApp:
    def __init__(self, master):
        self.master = master
        master.title("汇率预测系统 v1.0")
        master.geometry("1200x800")
        
        # 设置窗口最小尺寸
        master.minsize(800, 600)
        
        # 初始化预测系统
        self.forecast_system = None
        self.current_predictions = None
        
        # 创建GUI组件
        self.create_widgets()
        
        # 样式配置
        style = ttk.Style()
        style.configure("TButton", padding=6, font=('Arial', 10))
        style.configure("TLabel", font=('Arial', 10))
        
    def create_widgets(self):
        # 顶部控制面板
        control_frame = ttk.Frame(self.master)
        control_frame.pack(pady=10, fill=tk.X, padx=10)
        
        # 第一行控件
        row1_frame = ttk.Frame(control_frame)
        row1_frame.pack(fill=tk.X, pady=5)
        
        # 货币选择
        ttk.Label(row1_frame, text="源货币:").pack(side=tk.LEFT, padx=5)
        self.from_currency = ttk.Combobox(row1_frame, width=8, values=[
            'USD', 'CNY', 'EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF',
            'HKD', 'NZD', 'SEK', 'KRW', 'SGD', 'NOK', 'MXN', 'INR',
            'RUB', 'ZAR', 'TRY', 'BRL'
        ])
        self.from_currency.set('USD')
        self.from_currency.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row1_frame, text="目标货币:").pack(side=tk.LEFT, padx=5)
        self.to_currency = ttk.Combobox(row1_frame, width=8, values=[
            'USD', 'CNY', 'EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF',
            'HKD', 'NZD', 'SEK', 'KRW', 'SGD', 'NOK', 'MXN', 'INR',
            'RUB', 'ZAR', 'TRY', 'BRL'
        ])
        self.to_currency.set('CNY')
        self.to_currency.pack(side=tk.LEFT, padx=5)
        
        # 开始日期
        ttk.Label(row1_frame, text="起始日期:").pack(side=tk.LEFT, padx=5)
        self.start_date_entry = ttk.Entry(row1_frame, width=12)
        self.start_date_entry.pack(side=tk.LEFT, padx=5)
        self.start_date_entry.insert(0, "2020-01-01")
        
        # 预测天数
        ttk.Label(row1_frame, text="预测天数:").pack(side=tk.LEFT, padx=5)
        self.forecast_days = ttk.Combobox(row1_frame, width=5, values=[7, 14, 30, 60])
        self.forecast_days.current(2)
        self.forecast_days.pack(side=tk.LEFT, padx=5)
        
        # 第二行按钮
        row2_frame = ttk.Frame(control_frame)
        row2_frame.pack(fill=tk.X, pady=5)
        
        self.load_btn = ttk.Button(row2_frame, text="加载数据", command=self.load_data)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.train_btn = ttk.Button(row2_frame, text="训练模型", command=self.train_models, state=tk.DISABLED)
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.predict_btn = ttk.Button(row2_frame, text="执行预测", command=self.run_prediction, state=tk.DISABLED)
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # 图表区域
        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 结果展示区域
        result_frame = ttk.Frame(self.master)
        result_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 创建左右两列
        left_frame = ttk.Frame(result_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        right_frame = ttk.Frame(result_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # 模型评估结果
        ttk.Label(left_frame, text="模型性能:").pack(anchor=tk.W)
        self.metrics_text = tk.Text(left_frame, height=6, width=40)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        
        # 预测结果
        ttk.Label(right_frame, text="最新预测:").pack(anchor=tk.W)
        self.forecast_text = tk.Text(right_frame, height=6, width=40)
        self.forecast_text.pack(fill=tk.BOTH, expand=True)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(self.master, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def set_status(self, message):
        self.status_var.set(message)
        self.master.update_idletasks()
        
    def load_data(self):
        def load_task():
            try:
                from_curr = self.from_currency.get()
                to_curr = self.to_currency.get()
                
                if not from_curr or not to_curr:
                    messagebox.showerror("错误", "请选择货币对")
                    return
                
                if from_curr == to_curr:
                    messagebox.showerror("错误", "源货币和目标货币不能相同")
                    return
                
                self.set_status(f"正在加载 {from_curr}/{to_curr} 数据...")
                self.forecast_system = CurrencyForecastSystem(from_curr, to_curr)
                
                success = self.forecast_system.load_data(
                    start_date=self.start_date_entry.get()
                )
                
                if success:
                    self.set_status("数据加载成功！")
                    self.train_btn.config(state=tk.NORMAL)
                    self.plot_raw_data()
                    
                    # 尝试加载已有模型
                    if self.forecast_system.load_all_models():
                        self.predict_btn.config(state=tk.NORMAL)
                        self.set_status("数据加载成功，已找到预训练模型！")
                else:
                    messagebox.showerror("错误", "数据加载失败，请检查网络连接或货币对是否支持")
                
            except Exception as e:
                messagebox.showerror("错误", f"加载数据时出错: {str(e)}")
            finally:
                self.load_btn.config(state=tk.NORMAL)
        
        self.load_btn.config(state=tk.DISABLED)
        threading.Thread(target=load_task).start()
        
    def plot_raw_data(self):
        if self.forecast_system and self.forecast_system.processed_df is not None:
            self.ax.clear()
            df = self.forecast_system.processed_df
            self.ax.plot(df.index, df['exchange_rate'].values)
            self.ax.set_title(f'{self.from_currency.get()}/{self.to_currency.get()} Historical Exchange Rate')
            self.ax.set_xlabel("Date")
            self.ax.set_ylabel("Exchange Rate")
            self.ax.grid(True, alpha=0.3)
            self.figure.tight_layout()
            self.canvas.draw()
        
    def train_models(self):
        def train_task():
            try:
                self.set_status("正在训练模型...")
                success = self.forecast_system.train_all_models()
                if success:
                    self.set_status("模型训练完成！")
                    self.predict_btn.config(state=tk.NORMAL)
                else:
                    messagebox.showwarning("警告", "部分模型训练失败，但可以继续使用成功的模型")
                    self.predict_btn.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("错误", f"训练模型时出错: {str(e)}")
            finally:
                self.train_btn.config(state=tk.NORMAL)
        
        self.train_btn.config(state=tk.DISABLED)
        threading.Thread(target=train_task).start()
        
    def run_prediction(self):
        def predict_task():
            try:
                steps = int(self.forecast_days.get())
                self.set_status(f"正在进行{steps}天预测...")
                
                # 执行预测
                predictions = self.forecast_system.predict_all(steps=steps)
                
                if not predictions:
                    messagebox.showerror("错误", "没有可用的模型进行预测")
                    return
                
                self.current_predictions = predictions
                
                # 更新图表
                self.plot_predictions(predictions)
                
                # 显示评估结果
                metrics = self.forecast_system.evaluate_models()
                self.show_metrics(metrics)
                
                # 显示预测结果
                self.show_forecast_details(predictions)
                
                self.set_status("预测完成！")
            except Exception as e:
                messagebox.showerror("错误", f"预测时出错: {str(e)}")
            finally:
                self.predict_btn.config(state=tk.NORMAL)
        
        self.predict_btn.config(state=tk.DISABLED)
        threading.Thread(target=predict_task).start()
        
    def plot_predictions(self, predictions):
        self.ax.clear()
        
        # 绘制历史数据
        df = self.forecast_system.processed_df
        history_days = min(90, len(df))  # 显示最近90天历史数据
        history = df['exchange_rate'].iloc[-history_days:]
        self.ax.plot(history.index, history.values, label='Historical', color='blue', linewidth=2)
        
        # 绘制预测数据
        if predictions:
            steps = int(self.forecast_days.get())
            last_date = df.index[-1]
            pred_dates = pd.date_range(start=last_date, periods=steps+1)[1:]
            
            colors = {'ARIMA': 'red', 'Prophet': 'green', 'LSTM': 'orange'}
            
            for model_name in ['ARIMA', 'Prophet', 'LSTM']:
                if model_name in predictions:
                    try:
                        pred_data = predictions[model_name]['prediction']
                        if hasattr(pred_data, 'values'):
                            pred_data = pred_data.values
                        
                        # 确保数据长度匹配
                        if len(pred_data) >= steps:
                            pred_data = pred_data[:steps]
                        else:
                            continue
                            
                        self.ax.plot(pred_dates[:len(pred_data)], pred_data, 
                                   label=f'{model_name} Forecast', 
                                   color=colors[model_name],
                                   linestyle='--', 
                                   linewidth=2)
                        
                        # 绘制置信区间（如果有）
                        if model_name == 'ARIMA' and 'confidence_interval' in predictions[model_name]:
                            ci = predictions[model_name]['confidence_interval']
                            if len(ci) >= steps:
                                self.ax.fill_between(pred_dates[:steps], 
                                                   ci.iloc[:steps, 0], 
                                                   ci.iloc[:steps, 1],
                                                   color='gray', alpha=0.2)
                    except Exception as e:
                        print(f"绘制{model_name}预测失败: {e}")
        
        self.ax.set_title(f"{self.from_currency.get()}/{self.to_currency.get()} Exchange Rate Forecast")
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Exchange Rate')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        # 设置日期格式
        self.figure.autofmt_xdate()
        self.figure.tight_layout()
        self.canvas.draw()
        
    def show_metrics(self, metrics):
        self.metrics_text.delete(1.0, tk.END)
        if metrics:
            text_lines = []
            for model, scores in metrics.items():
                text_lines.append(f"{model}:")
                text_lines.append(f"  MAE: {scores['MAE']:.4f}")
                text_lines.append(f"  RMSE: {scores['RMSE']:.4f}")
                text_lines.append("")
            self.metrics_text.insert(tk.END, "\n".join(text_lines))
        else:
            self.metrics_text.insert(tk.END, "无法计算模型性能指标")
            
    def show_forecast_details(self, predictions):
        self.forecast_text.delete(1.0, tk.END)
        if not predictions:
            return
            
        steps = int(self.forecast_days.get())
        last_date = self.forecast_system.processed_df.index[-1]
        
        text_lines = []
        display_days = min(10, steps) # 显示前10天预测
        
        for i in range(display_days):
            date = (last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d')
            line = f"{date}: "
            values = []
            
            for model_name in ['ARIMA', 'Prophet', 'LSTM']:
                if model_name in predictions:
                    try:
                        pred = predictions[model_name]['prediction']
                        if hasattr(pred, 'iloc'):
                            value = pred.iloc[i]
                        elif hasattr(pred, 'values'):
                            value = pred.values[i]
                        else:
                            value = pred[i]
                        values.append(f"{model_name}={value:.4f}")
                    except:
                        continue
            
            if values:
                line += " | ".join(values)
                text_lines.append(line)
        
        if steps > 10:
            text_lines.append(f"\n... 还有 {steps-10} 天预测结果")
            
        self.forecast_text.insert(tk.END, "\n".join(text_lines))

# ==================== 主程序入口 ====================
def main():
    # 设置异常处理
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        messagebox.showerror(
            "错误",
            f"程序出现异常:\n{exc_type.__name__}: {exc_value}"
        )
    
    sys.excepthook = handle_exception
    
    # 创建主窗口
    root = tk.Tk()
    
    # 设置窗口图标（如果有）
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    
    # 创建应用
    app = ExchangeRateForecastApp(root)
    
    # 运行主循环
    root.mainloop()

if __name__ == "__main__":
    main()