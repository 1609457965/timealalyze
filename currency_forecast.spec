# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['currency_forecast.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.utils._sorting',
        'sklearn.utils._vector_sentinel',
        'sklearn.neighbors._partition_nodes',
        'sklearn.metrics._pairwise_distances_reduction._datasets_pair',
        'sklearn.metrics._pairwise_distances_reduction._middle_term_computer',
        'prophet.models',
        'prophet.forecaster',
        'matplotlib.backends.backend_tkagg',
        'tensorflow',
        'pandas',
        'numpy',
        'yfinance',
        'statsmodels.tsa.stattools',
        'statsmodels.tsa.arima.model',
        'joblib'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='汇率预测系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # False表示不显示控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico'  # 如果有图标文件
)