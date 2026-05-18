# Bitcoin-HMM: Hidden Markov Model for Cryptocurrency Market Regime Detection

A sophisticated machine learning framework for detecting Bitcoin market regimes using Hidden Markov Models (HMM). This project identifies bull, sideways, and crash market conditions to inform trading strategies through advanced regime detection and backtesting.

## 📋 Project Overview

Bitcoin-HMM is a comprehensive toolkit for:
- **Market Regime Detection**: Classify Bitcoin price action into 3 distinct regimes (Crash, Sideways, Bull)
- **Continuous Model Adaptation**: Implements warm-start training with state label tracking to maintain consistency across retraining cycles
- **Walk-Forward Validation**: Simulates real-world trading conditions with proper train/test splits
- **Strategy Backtesting**: Evaluates HMM-based trading signals against buy-and-hold benchmark
- **Feature Engineering**: Multiple versions of technical indicators optimized for regime prediction

## 🎯 Key Features

### 1. **Smart Regime Tracking**
The `RegimeHMM` class (in `src/regime_HMM.py`) implements:
- **Warm Start Mechanism**: Continues training from previous parameters to ensure model convergence
- **Label Consistency**: Uses Hungarian Algorithm + Euclidean distance to maintain regime identity across retraining
- **Drift Tracking**: Adapts reference state profiles with 80/20 exponential weighting to handle market evolution
- **Three Market States**:
  - **Regime 0 (Crash)**: High volatility, negative returns - SELL signal
  - **Regime 1 (Sideways)**: Low returns, moderate volatility - HOLD signal
  - **Regime 2 (Bull)**: Positive returns, low volatility - BUY signal

### 2. **Data Pipeline**
- **Data Loader** (`src/data_loader.py`): Fetches Bitcoin OHLCV data from Coinbase via CCXT library (2015-present)
- **Feature Engineering** (`src/features.py`): Multiple feature versions
  - v1: Basic returns, volatility, trend slope, volume z-score
  - v2: Normalized metrics and price distance from MA
  - v3: Enhanced with fast volatility and daily range
  - Micro Features: Aggressive 1H-level features for short-term anomalies

### 3. **Technical Indicators**
All features leverage industry-standard technical analysis:
- **Log Returns**: Normalized price changes for statistical stability
- **Volatility**: Rolling standard deviation (fast & regular)
- **Trend Slope**: SMA-based trend strength as percentage
- **Volume Z-Score**: Normalized volume deviation to detect unusual activity
- **Price Distance**: Relative positioning from moving average (overbought/oversold)
- **Momentum**: Rate of Change (ROC) over multiple horizons (3H, 6H, 12H)
- **Candlestick Wicks**: Upper/lower wick ratios for price rejection signals
- **EMA Cross Signals**: Exponential moving average crossover indicators

### 4. **Model Validation & Backtesting**
- **Walk-Forward Validation** (`src/walk_forward_validation.py`): Rolling train/test windows
- **Backtester** (`src/backtester.py`): Simulates trading strategy with:
  - No look-ahead bias (1-day lag on signals)
  - Maximum Drawdown calculation
  - Cumulative return curves
  - Performance metrics vs Buy-and-Hold

## 📁 Repository Structure

```
Bitcoin-HMM/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore rules
│
├── data/                                  # Historical price data & features
│   ├── btc_usd_coinbase.csv              # Raw OHLCV data
│   ├── btc_usd_coinbase_features_v2.csv  # Features v2
│   └── btc_usd_coinbase_features_with_wf.csv  # With walk-forward labels
│
├── models/                                # Trained HMM models
│   ├── model_btc_v1.joblib               # HMM model version 1
│   ├── model_btc_v2.joblib               # HMM model version 2
│   ├── model_btc_v3.joblib               # HMM model version 3
│   └── scaler.joblib                     # StandardScaler for feature normalization
│
├── src/                                   # Core source code
│   ├── data_loader.py                    # OHLCV data fetching from CCXT
│   ├── features.py                       # Feature engineering (v1, v2, v3, micro)
│   ├── regime_HMM.py                     # Main HMM wrapper class
│   ├── walk_forward_validation.py        # Rolling window validation
│   └── backtester.py                     # Strategy backtesting engine
│
└── notebooks/                             # Jupyter analysis notebooks
    ├── 01_data_and_features.ipynb        # Data loading & feature exploration
    ├── 02_hmm_in_sample.ipynb            # HMM training & regime visualization
    ├── 03_walk_forward_validation.ipynb  # Walk-forward testing methodology
    └── 04_backtest_strategy.ipynb        # Strategy performance analysis
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Elvin-Aurelio/Bitcoin-HMM.git
cd Bitcoin-HMM

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
scikit-learn>=1.7.2      # Machine learning utilities & HMM
hmmlearn>=0.3.3          # Hidden Markov Model implementation
matplotlib>=3.10.8       # Data visualization
ccxt==4.5.42             # Cryptocurrency data fetching
```

### Basic Usage

```python
from src.regime_HMM import RegimeHMM
from src.data_loader import fetch_full_history
from src.features import add_features_v2

# 1. Fetch Bitcoin data
df = fetch_full_history(symbol='BTC/USD', time_frame='1h')

# 2. Add features
df = add_features_v2(df)

# 3. Initialize and train HMM
hmm = RegimeHMM(n_components=3, random_state=42)
hmm.fit(df, feature_cols=['log_return', 'volatility', 'trend_slope', 'volume_zscore'])

# 4. Predict regimes
regimes = hmm.predict(df, feature_cols=['log_return', 'volatility', 'trend_slope', 'volume_zscore'])
regime_proba = hmm.predict_proba(df, feature_cols=['log_return', 'volatility', 'trend_slope', 'volume_zscore'])
```

## 📊 Notebooks Overview

### 01_data_and_features.ipynb
- Downloads historical Bitcoin data from Coinbase (2015-present)
- Explores raw OHLCV data structure
- Demonstrates feature engineering pipeline
- Visualizes technical indicators

### 02_hmm_in_sample.ipynb
- Trains HMM on full dataset
- Visualizes detected market regimes
- Shows regime probability distributions
- Analyzes state transition patterns

### 03_walk_forward_validation.ipynb
- Implements rolling train/test windows (walk-forward analysis)
- Demonstrates model adaptation across time periods
- Compares in-sample vs out-of-sample performance
- Shows label consistency maintenance

### 04_backtest_strategy.ipynb
- Backtests regime-based trading strategy
- Calculates performance metrics (returns, drawdown, Sharpe ratio)
- Compares HMM strategy vs Buy-and-Hold benchmark
- Visualizes equity curves and regime signals

## 🧠 HMM Architecture Details

### State Mapping Logic
States are assigned based on Risk/Reward ratio (Return/Volatility):
1. **Cold Start** (first training):
   - Calculate risk/reward score for each HMM state
   - Assign labels: 0=Lowest score (Crash), 1=Middle (Sideways), 2=Highest (Bull)
   - Store reference profiles (means)

2. **Warm Start** (subsequent trainings):
   - Measure Euclidean distance between new and reference states
   - Apply Hungarian Algorithm for optimal 1-1 mapping
   - Update references with exponential weighting (α=0.2) for drift tracking
   - Ensures consistent regime labels despite parameter changes

### Probability Remapping
The `predict_proba()` method remaps raw HMM state probabilities to logical regime labels, so:
- Regime 0 probability = Crash likelihood
- Regime 1 probability = Sideways likelihood  
- Regime 2 probability = Bull likelihood

## 📈 Key Performance Metrics

The backtester calculates:
- **Final Portfolio Value**: Cumulative return of strategy
- **Buy & Hold Benchmark**: Passive Bitcoin holding
- **Max Drawdown**: Largest peak-to-trough decline
- **Regime Accuracy**: Measured via out-of-sample testing

## 🔄 Warm Start Mechanism

Why warm start matters:
- **Problem**: Retraining from scratch causes regime labels to shuffle randomly
- **Solution**: Initialize new HMM with previous parameters + Hungarian Algorithm matching
- **Benefit**: Smooth transitions, stable backtesting, consistent decision-making

```
Iteration 1 (Cold): Find initial regime profiles
    ↓
Iteration 2+ (Warm): 
    - Train new HMM
    - Match new states → reference states (minimize distance)
    - Update references gradually (drift tracking)
    - Maintain label consistency
```

## 📝 Feature Engineering Versions

| Version | Returns | Volatility | Trend | Volume | Distance MA | Daily Range | Momentum |
|---------|---------|------------|-------|--------|-------------|-------------|----------|
| v1      | ✓       | ✓ (21-day) | ✓     | ✓      | -           | -           | -        |
| v2      | ✓       | ✓ (21-day) | ✓ %   | ✓      | ✓           | -           | -        |
| v3      | ✓       | ✓ (7-day)  | ✓ %   | ✓      | ✓           | ✓           | -        |
| Micro   | ✓ (1H)  | ✓ (12/24H) | -     | ✓      | ✓           | ✓           | ✓ (3/6/12H) |

## 🛠️ Development & Customization

### Adding New Features
Edit `src/features.py`:
```python
def add_features_custom(df):
    data = df.copy()
    # Add your feature calculation here
    data['my_feature'] = ...
    data.dropna(inplace=True)
    return data
```

### Adjusting HMM Parameters
```python
hmm = RegimeHMM(
    n_components=3,        # Number of regimes (3 = Crash, Sideways, Bull)
    random_state=42        # Reproducibility seed
)
```

### Modifying Walk-Forward Windows
Edit `src/walk_forward_validation.py` to adjust:
- Train window size
- Test window size
- Step size (overlap)

## 📚 Related Literature

This project implements concepts from:
- **Hidden Markov Models**: Rabiner (1989), Viterbi algorithm
- **Regime Detection**: Hamilton (1989), regime-switching models
- **Cryptocurrency Trading**: Atsalakis & Valavanis (2009)
- **Hungarian Algorithm**: Munkres (1957), optimal assignment

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. 
- Past performance does not guarantee future results
- Use at your own risk with real capital
- Always perform thorough backtesting before live trading

## 📄 License

Open source - feel free to use, modify, and distribute

## 👤 Author

**Elvin-Aurelio**  
GitHub: [@Elvin-Aurelio](https://github.com/Elvin-Aurelio)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

**Last Updated**: May 2026  
**Bitcoin Data Range**: 2015-01-01 to Present  
**Data Source**: Coinbase (via CCXT)
