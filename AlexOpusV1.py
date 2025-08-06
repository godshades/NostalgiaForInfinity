import logging
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import talib.abstract as ta
from scipy.signal import argrelextrema

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter
from freqtrade.persistence import Trade

# Import our new modules
from .strategy_constants import *
from .strategy_helpers import MarketAnalyzer, SignalGenerator, RiskManager, PerformanceOptimizer
from .trade_state_manager import TradeStateManager, TradeState
from .mml_exit_system import MMLExitSystem

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)

class AlexOpusV1(IStrategy):
    """
    Improved version with better organization, performance, and maintainability
    """
    
    # === STRATEGY CONFIGURATION ===
    timeframe = "15m"
    startup_candle_count: int = 100
    stoploss = -0.15
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    use_custom_stoploss = False
    stoploss_on_exchange = True
    stoploss_on_exchange_interval = STOPLOSS_CHECK_INTERVAL
    position_adjustment_enable = True
    can_short = True
    use_exit_signal = True
    ignore_roi_if_entry_signal = True
    process_only_new_candles = True
    
    # === INITIALIZE COMPONENTS ===
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.state_manager = TradeStateManager()
        self.market_analyzer = MarketAnalyzer()
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager()
        self.performance_optimizer = PerformanceOptimizer()
        self.mml_exit_system = MMLExitSystem(
            use_emergency_exits=self.use_emergency_exits.value
        )
        
        # Cache for expensive calculations
        self._indicator_cache = {}
        self._cache_max_size = 100
    
    # === HYPERPARAMETERS (Reduced and organized) ===
    
    # Core Strategy Parameters
    confluence_threshold = DecimalParameter(2.0, 4.0, default=3.0, space="buy")
    momentum_threshold = IntParameter(2, 4, default=3, space="buy")
    volume_threshold = DecimalParameter(1.1, 2.0, default=1.3, space="buy")
    
    # Risk Parameters
    risk_per_trade = DecimalParameter(0.01, 0.03, default=0.02, space="sell")
    max_correlation = DecimalParameter(0.5, 0.9, default=0.7, space="sell")
    
    # Exit Parameters
    profit_target_multiplier = DecimalParameter(1.5, 3.0, default=2.0, space="sell")
    
    # Simplified ROI
    minimal_roi = {
        "0": 0.08,
        "10": 0.04,
        "30": 0.02,
        "60": 0.01
    }
    
    # Add parameter to choose exit system
    use_mml_exits = BooleanParameter(default=True, space="sell", optimize=False)
    use_simple_exits = BooleanParameter(default=False, space="sell", optimize=False)
    use_emergency_exits = BooleanParameter(default=True, space="sell", optimize=False)
    
    # 🚨  REGIME CHANGE DETECTION PARAMETERS (NEU)
    regime_change_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    regime_change_sensitivity = DecimalParameter(0.3, 0.8, default=0.5, decimals=2, space="sell", optimize=True, load=True)
    
    # Flash Move Detection
    flash_move_threshold = DecimalParameter(0.03, 0.08, default=0.05, decimals=3, space="sell", optimize=True, load=True)
    flash_move_candles = IntParameter(3, 10, default=5, space="sell", optimize=True, load=True)
    
    # Volume Spike Detection
    volume_spike_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    volume_spike_multiplier = DecimalParameter(2.0, 5.0, default=3.0, decimals=1, space="sell", optimize=True, load=True)
    
    # Emergency Exit Protection
    emergency_exit_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    emergency_exit_profit_threshold = DecimalParameter(0.005, 0.03, default=0.015, decimals=3, space="sell", optimize=True, load=True)
    
    # Market Sentiment Protection
    sentiment_protection_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    sentiment_shift_threshold = DecimalParameter(0.2, 0.4, default=0.3, decimals=2, space="sell", optimize=True, load=True)

    # 🔧ATR STOPLOSS PARAMETERS (Anpassbar machen)
    atr_stoploss_multiplier = DecimalParameter(0.8, 2.0, default=1.0, decimals=1, space="sell", optimize=True, load=True)
    atr_stoploss_minimum = DecimalParameter(-0.25, -0.10, default=-0.12, decimals=2, space="sell", optimize=True, load=True)
    atr_stoploss_maximum = DecimalParameter(-0.30, -0.15, default=-0.18, decimals=2, space="sell", optimize=True, load=True)
    atr_stoploss_ceiling = DecimalParameter(-0.10, -0.06, default=-0.06, decimals=2, space="sell", optimize=True, load=True)
    # DCA parameters
    initial_safety_order_trigger = DecimalParameter(
        low=-0.02, high=-0.01, default=-0.018, decimals=3, space="buy", optimize=True, load=True
    )
    max_safety_orders = IntParameter(1, 3, default=1, space="buy", optimize=True, load=True)
    safety_order_step_scale = DecimalParameter(
        low=1.05, high=1.5, default=1.25, decimals=2, space="buy", optimize=True, load=True
    )
    safety_order_volume_scale = DecimalParameter(
        low=1.1, high=2.0, default=1.4, decimals=1, space="buy", optimize=True, load=True
    )
    h2 = IntParameter(20, 60, default=40, space="buy", optimize=True, load=True)
    h1 = IntParameter(10, 40, default=20, space="buy", optimize=True, load=True)
    h0 = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)
    cp = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)

    # Entry parameters
    increment_for_unique_price = DecimalParameter(
        low=1.0005, high=1.002, default=1.001, decimals=4, space="buy", optimize=True, load=True
    )
    last_entry_price: Optional[float] = None

    # Protection parameters
    cooldown_lookback = IntParameter(2, 48, default=1, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=4, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # Dynamic Leverage parameters
    leverage_window_size = IntParameter(20, 100, default=50, space="buy", optimize=True, load=True)
    leverage_base = DecimalParameter(5.0, 20.0, default=5.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_rsi_low = DecimalParameter(20.0, 40.0, default=30.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_rsi_high = DecimalParameter(60.0, 80.0, default=70.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_long_increase_factor = DecimalParameter(1.1, 2.0, default=1.5, decimals=1, space="buy", optimize=True,
                                                     load=True)
    leverage_long_decrease_factor = DecimalParameter(0.3, 0.9, default=0.5, decimals=1, space="buy", optimize=True,
                                                     load=True)
    leverage_volatility_decrease_factor = DecimalParameter(0.5, 0.95, default=0.8, decimals=2, space="buy",
                                                           optimize=True, load=True)
    leverage_atr_threshold_pct = DecimalParameter(0.01, 0.05, default=0.03, decimals=3, space="buy", optimize=True,
                                                  load=True)

    # Indicator parameters
    indicator_extrema_order = IntParameter(3, 15, default=8, space="buy", optimize=True, load=True)  # War 5
    indicator_mml_window = IntParameter(50, 200, default=50, space="buy", optimize=True, load=True)  # War 50
    indicator_rolling_window_threshold = IntParameter(20, 100, default=50, space="buy", optimize=True, load=True)  # War 20
    indicator_rolling_check_window = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)  # War 5


    
    # Market breadth parameters
    market_breadth_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    market_breadth_threshold = DecimalParameter(0.3, 0.6, default=0.45, space="buy", optimize=True)
    
    # Total market cap parameters
    total_mcap_filter_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    total_mcap_ma_period = IntParameter(20, 100, default=50, space="buy", optimize=True)
    
    # Market regime parameters
    regime_filter_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    regime_lookback_period = IntParameter(24, 168, default=48, space="buy", optimize=True)  # hours
    
    # Fear & Greed parameters
    fear_greed_enabled = BooleanParameter(default=False, space="buy", optimize=True)  # Optional
    fear_greed_extreme_threshold = IntParameter(20, 30, default=25, space="buy", optimize=True)
    fear_greed_greed_threshold = IntParameter(70, 80, default=75, space="buy", optimize=True)
    # Momentum
    avoid_strong_trends = BooleanParameter(default=True, space="buy", optimize=True)
    trend_strength_threshold = DecimalParameter(0.01, 0.05, default=0.02, space="buy", optimize=True)
    momentum_confirmation_candles = IntParameter(1, 5, default=2, space="buy", optimize=True)

    # Dynamic exit based on entry quality
    dynamic_exit_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    exit_on_confluence_loss = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    exit_on_structure_break = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    
    # Profit target multipliers based on entry type
    high_quality_profit_multiplier = DecimalParameter(1.2, 3.0, default=2.0, space="sell", optimize=True, load=True)
    medium_quality_profit_multiplier = DecimalParameter(1.0, 2.5, default=1.5, space="sell", optimize=True, load=True)
    backup_profit_multiplier = DecimalParameter(0.8, 2.0, default=1.2, space="sell", optimize=True, load=True)
    
    # Advanced exit thresholds
    volume_decline_exit_threshold = DecimalParameter(0.3, 0.8, default=0.5, space="sell", optimize=True, load=True)
    momentum_decline_exit_threshold = IntParameter(1, 4, default=2, space="sell", optimize=True, load=True)
    structure_deterioration_threshold = DecimalParameter(-3.0, 0.0, default=-1.5, space="sell", optimize=True, load=True)
    
    # RSI exit levels
    rsi_overbought_exit = IntParameter(70, 85, default=75, space="sell", optimize=True, load=True)
    rsi_divergence_exit_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    
    # Trailing stop improvements
    use_advanced_trailing = BooleanParameter(default=False, space="sell", optimize=False, load=True)
    trailing_stop_positive_offset_high_quality = DecimalParameter(0.02, 0.08, default=0.04, space="sell", optimize=True, load=True)
    trailing_stop_positive_offset_medium_quality = DecimalParameter(0.015, 0.06, default=0.03, space="sell", optimize=True, load=True)
    
    # === NEUE ADVANCED PARAMETERS ===
    # Confluence Analysis
    confluence_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    confluence_threshold = DecimalParameter(2.0, 4.0, default=2.5, space="buy", optimize=True, load=True)  # War 3.0
    
    # Volume Analysis
    volume_analysis_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    volume_strength_threshold = DecimalParameter(1.1, 2.0, default=1.3, space="buy", optimize=True, load=True)
    volume_pressure_threshold = IntParameter(1, 3, default=1, space="buy", optimize=True, load=True)  # War 2

    
    # Momentum Analysis
    momentum_analysis_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    momentum_quality_threshold = IntParameter(2, 4, default=2, space="buy", optimize=True, load=True)  # War 3
    
    # Market Structure Analysis
    structure_analysis_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    structure_score_threshold = DecimalParameter(-2.0, 5.0, default=0.5, space="buy", optimize=True, load=True)
    
    # Ultimate Score
    ultimate_score_threshold = DecimalParameter(0.5, 3.0, default=1.5, space="buy", optimize=True, load=True)
    
    # Advanced Entry Filters
    require_volume_confirmation = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    require_momentum_confirmation = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    require_structure_confirmation = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Optimized indicator calculation"""
        
        # Check cache
        cache_key = f"{metadata['pair']}_{len(dataframe)}"
        if cache_key in self._indicator_cache:
            cached_df = self._indicator_cache[cache_key]
            if len(cached_df) == len(dataframe):
                return cached_df
        
        # === BATCH CALCULATE BASIC INDICATORS ===
        dataframe = self.performance_optimizer.batch_calculate_indicators(
            dataframe, 
            ['sma', 'rsi', 'volume_profile']
        )
        
        # === CORE INDICATORS ===
        dataframe['ema50'] = ta.EMA(dataframe['close'], timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe['close'], timeperiod=100)
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'])
        
        # === MARKET STRUCTURE ===
        support, resistance = self.market_analyzer.calculate_support_resistance(dataframe)
        dataframe['support'] = support
        dataframe['resistance'] = resistance
        dataframe['trend_change'] = self.market_analyzer.detect_trend_change(dataframe)
        
        dataframe["avg_volume"] = dataframe["volume"].rolling(window=50).mean()
        
        # === TREND STRENGTH INDICATORS ===
        def calc_slope(series, period):
            """Calculate linear regression slope"""
            if len(series) < period:
                return 0
            x = np.arange(period)
            y = series.values
            if np.isnan(y).any():
                return 0
            try:
                slope = np.polyfit(x, y, 1)[0]
                return slope
            except:
                return 0

        dataframe['slope_5'] = dataframe['close'].rolling(5).apply(lambda x: calc_slope(x, 5), raw=False)
        dataframe['slope_10'] = dataframe['close'].rolling(10).apply(lambda x: calc_slope(x, 10), raw=False)
        dataframe['slope_20'] = dataframe['close'].rolling(20).apply(lambda x: calc_slope(x, 20), raw=False)

        dataframe['trend_strength_5'] = dataframe['slope_5'] / dataframe['close'] * 100
        dataframe['trend_strength_10'] = dataframe['slope_10'] / dataframe['close'] * 100
        dataframe['trend_strength_20'] = dataframe['slope_20'] / dataframe['close'] * 100

        dataframe['trend_strength'] = (dataframe['trend_strength_5'] + dataframe['trend_strength_10'] + dataframe['trend_strength_20']) / 3

        strong_threshold = 0.02
        dataframe['strong_uptrend'] = dataframe['trend_strength'] > strong_threshold
        dataframe['strong_downtrend'] = dataframe['trend_strength'] < -strong_threshold
        dataframe['ranging'] = dataframe['trend_strength'].abs() < (strong_threshold * 0.5)
        
        # === ADVANCED INDICATORS (Only if needed) ===
        if metadata['pair'] in DEBUG_PAIRS or self.dp.runmode.value in ['live', 'dry_run']:
            dataframe = self._calculate_advanced_indicators(dataframe)
        
        # Cache result
        self._update_cache(cache_key, dataframe)
        
        return dataframe
    
    def _calculate_advanced_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all advanced indicators from your original strategy
        Optimized for performance with caching and vectorization
        """
        
        # === EXTREMA DETECTION (Optimized) ===
        dataframe = self._calculate_extrema_indicators(dataframe)
        
        # === MURREY MATH LEVELS (Cached) ===
        dataframe = self._calculate_mml_indicators(dataframe)
        
        # === ADVANCED MARKET ANALYSIS ===
        if self.confluence_enabled.value:
            dataframe = self._calculate_confluence_indicators(dataframe)
        
        if self.volume_analysis_enabled.value:
            dataframe = self._calculate_volume_indicators(dataframe)
        
        if self.momentum_analysis_enabled.value:
            dataframe = self._calculate_momentum_indicators(dataframe)
        
        if self.structure_analysis_enabled.value:
            dataframe = self._calculate_structure_indicators(dataframe)
        
        # === REGIME DETECTION ===
        if self.regime_change_enabled.value:
            dataframe = self._calculate_regime_indicators(dataframe)
        
        # === ENTRY/EXIT SIGNALS ===
        dataframe = self._calculate_signal_indicators(dataframe)
        
        # === ULTIMATE SCORE ===
        dataframe = self._calculate_ultimate_score(dataframe)
        
        return dataframe
    
    def _calculate_extrema_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Calculate extrema and related indicators (from your original code)"""
        
        # Basic extrema detection
        extrema_order = self.indicator_extrema_order.value
        
        # Vectorized extrema detection
        dataframe["maxima"] = (
            dataframe["close"] == dataframe["close"].shift(1).rolling(window=extrema_order).max()
        ).astype(np.int8)
        
        dataframe["minima"] = (
            dataframe["close"] == dataframe["close"].shift(1).rolling(window=extrema_order).min()
        ).astype(np.int8)
        
        # S_extrema indicator
        dataframe["s_extrema"] = 0
        dataframe.loc[dataframe["minima"] == 1, "s_extrema"] = -1
        dataframe.loc[dataframe["maxima"] == 1, "s_extrema"] = 1
        
        # Heikin-Ashi
        dataframe["ha_close"] = (
            dataframe["open"] + dataframe["high"] + 
            dataframe["low"] + dataframe["close"]
        ) / 4
        
        # Rolling extrema (optimized version)
        dataframe = self._calculate_rolling_extrema_vectorized(dataframe)
        
        # DI indicators
        dataframe["plus_di"] = ta.PLUS_DI(dataframe)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe)
        dataframe["DI_values"] = dataframe["plus_di"] - dataframe["minus_di"]
        dataframe["DI_cutoff"] = 0
        dataframe["DI_catch"] = np.where(
            dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1
        ).astype(np.int8)
        
        # Rolling thresholds
        rolling_window = self.indicator_rolling_window_threshold.value
        dataframe["minima_sort_threshold"] = dataframe["close"].rolling(
            window=rolling_window, min_periods=1
        ).min()
        
        dataframe["maxima_sort_threshold"] = dataframe["close"].rolling(
            window=rolling_window, min_periods=1
        ).max()
        
        # Extrema checks
        check_window = self.indicator_rolling_check_window.value
        dataframe["minima_check"] = (
            dataframe["minima"].rolling(window=check_window, min_periods=1).sum() == 0
        ).astype(np.int8)
        
        dataframe["maxima_check"] = (
            dataframe["maxima"].rolling(window=check_window, min_periods=1).sum() == 0
        ).astype(np.int8)
        
        return dataframe
    
    def _calculate_rolling_extrema_vectorized(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Vectorized rolling extrema calculation (massive performance improvement)"""
        
        # Pre-calculate all windows at once
        windows = {
            'h2': self.h2.value,
            'h1': self.h1.value,
            'h0': self.h0.value,
            'cp': self.cp.value
        }
        
        for name, window in windows.items():
            # Vectorized min/max detection
            rolling_min = dataframe['ha_close'].rolling(window=window, center=True).min()
            rolling_max = dataframe['ha_close'].rolling(window=window, center=True).max()
            
            is_min = (dataframe['ha_close'] == rolling_min) & (dataframe['ha_close'] != dataframe['ha_close'].shift(1))
            is_max = (dataframe['ha_close'] == rolling_max) & (dataframe['ha_close'] != dataframe['ha_close'].shift(1))
            
            dataframe[f'min{name}'] = np.where(is_min, -window, 0).astype(np.int16)
            dataframe[f'max{name}'] = np.where(is_max, window, 0).astype(np.int16)
        
        return dataframe
    
    def _calculate_mml_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Calculate Murrey Math Levels with caching"""
        
        # Use the optimized MML calculation
        mml_window = self.indicator_mml_window.value
        murrey_levels = self.calculate_rolling_murrey_math_levels_optimized(
            dataframe, 
            window_size=mml_window
        )
        
        # Add all MML levels to dataframe
        for level_name in MML_LEVEL_NAMES:
            if level_name in murrey_levels:
                dataframe[level_name] = murrey_levels[level_name]
            else:
                dataframe[level_name] = dataframe["close"]
        
        # MML Oscillator
        mml_4_8 = dataframe.get("[4/8]P")
        mml_plus_3_8 = dataframe.get("[+3/8]P")
        mml_minus_3_8 = dataframe.get("[-3/8]P")
        
        if all(x is not None for x in [mml_4_8, mml_plus_3_8, mml_minus_3_8]):
            osc_denominator = (mml_plus_3_8 - mml_minus_3_8).replace(0, np.nan)
            dataframe["mmlextreme_oscillator"] = 100 * (
                (dataframe["close"] - mml_4_8) / osc_denominator
            )
        else:
            dataframe["mmlextreme_oscillator"] = 0
        
        return dataframe
    
    def calculate_rolling_murrey_math_levels_optimized(self, df: pd.DataFrame, window_size: int) -> Dict[str, pd.Series]:
        """
        OPTIMIZED Version - Calculate MML levels every 5 candles using only past data
        """
        murrey_levels_data: Dict[str, list] = {key: [np.nan] * len(df) for key in MML_LEVEL_NAMES}
        mml_c1 = self.mml_const1.value
        mml_c2 = self.mml_const2.value
        
        calculation_step = 5
        
        for i in range(0, len(df), calculation_step):
            if i < window_size:
                continue
                
            # Use data up to the previous candle for the rolling window
            window_end = i - 1
            window_start = window_end - window_size + 1
            if window_start < 0:
                window_start = 0
                
            window_data = df.iloc[window_start:window_end]
            mn_period = window_data["low"].min()
            mx_period = window_data["high"].max()
            current_close = df["close"].iloc[window_end] if window_end > 0 else df["close"].iloc[0]
            
            if pd.isna(mn_period) or pd.isna(mx_period) or mn_period == mx_period:
                for key in MML_LEVEL_NAMES:
                    murrey_levels_data[key][window_end] = current_close
                continue
                
            levels = self._calculate_mml_core(mn_period, mx_period, mx_period, mn_period, mml_c1, mml_c2)
            
            for key in MML_LEVEL_NAMES:
                murrey_levels_data[key][window_end] = levels.get(key, current_close)
        
        # Interpolate using only past data up to each point
        for key in MML_LEVEL_NAMES:
            series = pd.Series(murrey_levels_data[key], index=df.index)
            # Interpolate forward only up to the current point, avoiding future data
            series = series.expanding().mean().ffill()  # Use expanding mean as a safe alternative
            murrey_levels_data[key] = series.tolist()
        
        return {key: pd.Series(data, index=df.index) for key, data in murrey_levels_data.items()}
        
    def _calculate_confluence_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Calculate confluence score and related indicators"""
        
        # Support/Resistance Confluence
        dataframe['near_support'] = (
            (dataframe['close'] <= dataframe['minima_sort_threshold'] * 1.02) &
            (dataframe['close'] >= dataframe['minima_sort_threshold'] * 0.98)
        ).astype(np.int8)
        
        dataframe['near_resistance'] = (
            (dataframe['close'] <= dataframe['maxima_sort_threshold'] * 1.02) &
            (dataframe['close'] >= dataframe['maxima_sort_threshold'] * 0.98)
        ).astype(np.int8)
        
        # MML Level Confluence (vectorized)
        mml_levels = ['[0/8]P', '[2/8]P', '[4/8]P', '[6/8]P', '[8/8]P']
        near_mml_sum = np.zeros(len(dataframe), dtype=np.int8)
        
        for level in mml_levels:
            if level in dataframe.columns:
                level_values = dataframe[level].values
                near_level = (
                    (dataframe['close'].values <= level_values * 1.015) &
                    (dataframe['close'].values >= level_values * 0.985)
                )
                near_mml_sum += near_level.astype(np.int8)
        
        dataframe['near_mml'] = near_mml_sum
        
        # Volume Confluence
        dataframe['volume_spike'] = (
            dataframe['volume'] > dataframe['avg_volume'] * VOLUME_SPIKE_MULTIPLIER
        ).astype(np.int8)
        
        # RSI Confluence Zones
        dataframe['rsi_oversold'] = (dataframe['rsi'] < RSI_OVERSOLD).astype(np.int8)
        dataframe['rsi_overbought'] = (dataframe['rsi'] > RSI_OVERBOUGHT).astype(np.int8)
        dataframe['rsi_neutral'] = (
            (dataframe['rsi'] >= 40) & (dataframe['rsi'] <= 60)
        ).astype(np.int8)
        
        # EMA Confluence
        dataframe['above_ema'] = (dataframe['close'] > dataframe['ema50']).astype(np.int8)
        
        # CONFLUENCE SCORE (0-6)
        dataframe['confluence_score'] = (
            dataframe['near_support'] +
            dataframe['near_mml'].clip(0, 2) +
            dataframe['volume_spike'] +
            dataframe['rsi_oversold'] +
            dataframe['above_ema'] +
            (dataframe['trend_strength'] > 0.01).astype(np.int8)
        )
        
        return dataframe
    
    def _calculate_volume_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Calculate smart volume indicators"""
        
        # Volume-Price Trend (VPT) - vectorized
        price_change_pct = dataframe['close'].pct_change()
        dataframe['vpt'] = (dataframe['volume'] * price_change_pct).fillna(0).cumsum()
        
        # Volume moving averages
        dataframe['volume_sma20'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_sma50'] = dataframe['volume'].rolling(50).mean()
        
        # Volume strength
        dataframe['volume_strength'] = dataframe['volume'] / dataframe['volume_sma20']
        
        # Smart money indicators (vectorized)
        green_candle = dataframe['close'] > dataframe['open']
        red_candle = ~green_candle
        high_volume = dataframe['volume'] > dataframe['volume_sma20'] * 1.2
        upper_half = dataframe['close'] > (dataframe['high'] + dataframe['low']) / 2
        lower_half = ~upper_half
        
        dataframe['accumulation'] = (
            green_candle & high_volume & upper_half
        ).astype(np.int8)
        
        dataframe['distribution'] = (
            red_candle & high_volume & lower_half
        ).astype(np.int8)
        
        # Buying/Selling pressure
        dataframe['buying_pressure'] = dataframe['accumulation'].rolling(5).sum()
        dataframe['selling_pressure'] = dataframe['distribution'].rolling(5).sum()
        
        # Net volume pressure
        dataframe['volume_pressure'] = dataframe['buying_pressure'] - dataframe['selling_pressure']
        
        # Volume trend
        dataframe['volume_trend'] = (
            dataframe['volume_sma20'] > dataframe['volume_sma50']
        ).astype(np.int8)
        
        # Money Flow Index (vectorized)
        typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        money_flow = typical_price * dataframe['volume']
        
        # Vectorized positive/negative flow
        price_up = typical_price > typical_price.shift(1)
        positive_flow = np.where(price_up, money_flow, 0)
        negative_flow = np.where(~price_up, money_flow, 0)
        
        positive_flow_sum = pd.Series(positive_flow).rolling(14).sum()
        negative_flow_sum = pd.Series(negative_flow).rolling(14).sum()
        
        dataframe['money_flow_ratio'] = positive_flow_sum / (negative_flow_sum + 1e-10)
        dataframe['money_flow_index'] = 100 - (100 / (1 + dataframe['money_flow_ratio']))
        
        return dataframe
    
    def _calculate_momentum_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced momentum indicators"""
        
        # Multi-timeframe momentum
        for period, days in [(3, 6), (7, 14), (14, 28), (21, 21)]:
            dataframe[f'momentum_{period}'] = dataframe['close'].pct_change(days)
        
        # Momentum acceleration
        dataframe['momentum_acceleration'] = (
            dataframe['momentum_3'] - dataframe['momentum_3'].shift(3)
        )
        
        # Momentum consistency
        dataframe['momentum_consistency'] = (
            (dataframe['momentum_3'] > 0).astype(np.int8) +
            (dataframe['momentum_7'] > 0).astype(np.int8) +
            (dataframe['momentum_14'] > 0).astype(np.int8)
        )
        
        # Momentum divergence with volume
        dataframe['price_momentum_rank'] = dataframe['momentum_7'].rolling(20).rank(pct=True)
        dataframe['volume_momentum_rank'] = dataframe['volume_strength'].rolling(20).rank(pct=True)
        
        dataframe['momentum_divergence'] = (
            dataframe['price_momentum_rank'] - dataframe['volume_momentum_rank']
        ).abs()
        
        # Momentum strength
        dataframe['momentum_strength'] = (
            dataframe['momentum_3'].abs() +
            dataframe['momentum_7'].abs() +
            dataframe['momentum_14'].abs()
        ) / 3
        
        # Momentum quality score (0-5)
        dataframe['momentum_quality'] = (
            (dataframe['momentum_3'] > 0).astype(np.int8) +
            (dataframe['momentum_7'] > 0).astype(np.int8) +
            (dataframe['momentum_acceleration'] > 0).astype(np.int8) +
            (dataframe['volume_strength'] > 1.1).astype(np.int8) +
            (dataframe['momentum_divergence'] < 0.3).astype(np.int8)
        )
        
        # Rate of Change
        for period in [5, 10, 20]:
            dataframe[f'roc_{period}'] = dataframe['close'].pct_change(period) * 100
        
        # Momentum oscillator
        dataframe['momentum_oscillator'] = (
            dataframe['roc_5'] + dataframe['roc_10'] + dataframe['roc_20']
        ) / 3
        
        # Price momentum for other calculations
        dataframe['price_momentum'] = dataframe['close'].pct_change(3)
        dataframe['momentum_increasing'] = (
            dataframe['price_momentum'] > dataframe['price_momentum'].shift(1)
        )
        dataframe['momentum_decreasing'] = (
            dataframe['price_momentum'] < dataframe['price_momentum'].shift(1)
        )
        
        return dataframe
    
    def _calculate_structure_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Calculate market structure indicators"""
        
        # Higher highs, higher lows detection (vectorized)
        high_1 = dataframe['high'].shift(1)
        high_2 = dataframe['high'].shift(2)
        low_1 = dataframe['low'].shift(1)
        low_2 = dataframe['low'].shift(2)
        
        dataframe['higher_high'] = (
            (dataframe['high'] > high_1) & (high_1 > high_2)
        ).astype(np.int8)
        
        dataframe['higher_low'] = (
            (dataframe['low'] > low_1) & (low_1 > low_2)
        ).astype(np.int8)
        
        dataframe['lower_high'] = (
            (dataframe['high'] < high_1) & (high_1 < high_2)
        ).astype(np.int8)
        
        dataframe['lower_low'] = (
            (dataframe['low'] < low_1) & (low_1 < low_2)
        ).astype(np.int8)
        
        # Market structure scores
        dataframe['bullish_structure'] = (
            dataframe['higher_high'].rolling(5).sum() +
            dataframe['higher_low'].rolling(5).sum()
        )
        
        dataframe['bearish_structure'] = (
            dataframe['lower_high'].rolling(5).sum() +
            dataframe['lower_low'].rolling(5).sum()
        )
        
        dataframe['structure_score'] = (
            dataframe['bullish_structure'] - dataframe['bearish_structure']
        )
        
        # Swing highs and lows
        dataframe['swing_high'] = (
            (dataframe['high'] > high_1) &
            (dataframe['high'] > dataframe['high'].shift(-1))
        ).astype(np.int8)
        
        dataframe['swing_low'] = (
            (dataframe['low'] < low_1) &
            (dataframe['low'] < dataframe['low'].shift(-1))
        ).astype(np.int8)
        
        # Market structure breaks
        swing_highs = dataframe['high'].where(dataframe['swing_high'] == 1)
        swing_lows = dataframe['low'].where(dataframe['swing_low'] == 1)
        
        # Structure break detection
        dataframe['structure_break_up'] = (
            dataframe['close'] > swing_highs.ffill()
        ).astype(np.int8)
        
        dataframe['structure_break_down'] = (
            dataframe['close'] < swing_lows.ffill()
        ).astype(np.int8)
        
        # Trend strength based on structure
        dataframe['structure_trend_strength'] = (
            dataframe['structure_score'] / 10
        ).clip(-1, 1)
        
        # Support and resistance strength
        dataframe['support_strength'] = dataframe['swing_low'].rolling(20).sum()
        dataframe['resistance_strength'] = dataframe['swing_high'].rolling(20).sum()
        
        return dataframe
    
    def _calculate_regime_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Calculate regime change detection indicators"""
        
        flash_candles = self.flash_move_candles.value
        flash_threshold = self.flash_move_threshold.value
        
        # Flash move detection
        dataframe['price_change_fast'] = dataframe['close'].pct_change(flash_candles)
        dataframe['flash_pump'] = dataframe['price_change_fast'] > flash_threshold
        dataframe['flash_dump'] = dataframe['price_change_fast'] < -flash_threshold
        dataframe['flash_move'] = dataframe['flash_pump'] | dataframe['flash_dump']
        
        # Volume spike detection
        volume_ma20 = dataframe['volume'].rolling(20).mean()
        volume_multiplier = self.volume_spike_multiplier.value
        dataframe['volume_spike_regime'] = dataframe['volume'] > (volume_ma20 * volume_multiplier)
        
        # Volume + movement combined
        dataframe['volume_pump'] = dataframe['volume_spike_regime'] & dataframe['flash_pump']
        dataframe['volume_dump'] = dataframe['volume_spike_regime'] & dataframe['flash_dump']
        
        # Market sentiment (if available)
        if 'market_breadth' in dataframe.columns:
            dataframe['market_breadth_change'] = dataframe['market_breadth'].diff(3)
            sentiment_threshold = self.sentiment_shift_threshold.value
            dataframe['sentiment_shift_bull'] = dataframe['market_breadth_change'] > sentiment_threshold
            dataframe['sentiment_shift_bear'] = dataframe['market_breadth_change'] < -sentiment_threshold
        else:
            dataframe['sentiment_shift_bull'] = False
            dataframe['sentiment_shift_bear'] = False
        
        # BTC correlation monitoring (if available)
        if 'btc_close' in dataframe.columns:
            dataframe['btc_change_fast'] = dataframe['btc_close'].pct_change(flash_candles)
            dataframe['btc_flash_pump'] = dataframe['btc_change_fast'] > flash_threshold
            dataframe['btc_flash_dump'] = dataframe['btc_change_fast'] < -flash_threshold
            
            pair_movement = dataframe['price_change_fast'].abs()
            btc_movement = dataframe['btc_change_fast'].abs()
            dataframe['correlation_break'] = (
                (btc_movement > flash_threshold) & 
                (pair_movement < flash_threshold * 0.4)
            )
        else:
            dataframe['btc_flash_pump'] = False
            dataframe['btc_flash_dump'] = False
            dataframe['correlation_break'] = False
        
        # Regime change score
        regime_signals = [
            'flash_move', 'volume_spike_regime',
            'sentiment_shift_bull', 'sentiment_shift_bear',
            'btc_flash_pump', 'btc_flash_dump', 'correlation_break'
        ]
        
        dataframe['regime_change_score'] = 0
        for signal in regime_signals:
            if signal in dataframe.columns:
                dataframe['regime_change_score'] += dataframe[signal].astype(int)
        
        # Normalize and alert
        max_signals = len(regime_signals)
        dataframe['regime_change_intensity'] = dataframe['regime_change_score'] / max_signals
        
        sensitivity = self.regime_change_sensitivity.value
        dataframe['regime_alert'] = dataframe['regime_change_intensity'] >= sensitivity
        
        return dataframe
    
    def _calculate_signal_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Calculate entry/exit signal indicators"""
        
        # Multi-factor signal strength
        dataframe['signal_strength'] = 0
        
        # Confluence signals
        dataframe['confluence_signal'] = (
            dataframe['confluence_score'] >= self.confluence_threshold.value
        ).astype(np.int8)
        dataframe['signal_strength'] += dataframe['confluence_signal'] * 2
        
        # Volume signals
        dataframe['volume_signal'] = (
            (dataframe['volume_pressure'] >= 2) &
            (dataframe['volume_strength'] > 1.2)
        ).astype(np.int8)
        dataframe['signal_strength'] += dataframe['volume_signal'] * 2
        
        # Momentum signals
        dataframe['momentum_signal'] = (
            (dataframe['momentum_quality'] >= 3) &
            (dataframe['momentum_acceleration'] > 0)
        ).astype(np.int8)
        dataframe['signal_strength'] += dataframe['momentum_signal'] * 2
        
        # Structure signals
        dataframe['structure_signal'] = (
            (dataframe['structure_score'] > 0) &
            (dataframe['structure_break_up'] == 1)
        ).astype(np.int8)
        dataframe['signal_strength'] += dataframe['structure_signal'] * 1
        
        # RSI position signal
        dataframe['rsi_signal'] = (
            (dataframe['rsi'] > 30) & (dataframe['rsi'] < 70)
        ).astype(np.int8)
        dataframe['signal_strength'] += dataframe['rsi_signal'] * 1
        
        # Trend alignment signal
        dataframe['trend_signal'] = (
            (dataframe['close'] > dataframe['ema50']) &
            (dataframe['trend_strength'] > 0)
        ).astype(np.int8)
        dataframe['signal_strength'] += dataframe['trend_signal'] * 1
        
        # Money flow signal
        dataframe['money_flow_signal'] = (
            dataframe['money_flow_index'] > 50
        ).astype(np.int8)
        dataframe['signal_strength'] += dataframe['money_flow_signal'] * 1
        
        # Exit signals (from your original calculate_exit_signals)
        dataframe = self._calculate_exit_signal_indicators(dataframe)
        
        return dataframe
    
    def _calculate_exit_signal_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Calculate exit signal indicators (from your original code)"""
        
        # Momentum deterioration
        dataframe['momentum_deteriorating'] = (
            (dataframe['momentum_quality'] < dataframe['momentum_quality'].shift(1)) &
            (dataframe['momentum_acceleration'] < 0) &
            (dataframe['price_momentum'] < dataframe['price_momentum'].shift(1))
        ).astype(np.int8)
        
        # Volume deterioration
        dataframe['volume_deteriorating'] = (
            (dataframe['volume_strength'] < 0.8) &
            (dataframe['selling_pressure'] > dataframe['buying_pressure']) &
            (dataframe['volume_pressure'] < 0)
        ).astype(np.int8)
        
        # Structure deterioration
        dataframe['structure_deteriorating'] = (
            (dataframe['structure_score'] < -1) &
            (dataframe['bearish_structure'] > dataframe['bullish_structure']) &
            (dataframe['structure_break_down'] == 1)
        ).astype(np.int8)
        
        # Confluence breakdown
        dataframe['confluence_breakdown'] = (
            (dataframe['confluence_score'] < 2) &
            (dataframe['near_resistance'] == 1) &
            (dataframe['volume_spike'] == 0)
        ).astype(np.int8)
        
        # Trend weakness
        dataframe['trend_weakening'] = (
            (dataframe['trend_strength'] < 0) &
            (dataframe['close'] < dataframe['ema50']) &
            (dataframe['strong_downtrend'] == 1)
        ).astype(np.int8)
        
        # Exit pressure score
        dataframe['exit_pressure'] = (
            dataframe['momentum_deteriorating'] * 2 +
            dataframe['volume_deteriorating'] * 2 +
            dataframe['structure_deteriorating'] * 2 +
            dataframe['confluence_breakdown'] * 1 +
            dataframe['trend_weakening'] * 1
        )
        
        # RSI overbought with divergence
        dataframe['rsi_exit_signal'] = (
            (dataframe['rsi'] > 75) &
            (
                (dataframe.get('rsi_divergence_bear', 0) == 1) |
                (
                    (dataframe['rsi'] > dataframe['rsi'].shift(1)) &
                    (dataframe['close'] < dataframe['close'].shift(1))
                )
            )
        ).astype(np.int8)
        
        # Volatility spike exit
        dataframe['volatility_spike'] = (
            dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 1.5
        ).astype(np.int8)
        
        # Bullish exhaustion
        dataframe['bullish_exhaustion'] = (
            (dataframe.get('consecutive_green', 0) >= 4) &
            (dataframe['rsi'] > 70) &
            (dataframe['volume'] < dataframe['avg_volume'] * 0.8) &
            (dataframe['momentum_acceleration'] < 0)
        ).astype(np.int8)
        
        return dataframe
    
    def _calculate_ultimate_score(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Calculate ultimate market score"""
        
        # Ultimate market score (weighted combination)
        dataframe['ultimate_score'] = (
            dataframe['confluence_score'] * 0.25 +
            dataframe['volume_pressure'] * 0.2 +
            dataframe['momentum_quality'] * 0.2 +
            (dataframe['structure_score'] / 5) * 0.15 +
            (dataframe['signal_strength'] / 10) * 0.2
        )
        
        # Normalize to 0-1 range
        dataframe['ultimate_score'] = dataframe['ultimate_score'].clip(0, 5) / 5
        
        # High quality setup detection
        dataframe['high_quality_setup'] = (
            (dataframe['ultimate_score'] > self.ultimate_score_threshold.value) &
            (dataframe['signal_strength'] >= 5) &
            (dataframe['volume_strength'] > 1.1) &
            (dataframe['rsi'] > 30) & (dataframe['rsi'] < 70)
        ).astype(np.int8)
        
        # Additional indicators from original
        dataframe['green_candle'] = (dataframe['close'] > dataframe['open']).astype(np.int8)
        dataframe['red_candle'] = (dataframe['close'] < dataframe['open']).astype(np.int8)
        dataframe['consecutive_green'] = dataframe['green_candle'].rolling(3).sum()
        dataframe['consecutive_red'] = dataframe['red_candle'].rolling(3).sum()
        
        dataframe['strong_up_momentum'] = (
            (dataframe['consecutive_green'] >= 3) &
            (dataframe['volume'] > dataframe['avg_volume']) &
            (dataframe['trend_strength'] > self.trend_strength_threshold.value)
        ).astype(np.int8)
        
        dataframe['strong_down_momentum'] = (
            (dataframe['consecutive_red'] >= 3) &
            (dataframe['volume'] > dataframe['avg_volume']) &
            (dataframe['trend_strength'] < -self.trend_strength_threshold.value)
        ).astype(np.int8)
        
        # RSI divergence
        dataframe['rsi_divergence_bull'] = (
            (dataframe['close'] < dataframe['close'].shift(5)) &
            (dataframe['rsi'] > dataframe['rsi'].shift(5))
        ).astype(np.int8)
        
        dataframe['rsi_divergence_bear'] = (
            (dataframe['close'] > dataframe['close'].shift(5)) &
            (dataframe['rsi'] < dataframe['rsi'].shift(5))
        ).astype(np.int8)
        
        # Volume momentum
        dataframe['volume_momentum'] = (
            dataframe['volume'].rolling(3).mean() / 
            dataframe['volume'].rolling(20).mean()
        )
        
        # Entry type placeholder
        dataframe['entry_type'] = 0
        
        return dataframe
    
    def _update_cache(self, key: str, dataframe: pd.DataFrame):
        """Update cache with size limit"""
        if len(self._indicator_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._indicator_cache))
            del self._indicator_cache[oldest_key]
        
        self._indicator_cache[key] = dataframe.copy()
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Vectorized entry logic with state management"""
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = ''
        
        # Check if entry is allowed - YOUR STATE CHECK
        if not self.state_manager.should_allow_entry(metadata['pair']):
            return dataframe
        
        # Vectorized high quality long conditions
        high_quality_long = (
            (dataframe['ultimate_score'] > self.ultimate_score_threshold.value) &
            (dataframe['signal_strength'] >= 5) &
            (dataframe['volume_strength'] > self.volume_threshold.value) &
            (dataframe['momentum_quality'] >= self.momentum_threshold.value) &
            (dataframe['confluence_score'] >= self.confluence_threshold.value) &
            (dataframe['close'] > dataframe['ema50']) &
            (dataframe['rsi'] > 30) & (dataframe['rsi'] < 70) &
            (dataframe['minima'] == 1)
        )
        
        # Medium quality long conditions
        medium_quality_long = (
            ~high_quality_long &  # Not already high quality
            (dataframe['ultimate_score'] > self.ultimate_score_threshold.value * 0.8) &
            (dataframe['signal_strength'] >= 4) &
            (dataframe['volume_strength'] > 1.0) &
            (dataframe['momentum_quality'] >= 2) &
            (dataframe['close'] > dataframe['ema50']) &
            (dataframe['rsi'] > 25) & (dataframe['rsi'] < 75) &
            (dataframe['DI_catch'] == 0)
        )
        
        # Apply long entries
        dataframe.loc[high_quality_long, 'enter_long'] = 1
        dataframe.loc[high_quality_long, 'enter_tag'] = 'high_quality_long'
        
        dataframe.loc[medium_quality_long, 'enter_long'] = 1
        dataframe.loc[medium_quality_long, 'enter_tag'] = 'medium_quality_long'
        
        # Short entries if enabled
        if self.can_short:
            high_quality_short = (
                (dataframe['ultimate_score'] < 0.3) &
                (dataframe['signal_strength'] <= 2) &
                (dataframe['volume_pressure'] < -self.volume_threshold.value) &
                (dataframe['momentum_quality'] <= 1) &
                (dataframe['confluence_score'] <= 1) &
                (dataframe['close'] < dataframe['ema50']) &
                (dataframe['rsi'] > 30) & (dataframe['rsi'] < 70) &
                (dataframe['maxima'] == 1)
            )
            
            dataframe.loc[high_quality_short, 'enter_short'] = 1
            dataframe.loc[high_quality_short, 'enter_tag'] = 'high_quality_short'
        
        # Update state manager for any new entries - YOUR STATE TRANSITIONS
        if dataframe['enter_long'].any():
            # Get the quality of the last signal
            last_long_idx = dataframe[dataframe['enter_long'] == 1].index[-1]
            quality = 'high' if 'high_quality' in dataframe.loc[last_long_idx, 'enter_tag'] else 'medium'
            score = dataframe.loc[last_long_idx, 'ultimate_score']
            
            self.state_manager.transition(
                metadata['pair'], 
                TradeState.ENTERING,
                {
                    'quality': quality, 
                    'score': float(score),
                    'entry_type': 'long',
                    'entry_time': current_time
                }
            )
        
        if self.can_short and dataframe['enter_short'].any():
            last_short_idx = dataframe[dataframe['enter_short'] == 1].index[-1]
            quality = 'high' if 'high_quality' in dataframe.loc[last_short_idx, 'enter_tag'] else 'medium'
            score = dataframe.loc[last_short_idx, 'ultimate_score']
            
            self.state_manager.transition(
                metadata['pair'], 
                TradeState.ENTERING,
                {
                    'quality': quality,
                    'score': float(score),
                    'entry_type': 'short',
                    'entry_time': current_time
                }
            )
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Enhanced exit logic with multiple exit systems
        Now properly uses your MML exits!
        """
        
        # Initialize exit columns
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = ''
        
        # Check for forced exits from state manager
        if self.state_manager.should_force_exit(metadata['pair']):
            dataframe.iloc[-1, dataframe.columns.get_loc('exit_long')] = 1
            dataframe.iloc[-1, dataframe.columns.get_loc('exit_short')] = 1
            dataframe.iloc[-1, dataframe.columns.get_loc('exit_tag')] = 'forced_timeout'
            
            self.state_manager.transition(metadata['pair'], TradeState.EMERGENCY_EXIT)
            return dataframe
        
        # Choose exit system based on parameters
        if self.use_mml_exits.value:
            # Use your sophisticated MML exit system
            dataframe = self.mml_exit_system.calculate_exits_with_state(
                dataframe, 
                can_short=self.can_short
            )
            
            # Log MML exit details for major pairs
            if metadata['pair'] in DEBUG_PAIRS:
                recent_exits = dataframe['exit_long'].tail(5).sum() + dataframe['exit_short'].tail(5).sum()
                if recent_exits > 0:
                    idx = dataframe[dataframe['exit_long'] == 1].index[-1] if any(dataframe['exit_long']) else -1
                    if idx >= 0:
                        reason = self.mml_exit_system.get_exit_reason(dataframe, idx, 'long')
                        logger.info(f"{metadata['pair']} MML Exit: {reason}")
        
        elif self.use_simple_exits.value:
            # Use simple opposite signal exits
            dataframe = self._apply_simple_exits(dataframe)
        
        else:
            # Use hybrid approach - combine both systems
            mml_df = self.mml_exit_system.calculate_exits_with_state(
                dataframe.copy(), 
                can_short=self.can_short
            )
            simple_df = self._apply_simple_exits(dataframe.copy())
            
            # Combine signals (MML takes priority)
            dataframe['exit_long'] = (mml_df['exit_long'] | simple_df['exit_long']).astype(int)
            dataframe['exit_short'] = (mml_df['exit_short'] | simple_df['exit_short']).astype(int)
            
            # Use MML tags where available, otherwise simple tags
            dataframe['exit_tag'] = mml_df['exit_tag'].where(
                mml_df['exit_tag'] != '', 
                simple_df['exit_tag']
            )
        
        # Update state manager on exits
        if any(dataframe['exit_long']) or any(dataframe['exit_short']):
            self.state_manager.transition(metadata['pair'], TradeState.EXITING)
        
        return dataframe
    
    def _apply_simple_exits(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Simple exit system as fallback"""
        
        # Exit on opposite signal
        if 'enter_short' in dataframe.columns:
            dataframe.loc[dataframe['enter_short'] == 1, 'exit_long'] = 1
            dataframe.loc[dataframe['enter_short'] == 1, 'exit_tag'] = 'opposite_signal'
        
        if 'enter_long' in dataframe.columns:
            dataframe.loc[dataframe['enter_long'] == 1, 'exit_short'] = 1
            dataframe.loc[dataframe['enter_long'] == 1, 'exit_tag'] = 'opposite_signal'
        
        return dataframe

    @property
    def protections(self):
        prot = [{"method": "CooldownPeriod", "stop_duration_candles": self.cooldown_lookback.value}]
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 72,
                "trade_limit": 2,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False,
            })
        return prot
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: Optional[float], max_stake: float,
                           leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """Improved stake calculation using risk manager"""
        
        try:
            portfolio_value = self.wallets.get_total_stake_amount()
        except:
            portfolio_value = 1000.0  # Fallback
        
        # Get dynamic stop loss
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if not dataframe.empty:
            stop_loss_pct = self.risk_manager.calculate_dynamic_stop_loss(
                dataframe, 
                len(dataframe) - 1
            )
        else:
            stop_loss_pct = abs(self.stoploss)
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            portfolio_value=portfolio_value,
            risk_per_trade=self.risk_per_trade.value,
            stop_loss_pct=stop_loss_pct,
            max_position_pct=0.1
        )
        
        # Apply state-based adjustments
        trade_quality = self.state_manager.get_trade_quality(pair)
        if trade_quality == 'high':
            position_size *= 1.2
        elif trade_quality == 'medium':
            position_size *= 1.0
        else:
            position_size *= 0.8
        
        # Ensure within limits
        position_size = max(min_stake or 0, min(position_size, max_stake))
        
        logger.info(f"{pair} Position size: {position_size:.2f} USDT "
                   f"(Quality: {trade_quality}, Stop: {stop_loss_pct:.2%})")
        
        return position_size
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime, 
                            current_rate: float, current_profit: float,
                            min_stake: Optional[float], max_stake: float,
                            current_entry_rate: float, current_exit_rate: float,
                            current_entry_profit: float, current_exit_profit: float,
                            **kwargs) -> Optional[float]:
        """Improved position adjustment with state management preserved"""
        
        # Check state - PRESERVE YOUR LOGIC
        if not self.state_manager.should_allow_dca(trade.pair):
            return None
        
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            return None
        
        last_candle = dataframe.iloc[-1]
        
        # Dynamic profit taking based on market conditions
        if current_profit > 0:
            # Take profits at resistance
            if last_candle.get('near_resistance', 0) == 1 and current_profit > 0.01:
                self.state_manager.transition(trade.pair, TradeState.SCALING_OUT)  # YOUR STATE TRANSITION
                return -(trade.amount * 0.25)
            
            # Take profits on momentum loss
            if (last_candle.get('momentum_deteriorating', 0) == 1 and 
                current_profit > 0.015):
                self.state_manager.transition(trade.pair, TradeState.SCALING_OUT)
                return -(trade.amount * 0.33)
            
            # Standard profit taking
            if current_profit > STANDARD_PROFIT_TAKING:
                if trade.nr_of_successful_exits == 0:
                    self.state_manager.transition(trade.pair, TradeState.SCALING_OUT)  # YOUR STATE TRANSITION
                    return -(trade.amount * STANDARD_SELL_PCT)
        
        # DCA logic with state management
        elif current_profit < self.initial_safety_order_trigger.value:
            # Additional check for DCA permission
            if self.state_manager.should_allow_dca(trade.pair):  # YOUR STATE CHECK
                # DCA only at strong support levels
                if (last_candle.get('near_support', 0) == 1 or 
                    last_candle.get('near_mml', 0) >= 1):
                    if trade.nr_of_successful_entries <= self.max_safety_orders.value:
                        # Transition to scaling in state
                        self.state_manager.transition(trade.pair, TradeState.SCALING_IN)  # YOUR STATE TRANSITION
                        
                        # Scale position size
                        order_num = trade.nr_of_successful_entries
                        scale = self.safety_order_volume_scale.value ** order_num
                        dca_amount = (min_stake or 10.0) * scale
                        
                        # Log DCA decision
                        logger.info(f"{trade.pair}: DCA #{order_num} at support. "
                                f"Amount: {dca_amount:.2f}, Current loss: {current_profit:.2%}")
                        
                        return dca_amount
        
        return None
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                           rate: float, time_in_force: str, current_time: datetime,
                           entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """Enhanced entry confirmation with state management"""
        
        # Check correlation risk
        open_trades = Trade.get_open_trades()
        if len(open_trades) > 3:
            correlation_check = self.risk_manager.check_correlation_risk(
                [t.pair for t in open_trades]
            )
            if correlation_check['risk_level'] == 'high':
                logger.warning(f"{pair} Entry blocked due to high correlation risk")
                return False
        
        # Update state to managing after successful entry
        self.state_manager.transition(pair, TradeState.MANAGING)
        
        return True
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str,
                          amount: float, rate: float, time_in_force: str,
                          exit_reason: str, current_time: datetime, **kwargs) -> bool:
        """Enhanced exit confirmation with state management"""
        
        # Always allow emergency exits
        if self.state_manager.get_state(pair) == TradeState.EMERGENCY_EXIT:
            self.state_manager.reset_state(pair)
            return True
        
        # Check exit reason
        always_allow = ['roi', 'stop_loss', 'stoploss', 'trailing_stop_loss']
        if exit_reason in always_allow:
            self.state_manager.reset_state(pair)
            return True
        
        # For partial exits, update state
        if amount < trade.amount:
            self.state_manager.transition(pair, TradeState.SCALING_OUT)
        else:
            self.state_manager.transition(pair, TradeState.EXITING)
        
        return True