import logging
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from typing import Optional, Any

import talib.abstract as ta
from scipy.signal import argrelextrema

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter
from freqtrade.persistence import Trade

# Import our new modules
from strategy_constants import *
from strategy_helpers import MarketAnalyzer, SignalGenerator, RiskManager, PerformanceOptimizer
from trade_state_manager import TradeStateManager, TradeState
from mml_exit_system import MMLExitSystem

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
    can_short = False
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
            use_emergency_exits=self.use_emergency_exits
        )
        
        if ("trading_mode" in self.config) and (self.config["trading_mode"] in ["futures", "margin"]):
            self.can_short = True
        
        # Cache for expensive calculations
        self._indicator_cache = {}
        self._cache_max_size = 100
        self._last_portfolio_check = None
        self._cached_risk_report = None
    
    # === HYPERPARAMETERS (Reduced and organized) ===
    
    # Core Strategy Parameters
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
    use_emergency_exits = True
    
    # 🚨  REGIME CHANGE DETECTION PARAMETERS (NEU)
    regime_change_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    regime_change_sensitivity = DecimalParameter(0.3, 0.8, default=0.5, decimals=2, space="sell", optimize=False, load=True)
    
    # Flash Move Detection
    flash_move_threshold = DecimalParameter(0.03, 0.08, default=0.05, decimals=3, space="sell", optimize=False, load=True)
    flash_move_candles = IntParameter(3, 10, default=5, space="sell", optimize=False, load=True)
    
    # Volume Spike Detection
    volume_spike_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    volume_spike_multiplier = DecimalParameter(2.0, 5.0, default=3.0, decimals=1, space="sell", optimize=False, load=True)
    
    # Emergency Exit Protection
    emergency_exit_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    emergency_exit_profit_threshold = DecimalParameter(0.005, 0.03, default=0.015, decimals=3, space="sell", optimize=False, load=True)
    
    # Market Sentiment Protection
    sentiment_protection_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    sentiment_shift_threshold = DecimalParameter(0.2, 0.4, default=0.3, decimals=2, space="sell", optimize=False, load=True)

    # 🔧ATR STOPLOSS PARAMETERS (Anpassbar machen)
    atr_stoploss_multiplier = DecimalParameter(0.8, 2.0, default=1.0, decimals=1, space="sell", optimize=False, load=True)
    atr_stoploss_minimum = DecimalParameter(-0.25, -0.10, default=-0.12, decimals=2, space="sell", optimize=False, load=True)
    atr_stoploss_maximum = DecimalParameter(-0.30, -0.15, default=-0.18, decimals=2, space="sell", optimize=False, load=True)
    atr_stoploss_ceiling = DecimalParameter(-0.10, -0.06, default=-0.06, decimals=2, space="sell", optimize=False, load=True)
    # DCA parameters
    initial_safety_order_trigger = DecimalParameter(
        low=-0.02, high=-0.01, default=-0.018, decimals=3, space="buy", optimize=False, load=True
    )
    max_safety_orders = IntParameter(1, 3, default=1, space="buy", optimize=False, load=True)
    safety_order_step_scale = DecimalParameter(
        low=1.05, high=1.5, default=1.25, decimals=2, space="buy", optimize=False, load=True
    )
    safety_order_volume_scale = DecimalParameter(
        low=1.1, high=2.0, default=1.4, decimals=1, space="buy", optimize=False, load=True
    )
    h2 = IntParameter(20, 60, default=40, space="buy", optimize=False, load=True)
    h1 = IntParameter(10, 40, default=20, space="buy", optimize=False, load=True)
    h0 = IntParameter(5, 20, default=10, space="buy", optimize=False, load=True)
    cp = IntParameter(5, 20, default=10, space="buy", optimize=False, load=True)

    # Entry parameters
    increment_for_unique_price = DecimalParameter(
        low=1.0005, high=1.002, default=1.001, decimals=4, space="buy", optimize=False, load=True
    )
    last_entry_price: Optional[float] = None

    # Protection parameters
    cooldown_lookback = IntParameter(2, 48, default=1, space="protection", optimize=False)
    stop_duration = IntParameter(12, 200, default=4, space="protection", optimize=False)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=False)
    
    # Murrey Math level parameters
    mml_const1 = DecimalParameter(1.0, 1.1, default=1.0699, decimals=4, space="buy", optimize=False, load=True)
    mml_const2 = DecimalParameter(0.99, 1.0, default=0.99875, decimals=5, space="buy", optimize=False, load=True)

    # Dynamic Leverage parameters
    leverage_window_size = IntParameter(20, 100, default=50, space="buy", optimize=False, load=True)
    leverage_base = DecimalParameter(5.0, 20.0, default=5.0, decimals=1, space="buy", optimize=False, load=True)
    leverage_rsi_low = DecimalParameter(20.0, 40.0, default=30.0, decimals=1, space="buy", optimize=False, load=True)
    leverage_rsi_high = DecimalParameter(60.0, 80.0, default=70.0, decimals=1, space="buy", optimize=False, load=True)
    leverage_long_increase_factor = DecimalParameter(1.1, 2.0, default=1.5, decimals=1, space="buy", optimize=False,
                                                     load=True)
    leverage_long_decrease_factor = DecimalParameter(0.3, 0.9, default=0.5, decimals=1, space="buy", optimize=False,
                                                     load=True)
    leverage_volatility_decrease_factor = DecimalParameter(0.5, 0.95, default=0.8, decimals=2, space="buy",
                                                           optimize=False, load=True)
    leverage_atr_threshold_pct = DecimalParameter(0.01, 0.05, default=0.03, decimals=3, space="buy", optimize=False,
                                                  load=True)

    # Indicator parameters
    indicator_extrema_order = IntParameter(3, 15, default=8, space="buy", optimize=False, load=True)  # War 5
    indicator_mml_window = IntParameter(50, 200, default=50, space="buy", optimize=False, load=True)  # War 50
    indicator_rolling_window_threshold = IntParameter(20, 100, default=50, space="buy", optimize=False, load=True)  # War 20
    indicator_rolling_check_window = IntParameter(5, 20, default=10, space="buy", optimize=False, load=True)  # War 5


    
    # Market breadth parameters
    market_breadth_enabled = BooleanParameter(default=True, space="buy", optimize=False)
    market_breadth_threshold = DecimalParameter(0.3, 0.6, default=0.45, space="buy", optimize=False)
    
    # Total market cap parameters
    total_mcap_filter_enabled = BooleanParameter(default=True, space="buy", optimize=False)
    total_mcap_ma_period = IntParameter(20, 100, default=50, space="buy", optimize=False)
    
    # Market regime parameters
    regime_filter_enabled = BooleanParameter(default=True, space="buy", optimize=False)
    regime_lookback_period = IntParameter(24, 168, default=48, space="buy", optimize=False)  # hours
    
    # Fear & Greed parameters
    fear_greed_enabled = BooleanParameter(default=False, space="buy", optimize=False)  # Optional
    fear_greed_extreme_threshold = IntParameter(20, 30, default=25, space="buy", optimize=False)
    fear_greed_greed_threshold = IntParameter(70, 80, default=75, space="buy", optimize=False)
    # Momentum
    avoid_strong_trends = BooleanParameter(default=True, space="buy", optimize=False)
    trend_strength_threshold = DecimalParameter(0.01, 0.05, default=0.02, space="buy", optimize=False)
    momentum_confirmation_candles = IntParameter(1, 5, default=2, space="buy", optimize=False)

    # Dynamic exit based on entry quality
    dynamic_exit_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    exit_on_confluence_loss = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    exit_on_structure_break = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    
    # Profit target multipliers based on entry type
    high_quality_profit_multiplier = DecimalParameter(1.2, 3.0, default=2.0, space="sell", optimize=False, load=True)
    medium_quality_profit_multiplier = DecimalParameter(1.0, 2.5, default=1.5, space="sell", optimize=False, load=True)
    backup_profit_multiplier = DecimalParameter(0.8, 2.0, default=1.2, space="sell", optimize=False, load=True)
    
    # Advanced exit thresholds
    volume_decline_exit_threshold = DecimalParameter(0.3, 0.8, default=0.5, space="sell", optimize=False, load=True)
    momentum_decline_exit_threshold = IntParameter(1, 4, default=2, space="sell", optimize=False, load=True)
    structure_deterioration_threshold = DecimalParameter(-3.0, 0.0, default=-1.5, space="sell", optimize=False, load=True)
    
    # RSI exit levels
    rsi_overbought_exit = IntParameter(70, 85, default=75, space="sell", optimize=False, load=True)
    rsi_divergence_exit_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    
    # Trailing stop improvements
    use_advanced_trailing = BooleanParameter(default=False, space="sell", optimize=False, load=True)
    trailing_stop_positive_offset_high_quality = DecimalParameter(0.02, 0.08, default=0.04, space="sell", optimize=False, load=True)
    trailing_stop_positive_offset_medium_quality = DecimalParameter(0.015, 0.06, default=0.03, space="sell", optimize=False, load=True)
    
    # === NEUE ADVANCED PARAMETERS ===
    # Confluence Analysis
    confluence_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    confluence_threshold = DecimalParameter(2.0, 4.0, default=2.5, space="buy", optimize=False, load=True)  # War 3.0
    
    # Volume Analysis
    volume_analysis_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    volume_strength_threshold = DecimalParameter(1.1, 2.0, default=1.3, space="buy", optimize=False, load=True)
    volume_pressure_threshold = IntParameter(1, 3, default=1, space="buy", optimize=False, load=True)  # War 2

    
    # Momentum Analysis
    momentum_analysis_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    momentum_quality_threshold = IntParameter(2, 4, default=2, space="buy", optimize=False, load=True)  # War 3
    
    # Market Structure Analysis
    structure_analysis_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    structure_score_threshold = DecimalParameter(-2.0, 5.0, default=0.5, space="buy", optimize=False, load=True)
    
    # Ultimate Score
    ultimate_score_threshold = DecimalParameter(0.5, 3.0, default=1.5, space="buy", optimize=False, load=True)
    
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
            rolling_min = dataframe['ha_close'].rolling(window=window).min()
            rolling_max = dataframe['ha_close'].rolling(window=window).max()
            
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
    
    def calculate_rolling_murrey_math_levels_optimized(self, df: pd.DataFrame, window_size: int) -> dict[str, pd.Series]:
        """
        OPTIMIZED Version - Calculate MML levels every 5 candles using only past data
        """
        murrey_levels_data: dict[str, list] = {key: [np.nan] * len(df) for key in MML_LEVEL_NAMES}
        mml_c1 = self.mml_const1.value
        mml_c2 = self.mml_const2.value
        
        calculation_step = 1
        
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
            series = series.ffill() # Forward-fill the last known value
            murrey_levels_data[key] = series.tolist()
        
        return {key: pd.Series(data, index=df.index) for key, data in murrey_levels_data.items()}
    
    @staticmethod
    def _calculate_mml_core(mn: float, finalH: float, mx: float, finalL: float,
                            mml_c1: float, mml_c2: float) -> dict[str, float]:
        dmml_calc = ((finalH - finalL) / 8.0) * mml_c1
        if dmml_calc == 0 or np.isinf(dmml_calc) or np.isnan(dmml_calc) or finalH == finalL:
            return {key: finalL for key in MML_LEVEL_NAMES}
        mml_val = (mx * mml_c2) + (dmml_calc * 3)
        if np.isinf(mml_val) or np.isnan(mml_val):
            return {key: finalL for key in MML_LEVEL_NAMES}
        ml = [mml_val - (dmml_calc * i) for i in range(16)]
        return {
            "[-3/8]P": ml[14], "[-2/8]P": ml[13], "[-1/8]P": ml[12],
            "[0/8]P": ml[11], "[1/8]P": ml[10], "[2/8]P": ml[9],
            "[3/8]P": ml[8], "[4/8]P": ml[7], "[5/8]P": ml[6],
            "[6/8]P": ml[5], "[7/8]P": ml[4], "[8/8]P": ml[3],
            "[+1/8]P": ml[2], "[+2/8]P": ml[1], "[+3/8]P": ml[0],
        }
      
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
            dataframe[f'momentum_{period}'] = dataframe['close'].pct_change(period)
        
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
            (dataframe['high'].shift(1) > dataframe['high'].shift(2)) &
            (dataframe['high'].shift(1) > dataframe['high'])
        ).astype(np.int8)

        dataframe['swing_low'] = (
            (dataframe['low'].shift(1) < dataframe['low'].shift(2)) &
            (dataframe['low'].shift(1) < dataframe['low'])
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

    def _calculate_entry_zones(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Improved entry zone calculation with dynamic sizing"""
        
        # Calculate zone sizes based on ATR
        atr_pct = dataframe['atr'] / dataframe['close']
        zone_size = 0.01 + (atr_pct * 0.5)  # Dynamic zone size
        
        # Calculate key levels
        dataframe['zone_upper'] = dataframe['close'] * (1 + zone_size)
        dataframe['zone_lower'] = dataframe['close'] * (1 - zone_size)
        
        # Long zones with better logic
        dataframe['long_zone_price'] = (
            # Near support levels
            (dataframe['close'] <= dataframe['support'] * (1 + zone_size)) |
            # MML support zones (weighted by importance)
            (dataframe['close'] <= dataframe['[0/8]P'] * 1.01) |  # Strong support
            (dataframe['close'] <= dataframe['[2/8]P'] * (1 + zone_size * 0.8)) |  # Medium support
            (dataframe['close'] <= dataframe['[1/8]P'] * 1.005)  # Extreme support
        )
        
        # Quality factors for zones
        dataframe['long_zone_quality'] = (
            dataframe['long_zone_price'].astype(int) +
            (dataframe['rsi'] < 35).astype(int) +
            (dataframe['volume_pressure'] > 0).astype(int) +
            (dataframe['structure_score'] > -1).astype(int) +
            (dataframe['momentum_acceleration'] >= 0).astype(int)
        )
        
        # Define zones by quality
        dataframe['strong_long_zone'] = (dataframe['long_zone_quality'] >= 4).astype(int)
        dataframe['long_zone'] = (dataframe['long_zone_quality'] >= 3).astype(int)
        
        # Short zones (mirror logic)
        dataframe['short_zone_price'] = (
            (dataframe['close'] >= dataframe['resistance'] * (1 - zone_size)) |
            (dataframe['close'] >= dataframe['[8/8]P'] * 0.99) |
            (dataframe['close'] >= dataframe['[6/8]P'] * (1 - zone_size * 0.8)) |
            (dataframe['close'] >= dataframe['[7/8]P'] * 0.995)
        )
        
        dataframe['short_zone_quality'] = (
            dataframe['short_zone_price'].astype(int) +
            (dataframe['rsi'] > 65).astype(int) +
            (dataframe['volume_pressure'] < 0).astype(int) +
            (dataframe['structure_score'] < 1).astype(int) +
            (dataframe['momentum_acceleration'] <= 0).astype(int)
        )
        
        dataframe['strong_short_zone'] = (dataframe['short_zone_quality'] >= 4).astype(int)
        dataframe['short_zone'] = (dataframe['short_zone_quality'] >= 3).astype(int)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Comprehensive entry logic using zones with all filters and state management"""
        
        # Initialize entry columns
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = ''
        
        # State check - don't enter if not allowed
        if not self.state_manager.should_allow_entry(metadata['pair']):
            return dataframe
        
        # Portfolio risk check
        allow_trade, risk_report = self.check_portfolio_risk()
        if not allow_trade:
            logger.info(f"{metadata['pair']} Entries blocked by portfolio risk: {risk_report['reasons']}")
            return dataframe
        
        # Calculate entry zones
        dataframe = self._calculate_entry_zones(dataframe)
        
        # Get market conditions
        market_conditions = self.get_market_conditions(metadata['pair'], dataframe)
        
        # === LONG ENTRY CONDITIONS ===
        
        # High Quality Long Entry (using zones)
        high_quality_long = (
            # Must be in a zone
            (dataframe['strong_long_zone'] == 1) &  # Strong zone preferred
            # Core requirements
            (dataframe['ultimate_score'] > self.ultimate_score_threshold.value) &
            (dataframe['signal_strength'] >= 5) &
            (dataframe['confluence_score'] >= self.confluence_threshold.value) &
            # Volume and momentum
            (dataframe['volume_strength'] > self.volume_threshold.value) &
            (dataframe['momentum_quality'] >= self.momentum_threshold.value) &
            (dataframe['volume_pressure'] > self.volume_pressure_threshold.value) &
            # Trend alignment
            (dataframe['close'] > dataframe['ema50']) &
            # Risk filters
            (dataframe['rsi'] > 25) & (dataframe['rsi'] < 65) &
            # Structure confirmation
            (dataframe['structure_score'] > self.structure_score_threshold.value) &
            # No regime change
            (~dataframe.get('regime_alert', False))
        )
        
        # Apply additional filters based on parameters
        if self.require_volume_confirmation.value:
            high_quality_long &= (dataframe['volume_trend'] == 1)
        
        if self.require_momentum_confirmation.value:
            high_quality_long &= (dataframe['momentum_acceleration'] > 0)
        
        if self.require_structure_confirmation.value:
            high_quality_long &= (dataframe['structure_break_up'] == 1)
        
        # Medium Quality Long Entry (zones but relaxed conditions)
        medium_quality_long = (
            # Not already high quality
            ~high_quality_long &
            # Must be in a zone (regular zone ok)
            (dataframe['long_zone'] == 1) &
            # Relaxed core requirements
            (dataframe['ultimate_score'] > self.ultimate_score_threshold.value * 0.7) &
            (dataframe['signal_strength'] >= 4) &
            (dataframe['confluence_score'] >= 2) &
            # Basic volume and momentum
            (dataframe['volume_strength'] > 1.0) &
            (dataframe['momentum_quality'] >= 2) &
            # Trend can be neutral
            (dataframe['close'] > dataframe['ema100']) &
            # Wider RSI range
            (dataframe['rsi'] > 20) & (dataframe['rsi'] < 70) &
            # Basic DI confirmation
            (dataframe['DI_catch'] == 0)
        )
        
        # Backup Long Entry (zone-based opportunistic)
        backup_long = (
            # Not already entered
            ~high_quality_long & ~medium_quality_long &
            # Strong zone is enough with basic confirmation
            (dataframe['strong_long_zone'] == 1) &
            # Minimal requirements
            (dataframe['ultimate_score'] > 0.5) &
            (dataframe['signal_strength'] >= 3) &
            # Oversold bounce
            (dataframe['rsi'] < 30) &
            (dataframe['minima'] == 1) &  # Actual bottom for backup
            # Volume spike (panic selling)
            (dataframe['volume'] > dataframe['avg_volume'] * 2.0) &
            # Not in strong downtrend
            (~dataframe.get('strong_downtrend', False))
        )
        
        # === SHORT ENTRY CONDITIONS (if enabled) ===
        
        if self.can_short:
            # High Quality Short Entry
            high_quality_short = (
                # Must be in a zone
                (dataframe['strong_short_zone'] == 1) &
                # Core requirements (inverted)
                (dataframe['ultimate_score'] < 0.5) &
                (dataframe['signal_strength'] <= 2) &
                (dataframe['confluence_score'] <= 1) &
                # Volume and momentum
                (dataframe['volume_pressure'] < -self.volume_pressure_threshold.value) &
                (dataframe['momentum_quality'] <= 1) &
                (dataframe['volume_strength'] > self.volume_threshold.value) &
                # Trend alignment
                (dataframe['close'] < dataframe['ema50']) &
                # Risk filters
                (dataframe['rsi'] > 35) & (dataframe['rsi'] < 75) &
                # Structure confirmation
                (dataframe['structure_score'] < -self.structure_score_threshold.value) &
                # No regime change
                (~dataframe.get('regime_alert', False))
            )
            
            # Medium Quality Short Entry
            medium_quality_short = (
                ~high_quality_short &
                (dataframe['short_zone'] == 1) &
                (dataframe['ultimate_score'] < 0.7) &
                (dataframe['signal_strength'] <= 3) &
                (dataframe['volume_pressure'] < 0) &
                (dataframe['momentum_quality'] <= 2) &
                (dataframe['close'] < dataframe['ema100']) &
                (dataframe['rsi'] > 30) & (dataframe['rsi'] < 80) &
                (dataframe['DI_catch'] == 1)
            )
                
        # === AVOID PROBLEMATIC CONDITIONS ===
        
        # Don't enter during flash moves or high volatility
        avoid_entry = (
            dataframe.get('flash_move', False) |
            dataframe.get('regime_alert', False) |
            (dataframe.get('volatility', 0) > HIGH_VOLATILITY_THRESHOLD * 1.5)
        )
        
        high_quality_long &= ~avoid_entry
        medium_quality_long &= ~avoid_entry
        backup_long &= ~avoid_entry
        
        if self.can_short:
            high_quality_short &= ~avoid_entry
            medium_quality_short &= ~avoid_entry
        
        # === APPLY ENTRIES WITH PROPER TAGS ===
        
        # Long entries (priority: high > medium > backup)
        dataframe.loc[high_quality_long, 'enter_long'] = 1
        dataframe.loc[high_quality_long, 'enter_tag'] = f'HQ_zone'
        
        # Only apply medium if no high quality
        medium_long_mask = medium_quality_long & (dataframe['enter_long'] == 0)
        dataframe.loc[medium_long_mask, 'enter_long'] = 1
        dataframe.loc[medium_long_mask, 'enter_tag'] = f'MQ_zone'
        
        # Only apply backup if no other entry
        backup_long_mask = backup_long & (dataframe['enter_long'] == 0)
        dataframe.loc[backup_long_mask, 'enter_long'] = 1
        dataframe.loc[backup_long_mask, 'enter_tag'] = f'backup_zone'
        
        # Short entries
        if self.can_short:
            dataframe.loc[high_quality_short, 'enter_short'] = 1
            dataframe.loc[high_quality_short, 'enter_tag'] = f'HQ_short_zone'
            
            medium_short_mask = medium_quality_short & (dataframe['enter_short'] == 0)
            dataframe.loc[medium_short_mask, 'enter_short'] = 1
            dataframe.loc[medium_short_mask, 'enter_tag'] = f'MQ_short_zone'
        
        # === STATE MANAGEMENT UPDATES ===
        
        if dataframe['enter_long'].any():
            # Get the last signal details
            last_long_idx = dataframe[dataframe['enter_long'] == 1].index[-1]
            last_long = dataframe.loc[last_long_idx]
            
            # Determine quality from tag
            quality = 'high' if 'HQ' in last_long['enter_tag'] else 'medium' if 'MQ' in last_long['enter_tag'] else 'backup'
            
            # Comprehensive metadata
            entry_metadata = {
                'quality': quality,
                'score': float(last_long.get('ultimate_score', 0)),
                'confluence': int(last_long.get('confluence_score', 0)),
                'signal_strength': int(last_long.get('signal_strength', 0)),
                'momentum_quality': int(last_long.get('momentum_quality', 0)),
                'volume_pressure': float(last_long.get('volume_pressure', 0)),
                'structure_score': float(last_long.get('structure_score', 0)),
                'entry_type': 'long',
                'entry_zone': 'strong' if last_long.get('strong_long_zone', 0) == 1 else 'regular',
                'entry_price': float(last_long['close']),
                'entry_rsi': float(last_long['rsi']),
                'entry_time': datetime.now(),
                'market_health': market_conditions['market_health']
            }
            
            self.state_manager.transition(
                metadata['pair'], 
                TradeState.ENTERING,
                entry_metadata
            )
            
            # Log entry details for major pairs
            if metadata['pair'] in DEBUG_PAIRS:
                logger.info(f"{metadata['pair']} LONG Entry: {last_long['enter_tag']} | "
                        f"Score: {entry_metadata['score']:.2f} | "
                        f"Zone: {entry_metadata['entry_zone']} | "
                        f"Market Health: {entry_metadata['market_health']}/10")
        
        if self.can_short and dataframe['enter_short'].any():
            last_short_idx = dataframe[dataframe['enter_short'] == 1].index[-1]
            last_short = dataframe.loc[last_short_idx]
            
            quality = 'high' if 'HQ' in last_short['enter_tag'] else 'medium'
            
            entry_metadata = {
                'quality': quality,
                'score': float(last_short.get('ultimate_score', 0)),
                'confluence': int(last_short.get('confluence_score', 0)),
                'signal_strength': int(last_short.get('signal_strength', 0)),
                'momentum_quality': int(last_short.get('momentum_quality', 0)),
                'volume_pressure': float(last_short.get('volume_pressure', 0)),
                'structure_score': float(last_short.get('structure_score', 0)),
                'entry_type': 'short',
                'entry_zone': 'strong' if last_short.get('strong_short_zone', 0) == 1 else 'regular',
                'entry_price': float(last_short['close']),
                'entry_rsi': float(last_short['rsi']),
                'entry_time': datetime.now(),
                'market_health': market_conditions['market_health']
            }
            
            self.state_manager.transition(
                metadata['pair'], 
                TradeState.ENTERING,
                entry_metadata
            )
        
        # === PERFORMANCE LOGGING ===
        
        if metadata['pair'] in DEBUG_PAIRS and (dataframe['enter_long'].any() or dataframe['enter_short'].any()):
            zones_found = {
                'long_zones': dataframe['long_zone'].sum(),
                'strong_long_zones': dataframe['strong_long_zone'].sum(),
                'short_zones': dataframe['short_zone'].sum(),
                'strong_short_zones': dataframe['strong_short_zone'].sum()
            }
            logger.debug(f"{metadata['pair']} Zones in period: {zones_found}")
        
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
            dataframe = self.mml_exit_system.calculate_exits(df=dataframe, can_short=self.can_short)
            
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
            mml_df = self.mml_exit_system.calculate_exits(dataframe.copy(), can_short=self.can_short)
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
            # Determine the correct state transition based on exit tag
            last_exit_idx = dataframe[(dataframe['exit_long'] == 1) | (dataframe['exit_short'] == 1)].index[-1]
            exit_tag = dataframe.loc[last_exit_idx, 'exit_tag']

            if 'Emergency' in exit_tag:
                self.state_manager.transition(metadata['pair'], TradeState.EMERGENCY_EXIT)
            else:
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
    
    def _check_time_based_dca(self, trade: Trade) -> bool:
        """Check if enough time has passed for DCA"""
        if trade.open_date_utc:
            time_open = (datetime.now() - trade.open_date_utc.replace(tzinfo=None)).total_seconds() / 3600
            # Allow DCA every 4 hours minimum
            min_hours_between_dca = 4 * max(1, trade.nr_of_successful_entries)
            return time_open >= min_hours_between_dca
        return False

    def should_dca(self, trade: Trade, current_profit: float, last_candle: pd.Series) -> tuple[bool, str]:
        """More flexible DCA logic with reason"""
        
        # Basic checks
        if current_profit >= self.initial_safety_order_trigger.value:
            return False, "profit_above_threshold"
        
        if trade.nr_of_successful_entries > self.max_safety_orders.value:
            return False, "max_orders_reached"
        
        if not self.state_manager.should_allow_dca(trade.pair):
            return False, "state_not_allowing"
        
        # Multiple DCA triggers with scores
        dca_triggers = {
            'support_bounce': {
                'condition': last_candle.get('near_support', 0) == 1,
                'score': 3,
                'reason': 'at_support'
            },
            'mml_support': {
                'condition': last_candle.get('near_mml', 0) >= 1,
                'score': 3,
                'reason': 'at_mml_level'
            },
            'oversold': {
                'condition': last_candle['rsi'] < 30,
                'score': 2,
                'reason': 'rsi_oversold'
            },
            'extreme_oversold': {
                'condition': last_candle['rsi'] < 20,
                'score': 4,
                'reason': 'rsi_extreme_oversold'
            },
            'volume_spike': {
                'condition': last_candle.get('volume_spike', 0) == 1 and last_candle['close'] < last_candle['open'],
                'score': 2,
                'reason': 'panic_selling'
            },
            'momentum_reversal': {
                'condition': last_candle.get('minima', 0) == 1 or last_candle.get('momentum_acceleration', 0) > 0,
                'score': 2,
                'reason': 'momentum_reversal'
            },
            'structure_support': {
                'condition': last_candle.get('structure_score', 0) > 0 and current_profit < -0.05,
                'score': 2,
                'reason': 'bullish_structure'
            },
            'time_based': {
                'condition': self._check_time_based_dca(trade) and current_profit < -0.03,
                'score': 1,
                'reason': 'time_based_dca'
            }
        }
        
        # Calculate total score and active triggers
        total_score = 0
        active_triggers = []
        
        for trigger_name, trigger_data in dca_triggers.items():
            if trigger_data['condition']:
                total_score += trigger_data['score']
                active_triggers.append(trigger_data['reason'])
        
        # Need minimum score of 4 (at least 2 meaningful triggers)
        min_score_required = 4
        
        # Adjust requirement based on number of existing DCAs
        if trade.nr_of_successful_entries >= 2:
            min_score_required = 6  # Higher requirement for 3rd+ DCA
        
        should_dca_now = total_score >= min_score_required
        reason = f"score_{total_score}_triggers_{'_'.join(active_triggers)}" if active_triggers else "no_triggers"
        
        return should_dca_now, reason

    # Use in adjust_trade_position:
    def adjust_trade_position(self, trade: Trade, current_time: datetime, 
                            current_rate: float, current_profit: float,
                            min_stake: Optional[float], max_stake: float,
                            current_entry_rate: float, current_exit_rate: float,
                            current_entry_profit: float, current_exit_profit: float,
                            **kwargs) -> Optional[float]:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            return None
        
        last_candle = dataframe.iloc[-1]
        
        # ... profit taking logic ...
        
        # DCA logic with new implementation
        should_dca_now, dca_reason = self.should_dca(trade, current_profit, last_candle)
        
        if should_dca_now:
            self.state_manager.transition(trade.pair, TradeState.SCALING_IN)
            
            # Calculate DCA amount with dynamic scaling
            order_num = trade.nr_of_successful_entries
            
            # Base amount with scaling
            base_amount = min_stake or 10.0
            scale_factor = self.safety_order_volume_scale.value ** order_num
            
            # Adjust scale based on loss depth
            if current_profit < -0.10:  # Deep loss
                scale_factor *= 1.5  # Larger DCA
            
            dca_amount = base_amount * scale_factor
            
            # Log detailed DCA info
            logger.info(f"{trade.pair} DCA #{order_num}: {dca_reason} | "
                    f"Loss: {current_profit:.2%} | Amount: {dca_amount:.2f}")
            
            return dca_amount
        
        return None
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                           rate: float, time_in_force: str, current_time: datetime,
                           entry_tag: Optional[str], side: str, **kwargs) -> bool:
        # Portfolio risk check
        allow_trade, risk_report = self.check_portfolio_risk()
        
        if not allow_trade:
            logger.warning(f"{pair} Entry blocked - Portfolio risk: {risk_report['risk_level']} "
                        f"Reasons: {', '.join(risk_report['reasons'])}")
            return False
        
        # Log portfolio state
        if risk_report['risk_level'] != 'low':
            logger.info(f"Portfolio risk level: {risk_report['risk_level']} - {risk_report}")
        
        # Additional pair-specific checks
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if not dataframe.empty:
            market_conditions = self.get_market_conditions(pair, dataframe)
            
            # Don't enter in extreme conditions
            if market_conditions['regime_change']:
                logger.warning(f"{pair} Entry blocked - Regime change detected")
                return False
            
            if market_conditions['market_health'] < 3:
                logger.warning(f"{pair} Entry blocked - Poor market health: {market_conditions['market_health']}")
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
    
    def get_market_conditions(self, pair: str, dataframe: pd.DataFrame) -> dict[str, Any]:
        """Analyze current market conditions"""
        
        # Get recent data
        recent_data = dataframe.tail(20)
        current = dataframe.iloc[-1]
        
        conditions = {
            # Trend Analysis
            'bullish': current['close'] > current['ema50'] and current['ema50'] > current['ema100'],
            'bearish': current['close'] < current['ema50'] and current['ema50'] < current['ema100'],
            'ranging': abs(current.get('trend_strength', 0)) < 0.01,
            
            # Trend Strength
            'strong_trend': abs(current.get('trend_strength', 0)) > self.trend_strength_threshold.value,
            'trend_direction': 'up' if current.get('trend_strength', 0) > 0 else 'down',
            'trend_strength_value': current.get('trend_strength', 0),
            
            # Volatility
            'high_volatility': current.get('volatility', 0) > HIGH_VOLATILITY_THRESHOLD,
            'volatility_value': current.get('volatility', 0),
            'atr_percentage': (current['atr'] / current['close']) if current['close'] > 0 else 0,
            
            # Volume
            'high_volume': current['volume'] > current.get('avg_volume', current['volume']) * 1.5,
            'low_volume': current['volume'] < current.get('avg_volume', current['volume']) * 0.7,
            'volume_trend': 'increasing' if recent_data['volume'].is_monotonic_increasing else 'decreasing',
            
            # Momentum
            'strong_momentum': current.get('momentum_quality', 0) >= 4,
            'weak_momentum': current.get('momentum_quality', 0) <= 1,
            'momentum_shifting': current.get('momentum_acceleration', 0) != 0,
            
            # Market Structure
            'at_resistance': current.get('near_resistance', 0) == 1,
            'at_support': current.get('near_support', 0) == 1,
            'structure_bullish': current.get('structure_score', 0) > 2,
            'structure_bearish': current.get('structure_score', 0) < -2,
            
            # Risk Indicators
            'overbought': current['rsi'] > 70,
            'oversold': current['rsi'] < 30,
            'extreme_move': abs(recent_data['close'].pct_change().sum()) > 0.10,  # 10% move
            
            # Regime
            'regime_change': current.get('regime_alert', False),
            'flash_move': current.get('flash_move', False),
            
            # Overall Market Health Score (0-10)
            'market_health': 0  # Calculated below
        }
        
        # Calculate market health score
        health_score = 5  # Neutral start
        
        # Positive factors
        if conditions['bullish']: health_score += 1
        if conditions['strong_momentum']: health_score += 1
        if conditions['structure_bullish']: health_score += 1
        if not conditions['high_volatility']: health_score += 1
        
        # Negative factors
        if conditions['bearish']: health_score -= 1
        if conditions['weak_momentum']: health_score -= 1
        if conditions['structure_bearish']: health_score -= 1
        if conditions['regime_change']: health_score -= 2
        if conditions['extreme_move']: health_score -= 1
        
        conditions['market_health'] = max(0, min(10, health_score))
        
        return conditions
    
    def check_portfolio_risk(self) -> tuple[bool, dict[str, Any]]:
        
        current_time = datetime.now()
        # Only recalculate every 5 minutes
        if self._last_portfolio_check and (current_time - self._last_portfolio_check).seconds < 300:
            return True, self._cached_risk_report
        
        """Portfolio-wide risk checks"""
        try:
            open_trades = Trade.get_open_trades()
            
            risk_report = {
                'allow_new_trades': True,
                'risk_level': 'low',
                'reasons': [],
                'metrics': {}
            }
            
            if not open_trades:
                return True, risk_report
            
            # 1. Maximum position check
            max_open_trades = 6  # Adjust as needed
            if len(open_trades) >= max_open_trades:
                risk_report['allow_new_trades'] = False
                risk_report['reasons'].append(f"max_positions_reached_{len(open_trades)}")
            
            # 2. Correlation and Concentration Risk
            open_pairs = [t.pair for t in open_trades]
            if len(open_pairs) > 1:
                # Call the corrected correlation check from the RiskManager
                correlation_report = self.risk_manager.check_correlation_risk(
                    pairs=open_pairs,
                    dp=self.dp,
                    timeframe=self.timeframe,
                    correlation_threshold=self.max_correlation.value
                )
                if correlation_report['risk_level'] == 'high':
                    risk_report['allow_new_trades'] = False
                    # Add a descriptive reason about which pairs are correlated
                    reason_str = f"high_correlation_{correlation_report['high_correlation_pairs']}"
                    risk_report['reasons'].append(reason_str)
            
            # 3. Calculate current drawdown
            total_profit_loss = sum(trade.calc_profit() for trade in open_trades)
            total_stake = sum(trade.stake_amount for trade in open_trades)
            
            if total_stake > 0:
                current_drawdown = abs(min(0, total_profit_loss / total_stake))
                risk_report['metrics']['drawdown_pct'] = current_drawdown
                
                if current_drawdown > 0.15:  # 15% drawdown
                    risk_report['allow_new_trades'] = False
                    risk_report['reasons'].append(f"high_drawdown_{current_drawdown:.1%}")
                    risk_report['risk_level'] = 'high'
            
            # 4. Check losing trades
            losing_trades = [t for t in open_trades if t.calc_profit_ratio() < 0]
            winning_trades = [t for t in open_trades if t.calc_profit_ratio() > 0]
            
            risk_report['metrics']['losing_trades'] = len(losing_trades)
            risk_report['metrics']['winning_trades'] = len(winning_trades)
            
            if len(losing_trades) > 3:
                risk_report['allow_new_trades'] = False
                risk_report['reasons'].append(f"too_many_losers_{len(losing_trades)}")
                risk_report['risk_level'] = 'medium'
            
            # 5. Check average loss
            if losing_trades:
                avg_loss = np.mean([t.calc_profit_ratio() for t in losing_trades])
                risk_report['metrics']['avg_loss'] = avg_loss
                
                if avg_loss < -0.05:  # Average loss > 5%
                    risk_report['allow_new_trades'] = False
                    risk_report['reasons'].append(f"high_avg_loss_{avg_loss:.1%}")
            
            # 6. Time exposure check
            oldest_trade = min(open_trades, key=lambda t: t.open_date_utc) if open_trades else None
            if oldest_trade:
                hours_open = (datetime.now() - oldest_trade.open_date_utc.replace(tzinfo=None)).total_seconds() / 3600
                if hours_open > 72:  # 3 days
                    risk_report['risk_level'] = 'medium'
                    risk_report['reasons'].append(f"old_positions_{hours_open:.0f}h")
            
            # 7. Volatility exposure
            high_volatility_trades = 0
            for trade in open_trades:
                df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                if not df.empty and df.iloc[-1].get('volatility', 0) > HIGH_VOLATILITY_THRESHOLD:
                    high_volatility_trades += 1
            
            if high_volatility_trades > 2:
                risk_report['allow_new_trades'] = False
                risk_report['reasons'].append(f"high_volatility_exposure_{high_volatility_trades}")
            
            # Determine overall risk level
            if not risk_report['allow_new_trades']:
                if len(risk_report['reasons']) > 2:
                    risk_report['risk_level'] = 'high'
                else:
                    risk_report['risk_level'] = 'medium'
            
            self._last_portfolio_check = current_time
            self._cached_risk_report = risk_report
            return risk_report['allow_new_trades'], risk_report
            
        except Exception as e:
            logger.error(f"Portfolio risk check failed: {e}")
            return True, {'allow_new_trades': True, 'risk_level': 'unknown', 'error': str(e)}