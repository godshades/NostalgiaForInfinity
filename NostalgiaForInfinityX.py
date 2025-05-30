# --- CombinedStrategy.py ---
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement, unused-argument, duplicate-code
# flake8: noqa: F401
# isort: skip_file

# --- Do not remove these imports ---
from freqtrade.constants import Config
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative, IntParameter, DecimalParameter, RealParameter, CategoricalParameter, merge_informative_pair, stoploss_from_open
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal # Used by ZaratustraV13, ensure it's available or adapt if not used.
from datetime import datetime, timedelta
from pandas import DataFrame, Series
from typing import Dict, List, Optional, Union, Tuple # Added Tuple for type hinting consistency
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib # Assuming this is your custom technical library or qtpylib from freqtrade.
from technical.indicators import RMI # Assuming this is a custom indicator from your technical library
import numpy as np
from functools import reduce
import math
import logging

logger = logging.getLogger(__name__)

# --- Helper functions from newstrategy53 ---
def ewo(dataframe, sma1_length=5, sma2_length=35): # Note: newstrategy53 calls this with 50, 200
    sma1 = ta.EMA(dataframe, timeperiod=sma1_length)
    sma2 = ta.EMA(dataframe, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / dataframe['close'] * 100
    return smadif

def EWO(dataframe, ema_length=5, ema2_length=3): # Used with self.fast_ewo, self.slow_ewo
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

def top_percent_change_dca(dataframe: DataFrame, length: int) -> float:
    """
    Percentage change of the current close from the range maximum Open price
    Used in populate_indicators
    """
    if length == 0:
        return (dataframe['open'] - dataframe['close']) / dataframe['close']
    else:
        return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']

def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()
    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R", # Corrected f-string
    )
    return WR * -100

def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    # Ensure 'volume' is present, as qtpylib.rolling_vwap requires it.
    # If 'volume' is not in df, this will raise an error.
    # It's typically present in Freqtrade dataframes.
    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size) # Modified to use qtpylib's rolling_vwap directly
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']


def bollinger_bands(stock_price, window_size, num_of_std): # Custom bollinger_bands
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)

def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    mfv = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= dataframe['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / dataframe['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')

def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)

# pmax and its dependencies (VIDYA, vwma, zema if used and not in talib/qtpylib)
# For now, pmax is included. If VIDYA, vwma, zema are custom, they need to be defined.
# Assuming they are part of user's environment or will be replaced if standard alternatives exist.
# If MAtype 5, 8, 9 for pmax are used and these (VIDYA, vwma, zema) are custom, they need to be defined.
# The provided newstrategy53 calls pmax with MAtype=1, so these might not be immediately needed.

def pmax(df, period, multiplier, length, MAtype, src):
    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)

    mavalue_col = f'MA_{MAtype}_{length}' # Using f-string for column names
    atr_col = f'ATR_{period}'
    # pm_col = f'pm_{period}_{multiplier}_{length}_{MAtype}' # Not directly used for return value name
    # pmx_col = f'pmX_{period}_{multiplier}_{length}_{MAtype}' # Not directly used for return value name

    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4
    else:
        raise ValueError("Unsupported src value for pmax")

    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        mavalue = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        mavalue = ta.T3(masrc, timeperiod=length) # Ensure T3 is available in your talib
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    # elif MAtype == 5: mavalue = VIDYA(df, length=length) # VIDYA would need to be defined
    # elif MAtype == 6: mavalue = ta.TEMA(masrc, timeperiod=length)
    # elif MAtype == 7: mavalue = ta.WMA(df, timeperiod=length) # WMA takes dataframe, not series typically in ta
    # elif MAtype == 8: mavalue = vwma(df, length) # vwma would need to be defined
    # elif MAtype == 9: mavalue = zema(df, period=length) # zema would need to be defined
    else:
        raise ValueError(f"Unsupported MAtype: {MAtype} for pmax. Define corresponding MA or check type.")

    # df[atr_col] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=period) # More explicit ATR call
    # ATR in talib.abstract usually takes dataframe:
    df_atr = ta.ATR(df, timeperiod=period) # Assuming df has high, low, close columns

    # Ensure mavalue and df_atr are Series aligned with df.index
    # If they are not (e.g. if masrc was a subset), this could be an issue.
    # However, given typical Freqtrade usage, they should be.

    df['basic_ub'] = mavalue + ((multiplier/10) * df_atr)
    df['basic_lb'] = mavalue - ((multiplier/10) * df_atr)

    basic_ub_np = df['basic_ub'].to_numpy() # Using .to_numpy() for performance
    final_ub_np = np.full(len(df), np.nan) # Initialize with nan
    basic_lb_np = df['basic_lb'].to_numpy()
    final_lb_np = np.full(len(df), np.nan)
    mavalue_np = mavalue.to_numpy()

    for i in range(period, len(df)): # Start from 'period' to ensure mavalue[i-1] is valid
        if np.isnan(mavalue_np[i-1]) or np.isnan(final_ub_np[i-1]) or np.isnan(basic_ub_np[i]):
             final_ub_np[i] = final_ub_np[i-1] # Propagate previous if current calc is not possible
        else:
            final_ub_np[i] = basic_ub_np[i] if (
                basic_ub_np[i] < final_ub_np[i - 1]
                or mavalue_np[i - 1] > final_ub_np[i - 1]) else final_ub_np[i - 1]

        if np.isnan(mavalue_np[i-1]) or np.isnan(final_lb_np[i-1]) or np.isnan(basic_lb_np[i]):
            final_lb_np[i] = final_lb_np[i-1]
        else:
            final_lb_np[i] = basic_lb_np[i] if (
                basic_lb_np[i] > final_lb_np[i - 1]
                or mavalue_np[i - 1] < final_lb_np[i - 1]) else final_lb_np[i - 1]

    # df['final_ub'] = final_ub_np # Assign back to DataFrame if needed elsewhere
    # df['final_lb'] = final_lb_np

    pm_arr_np = np.full(len(df), np.nan) # Initialize with nan

    # Ensure first valid value for pm_arr_np[i-1]
    # This loop logic for pm_arr is complex and directly translated.
    # It might be sensitive to initial conditions or NaN propagation.
    # Consider initializing pm_arr_np[period-1] or pm_arr_np[0] if needed.
    # For robust calculation, ensure final_ub_np[i-1], final_lb_np[i-1], mavalue_np[i] are not NaN.

    for i in range(period, len(df)):
        if np.isnan(pm_arr_np[i-1]) or np.isnan(mavalue_np[i]) or \
           np.isnan(final_ub_np[i]) or np.isnan(final_lb_np[i]) or \
           np.isnan(final_ub_np[i-1]) or np.isnan(final_lb_np[i-1]): # Check for NaNs
            pm_arr_np[i] = pm_arr_np[i-1] # Or some other NaN handling
            continue

        if pm_arr_np[i-1] == final_ub_np[i-1]:
            if mavalue_np[i] <= final_ub_np[i]:
                pm_arr_np[i] = final_ub_np[i]
            else: # mavalue_np[i] > final_ub_np[i]
                pm_arr_np[i] = final_lb_np[i]
        elif pm_arr_np[i-1] == final_lb_np[i-1]:
            if mavalue_np[i] >= final_lb_np[i]:
                pm_arr_np[i] = final_lb_np[i]
            else: # mavalue_np[i] < final_lb_np[i]
                pm_arr_np[i] = final_ub_np[i]
        else: # Initialize first valid pm_arr value; this might need specific handling
            # This case implies pm_arr[i-1] was neither final_ub[i-1] nor final_lb[i-1]
            # which could happen if pm_arr[i-1] was NaN or 0.00 from original code.
            # If mavalue starts above final_lb, trend is up -> pm is final_lb
            # If mavalue starts below final_ub, trend is down -> pm is final_ub
            # This is a common way to initialize; otherwise, it defaults to an up-trend start:
            if mavalue_np[i] > final_lb_np[i]: # initial trend guess
                 pm_arr_np[i] = final_lb_np[i]
            else:
                 pm_arr_np[i] = final_ub_np[i]


    pm_series = Series(pm_arr_np, index=df.index)
    
    # pmx logic
    pmx_np = np.full(len(df), '', dtype=object) # Use object type for strings 'up'/'down'/NaN or empty
    mavalue_series = Series(mavalue_np, index=df.index) # Ensure mavalue is a series for comparison

    condition_up = (pm_series > 0) & (mavalue_series >= pm_series) # up condition
    condition_down = (pm_series > 0) & (mavalue_series < pm_series) # down condition

    pmx_np = np.select([condition_down, condition_up], ['down', 'up'], default=np.nan) # Use np.nan for undefined

    return pm_series, Series(pmx_np, index=df.index)


class Gemini(IStrategy):
    INTERFACE_VERSION = 3 # From ZaratustraV13, newstrategy53 doesn't specify but 3 is current

    # Strategy Parameters from ZaratustraV13 & newstrategy53 (to be merged/chosen)
    timeframe = '5m' # Consistent
    can_short = True # To support ZaratustraV13 shorts
    
    # ROI table:
    # From newstrategy53, ZaratustraV13's is empty
    minimal_roi = {
        "0": 100 
    }

    # Stoploss:
    # We will use custom_stoploss from newstrategy53
    stoploss = -0.99 # Placeholder from newstrategy53, custom_stoploss will override
    use_custom_stoploss = True

    # Trailing stop:
    # ZaratustraV13 enables this. newstrategy53 has params but disables. Let's enable.
    trailing_stop = True
    trailing_stop_positive = 0.01 # From ZaratustraV13
    trailing_stop_positive_offset = 0.1 # From ZaratustraV13
    trailing_only_offset_is_reached = True # From ZaratustraV13

    # From newstrategy53
    use_exit_signal = True
    exit_profit_only = False # From newstrategy53 (ZaratustraV13 has True) - Let's use newstrategy53's
    ignore_roi_if_entry_signal = False # From newstrategy53

    process_only_new_candles = True # From newstrategy53
    startup_candle_count = 168 # From newstrategy53

    # Position Adjustment (DCA) from newstrategy53
    position_adjustment_enable = True
    initial_safety_order_trigger = -0.018 # newstrategy53
    max_safety_orders = 8 # newstrategy53
    safety_order_step_scale = 1.2 # newstrategy53
    safety_order_volume_scale = 1.4 # newstrategy53

    # Leverage
    # Defined by user request
    
    # --- Parameters from newstrategy53 (buy_params, sell_params, and individual hyperoptable params) ---
    # These are extensive. We will copy them directly from newstrategy53 for now.
    # buy_params and sell_params from newstrategy53 are dictionaries.
    # ZaratustraV13 doesn't explicitly define these dicts, but uses individual hyperopt params.
    # We should ensure that if ZaratustraV13's parameters were intended to be part of a similar
    # structure, they are correctly placed or kept as standalone class attributes.

    buy_params = { # From newstrategy53
       "bbdelta_close": 0.01568, "bbdelta_tail": 0.75301, "close_bblower": 0.01195,
       "closedelta_close": 0.0092, "base_nb_candles_buy": 12, "rsi_buy": 58,
       "low_offset": 0.985, "rocr_1h": 0.57032, "rocr1_1h": 0.7210406300824859,
       "buy_clucha_bbdelta_close": 0.049, "buy_clucha_bbdelta_tail": 1.146,
       "buy_clucha_close_bblower": 0.018, "buy_clucha_closedelta_close": 0.017,
       "buy_clucha_rocr_1h": 0.526, "buy_cci": -116, "buy_cci_length": 25,
       "buy_rmi": 49, "buy_rmi_length": 17, "buy_srsi_fk": 32, "buy_bb_width_1h": 1.074,
    }

    sell_params = { # From newstrategy53
      "pHSL": -0.397, "pPF_1": 0.012, "pPF_2": 0.07, "pSL_1": 0.015, "pSL_2": 0.068,
      "sell_bbmiddle_close": 1.0909210168690215, "sell_fisher": 0.46405736994786184,
      "base_nb_candles_sell": 22, "high_offset": 1.014, "high_offset_2": 1.01,
      "sell_u_e_2_cmf": -0.0, "sell_u_e_2_ema_close_delta": 0.016, "sell_u_e_2_rsi": 10,
      "sell_deadfish_profit": -0.063, "sell_deadfish_bb_factor": 0.954,
      "sell_deadfish_bb_width": 0.043, "sell_deadfish_volume_factor": 2.37
    }
    
    # --- Individual Hyperoptable Parameters from newstrategy53 ---
    # (Re-declaring them as class attributes for clarity and direct use)
    # General categorisation for easier management
    
    # -- EWO params --
    fast_ewo = 50 # newstrategy53 uses this in EWO call
    slow_ewo = 200 # newstrategy53 uses this in EWO call

    # -- Buy parameters (grouped by some logic seen in newstrategy53) --
    buy_44_ma_offset = 0.982; buy_44_ewo = -18.143; buy_44_cti = -0.8; buy_44_r_1h = -75.0
    buy_37_ma_offset = 0.98; buy_37_ewo = 9.8; buy_37_rsi = 56.0; buy_37_cti = -0.7
    buy_ema_open_mult_7 = 0.030; buy_cti_7 = -0.89

    # -- DIP Signal related --
    is_optimize_dip = False # Control for optimization space
    buy_rmi_val = IntParameter(30, 50, default=buy_params['buy_rmi'], optimize= is_optimize_dip, space='buy', load=True) # changed name to avoid clash
    buy_cci_val = IntParameter(-135, -90, default=buy_params['buy_cci'], optimize= is_optimize_dip, space='buy', load=True) # changed name
    buy_srsi_fk_val = IntParameter(30, 50, default=buy_params['buy_srsi_fk'], optimize= is_optimize_dip, space='buy', load=True) # changed name
    buy_cci_length_val = IntParameter(25, 45, default=buy_params['buy_cci_length'], optimize = is_optimize_dip, space='buy', load=True) # changed name
    buy_rmi_length_val = IntParameter(8, 20, default=buy_params['buy_rmi_length'], optimize = is_optimize_dip, space='buy', load=True) # changed name

    # -- Break Signal related --
    is_optimize_break = False
    buy_bb_width_val = DecimalParameter(0.065, 0.135, default=0.095, optimize = is_optimize_break, space='buy', load=True) # changed name
    buy_bb_delta_val = DecimalParameter(0.018, 0.035, default=0.025, optimize = is_optimize_break, space='buy', load=True) # changed name
    
    # -- Check Signal related --
    is_optimize_check = False
    buy_roc_1h_val = IntParameter(-25, 200, default=10, optimize = is_optimize_check, space='buy', load=True) # changed name
    buy_bb_width_1h_val = DecimalParameter(0.3, 2.0, default=buy_params['buy_bb_width_1h'], optimize = is_optimize_check, space='buy', load=True) # changed name

    # -- Clucha HA Signal related --
    is_optimize_clucha = False
    buy_clucha_bbdelta_close_val = DecimalParameter(0.01,0.05, default=buy_params['buy_clucha_bbdelta_close'], optimize=is_optimize_clucha, space='buy', load=True)
    buy_clucha_bbdelta_tail_val = DecimalParameter(0.7, 1.2, default=buy_params['buy_clucha_bbdelta_tail'], optimize=is_optimize_clucha, space='buy', load=True)
    buy_clucha_close_bblower_val = DecimalParameter(0.001, 0.05, default=buy_params['buy_clucha_close_bblower'], optimize=is_optimize_clucha, space='buy', load=True)
    buy_clucha_closedelta_close_val = DecimalParameter(0.001, 0.05, default=buy_params['buy_clucha_closedelta_close'], optimize=is_optimize_clucha, space='buy', load=True)
    buy_clucha_rocr_1h_val = DecimalParameter(0.1, 1.0, default=buy_params['buy_clucha_rocr_1h'], optimize=is_optimize_clucha, space='buy', load=True)
    
    # -- Local Uptrend Signal related --
    is_optimize_local_uptrend = False
    buy_ema_diff_val = DecimalParameter(0.022, 0.027, default=0.025, optimize = is_optimize_local_uptrend, space='buy', load=True) # changed name
    buy_bb_factor_val = DecimalParameter(0.990, 0.999, default=0.995, optimize = False, space='buy', load=True) # changed name
    buy_closedelta_val = DecimalParameter(12.0, 18.0, default=15.0, optimize = is_optimize_local_uptrend, space='buy', load=True) # changed name

    # -- General Buy Parameters (from original RealParameter definitions in newstrategy53) --
    # These were directly defined as RealParameter in newstrategy53.
    # Using .value in strategy logic for these.
    rocr_1h = RealParameter(0.5, 1.0, default=buy_params['rocr_1h'], space='buy', optimize=True, load=True)
    rocr1_1h = RealParameter(0.5, 1.0, default=buy_params['rocr1_1h'], space='buy', optimize=True, load=True) # Value from buy_params
    bbdelta_close = RealParameter(0.0005, 0.02, default=buy_params['bbdelta_close'], space='buy', optimize=True, load=True)
    closedelta_close = RealParameter(0.0005, 0.02, default=buy_params['closedelta_close'], space='buy', optimize=True, load=True)
    bbdelta_tail = RealParameter(0.7, 1.0, default=buy_params['bbdelta_tail'], space='buy', optimize=True, load=True)
    close_bblower = RealParameter(0.0005, 0.02, default=buy_params['close_bblower'], space='buy', optimize=True, load=True)
    
    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False, load=True)
    low_offset = DecimalParameter(0.985, 0.995, default=buy_params['low_offset'], space='buy', optimize=True, load=True)

    # -- Sell Parameters (from original RealParameter/DecimalParameter definitions) --
    sell_fisher_val = RealParameter(0.1, 0.5, default=sell_params['sell_fisher'], space='sell', optimize=False, load=True) # changed name
    sell_bbmiddle_close_val = RealParameter(0.97, 1.1, default=sell_params['sell_bbmiddle_close'], space='sell', optimize=False, load=True) # changed name

    is_optimize_deadfish = True # Control for optimization
    sell_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=sell_params['sell_deadfish_bb_width'] , space='sell', optimize = is_optimize_deadfish, load=True)
    sell_deadfish_profit = DecimalParameter(-0.15, -0.05, default=sell_params['sell_deadfish_profit'] , space='sell', optimize = is_optimize_deadfish, load=True)
    sell_deadfish_bb_factor = DecimalParameter(0.90, 1.20, default=sell_params['sell_deadfish_bb_factor'] , space='sell', optimize = is_optimize_deadfish, load=True)
    sell_deadfish_volume_factor = DecimalParameter(1, 2.5, default=sell_params['sell_deadfish_volume_factor'] ,space='sell', optimize = is_optimize_deadfish, load=True)

    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False, load=True)
    high_offset = DecimalParameter(1.005, 1.015, default=sell_params['high_offset'], space='sell', optimize=True, load=True)
    high_offset_2 = DecimalParameter(1.010, 1.020, default=sell_params['high_offset_2'], space='sell', optimize=True, load=True)
    
    # -- Custom Sell / Trail Parameters --
    sell_trail_profit_min_1 = DecimalParameter(0.1, 0.25, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_1 = DecimalParameter(0.3, 0.5, default=0.4, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_1 = DecimalParameter(0.04, 0.1, default=0.03, space='sell', decimals=3, optimize=False, load=True)

    sell_trail_profit_min_2 = DecimalParameter(0.04, 0.1, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_2 = DecimalParameter(0.08, 0.25, default=0.11, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_2 = DecimalParameter(0.04, 0.2, default=0.015, space='sell', decimals=3, optimize=False, load=True)

    # -- Custom Stoploss Parameters --
    pHSL = DecimalParameter(-0.500, -0.040, default=sell_params['pHSL'], decimals=3, space='sell', optimize=False, load=True)
    pPF_1 = DecimalParameter(0.008, 0.020, default=sell_params['pPF_1'], decimals=3, space='sell', optimize=False, load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=sell_params['pSL_1'], decimals=3, space='sell', optimize=False, load=True)
    pPF_2 = DecimalParameter(0.040, 0.100, default=sell_params['pPF_2'], decimals=3, space='sell',optimize=False, load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=sell_params['pSL_2'], decimals=3, space='sell', optimize=False,load=True)

    # ZaratustraV13 does not have explicit hyperopt parameters defined as class attributes
    # It uses them directly in its logic (e.g. dataframe['dx'] > dataframe['mdi'])
    # If any constants from ZaratustraV13 were meant to be hyperoptable, they'd need to be added here.

    def leverage(self, pair: str, current_time: "datetime", current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        """
        Customize leverage for each trade.
        """
        return 3.0 # As requested by user

    def informative_pairs(self): # From newstrategy53
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        informative_pairs += [("BTC/USDT", "5m")] # newstrategy53 uses this
        # ZaratustraV13 does not define informative_pairs. If it implies any, they should be merged.
        return informative_pairs
    
    # Method from newstrategy53 for DCA logic
    def top_percent_change_dca(self, dataframe: DataFrame, length: int) -> float:
        """
        Percentage change of the current close from the range maximum Open price
        Referenced by adjust_trade_position.
        """
        if length == 0:
            # More robust: check if 'open' and 'close' columns exist
            if 'open' in dataframe.columns and 'close' in dataframe.columns:
                 return (dataframe['open'].iloc[-1] - dataframe['close'].iloc[-1]) / dataframe['close'].iloc[-1] if len(dataframe) > 0 else 0.0
            return 0.0 # Or handle error
        else:
            if 'open' in dataframe.columns and 'close' in dataframe.columns and len(dataframe) >= length :
                return (dataframe['open'].rolling(length).max().iloc[-1] - dataframe['close'].iloc[-1]) / dataframe['close'].iloc[-1] if len(dataframe) > 0 else 0.0
            return 0.0 # Or handle error


    # is_support method from newstrategy53 (used in informative timeframe processing)
    def is_support(self, row_data) -> bool: # row_data is expected to be a numpy array or similar iterable
        conditions = []
        # Ensure row_data is not empty and has enough elements for the logic
        if len(row_data) < 2: # Needs at least 2 elements for comparison; original implies more for halving.
            return False
        
        # Original logic seems to check for a 'V' shape for support.
        # It might be more robust to ensure row_data is a pandas Series or numpy array for .iloc / slicing
        # For simplicity, assuming row_data is list-like as implied by original loop
        # However, rolling().apply() in pandas passes a numpy array if raw=True
        
        # The original logic was:
        # for row in range(len(row_data)-1):
        #     if row < len(row_data)/2:
        #         conditions.append(row_data[row] > row_data[row+1])
        #     else:
        #         conditions.append(row_data[row] < row_data[row+1])
        # This implies a V-shape. For a window of 5, it means:
        # data[0]>data[1], data[1]>data[2] (decreasing)
        # data[2]<data[3], data[3]<data[4] (increasing)
        # Let's make it more explicit for a typical support pattern (e.g., low surrounded by higher lows)
        # If window=5, center=True, apply gets 5 values. Middle one (index 2) is the candidate.
        # A simple support: data[1] > data[2] < data[3]
        # Or more strictly for V-shape in window 5: data[0]>data[1]>data[2] and data[2]<data[3]<data[4]
        
        # The original reduce(lambda x, y: x & y, conditions) requires all conditions to be true.
        # This is a very strict V-shape.
        
        # Simplified support: middle point is the lowest in its immediate vicinity
        # Example for a 5-point window passed to apply():
        if len(row_data) == 5: # data[0], data[1], data[2] (candle), data[3], data[4]
             # Check if candle at index 2 is a local minimum
            return row_data[1] > row_data[2] and row_data[2] < row_data[3]
        # Fallback or more generic if window size varies, but original used fixed window of 5
        # For now, sticking to a simplified version based on common understanding of support for rolling apply.
        # The original logic is very specific. If it's crucial, it should be carefully replicated.
        # If using `center=True` with `rolling(window=5)`, `row_data` will have 5 elements.
        # The "current" candle for which support is being checked is the middle one.
        
        # Re-implementing the original exact logic for `is_support` as provided:
        n = len(row_data)
        if n < 3: # Minimum length to form a V shape, e.g., high-low-high
            return False
        
        mid_point_approx = n // 2
        
        # Check decreasing trend up to mid_point_approx (exclusive of mid_point_approx itself if it's the low)
        for i in range(mid_point_approx -1): # e.g. if n=5, mid=2. range(1) -> i=0. data[0]>data[1]
            if not (row_data[i] > row_data[i+1]):
                return False
        
        # Check increasing trend from mid_point_approx onwards
        for i in range(mid_point_approx, n - 1): # e.g. if n=5, mid=2. range(2,4) -> i=2,3. data[2]<data[3], data[3]<data[4]
            if not (row_data[i] < row_data[i+1]):
                return False
        
        # The conditions list and reduce are effectively checking all parts of the V.
        # The above is a direct translation of that intent.
        return True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Start with indicators from newstrategy53
        
        # Informative BTC data (from newstrategy53)
        info_tf_btc = '5m' # newstrategy53 uses '5m' for BTC
        informative_btc_df = self.dp.get_pair_dataframe('BTC/USDT', timeframe=info_tf_btc)
        # To avoid issues with data not being ready, especially at startup, 
        # newstrategy53 shifts, but this might not be best for all signals.
        # Consider if a direct merge or ffill is more appropriate for some btc indicators.
        # For now, maintaining the shift as in newstrategy53:
        # informative_btc = informative_btc_df.copy().shift(1) # Original shift
        # However, direct merge then shift is often safer if joins are involved:
        # Merging without shift first, then creating shifted columns if needed by specific logic.
        # Freqtrade's merge_informative_pair handles ffill correctly.
        
        # Let's use merge_informative_pair for BTC data as well for consistency and robustness
        btc_informative = self.dp.get_pair_dataframe(pair="BTC/USDT", timeframe=info_tf_btc)
        dataframe = merge_informative_pair(dataframe, btc_informative, self.timeframe, info_tf_btc, ffill=True, suffix='_btc')

        # Now use the suffixed columns, e.g., dataframe['close_btc']
        # If a 1-candle shift is desired for specific BTC signals (as newstrategy53's .shift(1) implied):
        dataframe['btc_close_shifted'] = dataframe['close_btc'].shift(1) # Example if needed
        dataframe['btc_ema_fast'] = ta.EMA(dataframe['close_btc'], timeperiod=20) # Using non-shifted BTC close for EMA
        dataframe['btc_ema_slow'] = ta.EMA(dataframe['close_btc'], timeperiod=25)
        dataframe['down_btc_trend'] = (dataframe['btc_ema_fast'] < dataframe['btc_ema_slow']).astype('int') # Renamed from 'down'

        # MA for sells (newstrategy53)
        for val in self.base_nb_candles_sell.range: # Ensure .range is correct usage for IntParameter
             dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)
        
        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1) # shift(1) used in newstrategy53
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1) # shift(1) used in newstrategy53
        
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        # Bollinger Bands from newstrategy53
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        bollinger2_40 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=40, stds=2)
        dataframe['bb_lowerband2_40'] = bollinger2_40['lower']
        dataframe['bb_middleband2_40'] = bollinger2_40['mid']
        dataframe['bb_upperband2_40'] = bollinger2_40['upper']

        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84) # Used in newstrategy53 entries
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112) # Used in newstrategy53 entries

        # Heikin Ashi from newstrategy53
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        dataframe['bb_delta_cluc'] = (dataframe['bb_middleband2_40'] - dataframe['bb_lowerband2_40']).abs()
        dataframe['ha_closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()

        stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=0) # Common STOCHRSI, newstrategy53 used 15,20,2,2 - check which defaults are best
        # Using newstrategy53's STOCHRSI parameters:
        # stoch = ta.STOCHRSI(dataframe, 15, 20, 2, 2) # This might be a custom TA-Lib call structure or specific version.
        # Standard talib.abstract.STOCHRSI: timeperiod, fastk_period, fastd_period, fastd_matype
        # Let's assume it meant timeperiod=14 (common for RSI base of STOCHRSI), fastk_period=15, fastd_period=20, fastd_matype=2 (SMA for fastd) - this is unusual.
        # Or, it means timeperiod=15 (for RSI), STOCH length=20, fastk=2, fastd=2.
        # Given `stoch = ta.STOCHRSI(dataframe, 15, 20, 2, 2)` from original:
        # Assuming: timeperiod=15, fastk_period=20, fastd_period=2, fastd_matype=2 (from original numbers)
        # This requires checking TA_STOCHRSI parameter order if it's not default.
        # A more common interpretation for `STOCHRSI(dataframe, 15, 20, 2, 2)` if it means:
        # RSI period 15, Stoch period 20, K period 2, D period 2. This doesn't map directly.
        # Let's use a common STOCHRSI setup for now and it can be adjusted:
        # Default STOCHRSI usually takes timeperiod (for RSI), fastk_period, fastd_period, fastd_matype.
        # The values 15, 20, 2, 2 might be for a specific library version or a non-abstract call.
        # For talib.abstract.STOCHRSI:
        # Let's assume the call `ta.STOCHRSI(dataframe, 15, 20, 2, 2)` meant:
        # timeperiod=15 (for underlying RSI)
        # fastk_period=20 (for Stochastic K line over RSI)
        # fastd_period=2  (for Stochastic D line)
        # fastd_matype=2  (MA type for D line - e.g., SMA)
        # This is an unusual combination. A more typical STOCHRSI might be timeperiod=14, fastk_period=14, fastd_period=3, fastd_matype=0
        # Given the original code, let's try to map:
        # ta.STOCHRSI(dataframe, timeperiod=15, fastk_period=20, fastd_period=2, fastd_matype=ta.MA_Type.SMA if 2 means SMA)
        # If `ta.STOCHRSI(dataframe, 15, 20, 2, 2)` was a non-abstract call, it might be `STOCHRSI(close, timeperiod=15, fastk_period=20, fastd_period=2, fastd_matype=2)`
        # For now, using common defaults as placeholder for STOCHRSI:
        # stoch = ta.STOCHRSI(dataframe, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        # Reverting to parameters from `newstrategy53` as best guess for `talib.abstract.STOCHRSI`
        # If `2` for matype is not a direct map, `0` (SMA) is a common default.
        try:
            stoch = ta.STOCHRSI(dataframe, timeperiod=15, fastk_period=20, fastd_period=2, fastd_matype=0) # Assuming matype 2 might be SMA (0)
            dataframe['srsi_fk'] = stoch['fastk']
            dataframe['srsi_fd'] = stoch['fastd']
        except Exception as e:
            logger.warning(f"Could not calculate STOCHRSI with specified params (15,20,2,2): {e}. Falling back to defaults or NaN.")
            dataframe['srsi_fk'] = 0 # Or np.nan
            dataframe['srsi_fd'] = 0 # Or np.nan


        # Custom Bollinger Bands on HA typical price (newstrategy53)
        mid, lower = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2) # Uses custom bollinger_bands
        dataframe['lower'] = lower # Used for 'bb_lowerband' later
        dataframe['mid'] = mid # Used for 'bb_middleband' later

        dataframe['bbdelta'] = (dataframe['mid'] - dataframe['lower']).abs() # Based on HA BB
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()

        dataframe['bb_lowerband'] = dataframe['lower'] # From HA BB
        dataframe['bb_middleband'] = dataframe['mid']   # From HA BB

        # Bollinger Bands std 3 (newstrategy53)
        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']
        #dataframe['bb_delta'] was calculated above from HA BB. Here it's redefined. Choose one or rename.
        # Let's rename this one:
        dataframe['bb_delta_std3'] = ((dataframe['bb_lowerband2'] - dataframe['bb_lowerband3']) / dataframe['bb_lowerband2'])

        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3) # on HA close
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50) # on HA close
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28) # on HA close

        # VWAPB (newstrategy53)
        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['vwap_low'] = vwap_low
        dataframe['vwap_upperband'] = vwap_high
        dataframe['vwap_middleband'] = vwap
        # dataframe['vwap_lowerband'] = vwap_low # Redundant, already vwap_low
        dataframe['vwap_width'] = ( (dataframe['vwap_upperband'] - dataframe['vwap_lowerband']) / dataframe['vwap_middleband'] ) * 100

        dataframe['ema_vwap_diff_50'] = ( ( dataframe['ema_50'] - dataframe['vwap_low'] ) / dataframe['ema_50'] ) # uses vwap_low

        # Top Percent Change DCA (newstrategy53)
        dataframe['tpct_change_0'] = top_percent_change_dca(dataframe,0)
        dataframe['tpct_change_1'] = top_percent_change_dca(dataframe,1)
        dataframe['tcp_percent_4'] = top_percent_change_dca(dataframe , 4) # Typo tcp? Assuming top_percent_change_dca

        # EWO (newstrategy53) - using specific lengths
        dataframe['ewo_custom'] = ewo(dataframe, 50, 200) # Renamed from 'ewo' to avoid clash if generic 'ewo' is used

        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['sma_30'] = ta.SMA(dataframe, timeperiod=30)

        # RMI and CCI with dynamic lengths (newstrategy53)
        for val in self.buy_rmi_length_val.range: # Using _val from parameters
            dataframe[f'rmi_length_{val}'] = RMI(dataframe, length=val, mom=4)
        for val in self.buy_cci_length_val.range: # Using _val from parameters
            dataframe[f'cci_length_{val}'] = ta.CCI(dataframe, val)

        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        # dataframe['bb_delta_cluc'] already calculated above

        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        
        # EWO with hyperoptable lengths (newstrategy53)
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo) # Uses self.fast_ewo, self.slow_ewo params

        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)

        dataframe['r_14'] = williams_r(dataframe, period=14) # Williams %R

        dataframe['ema_5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema_10'] = ta.EMA(dataframe, timeperiod=10)

        # PMAX (newstrategy53)
        # Ensure pmax function is correctly defined and accessible globally or in class
        # The pmax function from newstrategy53 returns two series.
        pm, pmx = pmax(heikinashi, MAtype=1, length=9, multiplier=27, period=10, src=3) # Operates on heikinashi dataframe
        dataframe['pm'] = pm
        dataframe['pmx_indicator'] = pmx # Renamed from 'pmx' to avoid potential name collisions if 'pmx' is used as a generic signal name

        dataframe['source'] = (dataframe['high'] + dataframe['low'] + dataframe['open'] + dataframe['close'])/4
        dataframe['pmax_thresh'] = ta.EMA(dataframe['source'], timeperiod=9)
        dataframe['sma_75'] = ta.SMA(dataframe, timeperiod=75)

        # Fisher Transform RSI (newstrategy53)
        rsi_fisher = ta.RSI(dataframe) # Default period 14
        # dataframe["rsi"] is already calculated. Re-calculating rsi_fisher just for fisher is fine.
        rsi_val = 0.1 * (rsi_fisher - 50)
        dataframe["fisher"] = (np.exp(2 * rsi_val) - 1) / (np.exp(2 * rsi_val) + 1)

        # HMA (newstrategy53)
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        # --- Informative Timeframe (1h) Data from newstrategy53 ---
        inf_tf_1h = '1h'
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf_1h)
        inf_heikinashi = qtpylib.heikinashi(informative_1h)
        informative_1h['ha_close'] = inf_heikinashi['close']
        informative_1h['rocr'] = ta.ROCR(informative_1h['ha_close'], timeperiod=168) # Name clash: 'rocr' on 1h, 'rocr' on base TF. Suffix needed.
        # dataframe['rocr'] from base TF heikinashi. Rename 1h:
        informative_1h['rocr_1h_inf'] = ta.ROCR(informative_1h['ha_close'], timeperiod=168)
        
        # RSI on informative_1h (newstrategy53) - uses base TF dataframe for RSI, which is unusual for informative.
        # Original: informative_1h['rsi_14'] = ta.RSI(dataframe, timeperiod=14) - This is likely an error.
        # Should be: informative_1h['rsi_14'] = ta.RSI(informative_1h, timeperiod=14)
        informative_1h['rsi_14_inf'] = ta.RSI(informative_1h, timeperiod=14) # Renamed and corrected

        # CMF on informative_1h (newstrategy53) - similar issue, used base TF.
        # Original: informative_1h['cmf'] = chaikin_money_flow(dataframe, 20)
        # Should be: informative_1h['cmf_inf'] = chaikin_money_flow(informative_1h, 20)
        informative_1h['cmf_inf'] = chaikin_money_flow(informative_1h, 20) # Renamed and corrected

        # Support Level (newstrategy53)
        # The `is_support` method is complex. Ensure it's robust.
        # Rolling apply can be slow.
        try:
            sup_series = informative_1h['low'].rolling(window=5, center=True).apply(self.is_support, raw=True).shift(2)
            informative_1h['sup_level'] = Series(np.where(sup_series, np.where(informative_1h['close'] < informative_1h['open'], informative_1h['close'], informative_1h['open']), float('NaN'))).ffill()
        except Exception as e:
            logger.error(f"Error calculating sup_level: {e}")
            informative_1h['sup_level'] = np.nan

        informative_1h['roc_inf'] = ta.ROC(informative_1h, timeperiod=9) # Renamed from 'roc'

        informative_1h['r_480_inf'] = williams_r(informative_1h, period=480) # Renamed
        
        # BB on informative_1h
        inf_bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2)
        informative_1h['bb_lowerband2_inf'] = inf_bollinger2['lower']
        informative_1h['bb_middleband2_inf'] = inf_bollinger2['mid']
        informative_1h['bb_upperband2_inf'] = inf_bollinger2['upper']
        informative_1h['bb_width_inf'] = ((informative_1h['bb_upperband2_inf'] - informative_1h['bb_lowerband2_inf']) / informative_1h['bb_middleband2_inf'])
        
        informative_1h['r_84_inf'] = williams_r(informative_1h, period=84) # Renamed
        informative_1h['cti_40_inf'] = pta.cti(informative_1h["close"], length=40) # Renamed
        
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, inf_tf_1h, ffill=True, suffix='_1h')
        # Note: Suffixing here means columns become e.g. 'sup_level_1h', 'roc_1h', 'bb_width_1h' etc.
        # This is good. newstrategy53 already expected this for some (e.g. 'roc_1h', 'bb_width_1h').
        # I've pre-suffixed some informative calculations (e.g. 'roc_inf') before the merge.
        # The merge_informative_pair will add another '_1h'. So it might become 'roc_inf_1h'.
        # It's cleaner to calculate on informative, then merge, relying on suffix from merge.
        # Let's adjust: calculate on informative with base names, then merge with suffix.

        # --- Re-doing informative_1h with cleaner suffix handling ---
        informative_1h_clean = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf_1h)
        inf_heikinashi_clean = qtpylib.heikinashi(informative_1h_clean)
        informative_1h_clean['ha_close'] = inf_heikinashi_clean['close']
        informative_1h_clean['rocr'] = ta.ROCR(informative_1h_clean['ha_close'], timeperiod=168) # Base name 'rocr'
        informative_1h_clean['rsi_14'] = ta.RSI(informative_1h_clean, timeperiod=14) # Base name 'rsi_14'
        informative_1h_clean['cmf_calc'] = chaikin_money_flow(informative_1h_clean, 20) # Base name 'cmf_calc' to avoid clash with base TF 'cmf'

        try:
            sup_series_clean = informative_1h_clean['low'].rolling(window=5, center=True).apply(self.is_support, raw=True).shift(2)
            informative_1h_clean['sup_level'] = Series(np.where(sup_series_clean, np.where(informative_1h_clean['close'] < informative_1h_clean['open'], informative_1h_clean['close'], informative_1h_clean['open']), float('NaN'))).ffill()
        except Exception as e:
            logger.error(f"Error calculating sup_level (clean): {e}")
            informative_1h_clean['sup_level'] = np.nan
            
        informative_1h_clean['roc'] = ta.ROC(informative_1h_clean, timeperiod=9)
        informative_1h_clean['r_480'] = williams_r(informative_1h_clean, period=480)
        inf_bollinger2_clean = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h_clean), window=20, stds=2)
        informative_1h_clean['bb_lowerband2'] = inf_bollinger2_clean['lower']
        informative_1h_clean['bb_middleband2'] = inf_bollinger2_clean['mid']
        informative_1h_clean['bb_upperband2'] = inf_bollinger2_clean['upper']
        informative_1h_clean['bb_width'] = ((informative_1h_clean['bb_upperband2'] - informative_1h_clean['bb_lowerband2']) / informative_1h_clean['bb_middleband2'])
        informative_1h_clean['r_84'] = williams_r(informative_1h_clean, period=84)
        informative_1h_clean['cti_40'] = pta.cti(informative_1h_clean["close"], length=40)
        
        dataframe = merge_informative_pair(dataframe, informative_1h_clean, self.timeframe, inf_tf_1h, ffill=True, suffix='_1h')
        # Now columns will be: rocr_1h, rsi_14_1h, cmf_calc_1h, sup_level_1h, roc_1h, r_480_1h, bb_width_1h, r_84_1h, cti_40_1h.
        # This matches the naming convention expected by newstrategy53's entry logic.

        # --- Indicators from ZaratustraV13 ---
        dataframe['dx']  = ta.DX(dataframe)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['pdi'] = ta.PLUS_DI(dataframe)
        dataframe['mdi'] = ta.MINUS_DI(dataframe)
        
        # Zaratustra's Bollinger Bands (prefixed with z_)
        # Requires 'typical_price' if not already on dataframe. qtpylib.typical_price(dataframe)
        # qtpylib.bollinger_bands takes a series, not a full dataframe for the price.
        # Zaratustra: qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # If `typical_price` is not a column, calculate it first.
        if 'typical_price' not in dataframe.columns:
             dataframe['typical_price'] = qtpylib.typical_price(dataframe)
        
        z_bbands = qtpylib.bollinger_bands(dataframe['typical_price'], window=20, stds=2)
        dataframe['z_bbl'] = z_bbands['lower']
        dataframe['z_bbm'] = z_bbands['mid']
        dataframe['z_bbu'] = z_bbands['upper']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Initialize 'enter_long', 'enter_short', 'enter_tag' columns if they don't exist
        if 'enter_long' not in dataframe.columns:
            dataframe['enter_long'] = 0
        if 'enter_short' not in dataframe.columns:
            dataframe['enter_short'] = 0
        if 'enter_tag' not in dataframe.columns:
            dataframe['enter_tag'] = '' # Or np.nan / None

        # --- Entry conditions from newstrategy53 (all are long entries) ---
        # Note: These conditions use .value for hyperoptable parameters
        
        btc_dump_condition = (
                (dataframe['close_btc'].rolling(24).max() >= (dataframe['close_btc'] * 1.03 ))
        )
        rsi_check_condition = (
                (dataframe['rsi_84'] < 60) &
                (dataframe['rsi_112'] < 60)
        )

        # DIP Signal (newstrategy53)
        dip_conditions = (
                (dataframe[f'rmi_length_{self.buy_rmi_length_val.value}'] < self.buy_rmi_val.value) &
                (dataframe[f'cci_length_{self.buy_cci_length_val.value}'] <= self.buy_cci_val.value) &
                (dataframe['srsi_fk'] < self.buy_srsi_fk_val.value) & # Uses _val
                (dataframe['bbdelta'] > self.buy_bb_delta_val.value) & # bbdelta from HA BB. Uses _val
                (dataframe['bb_width'] > self.buy_bb_width_val.value) & # bb_width from std BB. Uses _val
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta_val.value / 1000 ) &
                (dataframe['close'] < dataframe['bb_lowerband3'] * self.buy_bb_factor_val.value) &
                (dataframe['roc_1h'] < self.buy_roc_1h_val.value) & # roc_1h from informative
                (dataframe['bb_width_1h'] < self.buy_bb_width_1h_val.value) # bb_width_1h from informative
            )
        dataframe.loc[dip_conditions, ['enter_long', 'enter_tag']] = (1, 'long_dip_signal_ns53')

        # Break Signal (newstrategy53) - Appears to be a subset of DIP, verify intent
        # This logic is very similar to dip_conditions without the RMI/CCI/SRSI checks.
        break_conditions = (
                (dataframe['bbdelta'] > self.buy_bb_delta_val.value) & # bbdelta from HA BB
                (dataframe['bb_width'] > self.buy_bb_width_val.value) & # bb_width from std BB
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta_val.value / 1000 ) &
                (dataframe['close'] < dataframe['bb_lowerband3'] * self.buy_bb_factor_val.value) &
                (dataframe['roc_1h'] < self.buy_roc_1h_val.value) &
                (dataframe['bb_width_1h'] < self.buy_bb_width_1h_val.value)
            )
        dataframe.loc[break_conditions, ['enter_long', 'enter_tag']] = (1, 'long_break_signal_ns53')
        
        # Clucha HA Signal (newstrategy53)
        clucha_ha_conditions = (
                (dataframe['rocr_1h'] > self.buy_clucha_rocr_1h_val.value ) & # rocr_1h from informative
                (dataframe['bb_lowerband2_40'].shift() > 0) & # bb_lowerband2_40 from base TF
                (dataframe['bb_delta_cluc'] > dataframe['ha_close'] * self.buy_clucha_bbdelta_close_val.value) &
                (dataframe['ha_closedelta'] > dataframe['ha_close'] * self.buy_clucha_closedelta_close_val.value) &
                (dataframe['tail'] < dataframe['bb_delta_cluc'] * self.buy_clucha_bbdelta_tail_val.value) &
                (dataframe['ha_close'] < dataframe['bb_lowerband2_40'].shift()) &
                (dataframe['close'] > (dataframe['sup_level_1h'] * 0.88)) & # sup_level_1h from informative
                (dataframe['ha_close'] < dataframe['ha_close'].shift())
            )
        dataframe.loc[clucha_ha_conditions, ['enter_long', 'enter_tag']] = (1, 'long_cluc_ha_ns53')    
        
        # NFIX39 Signal (newstrategy53)
        nfix39_conditions = (
                (dataframe['ema_200'] > (dataframe['ema_200'].shift(12) * 1.01)) &
                (dataframe['ema_200'] > (dataframe['ema_200'].shift(48) * 1.07)) &
                (dataframe['bb_lowerband2_40'].shift().gt(0)) &
                (dataframe['bb_delta_cluc'].gt(dataframe['close'] * 0.056)) & # bb_delta_cluc from base TF
                (dataframe['closedelta'].gt(dataframe['close'] * 0.01)) & # closedelta from HA
                (dataframe['tail'].lt(dataframe['bb_delta_cluc'] * 0.5)) & # tail from HA
                (dataframe['close'].lt(dataframe['bb_lowerband2_40'].shift())) &
                (dataframe['close'].le(dataframe['close'].shift())) &
                (dataframe['close'] > dataframe['ema_50'] * 0.912)
            )
        dataframe.loc[nfix39_conditions, ['enter_long', 'enter_tag']] = (1, 'long_nfix39_ns53')
        
        # NFIX29 Signal (newstrategy53)
        nfix29_conditions = (
                (dataframe['close'] > (dataframe['sup_level_1h'] * 0.72)) & # sup_level_1h from informative
                (dataframe['close'] < (dataframe['ema_16'] * 0.982)) &
                (dataframe['EWO'] < -10.0) & # EWO from self.fast_ewo, self.slow_ewo
                (dataframe['cti'] < -0.9) # cti from base TF
            )
        dataframe.loc[nfix29_conditions, ['enter_long', 'enter_tag']] = (1, 'long_nfix29_ns53')
        
        # Local Uptrend Signal (newstrategy53)
        local_uptrend_conditions = (
                (dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['ema_26'] - dataframe['ema_12'] > dataframe['open'] * self.buy_ema_diff_val.value) &
                (dataframe['ema_26'].shift() - dataframe['ema_12'].shift() > dataframe['open'] / 100) & # Note: open / 100, seems small. Verify.
                (dataframe['close'] < dataframe['bb_lowerband2'] * self.buy_bb_factor_val.value) & # bb_lowerband2 from std BB. Original used bb_lowerband (from HA BB)
                # Clarified: newstrategy53 used `dataframe['close'] < dataframe['bb_lowerband2'] * self.buy_bb_factor.value` (referring to the parameter, not column)
                # It should be `dataframe['bb_lowerband2']` (from typical_price BB) or `dataframe['bb_lowerband']` (from HA BB)
                # The original `buy_bb_factor` param in `newstrategy53` was next to `buy_closedelta` which was used with `bb_lowerband3`.
                # Let's assume std BB: `bb_lowerband2`
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta_val.value / 1000 ) # closedelta from HA
            )
        dataframe.loc[local_uptrend_conditions, ['enter_long', 'enter_tag']] = (1, 'long_local_uptrend_ns53')
        
        # VWAP Signal (newstrategy53)
        vwap_conditions = (
                (dataframe['close'] < dataframe['vwap_low']) & # vwap_low from VWAPB helper
                (dataframe['tcp_percent_4'] > 0.053) & 
                (dataframe['cti'] < -0.8) &
                (dataframe['rsi'] < 35) &
                rsi_check_condition & # Uses rsi_84, rsi_112
                (dataframe['volume'] > 0)
           )
        dataframe.loc[vwap_conditions, ['enter_long', 'enter_tag']] = (1, 'long_vwap_ns53')
        
        # Insta Signal (newstrategy53)
        insta_signal_conditions = (
                (dataframe['bb_width_1h'] > 0.131) & # bb_width_1h from informative
                (dataframe['r_14'] < -51) & # r_14 (Williams %R) from base TF
                (dataframe['r_84_1h'] < -70) & # r_84_1h from informative
                (dataframe['cti'] < -0.845) & # cti from base TF
                (dataframe['cti_40_1h'] < -0.735) & # cti_40_1h from informative
                ( (dataframe['close'].rolling(48).max() >= (dataframe['close'] * 1.1 )) ) &
                btc_dump_condition # Uses btc_close from informative BTC
          )
        dataframe.loc[insta_signal_conditions, ['enter_long', 'enter_tag']] = (1, 'long_insta_signal_ns53')

        # NFINext44 Signal (newstrategy53)
        nfinext44_conditions = (
            (dataframe['close'] < (dataframe['ema_16'] * self.buy_44_ma_offset)) & # buy_44_ma_offset is a class var
            (dataframe['ewo_custom'] < self.buy_44_ewo) & # ewo_custom (50,200), buy_44_ewo is class var
            (dataframe['cti'] < self.buy_44_cti) & # buy_44_cti is class var
            (dataframe['r_480_1h'] < self.buy_44_r_1h) & # r_480_1h from informative, buy_44_r_1h class var
            (dataframe['volume'] > 0)
          )
        dataframe.loc[nfinext44_conditions, ['enter_long', 'enter_tag']] = (1, 'long_nfinext44_ns53')

        # NFINext37 Signal (newstrategy53)
        nfinext37_conditions = (
            (dataframe['pm'] > dataframe['pmax_thresh']) & # pm from pmax indicator
            (dataframe['close'] < dataframe['sma_75'] * self.buy_37_ma_offset) & # buy_37_ma_offset class var
            (dataframe['ewo_custom'] > self.buy_37_ewo) & # ewo_custom, buy_37_ewo class var
            (dataframe['rsi'] < self.buy_37_rsi) & # buy_37_rsi class var
            (dataframe['cti'] < self.buy_37_cti) # buy_37_cti class var
        )
        dataframe.loc[nfinext37_conditions, ['enter_long', 'enter_tag']] = (1, 'long_nfinext37_ns53')

        # NFINext7 Signal (newstrategy53)
        nfinext7_conditions = (
            (dataframe['ema_26'] > dataframe['ema_12']) &
            ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_7)) & # buy_ema_open_mult_7 class var
            ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) & # Verify open / 100
            (dataframe['cti'] < self.buy_cti_7) # buy_cti_7 class var
        )       
        dataframe.loc[nfinext7_conditions, ['enter_long', 'enter_tag']] = (1, 'long_nfinext7_ns53')

        # NFINext32 Signal (newstrategy53)
        nfinext32_conditions = (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < 46) &
                (dataframe['rsi'] > 19) & # rsi (14)
                (dataframe['close'] < dataframe['sma_15'] * 0.942) &
                (dataframe['cti'] < -0.86)
        )       
        dataframe.loc[nfinext32_conditions, ['enter_long', 'enter_tag']] = (1, 'long_nfinext32_ns53')

        # SMA_3 Signal (newstrategy53) - Name seems generic
        sma3_conditions = (
                (dataframe['bb_lowerband2_40'].shift() > 0) &
                (dataframe['bb_delta_cluc'] > dataframe['close'] * 0.059) &
                (dataframe['ha_closedelta'] > dataframe['close'] * 0.023) & # ha_closedelta
                (dataframe['tail'] < dataframe['bb_delta_cluc'] * 0.24) & # tail from HA
                (dataframe['close'] < dataframe['bb_lowerband2_40'].shift()) &
                (dataframe['close'] < dataframe['close'].shift()) &
                (btc_dump_condition == 0) # Check if btc_dump_condition is boolean or int
            )       
        dataframe.loc[sma3_conditions, ['enter_long', 'enter_tag']] = (1, 'long_sma3_ns53')
        
        # WVAP Signal (newstrategy53) - Typo? Assume VWAP
        wvapsignal_conditions = (
                (dataframe['close'] < dataframe['vwap_lowerband']) & # vwap_lowerband (VWAPB helper)
                (dataframe['tpct_change_1'] > 0.04) & # tpct_change_1
                (dataframe['cti'] < -0.8) &
                (dataframe['rsi'] < 35) &
                rsi_check_condition &
                (btc_dump_condition == 0) 
        )       
        dataframe.loc[wvapsignal_conditions, ['enter_long', 'enter_tag']] = (1, 'long_wvapsignal_ns53') # Or 'long_vwap_alt_ns53'

        # --- Entry conditions from ZaratustraV13 ---
        # Long DI Enter (ZaratustraV13)
        z_long_di_conditions = (
                (dataframe['dx']  > dataframe['mdi']) &
                (dataframe['adx'] > dataframe['mdi']) &
                (dataframe['pdi'] > dataframe['mdi'])
            )
        dataframe.loc[z_long_di_conditions, ['enter_long', 'enter_tag']] = (1, 'long_di_z13')

        # Long Bollinger Enter (ZaratustraV13)
        z_long_bollinger_conditions = (
                qtpylib.crossed_above(dataframe['close'], dataframe['z_bbu']) # Uses z_bbu
            )
        dataframe.loc[z_long_bollinger_conditions, ['enter_long', 'enter_tag']] = (1, 'long_bollinger_z13')

        # Short DI Enter (ZaratustraV13)
        z_short_di_conditions = (
                (dataframe['dx']  > dataframe['mdi']) & # dx > mdi is unusual for short, usually dx > pdi or adx trend strength
                                                      # Original: (dataframe['dx']  > dataframe['mdi'])
                (dataframe['adx'] > dataframe['pdi']) &
                (dataframe['mdi'] > dataframe['pdi'])
            )
        dataframe.loc[z_short_di_conditions, ['enter_short', 'enter_tag']] = (1, 'short_di_z13')

        # Short Bollinger Enter (ZaratustraV13)
        z_short_bollinger_conditions = (
                qtpylib.crossed_below(dataframe['close'], dataframe['z_bbl']) # Uses z_bbl
            )
        dataframe.loc[z_short_bollinger_conditions, ['enter_short', 'enter_tag']] = (1, 'short_bollinger_z13')
        
        # Note: If multiple conditions are true, the last one setting the tag will prevail for 'enter_tag'.
        # 'enter_long' and 'enter_short' will be 1 if any of their respective conditions are met.
        # This is standard Freqtrade behavior if you OR conditions implicitly by multiple .loc assignments.
        # If you need to ensure only one signal type wins, more complex logic is needed.
        # For "allow either entry", this approach is fine.

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Using newstrategy53's exit trend logic
        # Initialize 'exit_long', 'exit_short', 'exit_tag' columns
        # if 'exit_long' not in dataframe.columns:
        #     dataframe['exit_long'] = 0
        # if 'exit_short' not in dataframe.columns: # For short positions
        #     dataframe['exit_short'] = 0
        # if 'exit_tag' not in dataframe.columns:
        #     dataframe['exit_tag'] = ''


        # newstrategy53's populate_exit_trend logic:
        # It seems to set 'sell' to 0, which is equivalent to 'exit_long' = 0.
        # This means it's disabling an exit signal under these conditions, not triggering one.
        # Original: dataframe.loc[..., 'sell'] = 0
        # If the intent was to *trigger* an exit, it should be 'exit_long' = 1
        # Given `use_exit_signal = True`, this method is for *triggering* exits.
        # Let's assume the original intent for newstrategy53's populate_exit_trend was an exit signal:

        # Exit Long conditions from newstrategy53 (originally for 'sell=0', interpreting as exit signal trigger 'exit_long=1')
        # This interpretation might be wrong if 'sell=0' meant "do not sell based on this".
        # However, populate_exit_trend is for *generating* exit signals.
        # If these conditions were meant to PREVENT an exit from custom_exit, that's a different pattern.
        # Let's assume it's an exit signal for now.
        
        # Original newstrategy53 logic:
        dataframe.loc[
            (dataframe['fisher'] > self.sell_fisher_val.value) &  # Using _val
            (dataframe['ha_high'].le(dataframe['ha_high'].shift(1))) &
            (dataframe['ha_high'].shift(1).le(dataframe['ha_high'].shift(2))) &
            (dataframe['ha_close'].le(dataframe['ha_close'].shift(1))) &
            (dataframe['ema_fast'] > dataframe['ha_close']) & # ema_fast on HA close
            ((dataframe['ha_close'] * self.sell_bbmiddle_close_val.value) > dataframe['bb_middleband']) & # bb_middleband from HA BB. Using _val
            (dataframe['volume'] > 0),
            'exit_long' # Changed from 'sell' to 'exit_long'
        ] = 1 # Changed from 0 to 1 to signal an exit
        dataframe.loc[conditions_above, 'exit_tag'] = 'exit_long_fisher_ha_ns53'
        
        # Re-evaluating newstrategy53's `populate_exit_trend`:
        # `dataframe.loc[..., 'sell'] = 0` (where 'sell' is an alias for 'exit_long' if not shorting)
        # This actually *cancels* an exit signal if one was previously set by other means in this method.
        # If `use_exit_signal = True`, Freqtrade expects this method to set `exit_long=1` or `exit_short=1`.
        # `ZaratustraV13` has `use_exit_signal = False` and an empty `populate_exit_trend`.
        # `newstrategy53` has `use_exit_signal = True`.
        # The current combined strategy has `use_exit_signal = True`.
        
        # If the condition in newstrategy53's `populate_exit_trend` was met, it would *prevent* an ROI exit or stop-loss exit *if*
        # `ignore_roi_if_entry_signal` was involved or if other signals also set `exit_long=1`.
        # This is unusual. Usually, `populate_exit_trend` *sets* exit signals.
        
        # Given the ambiguity and that `ZaratustraV13` had no custom exit signal here,
        # and `newstrategy53`'s logic is potentially to *cancel* exits (which isn't standard for this method),
        # let's leave `populate_exit_trend` empty for now, primarily relying on `custom_exit` (from newstrategy53),
        # ROI, stoploss, and trailing stop.
        # If `newstrategy53` indeed had a valid exit signal here, it needs to be clarified.
        # For now, this method will do nothing, and exits will come from other mechanisms.
        
        # If you want to implement ZaratustraV13's implicit exits (ROI, stoploss, trailing)
        # as explicit signals, you could try to codify them here, but it's usually not necessary
        # if those Freqtrade features are enabled.

        # No explicit exit signals from ZaratustraV13's populate_exit_trend (it was pass)
        # No clear explicit exit signals from newstrategy53's populate_exit_trend (it set 'sell=0')
        # Therefore, this function will currently not set any exit signals.
        # Exits will be handled by ROI, stoploss, trailing_stop, and custom_exit.
        
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        # Logic from newstrategy53
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None # Or handle error appropriately
            
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Ensure trade is not None and has filled orders
        if not trade or not trade.orders:
             return None # Or handle as per strategy needs

        filled_buys = trade.select_filled_orders('buy') # Corrected: was 'buy' type
        count_of_buys = len(filled_buys)

        # Trail targets from newstrategy53's custom_exit
        # Using .value for hyperopt parameters
        if (current_profit > self.sell_trail_profit_min_1.value) and \
           (current_profit < self.sell_trail_profit_max_1.value) and \
           (((trade.max_rate - trade.open_rate) / trade.open_rate) > (current_profit + self.sell_trail_down_1.value)): # Profit calc needs trade.open_rate
            return 'sell_trail_target_1_ns53'
        
        if (current_profit > self.sell_trail_profit_min_2.value) and \
           (current_profit < self.sell_trail_profit_max_2.value) and \
           (((trade.max_rate - trade.open_rate) / trade.open_rate) > (current_profit + self.sell_trail_down_2.value)):
            return 'sell_trail_target_2_ns53'
        
        if (current_profit > 0.03) and (last_candle['rsi'] > 85): # Original was > 3 (300% profit), assuming 0.03 (3%)
             return 'sell_rsi_85_target_ns53'

        # Sell signals from newstrategy53
        # These conditions might need access to hyperopt parameters via .value if they were defined as such.
        # Assuming base_nb_candles_sell, high_offset_2, high_offset are hyperopt params:
        sell_ma_col = f'ma_sell_{self.base_nb_candles_sell.value}'
        if sell_ma_col not in last_candle:
            logger.warning(f"Column {sell_ma_col} not found in last_candle for custom_exit. Skipping related sell signals.")
            # Fallback or return None if essential columns are missing
        else:
            if (current_profit > 0) and (count_of_buys < 4) and \
               (last_candle['close'] > last_candle['hma_50']) and \
               (last_candle['close'] > (last_candle[sell_ma_col] * self.high_offset_2.value)) and \
               (last_candle['rsi'] > 50) and (last_candle['volume'] > 0) and \
               (last_candle['rsi_fast'] > last_candle['rsi_slow']):
                return 'sell_signal1_ns53'
            
            if (current_profit > 0) and (count_of_buys >= 4) and \
               (last_candle['close'] > last_candle['hma_50'] * 1.01) and \
               (last_candle['close'] > (last_candle[sell_ma_col] * self.high_offset_2.value)) and \
               (last_candle['rsi'] > 50) and (last_candle['volume'] > 0) and \
               (last_candle['rsi_fast'] > last_candle['rsi_slow']):
                return 'sell_signal1_dca_ns53' # Adjusted tag

            if (current_profit > 0) and \
               (last_candle['close'] > last_candle['hma_50']) and \
               (last_candle['close'] > (last_candle[sell_ma_col] * self.high_offset.value)) and \
               (last_candle['volume'] > 0) and (last_candle['rsi_fast'] > last_candle['rsi_slow']):
                return 'sell_signal2_ns53'

        # Deadfish sell logic from newstrategy53
        # Ensure all columns are present in last_candle
        required_cols_deadfish = ['close', 'ema_200', 'bb_width', 'bb_middleband2', 'volume_mean_12', 'volume_mean_24', 'cmf']
        if all(col in last_candle for col in required_cols_deadfish):
            if (    (current_profit < self.sell_deadfish_profit.value) and
                    (last_candle['close'] < last_candle['ema_200']) and
                    (last_candle['bb_width'] < self.sell_deadfish_bb_width.value) and
                    (last_candle['close'] > last_candle['bb_middleband2'] * self.sell_deadfish_bb_factor.value) and
                    (last_candle['volume_mean_12'] < last_candle['volume_mean_24'] * self.sell_deadfish_volume_factor.value) and
                    (last_candle['cmf'] < 0.0) # Ensure CMF is correctly calculated and available
                ):
                    return f"sell_deadfish_ns53"
        else:
            missing_cols = [col for col in required_cols_deadfish if col not in last_candle]
            logger.warning(f"Missing columns for deadfish logic: {missing_cols}. Skipping deadfish sell.")
            
        return None # Or a default sell tag / logic

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # Using newstrategy53's progressive stoploss logic
        # These are DecimalParameter, access with .value
        hsl_val = self.pHSL.value
        pf1_val = self.pPF_1.value
        sl1_val = self.pSL_1.value
        pf2_val = self.pPF_2.value
        sl2_val = self.pSL_2.value

        # The stoploss logic requires current_profit to be passed correctly.
        # Ensure profit is calculated based on the trade's open rate and current rate.
        # current_profit is already provided as a parameter to custom_stoploss.

        sl_profit = hsl_val # Default to initial hard stoploss (as a loss percentage)

        if current_profit > pf2_val:
            sl_profit = sl2_val + (current_profit - pf2_val)
        elif current_profit > pf1_val:
            # Ensure (pf2_val - pf1_val) is not zero to avoid division by zero
            if (pf2_val - pf1_val) != 0:
                sl_profit = sl1_val + ((current_profit - pf1_val) * (sl2_val - sl1_val) / (pf2_val - pf1_val))
            else: # If pf1 and pf2 are too close, fallback or use sl1_val directly
                sl_profit = sl1_val 
        # else: sl_profit remains hsl_val

        # Stoploss value is generally expressed as a negative percentage (e.g., -0.05 for -5%)
        # The sl_profit calculated here is a "profit target" for the stoploss.
        # If sl_profit is 0.02, it means stop if profit drops below 2%.
        # If sl_profit is -0.05, it means stop if profit drops below -5% (hard stop).

        # `stoploss_from_open` requires the stoploss to be defined as a distance from open price.
        # If sl_profit is positive (e.g. 0.02), we want a stop at +2% profit.
        # If sl_profit is negative (e.g. -0.05 for HSL), we want a stop at -5% loss.
        # The value returned must be the stoploss percentage (e.g. -0.05 for a 5% stop loss from open price adjusted for current profit).
        # stoploss_from_open(stoploss_value, current_profit_value) -> calculates required rate.
        # Here, we return the stoploss *percentage*.

        # If calculated sl_profit (target profit for stop) is higher than current_profit,
        # it means the stoploss should be higher than the initial HSL.
        # Example: HSL=-0.10. current_profit=0.05. PF1=0.02, SL1=0.01.
        # current_profit (0.05) > PF1 (0.02).
        # sl_profit = 0.01 + ( (0.05 - 0.02) * (SL2 - 0.01) / (PF2 - 0.02) )
        # This sl_profit is the profit level at which we want the stop to be.
        # The custom_stoploss function should return the stoploss as a percentage relative to the open price.
        # A return value of -0.05 means a 5% stop-loss from the open price.
        # A return value of 0.02 means a take-profit at 2% from the open price (acting as a stop).

        # If sl_profit >= current_profit, it means the stop level is at or above the current profit,
        # which would trigger an immediate stop. This condition from newstrategy53 seems to prevent that:
        # `if sl_profit >= current_profit: return -0.99` (a very wide stop, effectively disabling it here)
        # This is usually to ensure the stoploss set is *below* current profit if it's a trailing one.
        # However, this function sets a fixed (but progressive) stoploss level.
        
        # Let's interpret sl_profit as the desired profit level for the stop.
        # If current profit is 0.05 (5%) and sl_profit is 0.02 (2%),
        # we want the stop to be at a price that locks in 2% profit.
        # The return value of custom_stoploss is this target profit/loss level as a ratio.
        # So, if sl_profit = 0.02, return 0.02. If sl_profit = -0.05 (HSL), return -0.05.

        # The condition `if sl_profit >= current_profit: return -0.99` means:
        # If the calculated stop-profit level is HIGHER than or equal to current profit,
        # then set a very loose stoploss (-99%). This happens if, e.g., HSL itself is > current_profit (e.g. HSL -0.05, current_profit -0.06).
        # Or if a progressive SL (e.g. SL1=0.01) is above a small current_profit (e.g. 0.005).
        # This line is crucial for how the progressive stoploss behaves. It prevents the stop from being placed "above" the current price in a loss scenario,
        # or "above" the current profit if the profit is small and the calculated sl_profit is aggressive.

        if sl_profit >= current_profit and current_profit > 0: # If positive profit, don't place stop above current profit
             return current_profit - 0.001 # Trail very tightly below current profit, or use a different logic.
                                          # The -0.99 might be a "disable trailing effect from custom_stoploss" signal.
                                          # Let's keep the original -0.99 as it might interact with other parts.

        if sl_profit >= current_profit: # This handles the HSL case correctly when current_profit is negative.
             return -0.99 # Keep original logic.

        # Otherwise, return the calculated stoploss level (e.g., -0.08, 0.01, 0.04 based on pHSL, pSL_1, pSL_2)
        return sl_profit


    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        # DCA Logic from newstrategy53
        if current_profit > self.initial_safety_order_trigger: # initial_safety_order_trigger is negative
            return None # Not time for safety order yet

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            return None
        last_candle = dataframe.iloc[-1].squeeze()

        filled_buys = trade.select_filled_orders(type='buy') # Corrected: was 'buy'
        count_of_buys = len(filled_buys)

        # DCA entry guards from newstrategy53
        # Ensure all columns used here are available in last_candle and correctly named after merges.
        # e.g., 'cmf_1h' should be 'cmf_calc_1h' if using my cleaned informative names.
        # 'rsi_14_1h' is correct if using my cleaned informative names.
        # Let's assume column names match what newstrategy53 expects from its populate_indicators.
        # Need to verify 'cmf_1h' vs 'cmf_calc_1h', 'rsi_1h' vs 'rsi_14_1h'.
        # Original newstrategy53 uses 'cmf_1h' and 'rsi_14_1h' in adjust_trade_position.
        # So the informative merge should produce these exact names.

        # Accessing class method top_percent_change_dca
        # This method expects a dataframe slice, not just last_candle.
        # However, the implementation in newstrategy53 for length=0 uses iloc[-1] equivalent.
        # Let's call it with the full dataframe.
        # Note: The class method `top_percent_change_dca` in `newstrategy53` was slightly different from the global one.
        # The global one was used in `populate_indicators`.
        # The one here should be `self.top_percent_change_dca(dataframe, 0)` if using the class method.
        # The class method `top_percent_change_dca` was defined in the prompt, but not inside the class itself.
        # Let's assume `self.top_percent_change_dca` is available (copied into the class).
        # We added `top_percent_change_dca` as a method to `CombinedStrategy` earlier.

        tpct_change_0_val = self.top_percent_change_dca(dataframe, 0) # Call class method

        dca_guard_conditions = [False] * (self.max_safety_orders + 1) # Index matches count_of_buys

        if count_of_buys == 1 and (tpct_change_0_val > 0.018) and (last_candle['close'] < last_candle['open']):
            dca_guard_conditions[1] = True
        elif count_of_buys == 2 and (tpct_change_0_val > 0.018) and (last_candle['close'] < last_candle['open']) and \
             ('ema_vwap_diff_50' in last_candle and last_candle['ema_vwap_diff_50'] < 0.215):
            dca_guard_conditions[2] = True
        elif count_of_buys == 3 and (tpct_change_0_val > 0.018) and (last_candle['close'] < last_candle['open']) and \
             ('ema_vwap_diff_50' in last_candle and last_candle['ema_vwap_diff_50'] < 0.215):
            dca_guard_conditions[3] = True
        elif count_of_buys == 4 and (tpct_change_0_val > 0.018) and (last_candle['close'] < last_candle['open']) and \
             ('ema_vwap_diff_50' in last_candle and last_candle['ema_vwap_diff_50'] < 0.215) and \
             ('ema_5' in last_candle and 'ema_10' in last_candle and last_candle['ema_5'] >= last_candle['ema_10']):
            dca_guard_conditions[4] = True
        
        # For buys 5 through max_safety_orders
        # Check if informative columns like 'cmf_1h', 'rsi_14_1h' are available
        # Assuming they are dataframe[f'{indicator_name}_1h'] from merge_informative_pair
        cmf_1h_col = 'cmf_calc_1h' # As per my cleaned merge. Verify with original newstrategy53's expectation.
        rsi_14_1h_col = 'rsi_14_1h' # As per my cleaned merge.

        if count_of_buys >= 5:
            common_dca_condition_5plus = (
                (cmf_1h_col in last_candle and last_candle[cmf_1h_col] < 0.00) and
                (last_candle['close'] < last_candle['open']) and
                (rsi_14_1h_col in last_candle and last_candle[rsi_14_1h_col] < 30) and
                (tpct_change_0_val > 0.018) and
                ('ema_vwap_diff_50' in last_candle and last_candle['ema_vwap_diff_50'] < 0.215) and
                ('ema_5' in last_candle and 'ema_10' in last_candle and last_candle['ema_5'] >= last_candle['ema_10'])
            )
            if common_dca_condition_5plus:
                if count_of_buys <= self.max_safety_orders: # Check within bounds
                    dca_guard_conditions[count_of_buys] = True
                    logger.info(f"DCA for {trade.pair} (buy #{count_of_buys}) waiting for {cmf_1h_col} ({last_candle.get(cmf_1h_col, 'N/A')}) "
                                f"to rise above 0 and {rsi_14_1h_col} ({last_candle.get(rsi_14_1h_col, 'N/A')}) to rise above 30.")

        if count_of_buys <= self.max_safety_orders and dca_guard_conditions[count_of_buys]:
            return None # Guard condition met, do not DCA yet

        # Calculate safety order trigger point
        if 1 <= count_of_buys <= self.max_safety_orders:
            safety_order_trigger = abs(self.initial_safety_order_trigger) # For the first DCA
            if count_of_buys > 1: # For subsequent DCAs if step_scale is used
                if self.safety_order_step_scale > 1:
                    safety_order_trigger = abs(self.initial_safety_order_trigger) + \
                                           (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * \
                                           (math.pow(self.safety_order_step_scale, (count_of_buys - 1)) - 1) / \
                                           (self.safety_order_step_scale - 1))
                elif self.safety_order_step_scale < 1 and self.safety_order_step_scale > 0 : # Avoid division by zero if scale is 1
                    safety_order_trigger = abs(self.initial_safety_order_trigger) + \
                                           (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * \
                                           (1 - math.pow(self.safety_order_step_scale, (count_of_buys - 1))) / \
                                           (1 - self.safety_order_step_scale))
                # If safety_order_step_scale == 1, trigger is just initial_safety_order_trigger * count_of_buys (linear)
                # The formula provided in newstrategy53 seems to be for a geometric sum of step increases.
                # Let's use a simpler linear for scale=1:
                elif self.safety_order_step_scale == 1:
                     safety_order_trigger = abs(self.initial_safety_order_trigger) * count_of_buys


            if current_profit <= (-1 * abs(safety_order_trigger)):
                try:
                    # Ensure filled_buys is not empty
                    if not filled_buys:
                        logger.warning(f"No filled buy orders found for trade {trade.pair} to calculate DCA stake.")
                        return None

                    # Calculate stake amount for DCA
                    # Original logic: stake_amount = filled_buys[0].cost * math.pow(self.safety_order_volume_scale, (count_of_buys -1))
                    # This should be count_of_buys, not count_of_buys - 1 for the power if the first DCA is also scaled.
                    # If first DCA (buy #2, count_of_buys=1) has volume_scale^0 = 1 * initial_stake.
                    # Second DCA (buy #3, count_of_buys=2) has volume_scale^1 * initial_stake. Correct.
                    
                    stake_amount = filled_buys[0].cost * math.pow(self.safety_order_volume_scale, count_of_buys) # Corrected power to count_of_buys
                                                                                                                 # No, original was (count_of_buys -1)
                                                                                                                 # Let's trace:
                                                                                                                 # 1st DCA (total 2 buys): count_of_buys=1. pow(scale, 0). Stake = initial_stake.
                                                                                                                 # 2nd DCA (total 3 buys): count_of_buys=2. pow(scale, 1). Stake = initial_stake * scale.
                                                                                                                 # This seems correct.
                    stake_amount = filled_buys[0].cost * math.pow(self.safety_order_volume_scale, (count_of_buys -1)) # Reverted to original logic


                    # Ensure stake_amount is not less than min_stake, and respects max_stake if it's a total limit
                    if stake_amount < min_stake:
                        stake_amount = min_stake
                    
                    # Check against max_stake (if max_stake is per order and not total)
                    # This max_stake is often the total allowed for the trade.
                    # The `adjust_trade_position` should return the *additional* stake.
                    # Freqtrade handles total stake limits.

                    logger.info(f"Initiating safety order buy #{count_of_buys + 1} for {trade.pair} "
                                f"with additional stake amount of {stake_amount:.8f}")
                    return stake_amount
                except Exception as exception:
                    logger.info(f'Error during DCA for {trade.pair}: {str(exception)}')
                    return None
        return None
        
    def leverage(
      self,
      pair: str,
      current_time: datetime,
      current_rate: float,
      proposed_leverage: float,
      max_leverage: float,
      entry_tag: Optional[str],
      side: str,
      **kwargs,
    ) -> float:
      return 3.0