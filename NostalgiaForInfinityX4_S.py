import copy
import logging
import pathlib
import rapidjson
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import pandas as pd
import pandas_ta as pta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy import DecimalParameter, CategoricalParameter
from pandas import DataFrame, Series
from functools import reduce
from freqtrade.persistence import Trade, LocalTrade
from datetime import datetime, timedelta
import time
from typing import Optional
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

#############################################################################################################
##                NostalgiaForInfinityX4 by iterativ                                                       ##
##           https://github.com/iterativv/NostalgiaForInfinity                                             ##
##                                                                                                         ##
##    Strategy for Freqtrade https://github.com/freqtrade/freqtrade                                        ##
##                                                                                                         ##
#############################################################################################################
##               GENERAL RECOMMENDATIONS                                                                   ##
##                                                                                                         ##
##   For optimal performance, suggested to use between 4 and 6 open trades, with unlimited stake.          ##
##   A pairlist with 40 to 80 pairs. Volume pairlist works well.                                           ##
##   Prefer stable coin (USDT, BUSDT etc) pairs, instead of BTC or ETH pairs.                              ##
##   Highly recommended to blacklist leveraged tokens (*BULL, *BEAR, *UP, *DOWN etc).                      ##
##   Ensure that you don't override any variables in you config.json. Especially                           ##
##   the timeframe (must be 5m).                                                                           ##
##     use_exit_signal must set to true (or not set at all).                                               ##
##     exit_profit_only must set to false (or not set at all).                                             ##
##     ignore_roi_if_entry_signal must set to true (or not set at all).                                    ##
##                                                                                                         ##
#############################################################################################################
##               DONATIONS                                                                                 ##
##                                                                                                         ##
##   BTC: bc1qvflsvddkmxh7eqhc4jyu5z5k6xcw3ay8jl49sk                                                       ##
##   ETH (ERC20): 0x83D3cFb8001BDC5d2211cBeBB8cB3461E5f7Ec91                                               ##
##   BEP20/BSC (USDT, ETH, BNB, ...): 0x86A0B21a20b39d16424B7c8003E4A7e12d78ABEe                           ##
##   TRC20/TRON (USDT, TRON, ...): TTAa9MX6zMLXNgWMhg7tkNormVHWCoq8Xk                                      ##
##                                                                                                         ##
##               REFERRAL LINKS                                                                            ##
##                                                                                                         ##
##  Binance: https://accounts.binance.com/en/register?ref=C68K26A9 (20% discount on trading fees)          ##
##  Kucoin: https://www.kucoin.com/r/af/QBSSS5J2 (20% lifetime discount on trading fees)                   ##
##  Gate.io: https://www.gate.io/signup/UAARUlhf/20pct?ref_type=103 (20% lifetime discount on trading fees)##
##  OKX: https://www.okx.com/join/11749725931 (20% discount on trading fees)                               ##
##  MEXC: https://promote.mexc.com/a/nfi  (10% discount on trading fees)                                   ##
##  ByBit: https://partner.bybit.com/b/nfi                                                                 ##
##  Bitget: https://bonus.bitget.com/nfi (lifetime 20% rebate all & 10% discount on spot fees)             ##
##  HTX: https://www.htx.com/invite/en-us/1f?invite_code=ubpt2223                                          ##
##         (Welcome Bonus worth 241 USDT upon completion of a deposit and trade)                           ##
##  Bitvavo: https://account.bitvavo.com/create?a=D22103A4BC (no fees for the first € 1000)                ##
#############################################################################################################


class NostalgiaForInfinityX4_S(IStrategy):
  INTERFACE_VERSION = 3

  def version(self) -> str:
    return "v14.1.225"

  stoploss = -0.99
  can_short = True

  # Trailing stoploss (not used)
  trailing_stop = True
  trailing_only_offset_is_reached = True
  trailing_stop_positive = 0.01
  trailing_stop_positive_offset = 0.1

  use_custom_stoploss = False

  # Optimal timeframe for the strategy.
  timeframe = "5m"
  info_timeframes = ["15m", "1h", "4h", "1d"]

  # BTC informatives
  btc_info_timeframes = ["5m", "15m", "1h", "4h", "1d"]

  # Backtest Age Filter emulation
  has_bt_agefilter = False
  bt_min_age_days = 3

  # Exchange Downtime protection
  has_downtime_protection = False

  # Do you want to use the hold feature? (with hold-trades.json)
  hold_support_enabled = True

  # Run "populate_indicators()" only for new candle.
  process_only_new_candles = True

  # These values can be overridden in the "ask_strategy" section in the config.
  use_exit_signal = True
  exit_profit_only = False
  ignore_roi_if_entry_signal = True

  # Number of candles the strategy requires before producing valid signals
  startup_candle_count: int = 800

  # Normal mode tags
  normal_mode_tags = ["force_entry", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
  # Pump mode tags
  pump_mode_tags = ["21", "22", "23", "24", "25", "26"]
  # Quick mode tags
  quick_mode_tags = ["41", "42", "43", "44", "45", "46", "47", "48", "49", "50"]
  # Long rebuy mode tags
  long_rebuy_mode_tags = ["61"]
  # Long mode tags
  long_mode_tags = ["81", "82"]
  # Long rapid mode tags
  long_rapid_mode_tags = ["101", "102", "103", "104", "105", "106", "107", "108", "109", "110"]

  normal_mode_name = "normal"
  pump_mode_name = "pump"
  quick_mode_name = "quick"
  long_rebuy_mode_name = "long_rebuy"
  long_mode_name = "long"
  long_rapid_mode_name = "long_rapid"

  # Shorting

  # Short normal mode tags
  short_normal_mode_tags = ["500"]

  short_normal_mode_name = "short_normal"

  is_futures_mode = False
  futures_mode_leverage = 10.0
  futures_mode_leverage_rebuy_mode = 5.0

  # Stop thresholds. 0: Doom Bull, 1: Doom Bear, 2: u_e Bull, 3: u_e Bear, 4: u_e mins Bull, 5: u_e mins Bear.
  # 6: u_e ema % Bull, 7: u_e ema % Bear, 8: u_e RSI diff Bull, 9: u_e RSI diff Bear.
  # 10: enable Doom Bull, 11: enable Doom Bear, 12: enable u_e Bull, 13: enable u_e Bear.
  stop_thresholds = [-0.2, -0.2, -0.025, -0.025, 720, 720, 0.016, 0.016, 24.0, 24.0, False, False, True, True]
  # Based on the the first entry (regardless of rebuys)
  stop_threshold = 4.0
  stop_threshold_futures = 12.0
  stop_threshold_futures_rapid = 12.0
  stop_threshold_spot_rapid = 4.0
  stop_threshold_spot_rebuy = 0.9
  stop_threshold_futures_rebuy = 3.9

  # Rebuy mode minimum number of free slots
  rebuy_mode_min_free_slots = 2

  # Position adjust feature
  position_adjustment_enable = True

  # Grinding feature
  grinding_enable = True

  # Grinding
  grind_derisk_spot = -0.40
  grind_derisk_futures = -0.50

  grind_1_stop_grinds_spot = -0.16
  grind_1_profit_threshold_spot = 0.018
  grind_1_stakes_spot = [
    [0.20, 0.20, 0.20, 0.20, 0.20],
    [0.3, 0.3, 0.3, 0.3],
    [0.35, 0.35, 0.35, 0.35],
    [0.4, 0.4, 0.4],
    [0.45, 0.45, 0.45],
    [0.5, 0.5, 0.5],
    [0.75, 0.75],
  ]
  grind_1_sub_thresholds_spot = [
    [-0.12, -0.12, -0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12, -0.14],
    [-0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12],
    [-0.12, -0.12],
  ]

  grind_1_stop_grinds_futures = -0.48
  grind_1_profit_threshold_futures = 0.018
  grind_1_stakes_futures = [
    [0.20, 0.20, 0.20, 0.20, 0.20],
    [0.3, 0.3, 0.3, 0.3],
    [0.35, 0.35, 0.35, 0.35],
    [0.4, 0.4, 0.4],
    [0.45, 0.45, 0.45],
    [0.5, 0.5, 0.5],
    [0.75, 0.75],
  ]
  grind_1_sub_thresholds_futures = [
    [-0.12, -0.12, -0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12, -0.14],
    [-0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12],
    [-0.12, -0.12],
  ]

  grind_2_stop_grinds_spot = -0.10
  grind_2_profit_threshold_spot = 0.018
  grind_2_stakes_spot = [
    [0.10, 0.15, 0.20, 0.25, 0.30],
  ]
  grind_2_sub_thresholds_spot = [
    [-0.08, -0.10, -0.12, -0.14, -0.16],
  ]

  grind_2_stop_grinds_futures = -0.30
  grind_2_profit_threshold_futures = 0.018
  grind_2_stakes_futures = [
    [0.10, 0.15, 0.20, 0.25, 0.30],
  ]
  grind_2_sub_thresholds_futures = [
    [-0.08, -0.10, -0.12, -0.14, -0.16],
  ]

  grind_3_stop_grinds_spot = -0.10
  grind_3_profit_threshold_spot = 0.018
  grind_3_stakes_spot = [
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
  ]
  grind_3_sub_thresholds_spot = [
    [-0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08],
    [-0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08],
  ]

  grind_3_stop_grinds_futures = -0.30
  grind_3_profit_threshold_futures = 0.018
  grind_3_stakes_futures = [
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
  ]
  grind_3_sub_thresholds_futures = [
    [-0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08],
    [-0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08, -0.08],
  ]

  # Non rebuy modes
  regular_mode_stake_multiplier_spot = [0.5, 0.75]
  regular_mode_stake_multiplier_futures = [0.5, 0.75]

  regular_mode_rebuy_stakes_spot = [
    [0.40, 0.40, 0.40, 0.40, 0.40],
    [0.50, 0.50, 0.50, 0.50, 0.50],
    [0.75, 0.75, 0.75, 0.75],
    [1.0, 1.0, 1.0],
  ]
  regular_mode_grind_1_stakes_spot = [
    [0.40, 0.40, 0.40, 0.40, 0.40],
    [0.50, 0.50, 0.50, 0.50, 0.50],
    [0.75, 0.75, 0.75, 0.75],
    [1.0, 1.0, 1.0],
  ]
  regular_mode_rebuy_thresholds_spot = [
    [-0.12, -0.12, -0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12],
  ]
  regular_mode_grind_1_thresholds_spot = [
    [-0.06, -0.12, -0.12, -0.12, -0.12, -0.12],
    [-0.06, -0.12, -0.12, -0.12, -0.12, -0.12],
    [-0.06, -0.12, -0.12, -0.12, -0.12],
    [-0.06, -0.12, -0.12, -0.12],
  ]
  regular_mode_grind_1_profit_threshold_spot = 0.018
  regular_mode_grind_2_stakes_spot = [
    [0.10, 0.15, 0.20, 0.25, 0.30],
  ]
  regular_mode_grind_2_thresholds_spot = [
    [-0.03, -0.08, -0.10, -0.12, -0.14, -0.16],
  ]
  regular_mode_grind_2_profit_threshold_spot = 0.018
  regular_mode_derisk_spot = -0.80

  regular_mode_rebuy_stakes_futures = [
    [0.40, 0.40, 0.40, 0.40, 0.40],
    [0.50, 0.50, 0.50, 0.50, 0.50],
    [0.75, 0.75, 0.75, 0.75],
    [1.0, 1.0, 1.0],
  ]
  regular_mode_grind_1_stakes_futures = [
    [0.40, 0.40, 0.40, 0.40, 0.40],
    [0.50, 0.50, 0.50, 0.50, 0.50],
    [0.75, 0.75, 0.75, 0.75],
    [1.0, 1.0, 1.0],
  ]
  regular_mode_rebuy_thresholds_futures = [
    [-0.12, -0.12, -0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12],
  ]
  regular_mode_grind_1_thresholds_futures = [
    [-0.06, -0.12, -0.12, -0.12, -0.12, -0.12],
    [-0.06, -0.12, -0.12, -0.12, -0.12, -0.12],
    [-0.06, -0.12, -0.12, -0.12, -0.12],
    [-0.06, -0.12, -0.12, -0.12],
  ]
  regular_mode_grind_1_profit_threshold_futures = 0.018
  regular_mode_grind_2_stakes_futures = [
    [0.10, 0.15, 0.20, 0.25, 0.30],
  ]
  regular_mode_grind_2_thresholds_futures = [
    [-0.03, -0.08, -0.10, -0.12, -0.14, -0.16],
  ]
  regular_mode_grind_2_profit_threshold_futures = 0.018
  regular_mode_derisk_futures = -2.40

  # Rebuy mode
  rebuy_mode_stake_multiplier = 0.2
  rebuy_mode_stake_multiplier_alt = 0.3
  rebuy_mode_max = 3
  rebuy_mode_derisk_spot = -0.9
  rebuy_mode_derisk_futures = -2.0
  rebuy_mode_stakes_spot = [1.0, 2.0, 4.0]
  rebuy_mode_stakes_futures = [1.0, 2.0, 4.0]
  rebuy_mode_thresholds_spot = [-0.08, -0.10, -0.12]
  rebuy_mode_thresholds_futures = [-0.08, -0.10, -0.12]

  # Profit max thresholds
  profit_max_thresholds = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.05]

  # Max allowed buy "slippage", how high to buy on the candle
  max_slippage = 0.012

  # BTC/ETH stakes
  btc_stakes = ["BTC", "ETH"]

  #############################################################
  # Buy side configuration

  entry_long_params = {
    # Enable/Disable conditions
    # -------------------------------------------------------
    "buy_condition_1_enable": True,
    "buy_condition_2_enable": True,
    "buy_condition_3_enable": True,
    "buy_condition_4_enable": True,
    "buy_condition_5_enable": True,
    "buy_condition_6_enable": True,
    "buy_condition_7_enable": True,
    "buy_condition_8_enable": True,
    "buy_condition_9_enable": True,
    "buy_condition_10_enable": True,
    "buy_condition_11_enable": True,
    "buy_condition_12_enable": True,
    "buy_condition_21_enable": True,
    "buy_condition_22_enable": True,
    "buy_condition_23_enable": True,
    "buy_condition_24_enable": True,
    "buy_condition_25_enable": True,
    "buy_condition_26_enable": True,
    "buy_condition_41_enable": True,
    "buy_condition_42_enable": True,
    "buy_condition_43_enable": True,
    "buy_condition_44_enable": True,
    "buy_condition_45_enable": True,
    "buy_condition_46_enable": True,
    "buy_condition_46_enable": True,
    "buy_condition_47_enable": True,
    "buy_condition_48_enable": True,
    "buy_condition_49_enable": True,
    "buy_condition_50_enable": True,
    "buy_condition_61_enable": True,
    "buy_condition_81_enable": True,
    "buy_condition_82_enable": True,
    "buy_condition_101_enable": True,
    "buy_condition_102_enable": True,
    "buy_condition_103_enable": True,
    "buy_condition_104_enable": True,
    "buy_condition_105_enable": True,
    "buy_condition_106_enable": True,
    "buy_condition_107_enable": True,
    "buy_condition_108_enable": True,
    "buy_condition_109_enable": True,
    "buy_condition_110_enable": True,
  }

  entry_short_params = {
    # Enable/Disable conditions
    # -------------------------------------------------------
    "entry_condition_500_enable": False,
  }

  buy_protection_params = {}

  #############################################################

  entry_10_close_max_12 = DecimalParameter(00.50, 0.95, default=0.80, decimals=2, space="buy", optimize=False)
  entry_10_close_max_24 = DecimalParameter(00.50, 0.95, default=0.75, decimals=2, space="buy", optimize=False)
  entry_10_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=False)
  entry_10_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.60, decimals=2, space="buy", optimize=False)
  entry_10_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.50, decimals=2, space="buy", optimize=False)
  entry_10_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.45, decimals=2, space="buy", optimize=False)
  entry_10_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.80, decimals=2, space="buy", optimize=False)
  entry_10_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.90, decimals=2, space="buy", optimize=False)
  entry_10_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=0.95, decimals=2, space="buy", optimize=False)
  entry_10_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.00, decimals=2, space="buy", optimize=False)
  entry_10_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_10_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_10_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_10_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_10_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_10_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_10_ema_200_not_dec_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_10_ema_200_not_dec_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_10_ema_200_not_dec_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_10_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_10_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_10_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_10_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_10_rsi_3_min = DecimalParameter(00.0, 30.0, default=4.0, decimals=0, space="buy", optimize=False)
  entry_10_rsi_3_max = DecimalParameter(30.0, 70.0, default=46.0, decimals=0, space="buy", optimize=False)
  entry_10_rsi_3_15m_min = DecimalParameter(00.0, 36.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_10_rsi_3_1h_min = DecimalParameter(00.0, 36.0, default=8.0, decimals=0, space="buy", optimize=False)
  entry_10_rsi_3_4h_min = DecimalParameter(00.0, 36.0, default=8.0, decimals=0, space="buy", optimize=False)
  entry_10_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=8.0, decimals=0, space="buy", optimize=False)
  entry_10_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.90, decimals=2, space="buy", optimize=False)
  entry_10_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_10_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.90, decimals=2, space="buy", optimize=False)
  entry_10_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_10_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_10_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_10_r_14_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_10_r_14_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_10_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_10_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_10_rsi_14_max = DecimalParameter(20.0, 60.0, default=30.0, decimals=0, space="buy", optimize=False)
  entry_10_ema_offset = DecimalParameter(0.940, 0.972, default=0.952, decimals=3, space="buy", optimize=False)
  entry_10_ema_open_offset = DecimalParameter(0.0100, 0.0400, default=0.0200, decimals=4, space="buy", optimize=False)

  entry_11_close_max_12 = DecimalParameter(00.50, 0.95, default=0.80, decimals=2, space="buy", optimize=False)
  entry_11_close_max_24 = DecimalParameter(00.50, 0.95, default=0.75, decimals=2, space="buy", optimize=False)
  entry_11_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=False)
  entry_11_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.60, decimals=2, space="buy", optimize=False)
  entry_11_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.50, decimals=2, space="buy", optimize=False)
  entry_11_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.45, decimals=2, space="buy", optimize=False)
  entry_11_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.80, decimals=2, space="buy", optimize=False)
  entry_11_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.90, decimals=2, space="buy", optimize=False)
  entry_11_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=0.95, decimals=2, space="buy", optimize=False)
  entry_11_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.00, decimals=2, space="buy", optimize=False)
  entry_11_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_11_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_11_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_11_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_11_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_11_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_11_ema_200_not_dec_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_11_ema_200_not_dec_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_11_ema_200_not_dec_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_11_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_11_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_11_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_11_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_11_ema_100_over_ema_200_enabled = CategoricalParameter(
    [True, False], default=True, space="buy", optimize=False
  )
  entry_11_ema_12_1h_over_ema_200_1h_enabled = CategoricalParameter(
    [True, False], default=True, space="buy", optimize=False
  )
  entry_11_rsi_3_min = DecimalParameter(00.0, 30.0, default=2.0, decimals=0, space="buy", optimize=False)
  entry_11_rsi_3_max = DecimalParameter(30.0, 70.0, default=46.0, decimals=0, space="buy", optimize=False)
  entry_11_rsi_3_15m_min = DecimalParameter(00.0, 36.0, default=16.0, decimals=0, space="buy", optimize=False)
  entry_11_rsi_3_1h_min = DecimalParameter(00.0, 36.0, default=8.0, decimals=0, space="buy", optimize=False)
  entry_11_rsi_3_4h_min = DecimalParameter(00.0, 36.0, default=8.0, decimals=0, space="buy", optimize=False)
  entry_11_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=8.0, decimals=0, space="buy", optimize=False)
  entry_11_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_11_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=85.0, decimals=0, space="buy", optimize=False)
  entry_11_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_11_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=85.0, decimals=0, space="buy", optimize=False)
  entry_11_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_11_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=85.0, decimals=0, space="buy", optimize=False)
  entry_11_r_14_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_11_r_14_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_11_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_11_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_11_rsi_14_max = DecimalParameter(20.0, 60.0, default=45.0, decimals=0, space="buy", optimize=False)
  entry_11_cti_20_max = DecimalParameter(-0.99, -0.60, default=-0.50, decimals=2, space="buy", optimize=False)
  entry_11_ema_open_offset = DecimalParameter(0.0200, 0.0400, default=0.0260, decimals=4, space="buy", optimize=False)
  entry_11_sma_offset = DecimalParameter(0.940, 0.988, default=0.978, decimals=3, space="buy", optimize=False)

  entry_12_close_max_12 = DecimalParameter(00.50, 0.95, default=0.80, decimals=2, space="buy", optimize=False)
  entry_12_close_max_24 = DecimalParameter(00.50, 0.95, default=0.75, decimals=2, space="buy", optimize=False)
  entry_12_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=False)
  entry_12_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.60, decimals=2, space="buy", optimize=False)
  entry_12_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.50, decimals=2, space="buy", optimize=False)
  entry_12_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.45, decimals=2, space="buy", optimize=False)
  entry_12_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.80, decimals=2, space="buy", optimize=False)
  entry_12_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.90, decimals=2, space="buy", optimize=False)
  entry_12_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=1.10, decimals=2, space="buy", optimize=False)
  entry_12_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.20, decimals=2, space="buy", optimize=False)
  entry_12_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_12_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_12_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_12_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_12_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_12_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_12_ema_200_not_dec_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_12_ema_200_not_dec_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_12_ema_200_not_dec_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_12_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_12_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_12_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_12_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_12_ema_50_1h_over_ema_200_1h_enabled = CategoricalParameter(
    [True, False], default=True, space="buy", optimize=False
  )
  entry_12_rsi_3_min = DecimalParameter(00.0, 30.0, default=2.0, decimals=0, space="buy", optimize=False)
  entry_12_rsi_3_max = DecimalParameter(30.0, 70.0, default=46.0, decimals=0, space="buy", optimize=False)
  entry_12_rsi_3_15m_min = DecimalParameter(00.0, 36.0, default=12.0, decimals=0, space="buy", optimize=False)
  entry_12_rsi_3_1h_min = DecimalParameter(00.0, 36.0, default=20.0, decimals=0, space="buy", optimize=False)
  entry_12_rsi_3_4h_min = DecimalParameter(00.0, 36.0, default=20.0, decimals=0, space="buy", optimize=False)
  entry_12_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=20.0, decimals=0, space="buy", optimize=False)
  # entry_12_cti_20_1h_min = DecimalParameter(-0.9, -0.0, default=-0.50, decimals=2, space="buy", optimize=False)
  entry_12_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_12_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_12_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_12_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_12_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_12_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_12_r_14_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_12_r_14_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_12_r_14_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_12_r_14_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_12_r_480_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_12_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_12_r_480_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_12_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_12_r_14_max = DecimalParameter(-100.0, 80.0, default=-88.0, decimals=0, space="buy", optimize=False)
  entry_12_bb_offset = DecimalParameter(0.970, 0.999, default=0.984, decimals=3, space="buy", optimize=False)
  entry_12_sma_offset = DecimalParameter(0.930, 0.960, default=0.940, decimals=3, space="buy", optimize=False)

  entry_24_close_max_12 = DecimalParameter(00.50, 0.95, default=0.80, decimals=2, space="buy", optimize=True)
  entry_24_close_max_24 = DecimalParameter(00.50, 0.95, default=0.75, decimals=2, space="buy", optimize=True)
  entry_24_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=True)
  entry_24_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.60, decimals=2, space="buy", optimize=True)
  entry_24_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.50, decimals=2, space="buy", optimize=True)
  entry_24_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.45, decimals=2, space="buy", optimize=True)
  entry_24_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.80, decimals=2, space="buy", optimize=True)
  entry_24_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.90, decimals=2, space="buy", optimize=True)
  entry_24_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=1.10, decimals=2, space="buy", optimize=True)
  entry_24_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.20, decimals=2, space="buy", optimize=True)
  entry_24_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=True)
  entry_24_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=True)
  entry_24_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=True)
  entry_24_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=True)
  entry_24_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=True)
  entry_24_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=True)
  entry_24_ema_200_not_dec_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=True)
  entry_24_ema_200_not_dec_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=True)
  entry_24_ema_200_not_dec_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=True)
  entry_24_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=True)
  entry_24_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=True)
  entry_24_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=True)
  entry_24_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=True)
  entry_24_ema_50_over_ema_200_enabled = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
  entry_24_ema_12_1h_over_ema_200_1h_enabled = CategoricalParameter(
    [True, False], default=True, space="buy", optimize=True
  )
  entry_24_rsi_3_min = DecimalParameter(00.0, 30.0, default=2.0, decimals=0, space="buy", optimize=True)
  entry_24_rsi_3_max = DecimalParameter(30.0, 70.0, default=46.0, decimals=0, space="buy", optimize=True)
  entry_24_rsi_3_15m_min = DecimalParameter(00.0, 36.0, default=2.0, decimals=0, space="buy", optimize=True)
  entry_24_rsi_3_1h_min = DecimalParameter(00.0, 36.0, default=8.0, decimals=0, space="buy", optimize=True)
  entry_24_rsi_3_4h_min = DecimalParameter(00.0, 36.0, default=8.0, decimals=0, space="buy", optimize=True)
  entry_24_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=8.0, decimals=0, space="buy", optimize=True)
  entry_24_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=True)
  entry_24_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=True)
  entry_24_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=True)
  entry_24_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=True)
  entry_24_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=True)
  entry_24_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=True)
  entry_24_r_14_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=True)
  entry_24_r_14_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=True)
  entry_24_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=True)
  entry_24_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=True)
  entry_24_rsi_14_min = DecimalParameter(20.0, 60.0, default=26.0, decimals=0, space="buy", optimize=True)
  entry_24_rsi_14_max = DecimalParameter(20.0, 60.0, default=46.0, decimals=0, space="buy", optimize=True)
  entry_24_cti_20_max = DecimalParameter(-0.99, -0.60, default=-0.75, decimals=2, space="buy", optimize=True)
  entry_24_r_14_max = DecimalParameter(-100.0, 80.0, default=-97.0, decimals=0, space="buy", optimize=True)
  entry_24_ewo_50_200_min = DecimalParameter(2.0, 10.0, default=7.0, decimals=1, space="buy", optimize=True)
  entry_24_ewo_50_200_max = DecimalParameter(10.0, 30.0, default=24.0, decimals=1, space="buy", optimize=True)
  entry_24_sma_offset = DecimalParameter(0.960, 0.999, default=0.984, decimals=3, space="buy", optimize=True)

  entry_25_close_max_12 = DecimalParameter(00.50, 0.95, default=0.80, decimals=2, space="buy", optimize=False)
  entry_25_close_max_24 = DecimalParameter(00.50, 0.95, default=0.75, decimals=2, space="buy", optimize=False)
  entry_25_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=False)
  entry_25_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.60, decimals=2, space="buy", optimize=False)
  entry_25_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.50, decimals=2, space="buy", optimize=False)
  entry_25_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.45, decimals=2, space="buy", optimize=False)
  entry_25_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.5, decimals=2, space="buy", optimize=False)
  entry_25_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.75, decimals=2, space="buy", optimize=False)
  entry_25_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=0.90, decimals=2, space="buy", optimize=False)
  entry_25_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.00, decimals=2, space="buy", optimize=False)
  entry_25_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_25_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_25_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_25_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_25_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_25_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_25_ema_200_not_dec_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_25_ema_200_not_dec_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_25_ema_200_not_dec_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_25_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_25_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_25_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_25_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_25_rsi_3_min = DecimalParameter(00.0, 30.0, default=2.0, decimals=0, space="buy", optimize=False)
  entry_25_rsi_3_max = DecimalParameter(30.0, 60.0, default=60.0, decimals=0, space="buy", optimize=False)
  entry_25_rsi_3_15m_min = DecimalParameter(00.0, 30.0, default=2.0, decimals=0, space="buy", optimize=False)
  entry_25_rsi_3_1h_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_25_rsi_3_4h_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_25_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_25_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_25_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_25_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_25_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_25_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_25_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_25_r_14_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_25_r_14_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_25_r_14_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_25_r_14_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_25_r_480_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_25_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_25_r_480_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_25_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_25_rsi_14_max = DecimalParameter(20.0, 46.0, default=46.0, decimals=0, space="buy", optimize=False)
  entry_25_cti_20_max = DecimalParameter(-0.9, 0.0, default=-0.9, decimals=1, space="buy", optimize=False)
  entry_25_ewo_50_200_min = DecimalParameter(1.0, 8.0, default=2.0, decimals=1, space="buy", optimize=False)
  entry_25_sma_offset = DecimalParameter(0.920, 0.950, default=0.948, decimals=3, space="buy", optimize=False)

  entry_26_close_max_12 = DecimalParameter(00.50, 0.95, default=0.80, decimals=2, space="buy", optimize=False)
  entry_26_close_max_24 = DecimalParameter(00.50, 0.95, default=0.75, decimals=2, space="buy", optimize=False)
  entry_26_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=False)
  entry_26_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.60, decimals=2, space="buy", optimize=False)
  entry_26_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.50, decimals=2, space="buy", optimize=False)
  entry_26_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.45, decimals=2, space="buy", optimize=False)
  entry_26_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.80, decimals=2, space="buy", optimize=False)
  entry_26_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.90, decimals=2, space="buy", optimize=False)
  entry_26_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=1.10, decimals=2, space="buy", optimize=False)
  entry_26_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.20, decimals=2, space="buy", optimize=False)
  entry_26_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_26_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_26_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_26_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_26_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_26_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_26_ema_200_not_dec_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_26_ema_200_not_dec_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_26_ema_200_not_dec_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_26_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_26_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_26_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_26_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_26_rsi_3_min = DecimalParameter(00.0, 30.0, default=2.0, decimals=0, space="buy", optimize=False)
  entry_26_rsi_3_max = DecimalParameter(30.0, 70.0, default=46.0, decimals=0, space="buy", optimize=False)
  entry_26_rsi_3_15m_min = DecimalParameter(00.0, 36.0, default=4.0, decimals=0, space="buy", optimize=False)
  entry_26_rsi_3_1h_min = DecimalParameter(00.0, 36.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_26_rsi_3_4h_min = DecimalParameter(00.0, 36.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_26_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_26_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_26_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_26_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_26_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_26_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_26_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_26_r_14_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_26_r_14_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_26_r_14_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_26_r_14_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_26_r_480_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_26_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_26_r_480_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_26_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_26_bb_offset = DecimalParameter(0.978, 0.999, default=0.992, decimals=3, space="buy", optimize=False)
  entry_26_ema_open_offset = DecimalParameter(0.0100, 0.0400, default=0.018, decimals=3, space="buy", optimize=False)
  entry_26_ewo_50_200_1h_min = DecimalParameter(1.0, 8.0, default=1.2, decimals=1, space="buy", optimize=False)

  entry_45_close_max_12 = DecimalParameter(00.50, 0.95, default=0.88, decimals=2, space="buy", optimize=False)
  entry_45_close_max_24 = DecimalParameter(00.50, 0.95, default=0.84, decimals=2, space="buy", optimize=False)
  entry_45_close_max_48 = DecimalParameter(00.50, 0.95, default=0.8, decimals=2, space="buy", optimize=False)
  entry_45_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.7, decimals=2, space="buy", optimize=False)
  entry_45_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.66, decimals=2, space="buy", optimize=False)
  entry_45_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.6, decimals=2, space="buy", optimize=False)
  entry_45_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.56, decimals=2, space="buy", optimize=False)
  entry_45_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.86, decimals=2, space="buy", optimize=False)
  entry_45_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=0.9, decimals=2, space="buy", optimize=False)
  entry_45_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.36, decimals=2, space="buy", optimize=False)
  entry_45_sup_level_1h_enabled = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
  entry_45_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_45_sup_level_4h_enabled = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
  entry_45_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_45_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_45_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_45_rsi_3_min = DecimalParameter(00.0, 30.0, default=5.0, decimals=0, space="buy", optimize=False)
  entry_45_rsi_3_max = DecimalParameter(30.0, 60.0, default=46.0, decimals=0, space="buy", optimize=False)
  entry_45_rsi_3_15m_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_45_rsi_3_1h_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_45_rsi_3_4h_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_45_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_45_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.92, decimals=2, space="buy", optimize=False)
  entry_45_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_45_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.92, decimals=2, space="buy", optimize=False)
  entry_45_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=75.0, decimals=0, space="buy", optimize=False)
  entry_45_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.92, decimals=2, space="buy", optimize=False)
  entry_45_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=75.0, decimals=0, space="buy", optimize=False)
  entry_45_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-6.0, decimals=0, space="buy", optimize=False)
  entry_45_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_45_rsi_14_min = DecimalParameter(10.0, 40.0, default=26.0, decimals=0, space="buy", optimize=False)
  entry_45_rsi_14_max = DecimalParameter(20.0, 60.0, default=40.0, decimals=0, space="buy", optimize=False)
  entry_45_cti_20_max = DecimalParameter(-0.99, -0.50, default=-0.54, decimals=2, space="buy", optimize=False)
  entry_45_sma_offset = DecimalParameter(0.940, 0.984, default=0.954, decimals=3, space="buy", optimize=False)

  entry_46_close_max_12 = DecimalParameter(00.50, 0.95, default=0.80, decimals=2, space="buy", optimize=False)
  entry_46_close_max_24 = DecimalParameter(00.50, 0.95, default=0.75, decimals=2, space="buy", optimize=False)
  entry_46_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=False)
  entry_46_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.60, decimals=2, space="buy", optimize=False)
  entry_46_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.50, decimals=2, space="buy", optimize=False)
  entry_46_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.45, decimals=2, space="buy", optimize=False)
  entry_46_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.5, decimals=2, space="buy", optimize=False)
  entry_46_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.75, decimals=2, space="buy", optimize=False)
  entry_46_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=0.90, decimals=2, space="buy", optimize=False)
  entry_46_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.00, decimals=2, space="buy", optimize=False)
  entry_46_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_46_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_46_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_46_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_46_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_46_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_46_ema_200_not_dec_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_46_ema_200_not_dec_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_46_ema_200_not_dec_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_46_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_46_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_46_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_46_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
  entry_46_rsi_3_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_46_rsi_3_max = DecimalParameter(30.0, 60.0, default=60.0, decimals=0, space="buy", optimize=False)
  entry_46_rsi_3_15m_min = DecimalParameter(00.0, 30.0, default=20.0, decimals=0, space="buy", optimize=False)
  entry_46_rsi_3_1h_min = DecimalParameter(00.0, 30.0, default=20.0, decimals=0, space="buy", optimize=False)
  entry_46_rsi_3_4h_min = DecimalParameter(00.0, 30.0, default=20.0, decimals=0, space="buy", optimize=False)
  entry_46_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=20.0, decimals=0, space="buy", optimize=False)
  entry_46_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_46_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_46_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_46_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_46_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_46_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_46_r_14_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_46_r_14_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_46_r_14_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_46_r_14_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_46_r_480_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_46_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_46_r_480_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_46_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_46_rsi_14_max = DecimalParameter(26.0, 60.0, default=60.0, decimals=0, space="buy", optimize=False)

  entry_47_close_max_12 = DecimalParameter(00.50, 0.95, default=0.80, decimals=2, space="buy", optimize=False)
  entry_47_close_max_24 = DecimalParameter(00.50, 0.95, default=0.75, decimals=2, space="buy", optimize=False)
  entry_47_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=False)
  entry_47_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.60, decimals=2, space="buy", optimize=False)
  entry_47_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.50, decimals=2, space="buy", optimize=False)
  entry_47_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.45, decimals=2, space="buy", optimize=False)
  entry_47_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.5, decimals=2, space="buy", optimize=False)
  entry_47_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.75, decimals=2, space="buy", optimize=False)
  entry_47_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=0.90, decimals=2, space="buy", optimize=False)
  entry_47_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.00, decimals=2, space="buy", optimize=False)
  entry_47_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_47_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_47_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_47_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_47_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_47_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_47_ema_200_not_dec_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_47_ema_200_not_dec_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_47_ema_200_not_dec_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_47_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
  entry_47_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_47_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_47_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_47_rsi_3_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_47_rsi_3_max = DecimalParameter(30.0, 60.0, default=60.0, decimals=0, space="buy", optimize=False)
  entry_47_rsi_3_15m_min = DecimalParameter(00.0, 30.0, default=16.0, decimals=0, space="buy", optimize=False)
  entry_47_rsi_3_1h_min = DecimalParameter(00.0, 30.0, default=20.0, decimals=0, space="buy", optimize=False)
  entry_47_rsi_3_4h_min = DecimalParameter(00.0, 30.0, default=20.0, decimals=0, space="buy", optimize=False)
  entry_47_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=20.0, decimals=0, space="buy", optimize=False)
  entry_47_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.85, decimals=2, space="buy", optimize=False)
  entry_47_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_47_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.85, decimals=2, space="buy", optimize=False)
  entry_47_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_47_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_47_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_47_r_14_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_47_r_14_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_47_r_14_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_47_r_14_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_47_r_480_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_47_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_47_r_480_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_47_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_47_rsi_14_min = DecimalParameter(20.0, 40.0, default=24.0, decimals=0, space="buy", optimize=False)
  entry_47_rsi_14_max = DecimalParameter(26.0, 60.0, default=60.0, decimals=0, space="buy", optimize=False)
  entry_47_rsi_20_min = DecimalParameter(20.0, 40.0, default=24.0, decimals=0, space="buy", optimize=False)
  entry_47_rsi_20_max = DecimalParameter(26.0, 60.0, default=60.0, decimals=0, space="buy", optimize=False)
  entry_47_cti_20_max = DecimalParameter(-0.8, 0.8, default=-0.5, decimals=1, space="buy", optimize=False)
  entry_47_ema_offset = DecimalParameter(0.980, 0.999, default=0.994, decimals=3, space="buy", optimize=False)
  entry_47_high_max_12_1h_max = DecimalParameter(00.70, 0.95, default=0.88, decimals=2, space="buy", optimize=False)

  entry_48_close_max_12 = DecimalParameter(00.50, 0.95, default=0.80, decimals=2, space="buy", optimize=False)
  entry_48_close_max_24 = DecimalParameter(00.50, 0.95, default=0.75, decimals=2, space="buy", optimize=False)
  entry_48_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=False)
  entry_48_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.60, decimals=2, space="buy", optimize=False)
  entry_48_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.50, decimals=2, space="buy", optimize=False)
  entry_48_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.45, decimals=2, space="buy", optimize=False)
  entry_48_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.5, decimals=2, space="buy", optimize=False)
  entry_48_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.75, decimals=2, space="buy", optimize=False)
  entry_48_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=0.90, decimals=2, space="buy", optimize=False)
  entry_48_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.00, decimals=2, space="buy", optimize=False)
  entry_48_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_48_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_48_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_48_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_48_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_48_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_48_ema_200_not_dec_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_48_ema_200_not_dec_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_48_ema_200_not_dec_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_48_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_48_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_48_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_48_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_48_ema_50_1h_over_ema_200_1h_enabled = CategoricalParameter(
    [True, False], default=True, space="buy", optimize=True
  )
  entry_48_rsi_3_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_48_rsi_3_max = DecimalParameter(30.0, 60.0, default=60.0, decimals=0, space="buy", optimize=False)
  entry_48_rsi_3_15m_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_48_rsi_3_1h_min = DecimalParameter(00.0, 30.0, default=10.0, decimals=0, space="buy", optimize=False)
  entry_48_rsi_3_4h_min = DecimalParameter(00.0, 30.0, default=10.0, decimals=0, space="buy", optimize=False)
  entry_48_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=10.0, decimals=0, space="buy", optimize=False)
  entry_48_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_48_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_48_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_48_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_48_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_48_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_48_r_14_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_48_r_14_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_48_r_14_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_48_r_14_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_48_r_480_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_48_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_48_r_480_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_48_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_48_rsi_14_max = DecimalParameter(26.0, 50.0, default=38.0, decimals=0, space="buy", optimize=False)
  entry_48_cci_20_max = DecimalParameter(-180.0, -80.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_48_sma_offset = DecimalParameter(0.940, 0.978, default=0.956, decimals=3, space="buy", optimize=False)
  entry_48_inc_min = DecimalParameter(0.01, 0.04, default=0.022, decimals=3, space="buy", optimize=False)

  entry_49_close_max_12 = DecimalParameter(00.50, 0.95, default=0.88, decimals=2, space="buy", optimize=False)
  entry_49_close_max_24 = DecimalParameter(00.50, 0.95, default=0.86, decimals=2, space="buy", optimize=False)
  entry_49_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=False)
  entry_49_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.60, decimals=2, space="buy", optimize=False)
  entry_49_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.50, decimals=2, space="buy", optimize=False)
  entry_49_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.45, decimals=2, space="buy", optimize=False)
  entry_49_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.5, decimals=2, space="buy", optimize=False)
  entry_49_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.75, decimals=2, space="buy", optimize=False)
  entry_49_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=0.90, decimals=2, space="buy", optimize=False)
  entry_49_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.00, decimals=2, space="buy", optimize=False)
  entry_49_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_49_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_49_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_49_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_49_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_49_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_49_ema_200_not_dec_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_49_ema_200_not_dec_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_49_ema_200_not_dec_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_49_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_49_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_49_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_49_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_49_ema_12_1h_over_ema_200_1h_enabled = CategoricalParameter(
    [True, False], default=True, space="buy", optimize=True
  )
  entry_49_rsi_3_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_49_rsi_3_max = DecimalParameter(30.0, 60.0, default=60.0, decimals=0, space="buy", optimize=False)
  entry_49_rsi_3_15m_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_49_rsi_3_1h_min = DecimalParameter(00.0, 30.0, default=10.0, decimals=0, space="buy", optimize=False)
  entry_49_rsi_3_4h_min = DecimalParameter(00.0, 30.0, default=10.0, decimals=0, space="buy", optimize=False)
  entry_49_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=10.0, decimals=0, space="buy", optimize=False)
  entry_49_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_49_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_49_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_49_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_49_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_49_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_49_r_14_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_49_r_14_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_49_r_14_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_49_r_14_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_49_r_480_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_49_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_49_r_480_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_49_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_49_rsi_14_max = DecimalParameter(26.0, 50.0, default=36.0, decimals=0, space="buy", optimize=False)
  entry_49_r_14_max = DecimalParameter(-99.0, -70.0, default=-50.0, decimals=0, space="buy", optimize=False)
  entry_49_inc_min = DecimalParameter(0.01, 0.04, default=0.028, decimals=3, space="buy", optimize=False)

  entry_50_close_max_12 = DecimalParameter(00.50, 0.95, default=0.80, decimals=2, space="buy", optimize=False)
  entry_50_close_max_24 = DecimalParameter(00.50, 0.95, default=0.75, decimals=2, space="buy", optimize=False)
  entry_50_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=False)
  entry_50_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.60, decimals=2, space="buy", optimize=False)
  entry_50_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.50, decimals=2, space="buy", optimize=False)
  entry_50_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.45, decimals=2, space="buy", optimize=False)
  entry_50_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.80, decimals=2, space="buy", optimize=False)
  entry_50_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.90, decimals=2, space="buy", optimize=False)
  entry_50_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=1.10, decimals=2, space="buy", optimize=False)
  entry_50_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.20, decimals=2, space="buy", optimize=False)
  entry_50_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_50_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_50_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_50_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_50_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_50_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_50_ema_200_not_dec_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_50_ema_200_not_dec_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_50_ema_200_not_dec_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_50_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_50_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_50_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_50_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_50_ema_12_1h_over_ema_200_1h_enabled = CategoricalParameter(
    [True, False], default=True, space="buy", optimize=False
  )
  entry_50_rsi_3_min = DecimalParameter(00.0, 30.0, default=0.0, decimals=0, space="buy", optimize=False)
  entry_50_rsi_3_max = DecimalParameter(30.0, 70.0, default=46.0, decimals=0, space="buy", optimize=False)
  entry_50_rsi_3_15m_min = DecimalParameter(00.0, 36.0, default=0.0, decimals=0, space="buy", optimize=False)
  entry_50_rsi_3_1h_min = DecimalParameter(00.0, 36.0, default=0.0, decimals=0, space="buy", optimize=False)
  entry_50_rsi_3_4h_min = DecimalParameter(00.0, 36.0, default=0.0, decimals=0, space="buy", optimize=False)
  entry_50_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=0.0, decimals=0, space="buy", optimize=False)
  entry_50_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.80, decimals=2, space="buy", optimize=False)
  entry_50_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_50_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_50_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_50_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_50_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_50_r_14_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_50_r_14_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_50_r_14_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_50_r_14_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_50_r_480_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_50_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_50_r_480_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_50_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_50_bb_offset = DecimalParameter(0.970, 0.999, default=0.995, decimals=3, space="buy", optimize=False)
  entry_50_ema_open_offset = DecimalParameter(0.020, 0.040, default=0.020, decimals=3, space="buy", optimize=False)

  entry_102_close_max_12 = DecimalParameter(00.50, 0.95, default=0.92, decimals=2, space="buy", optimize=False)
  entry_102_close_max_24 = DecimalParameter(00.50, 0.95, default=0.75, decimals=2, space="buy", optimize=False)
  entry_102_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=False)
  entry_102_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.65, decimals=2, space="buy", optimize=False)
  entry_102_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.60, decimals=2, space="buy", optimize=False)
  entry_102_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.55, decimals=2, space="buy", optimize=False)
  entry_102_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.5, decimals=2, space="buy", optimize=False)
  entry_102_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.75, decimals=2, space="buy", optimize=False)
  entry_102_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=0.90, decimals=2, space="buy", optimize=False)
  entry_102_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.00, decimals=2, space="buy", optimize=False)
  entry_102_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_102_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_102_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_102_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_102_sup_level_1d_enabled = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
  entry_102_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_102_ema_200_not_dec_1h_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_102_ema_200_not_dec_4h_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_102_ema_200_not_dec_1d_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_102_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
  entry_102_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_102_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
  entry_102_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_102_rsi_3_min = DecimalParameter(00.0, 30.0, default=12.0, decimals=0, space="buy", optimize=False)
  entry_102_rsi_3_max = DecimalParameter(30.0, 70.0, default=30.0, decimals=0, space="buy", optimize=False)
  entry_102_rsi_3_15m_min = DecimalParameter(00.0, 30.0, default=16.0, decimals=0, space="buy", optimize=False)
  entry_102_rsi_3_1h_min = DecimalParameter(00.0, 30.0, default=12.0, decimals=0, space="buy", optimize=False)
  entry_102_rsi_3_4h_min = DecimalParameter(00.0, 30.0, default=12.0, decimals=0, space="buy", optimize=False)
  entry_102_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_102_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_102_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_102_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_102_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_102_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_102_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_102_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_102_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_102_rsi_14_max = DecimalParameter(20.0, 60.0, default=46.0, decimals=0, space="buy", optimize=False)
  entry_102_ema_offset = DecimalParameter(0.940, 0.984, default=0.966, decimals=3, space="buy", optimize=False)
  entry_102_bb_offset = DecimalParameter(0.970, 1.010, default=0.999, decimals=3, space="buy", optimize=False)

  entry_103_close_max_12 = DecimalParameter(00.50, 0.95, default=0.80, decimals=2, space="buy", optimize=False)
  entry_103_close_max_24 = DecimalParameter(00.50, 0.95, default=0.75, decimals=2, space="buy", optimize=False)
  entry_103_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=False)
  entry_103_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.60, decimals=2, space="buy", optimize=False)
  entry_103_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.50, decimals=2, space="buy", optimize=False)
  entry_103_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.45, decimals=2, space="buy", optimize=False)
  entry_103_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.80, decimals=2, space="buy", optimize=False)
  entry_103_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.90, decimals=2, space="buy", optimize=False)
  entry_103_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=0.95, decimals=2, space="buy", optimize=False)
  entry_103_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.00, decimals=2, space="buy", optimize=False)
  entry_103_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_103_res_level_1h_enabled = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
  entry_103_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_103_res_level_4h_enabled = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
  entry_103_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_103_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_103_ema_200_not_dec_1h_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_103_ema_200_not_dec_4h_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_103_ema_200_not_dec_1d_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_103_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_103_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_103_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
  entry_103_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_103_rsi_3_min = DecimalParameter(00.0, 30.0, default=10.0, decimals=0, space="buy", optimize=False)
  entry_103_rsi_3_max = DecimalParameter(30.0, 70.0, default=65.0, decimals=0, space="buy", optimize=False)
  entry_103_rsi_3_15m_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_103_rsi_3_1h_min = DecimalParameter(00.0, 30.0, default=20.0, decimals=0, space="buy", optimize=False)
  entry_103_rsi_3_4h_min = DecimalParameter(00.0, 30.0, default=20.0, decimals=0, space="buy", optimize=False)
  entry_103_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=1.0, decimals=0, space="buy", optimize=False)
  entry_103_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.85, decimals=2, space="buy", optimize=False)
  entry_103_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=85.0, decimals=0, space="buy", optimize=False)
  entry_103_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_103_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=85.0, decimals=0, space="buy", optimize=False)
  entry_103_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_103_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=85.0, decimals=0, space="buy", optimize=False)
  entry_103_r_14_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_103_r_14_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_103_r_14_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_103_r_14_4h_max = DecimalParameter(-40.0, -0.0, default=-5.0, decimals=0, space="buy", optimize=False)
  entry_103_r_480_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_103_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-25.0, decimals=0, space="buy", optimize=False)
  entry_103_r_480_4h_min = DecimalParameter(-100.0, -70.0, default=-90.0, decimals=0, space="buy", optimize=False)
  entry_103_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-25.0, decimals=0, space="buy", optimize=False)
  entry_103_rsi_14_min = DecimalParameter(20.0, 60.0, default=24.0, decimals=0, space="buy", optimize=False)
  entry_103_sma_offset = DecimalParameter(0.930, 0.972, default=0.960, decimals=3, space="buy", optimize=False)
  entry_103_bb_offset = DecimalParameter(0.940, 1.010, default=0.948, decimals=3, space="buy", optimize=False)

  entry_104_close_max_12 = DecimalParameter(00.50, 0.95, default=0.80, decimals=2, space="buy", optimize=False)
  entry_104_close_max_24 = DecimalParameter(00.50, 0.95, default=0.75, decimals=2, space="buy", optimize=False)
  entry_104_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=False)
  entry_104_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.60, decimals=2, space="buy", optimize=False)
  entry_104_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.50, decimals=2, space="buy", optimize=False)
  entry_104_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.45, decimals=2, space="buy", optimize=False)
  entry_104_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.80, decimals=2, space="buy", optimize=False)
  entry_104_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.90, decimals=2, space="buy", optimize=False)
  entry_104_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=0.95, decimals=2, space="buy", optimize=False)
  entry_104_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.00, decimals=2, space="buy", optimize=False)
  entry_104_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_104_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_104_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_104_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_104_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_104_res_level_1d_enabled = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
  entry_104_ema_200_not_dec_1h_enabled = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
  entry_104_ema_200_not_dec_4h_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_104_ema_200_not_dec_1d_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_104_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_104_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_104_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_104_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_104_rsi_3_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_104_rsi_3_max = DecimalParameter(30.0, 70.0, default=46.0, decimals=0, space="buy", optimize=False)
  entry_104_rsi_3_15m_min = DecimalParameter(00.0, 36.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_104_rsi_3_1h_min = DecimalParameter(00.0, 36.0, default=8.0, decimals=0, space="buy", optimize=False)
  entry_104_rsi_3_4h_min = DecimalParameter(00.0, 36.0, default=8.0, decimals=0, space="buy", optimize=False)
  entry_104_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=8.0, decimals=0, space="buy", optimize=False)
  entry_104_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.80, decimals=2, space="buy", optimize=False)
  entry_104_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=70.0, decimals=0, space="buy", optimize=False)
  entry_104_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_104_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=70.0, decimals=0, space="buy", optimize=False)
  entry_104_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_104_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=70.0, decimals=0, space="buy", optimize=False)
  entry_104_r_14_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_104_r_14_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_104_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_104_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_104_rsi_14_min = DecimalParameter(20.0, 60.0, default=30.0, decimals=0, space="buy", optimize=False)
  entry_104_rsi_14_max = DecimalParameter(20.0, 60.0, default=46.0, decimals=0, space="buy", optimize=False)
  entry_104_sma_offset = DecimalParameter(0.940, 0.984, default=0.956, decimals=3, space="buy", optimize=False)

  entry_106_close_max_12 = DecimalParameter(00.50, 0.95, default=0.80, decimals=2, space="buy", optimize=False)
  entry_106_close_max_24 = DecimalParameter(00.50, 0.95, default=0.75, decimals=2, space="buy", optimize=False)
  entry_106_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=False)
  entry_106_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.60, decimals=2, space="buy", optimize=False)
  entry_106_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.50, decimals=2, space="buy", optimize=False)
  entry_106_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.45, decimals=2, space="buy", optimize=False)
  entry_106_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.5, decimals=2, space="buy", optimize=False)
  entry_106_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.75, decimals=2, space="buy", optimize=False)
  entry_106_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=0.90, decimals=2, space="buy", optimize=False)
  entry_106_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.00, decimals=2, space="buy", optimize=False)
  entry_106_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_106_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_106_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_106_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_106_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_106_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_106_ema_200_not_dec_1h_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_106_ema_200_not_dec_4h_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_106_ema_200_not_dec_1d_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_106_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_106_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_106_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_106_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_106_rsi_3_min = DecimalParameter(00.0, 30.0, default=2.0, decimals=0, space="buy", optimize=False)
  entry_106_rsi_3_max = DecimalParameter(30.0, 60.0, default=60.0, decimals=0, space="buy", optimize=False)
  entry_106_rsi_3_15m_min = DecimalParameter(00.0, 30.0, default=2.0, decimals=0, space="buy", optimize=False)
  entry_106_rsi_3_1h_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_106_rsi_3_4h_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_106_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_106_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_106_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_106_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_106_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_106_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_106_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_106_r_14_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_106_r_14_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_106_r_14_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_106_r_14_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_106_r_480_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_106_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_106_r_480_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_106_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_106_cti_20_max = DecimalParameter(-0.9, 0.0, default=-0.7, decimals=1, space="buy", optimize=False)
  entry_106_ewo_50_200_max = DecimalParameter(-2.0, -10.0, default=-8.0, decimals=1, space="buy", optimize=True)
  entry_106_sma_offset = DecimalParameter(0.980, 0.999, default=0.986, decimals=3, space="buy", optimize=True)

  entry_107_close_max_12 = DecimalParameter(00.50, 0.95, default=0.80, decimals=2, space="buy", optimize=False)
  entry_107_close_max_24 = DecimalParameter(00.50, 0.95, default=0.75, decimals=2, space="buy", optimize=False)
  entry_107_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=False)
  entry_107_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.60, decimals=2, space="buy", optimize=False)
  entry_107_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.50, decimals=2, space="buy", optimize=False)
  entry_107_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.45, decimals=2, space="buy", optimize=False)
  entry_107_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.80, decimals=2, space="buy", optimize=False)
  entry_107_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.90, decimals=2, space="buy", optimize=False)
  entry_107_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=1.10, decimals=2, space="buy", optimize=False)
  entry_107_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.20, decimals=2, space="buy", optimize=False)
  entry_107_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_107_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_107_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_107_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_107_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_107_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_107_ema_200_not_dec_1h_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_107_ema_200_not_dec_4h_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_107_ema_200_not_dec_1d_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_107_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_107_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_107_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_107_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_107_rsi_3_min = DecimalParameter(00.0, 30.0, default=2.0, decimals=0, space="buy", optimize=False)
  entry_107_rsi_3_max = DecimalParameter(30.0, 70.0, default=46.0, decimals=0, space="buy", optimize=False)
  entry_107_rsi_3_15m_min = DecimalParameter(00.0, 36.0, default=2.0, decimals=0, space="buy", optimize=False)
  entry_107_rsi_3_1h_min = DecimalParameter(00.0, 36.0, default=8.0, decimals=0, space="buy", optimize=False)
  entry_107_rsi_3_4h_min = DecimalParameter(00.0, 36.0, default=8.0, decimals=0, space="buy", optimize=False)
  entry_107_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=8.0, decimals=0, space="buy", optimize=False)
  entry_107_cti_20_1h_min = DecimalParameter(-0.99, -0.50, default=-0.95, decimals=2, space="buy", optimize=False)
  entry_107_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.80, decimals=2, space="buy", optimize=False)
  entry_107_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_107_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.90, decimals=2, space="buy", optimize=False)
  entry_107_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_107_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_107_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_107_r_14_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_107_r_14_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_107_r_14_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_107_r_14_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_107_r_480_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_107_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_107_r_480_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_107_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_107_bb40_bbdelta_close = DecimalParameter(0.030, 0.060, default=0.040, decimals=3, space="buy", optimize=False)
  entry_107_bb40_closedelta_close = DecimalParameter(
    0.010, 0.040, default=0.020, decimals=3, space="buy", optimize=False
  )
  entry_107_bb40_tail_bbdelta = DecimalParameter(0.10, 0.60, default=0.40, decimals=2, space="buy", optimize=False)
  entry_107_cti_20_max = DecimalParameter(-0.90, -0.50, default=-0.75, decimals=2, space="buy", optimize=False)
  entry_107_r_480_min = DecimalParameter(-100.0, -80.0, default=-94.0, decimals=0, space="buy", optimize=False)

  entry_108_close_max_12 = DecimalParameter(00.50, 0.95, default=0.80, decimals=2, space="buy", optimize=False)
  entry_108_close_max_24 = DecimalParameter(00.50, 0.95, default=0.75, decimals=2, space="buy", optimize=False)
  entry_108_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=False)
  entry_108_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.60, decimals=2, space="buy", optimize=False)
  entry_108_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.50, decimals=2, space="buy", optimize=False)
  entry_108_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.45, decimals=2, space="buy", optimize=False)
  entry_108_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.80, decimals=2, space="buy", optimize=False)
  entry_108_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.90, decimals=2, space="buy", optimize=False)
  entry_108_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=1.10, decimals=2, space="buy", optimize=False)
  entry_108_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.20, decimals=2, space="buy", optimize=False)
  entry_108_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_108_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_108_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_108_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_108_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_108_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_108_ema_200_not_dec_1h_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_108_ema_200_not_dec_4h_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_108_ema_200_not_dec_1d_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_108_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_108_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_108_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_108_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_108_rsi_3_min = DecimalParameter(00.0, 30.0, default=0.0, decimals=0, space="buy", optimize=False)
  entry_108_rsi_3_max = DecimalParameter(30.0, 70.0, default=46.0, decimals=0, space="buy", optimize=False)
  entry_108_rsi_3_15m_min = DecimalParameter(00.0, 36.0, default=2.0, decimals=0, space="buy", optimize=False)
  entry_108_rsi_3_1h_min = DecimalParameter(00.0, 36.0, default=8.0, decimals=0, space="buy", optimize=False)
  entry_108_rsi_3_4h_min = DecimalParameter(00.0, 36.0, default=8.0, decimals=0, space="buy", optimize=False)
  entry_108_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=8.0, decimals=0, space="buy", optimize=False)
  entry_108_cti_20_1h_min = DecimalParameter(-0.99, -0.50, default=-0.95, decimals=2, space="buy", optimize=False)
  entry_108_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.90, decimals=2, space="buy", optimize=False)
  entry_108_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_108_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.90, decimals=2, space="buy", optimize=False)
  entry_108_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_108_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_108_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_108_r_14_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_108_r_14_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_108_r_14_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_108_r_14_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_108_r_480_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_108_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_108_r_480_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_108_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_108_rsi_14_min = DecimalParameter(10.0, 40.0, default=15.0, decimals=0, space="buy", optimize=False)
  entry_108_cti_20_max = DecimalParameter(-0.70, -0.40, default=-0.50, decimals=2, space="buy", optimize=False)
  entry_108_r_14_max = DecimalParameter(-100.0, 80.0, default=-90.0, decimals=0, space="buy", optimize=False)
  entry_108_bb_offset = DecimalParameter(0.970, 0.999, default=0.999, decimals=3, space="buy", optimize=False)
  entry_108_ema_open_offset = DecimalParameter(0.0100, 0.0400, default=0.0200, decimals=4, space="buy", optimize=False)

  entry_109_close_max_12 = DecimalParameter(00.50, 0.95, default=0.80, decimals=2, space="buy", optimize=False)
  entry_109_close_max_24 = DecimalParameter(00.50, 0.95, default=0.75, decimals=2, space="buy", optimize=False)
  entry_109_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=False)
  entry_109_high_max_24_1h = DecimalParameter(00.40, 0.95, default=0.60, decimals=2, space="buy", optimize=False)
  entry_109_high_max_24_4h = DecimalParameter(00.40, 0.95, default=0.50, decimals=2, space="buy", optimize=False)
  entry_109_high_max_6_1d = DecimalParameter(00.30, 0.95, default=0.45, decimals=2, space="buy", optimize=False)
  entry_109_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.80, decimals=2, space="buy", optimize=False)
  entry_109_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.90, decimals=2, space="buy", optimize=False)
  entry_109_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=1.10, decimals=2, space="buy", optimize=False)
  entry_109_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.20, decimals=2, space="buy", optimize=False)
  entry_109_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_109_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_109_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_109_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_109_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_109_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_109_ema_200_not_dec_1h_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_109_ema_200_not_dec_4h_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_109_ema_200_not_dec_1d_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_109_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_109_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_109_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_109_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_109_ema_50_over_ema_200_enabled = CategoricalParameter(
    [True, False], default=True, space="buy", optimize=False
  )
  entry_109_ema_100_over_ema_200_enabled = CategoricalParameter(
    [True, False], default=True, space="buy", optimize=False
  )
  entry_109_rsi_3_min = DecimalParameter(00.0, 30.0, default=0.0, decimals=0, space="buy", optimize=False)
  entry_109_rsi_3_max = DecimalParameter(30.0, 70.0, default=46.0, decimals=0, space="buy", optimize=False)
  entry_109_rsi_3_15m_min = DecimalParameter(00.0, 36.0, default=6.0, decimals=0, space="buy", optimize=False)
  entry_109_rsi_3_1h_min = DecimalParameter(00.0, 36.0, default=10.0, decimals=0, space="buy", optimize=False)
  entry_109_rsi_3_4h_min = DecimalParameter(00.0, 36.0, default=10.0, decimals=0, space="buy", optimize=False)
  entry_109_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=10.0, decimals=0, space="buy", optimize=False)
  entry_109_cti_20_1h_min = DecimalParameter(-0.99, -0.50, default=-0.99, decimals=2, space="buy", optimize=False)
  entry_109_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.90, decimals=2, space="buy", optimize=False)
  entry_109_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_109_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.90, decimals=2, space="buy", optimize=False)
  entry_109_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_109_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.95, decimals=2, space="buy", optimize=False)
  entry_109_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=90.0, decimals=0, space="buy", optimize=False)
  entry_109_r_14_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_109_r_14_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_109_r_14_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_109_r_14_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_109_r_480_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_109_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_109_r_480_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_109_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_109_cti_20_max = DecimalParameter(-0.95, -0.70, default=-0.85, decimals=2, space="buy", optimize=False)
  entry_109_r_14_max = DecimalParameter(-100.0, 80.0, default=-90.0, decimals=0, space="buy", optimize=False)
  entry_109_bb_offset = DecimalParameter(0.970, 0.999, default=0.992, decimals=3, space="buy", optimize=False)
  entry_109_ema_offset = DecimalParameter(0.940, 0.972, default=0.966, decimals=3, space="buy", optimize=False)

  entry_110_close_max_12 = DecimalParameter(00.50, 0.95, default=0.80, decimals=2, space="buy", optimize=False)
  entry_110_close_max_24 = DecimalParameter(00.50, 0.95, default=0.75, decimals=2, space="buy", optimize=False)
  entry_110_close_max_48 = DecimalParameter(00.50, 0.95, default=0.70, decimals=2, space="buy", optimize=False)
  entry_110_high_max_24_1h = DecimalParameter(00.40, 0.90, default=0.65, decimals=2, space="buy", optimize=False)
  entry_110_high_max_24_4h = DecimalParameter(00.40, 0.85, default=0.60, decimals=2, space="buy", optimize=False)
  entry_110_high_max_6_1d = DecimalParameter(00.30, 0.80, default=0.55, decimals=2, space="buy", optimize=False)
  entry_110_hl_pct_change_6_1h = DecimalParameter(00.30, 0.90, default=0.5, decimals=2, space="buy", optimize=False)
  entry_110_hl_pct_change_12_1h = DecimalParameter(00.40, 1.00, default=0.75, decimals=2, space="buy", optimize=False)
  entry_110_hl_pct_change_24_1h = DecimalParameter(00.50, 1.20, default=0.90, decimals=2, space="buy", optimize=False)
  entry_110_hl_pct_change_48_1h = DecimalParameter(00.60, 1.60, default=1.00, decimals=2, space="buy", optimize=False)
  entry_110_sup_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_110_res_level_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_110_sup_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_110_res_level_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_110_sup_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_110_res_level_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_110_ema_200_not_dec_1h_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_110_ema_200_not_dec_4h_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_110_ema_200_not_dec_1d_enabled = CategoricalParameter(
    [True, False], default=False, space="buy", optimize=False
  )
  entry_110_not_downtrend_15m_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_110_not_downtrend_1h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_110_not_downtrend_4h_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_110_not_downtrend_1d_enabled = CategoricalParameter([True, False], default=False, space="buy", optimize=False)
  entry_110_rsi_3_min = DecimalParameter(00.0, 30.0, default=2.0, decimals=0, space="buy", optimize=False)
  entry_110_rsi_3_max = DecimalParameter(30.0, 60.0, default=60.0, decimals=0, space="buy", optimize=False)
  entry_110_rsi_3_15m_min = DecimalParameter(00.0, 30.0, default=8.0, decimals=0, space="buy", optimize=False)
  entry_110_rsi_3_1h_min = DecimalParameter(00.0, 30.0, default=16.0, decimals=0, space="buy", optimize=False)
  entry_110_rsi_3_4h_min = DecimalParameter(00.0, 30.0, default=10.0, decimals=0, space="buy", optimize=False)
  entry_110_rsi_3_1d_min = DecimalParameter(00.0, 30.0, default=10.0, decimals=0, space="buy", optimize=False)
  entry_110_cti_20_1h_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_110_rsi_14_1h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_110_cti_20_4h_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_110_rsi_14_4h_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_110_cti_20_1d_max = DecimalParameter(0.0, 0.99, default=0.9, decimals=2, space="buy", optimize=False)
  entry_110_rsi_14_1d_max = DecimalParameter(50.0, 90.0, default=80.0, decimals=0, space="buy", optimize=False)
  entry_110_r_14_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_110_r_14_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_110_r_14_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_110_r_14_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_110_r_480_1h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_110_r_480_1h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_110_r_480_4h_min = DecimalParameter(-100.0, -70.0, default=-100.0, decimals=0, space="buy", optimize=False)
  entry_110_r_480_4h_max = DecimalParameter(-40.0, -0.0, default=-0.0, decimals=0, space="buy", optimize=False)
  entry_110_cti_20_max = DecimalParameter(-0.99, -0.50, default=-0.95, decimals=1, space="buy", optimize=False)
  entry_110_ewo_50_200_max = DecimalParameter(-2.0, -10.0, default=-3.0, decimals=1, space="buy", optimize=True)
  entry_110_ema_offset = DecimalParameter(0.980, 0.999, default=0.994, decimals=3, space="buy", optimize=True)

  #############################################################
  # CACHES

  hold_trades_cache = None
  target_profit_cache = None
  #############################################################

  def __init__(self, config: dict) -> None:
    if "ccxt_config" not in config["exchange"]:
      config["exchange"]["ccxt_config"] = {}
    if "ccxt_async_config" not in config["exchange"]:
      config["exchange"]["ccxt_async_config"] = {}

    options = {
      "brokerId": None,
      "broker": {"spot": None, "margin": None, "future": None, "delivery": None},
      "partner": {
        "spot": {"id": None, "key": None},
        "future": {"id": None, "key": None},
        "id": None,
        "key": None,
      },
    }

    config["exchange"]["ccxt_config"]["options"] = options
    config["exchange"]["ccxt_async_config"]["options"] = options
    super().__init__(config)
    if ("exit_profit_only" in self.config and self.config["exit_profit_only"]) or (
      "sell_profit_only" in self.config and self.config["sell_profit_only"]
    ):
      self.exit_profit_only = True
    if "regular_mode_stake_multiplier_spot" in self.config:
      self.regular_mode_stake_multiplier_spot = self.config["regular_mode_stake_multiplier_spot"]
    if "regular_mode_stake_multiplier_futures" in self.config:
      self.regular_mode_stake_multiplier_futures = self.config["regular_mode_stake_multiplier_futures"]
    if "max_slippage" in self.config:
      self.max_slippage = self.config["max_slippage"]
    if self.target_profit_cache is None:
      bot_name = ""
      if "bot_name" in self.config:
        bot_name = self.config["bot_name"] + "-"
      self.target_profit_cache = Cache(
        self.config["user_data_dir"]
        / (
          "nfix4-profit_max-"
          + bot_name
          + self.config["exchange"]["name"]
          + "-"
          + self.config["stake_currency"]
          + ("-(backtest)" if (self.config["runmode"].value == "backtest") else "")
          + ("-(hyperopt)" if (self.config["runmode"].value == "hyperopt") else "")
          + ".json"
        )
      )

    # OKX, Kraken provides a lower number of candle data per API call
    if self.config["exchange"]["name"] in ["okx", "okex"]:
      self.startup_candle_count = 480
    elif self.config["exchange"]["name"] in ["kraken"]:
      self.startup_candle_count = 710
    elif self.config["exchange"]["name"] in ["bybit"]:
      self.startup_candle_count = 199
    elif self.config["exchange"]["name"] in ["bitget"]:
      self.startup_candle_count = 499

    if ("trading_mode" in self.config) and (self.config["trading_mode"] in ["futures", "margin"]):
      self.is_futures_mode = True

    # If the cached data hasn't changed, it's a no-op
    self.target_profit_cache.save()

  def get_ticker_indicator(self):
    return int(self.timeframe[:-1])

  def mark_profit_target(
    self,
    mode_name: str,
    pair: str,
    sell: bool,
    signal_name: str,
    trade: Trade,
    current_time: datetime,
    current_rate: float,
    current_profit: float,
    last_candle,
    previous_candle_1,
  ) -> tuple:
    if sell and (signal_name is not None):
      return pair, signal_name

    return None, None

  def exit_profit_target(
    self,
    mode_name: str,
    pair: str,
    trade: Trade,
    current_time: datetime,
    current_rate: float,
    profit_stake: float,
    profit_ratio: float,
    profit_current_stake_ratio: float,
    profit_init_ratio: float,
    last_candle,
    previous_candle_1,
    previous_rate,
    previous_profit,
    previous_sell_reason,
    previous_time_profit_reached,
    enter_tags,
  ) -> tuple:
    if previous_sell_reason in [f"exit_{mode_name}_stoploss_doom", f"exit_{mode_name}_stoploss"]:
      if profit_ratio > 0.04:
        # profit is over the threshold, don't exit
        self._remove_profit_target(pair)
        return False, None
      if profit_ratio < -0.18:
        if profit_ratio < (previous_profit - 0.04):
          return True, previous_sell_reason
      elif profit_ratio < -0.1:
        if profit_ratio < (previous_profit - 0.04):
          return True, previous_sell_reason
      elif profit_ratio < -0.04:
        if profit_ratio < (previous_profit - 0.04):
          return True, previous_sell_reason
      else:
        if profit_ratio < (previous_profit - 0.04):
          return True, previous_sell_reason
    elif previous_sell_reason in [f"exit_{mode_name}_stoploss_u_e"]:
      if profit_current_stake_ratio > 0.04:
        # profit is over the threshold, don't exit
        self._remove_profit_target(pair)
        return False, None
      if profit_ratio < (previous_profit - (0.20 if trade.realized_profit == 0.0 else 0.26)):
        return True, previous_sell_reason
    elif previous_sell_reason in [f"exit_profit_{mode_name}_max"]:
      if profit_current_stake_ratio < -0.08:
        # profit is under the threshold, cancel it
        self._remove_profit_target(pair)
        return False, None
      if self.is_futures_mode:
        if 0.01 <= profit_current_stake_ratio < 0.02:
          if profit_current_stake_ratio < (previous_profit * 0.5):
            return True, previous_sell_reason
        elif 0.02 <= profit_current_stake_ratio < 0.03:
          if profit_current_stake_ratio < (previous_profit * 0.6):
            return True, previous_sell_reason
        elif 0.03 <= profit_current_stake_ratio < 0.04:
          if profit_current_stake_ratio < (previous_profit * 0.7):
            return True, previous_sell_reason
        elif 0.04 <= profit_current_stake_ratio < 0.08:
          if profit_current_stake_ratio < (previous_profit * 0.8):
            return True, previous_sell_reason
        elif 0.08 <= profit_current_stake_ratio < 0.16:
          if profit_current_stake_ratio < (previous_profit * 0.9):
            return True, previous_sell_reason
        elif 0.16 <= profit_current_stake_ratio:
          if profit_current_stake_ratio < (previous_profit * 0.95):
            return True, previous_sell_reason
      else:
        if 0.01 <= profit_current_stake_ratio < 0.03:
          if profit_current_stake_ratio < (previous_profit * 0.6):
            return True, previous_sell_reason
        elif 0.03 <= profit_current_stake_ratio < 0.08:
          if profit_current_stake_ratio < (previous_profit * 0.65):
            return True, previous_sell_reason
        elif 0.08 <= profit_current_stake_ratio < 0.16:
          if profit_current_stake_ratio < (previous_profit * 0.7):
            return True, previous_sell_reason
        elif 0.16 <= profit_current_stake_ratio:
          if profit_current_stake_ratio < (previous_profit * 0.75):
            return True, previous_sell_reason
    else:
      return False, None

    return False, None

  def calc_total_profit(
    self, trade: "Trade", filled_entries: "Orders", filled_exits: "Orders", exit_rate: float
  ) -> tuple:
    """
    Calculates the absolute profit for open trades.

    :param trade: trade object.
    :param filled_entries: Filled entries list.
    :param filled_exits: Filled exits list.
    :param exit_rate: The exit rate.
    :return tuple: The total profit in stake, ratio, ratio based on current stake, and ratio based on the first entry stake.
    """
    total_stake = 0.0
    total_profit = 0.0
    for entry in filled_entries:
      entry_stake = entry.safe_filled * entry.safe_price * (1 + trade.fee_open)
      total_stake += entry_stake
      total_profit -= entry_stake
    for exit in filled_exits:
      exit_stake = exit.safe_filled * exit.safe_price * (1 - trade.fee_close)
      total_profit += exit_stake
    current_stake = trade.amount * exit_rate * (1 - trade.fee_close)
    if self.is_futures_mode:
      if trade.is_short:
        current_stake -= trade.funding_fees
      else:
        current_stake += trade.funding_fees
    total_profit += current_stake
    total_profit_ratio = total_profit / total_stake
    current_profit_ratio = total_profit / current_stake
    init_profit_ratio = total_profit / filled_entries[0].cost
    return total_profit, total_profit_ratio, current_profit_ratio, init_profit_ratio

  def informative_pairs(self):
    # get access to all pairs available in whitelist.
    pairs = self.dp.current_whitelist()
    # Assign tf to each pair so they can be downloaded and cached for strategy.
    informative_pairs = []
    for info_timeframe in self.info_timeframes:
      informative_pairs.extend([(pair, info_timeframe) for pair in pairs])

    if self.config["stake_currency"] in [
      "USDT",
      "BUSD",
      "USDC",
      "DAI",
      "TUSD",
      "FDUSD",
      "PAX",
      "USD",
      "EUR",
      "GBP",
      "TRY",
    ]:
      if ("trading_mode" in self.config) and (self.config["trading_mode"] in ["futures", "margin"]):
        btc_info_pair = f"BTC/{self.config['stake_currency']}:{self.config['stake_currency']}"
      else:
        btc_info_pair = f"BTC/{self.config['stake_currency']}"
    else:
      if ("trading_mode" in self.config) and (self.config["trading_mode"] in ["futures", "margin"]):
        btc_info_pair = "BTC/USDT:USDT"
      else:
        btc_info_pair = "BTC/USDT"

    informative_pairs.extend([(btc_info_pair, btc_info_timeframe) for btc_info_timeframe in self.btc_info_timeframes])

    return informative_pairs

  def informative_1d_indicators(self, metadata: dict, info_timeframe) -> DataFrame:
    tik = time.perf_counter()
    assert self.dp, "DataProvider is required for multiple timeframes."
    # Get the informative pair
    informative_1d = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=info_timeframe)

    # Indicators
    # -----------------------------------------------------------------------------------------
    # RSI
    informative_1d["rsi_3"] = ta.RSI(informative_1d, timeperiod=3, fillna=True)
    informative_1d["rsi_14"] = ta.RSI(informative_1d, timeperiod=14)

    informative_1d["rsi_14_max_6"] = informative_1d["rsi_14"].rolling(6).max()

    # EMA
    informative_1d["ema_200"] = ta.EMA(informative_1d, timeperiod=200)

    informative_1d["ema_200_dec_4"] = (informative_1d["ema_200"].isnull()) | (
      informative_1d["ema_200"] <= informative_1d["ema_200"].shift(4)
    )

    # CTI
    informative_1d["cti_20"] = pta.cti(informative_1d["close"], length=20)

    informative_1d["cti_20_dec_3"] = informative_1d["cti_20"] < informative_1d["cti_20"].shift(3)

    # Pivots
    (
      informative_1d["pivot"],
      informative_1d["res1"],
      informative_1d["res2"],
      informative_1d["res3"],
      informative_1d["sup1"],
      informative_1d["sup2"],
      informative_1d["sup3"],
    ) = pivot_points(informative_1d, mode="fibonacci")

    # S/R
    res_series = (
      informative_1d["high"].rolling(window=5, center=True).apply(lambda row: is_resistance(row), raw=True).shift(2)
    )
    sup_series = (
      informative_1d["low"].rolling(window=5, center=True).apply(lambda row: is_support(row), raw=True).shift(2)
    )
    informative_1d["res_level"] = Series(
      np.where(
        res_series,
        np.where(informative_1d["close"] > informative_1d["open"], informative_1d["close"], informative_1d["open"]),
        float("NaN"),
      )
    ).ffill()
    informative_1d["res_hlevel"] = Series(np.where(res_series, informative_1d["high"], float("NaN"))).ffill()
    informative_1d["sup_level"] = Series(
      np.where(
        sup_series,
        np.where(informative_1d["close"] < informative_1d["open"], informative_1d["close"], informative_1d["open"]),
        float("NaN"),
      )
    ).ffill()

    # Downtrend checks
    informative_1d["not_downtrend"] = (informative_1d["close"] > informative_1d["close"].shift(2)) | (
      informative_1d["rsi_14"] > 50.0
    )

    informative_1d["is_downtrend_3"] = (
      (informative_1d["close"] < informative_1d["open"])
      & (informative_1d["close"].shift(1) < informative_1d["open"].shift(1))
      & (informative_1d["close"].shift(2) < informative_1d["open"].shift(2))
    )

    informative_1d["is_downtrend_5"] = (
      (informative_1d["close"] < informative_1d["open"])
      & (informative_1d["close"].shift(1) < informative_1d["open"].shift(1))
      & (informative_1d["close"].shift(2) < informative_1d["open"].shift(2))
      & (informative_1d["close"].shift(3) < informative_1d["open"].shift(3))
      & (informative_1d["close"].shift(4) < informative_1d["open"].shift(4))
    )

    # Wicks
    informative_1d["top_wick_pct"] = (
      informative_1d["high"] - np.maximum(informative_1d["open"], informative_1d["close"])
    ) / np.maximum(informative_1d["open"], informative_1d["close"])
    informative_1d["bot_wick_pct"] = abs(
      (informative_1d["low"] - np.minimum(informative_1d["open"], informative_1d["close"]))
      / np.minimum(informative_1d["open"], informative_1d["close"])
    )

    # Candle change
    informative_1d["change_pct"] = (informative_1d["close"] - informative_1d["open"]) / informative_1d["open"]

    # Pump protections
    informative_1d["hl_pct_change_3"] = range_percent_change(self, informative_1d, "HL", 3)
    informative_1d["hl_pct_change_6"] = range_percent_change(self, informative_1d, "HL", 6)

    # Max highs
    informative_1d["high_max_6"] = informative_1d["high"].rolling(6).max()
    informative_1d["high_max_12"] = informative_1d["high"].rolling(12).max()

    # Performance logging
    # -----------------------------------------------------------------------------------------
    tok = time.perf_counter()
    log.debug(f"[{metadata['pair']}] informative_1d_indicators took: {tok - tik:0.4f} seconds.")

    return informative_1d

  def informative_4h_indicators(self, metadata: dict, info_timeframe) -> DataFrame:
    tik = time.perf_counter()
    assert self.dp, "DataProvider is required for multiple timeframes."
    # Get the informative pair
    informative_4h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=info_timeframe)

    # Indicators
    # -----------------------------------------------------------------------------------------
    # RSI
    informative_4h["rsi_3"] = ta.RSI(informative_4h, timeperiod=3, fillna=True)
    informative_4h["rsi_14"] = ta.RSI(informative_4h, timeperiod=14, fillna=True)

    informative_4h["rsi_14_max_3"] = informative_4h["rsi_14"].rolling(3).max()
    informative_4h["rsi_14_max_6"] = informative_4h["rsi_14"].rolling(6).max()

    # EMA
    informative_4h["ema_12"] = ta.EMA(informative_4h, timeperiod=12)
    informative_4h["ema_26"] = ta.EMA(informative_4h, timeperiod=26)
    informative_4h["ema_50"] = ta.EMA(informative_4h, timeperiod=50)
    informative_4h["ema_100"] = ta.EMA(informative_4h, timeperiod=100)
    informative_4h["ema_200"] = ta.EMA(informative_4h, timeperiod=200)

    informative_4h["ema_200_dec_24"] = (informative_4h["ema_200"].isnull()) | (
      informative_4h["ema_200"] <= informative_4h["ema_200"].shift(24)
    )

    # SMA
    informative_4h["sma_12"] = ta.SMA(informative_4h, timeperiod=12)
    informative_4h["sma_26"] = ta.SMA(informative_4h, timeperiod=26)
    informative_4h["sma_50"] = ta.SMA(informative_4h, timeperiod=50)
    informative_4h["sma_200"] = ta.SMA(informative_4h, timeperiod=200)

    # Williams %R
    informative_4h["r_14"] = williams_r(informative_4h, period=14)
    informative_4h["r_480"] = williams_r(informative_4h, period=480)

    # CTI
    informative_4h["cti_20"] = pta.cti(informative_4h["close"], length=20)

    # S/R
    res_series = (
      informative_4h["high"].rolling(window=5, center=True).apply(lambda row: is_resistance(row), raw=True).shift(2)
    )
    sup_series = (
      informative_4h["low"].rolling(window=5, center=True).apply(lambda row: is_support(row), raw=True).shift(2)
    )
    informative_4h["res_level"] = Series(
      np.where(
        res_series,
        np.where(informative_4h["close"] > informative_4h["open"], informative_4h["close"], informative_4h["open"]),
        float("NaN"),
      )
    ).ffill()
    informative_4h["res_hlevel"] = Series(np.where(res_series, informative_4h["high"], float("NaN"))).ffill()
    informative_4h["sup_level"] = Series(
      np.where(
        sup_series,
        np.where(informative_4h["close"] < informative_4h["open"], informative_4h["close"], informative_4h["open"]),
        float("NaN"),
      )
    ).ffill()

    # Downtrend checks
    informative_4h["not_downtrend"] = (informative_4h["close"] > informative_4h["close"].shift(2)) | (
      informative_4h["rsi_14"] > 50.0
    )

    informative_4h["is_downtrend_3"] = (
      (informative_4h["close"] < informative_4h["open"])
      & (informative_4h["close"].shift(1) < informative_4h["open"].shift(1))
      & (informative_4h["close"].shift(2) < informative_4h["open"].shift(2))
    )

    # Wicks
    informative_4h["top_wick_pct"] = (
      informative_4h["high"] - np.maximum(informative_4h["open"], informative_4h["close"])
    ) / np.maximum(informative_4h["open"], informative_4h["close"])

    # Candle change
    informative_4h["change_pct"] = (informative_4h["close"] - informative_4h["open"]) / informative_4h["open"]

    # Max highs
    informative_4h["high_max_3"] = informative_4h["high"].rolling(3).max()
    informative_4h["high_max_12"] = informative_4h["high"].rolling(12).max()
    informative_4h["high_max_24"] = informative_4h["high"].rolling(24).max()
    informative_4h["high_max_36"] = informative_4h["high"].rolling(36).max()
    informative_4h["high_max_48"] = informative_4h["high"].rolling(48).max()

    # Volume
    informative_4h["volume_mean_factor_6"] = informative_4h["volume"] / informative_4h["volume"].rolling(6).mean()

    # Performance logging
    # -----------------------------------------------------------------------------------------
    tok = time.perf_counter()
    log.debug(f"[{metadata['pair']}] informative_1d_indicators took: {tok - tik:0.4f} seconds.")

    return informative_4h

  def informative_1h_indicators(self, metadata: dict, info_timeframe) -> DataFrame:
    tik = time.perf_counter()
    assert self.dp, "DataProvider is required for multiple timeframes."
    # Get the informative pair
    informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=info_timeframe)

    # Indicators
    # -----------------------------------------------------------------------------------------
    # RSI
    informative_1h["rsi_3"] = ta.RSI(informative_1h, timeperiod=3)
    informative_1h["rsi_14"] = ta.RSI(informative_1h, timeperiod=14)

    # EMA
    informative_1h["ema_12"] = ta.EMA(informative_1h, timeperiod=12)
    informative_1h["ema_26"] = ta.EMA(informative_1h, timeperiod=26)
    informative_1h["ema_50"] = ta.EMA(informative_1h, timeperiod=50)
    informative_1h["ema_100"] = ta.EMA(informative_1h, timeperiod=100)
    informative_1h["ema_200"] = ta.EMA(informative_1h, timeperiod=200)

    informative_1h["ema_200_dec_48"] = (informative_1h["ema_200"].isnull()) | (
      informative_1h["ema_200"] <= informative_1h["ema_200"].shift(48)
    )

    # SMA
    informative_1h["sma_12"] = ta.SMA(informative_1h, timeperiod=12)
    informative_1h["sma_26"] = ta.SMA(informative_1h, timeperiod=26)
    informative_1h["sma_50"] = ta.SMA(informative_1h, timeperiod=50)
    informative_1h["sma_100"] = ta.SMA(informative_1h, timeperiod=100)
    informative_1h["sma_200"] = ta.SMA(informative_1h, timeperiod=200)

    # ZL MA
    informative_1h["zlma_50"] = pta.zlma(informative_1h["close"], length=50, matype="linreg", offset=0)
    informative_1h["zlma_50"].fillna(0.0, inplace=True)

    informative_1h["zlma_50_dec"] = (informative_1h["zlma_50"].isnull()) | (
      informative_1h["zlma_50"] <= informative_1h["zlma_50"].shift(1)
    )

    # BB
    bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2)
    informative_1h["bb20_2_low"] = bollinger["lower"]
    informative_1h["bb20_2_mid"] = bollinger["mid"]
    informative_1h["bb20_2_upp"] = bollinger["upper"]

    informative_1h["bb20_2_width"] = (informative_1h["bb20_2_upp"] - informative_1h["bb20_2_low"]) / informative_1h[
      "bb20_2_mid"
    ]

    # Williams %R
    informative_1h["r_14"] = williams_r(informative_1h, period=14)
    informative_1h["r_96"] = williams_r(informative_1h, period=96)
    informative_1h["r_480"] = williams_r(informative_1h, period=480)

    # CTI
    informative_1h["cti_20"] = pta.cti(informative_1h["close"], length=20)
    informative_1h["cti_40"] = pta.cti(informative_1h["close"], length=40)

    informative_1h["cti_20_dec_3"] = informative_1h["cti_20"] < informative_1h["cti_20"].shift(3)

    # SAR
    informative_1h["sar"] = ta.SAR(informative_1h)

    # EWO
    informative_1h["ewo_50_200"] = ewo(informative_1h, 50, 200)

    # EverGet ChandelierExit
    high = informative_1h["high"]
    low = informative_1h["low"]
    close = informative_1h["close"]
    chandelier_atr = ta.ATR(high, low, close, 22) * 3.0
    long_stop = (high.rolling(22).max() if True else high.rolling(22).apply(lambda x: x[:-1].max())) - chandelier_atr
    long_stop_prev = long_stop.shift(1).fillna(long_stop)
    long_stop = close.shift(1).where(close.shift(1) > long_stop_prev, long_stop)
    short_stop = (low.rolling(22).min() if True else low.rolling(22).apply(lambda x: x[:-1].min())) + chandelier_atr
    short_stop_prev = short_stop.shift(1).fillna(short_stop)
    short_stop = close.shift(1).where(close.shift(1) < short_stop_prev, short_stop)
    informative_1h["chandelier_dir"] = 1
    informative_1h.loc[informative_1h["close"] <= long_stop_prev, "chandelier_dir"] = -1
    informative_1h.loc[informative_1h["close"] > short_stop_prev, "chandelier_dir"] = 1

    # S/R
    res_series = (
      informative_1h["high"].rolling(window=5, center=True).apply(lambda row: is_resistance(row), raw=True).shift(2)
    )
    sup_series = (
      informative_1h["low"].rolling(window=5, center=True).apply(lambda row: is_support(row), raw=True).shift(2)
    )
    informative_1h["res_level"] = Series(
      np.where(
        res_series,
        np.where(informative_1h["close"] > informative_1h["open"], informative_1h["close"], informative_1h["open"]),
        float("NaN"),
      )
    ).ffill()
    informative_1h["res_hlevel"] = Series(np.where(res_series, informative_1h["high"], float("NaN"))).ffill()
    informative_1h["sup_level"] = Series(
      np.where(
        sup_series,
        np.where(informative_1h["close"] < informative_1h["open"], informative_1h["close"], informative_1h["open"]),
        float("NaN"),
      )
    ).ffill()

    # Pump protections
    informative_1h["hl_pct_change_48"] = range_percent_change(self, informative_1h, "HL", 48)
    informative_1h["hl_pct_change_36"] = range_percent_change(self, informative_1h, "HL", 36)
    informative_1h["hl_pct_change_24"] = range_percent_change(self, informative_1h, "HL", 24)
    informative_1h["hl_pct_change_12"] = range_percent_change(self, informative_1h, "HL", 12)
    informative_1h["hl_pct_change_6"] = range_percent_change(self, informative_1h, "HL", 6)

    # Downtrend checks
    informative_1h["not_downtrend"] = (informative_1h["close"] > informative_1h["close"].shift(2)) | (
      informative_1h["rsi_14"] > 50.0
    )

    informative_1h["is_downtrend_3"] = (
      (informative_1h["close"] < informative_1h["open"])
      & (informative_1h["close"].shift(1) < informative_1h["open"].shift(1))
      & (informative_1h["close"].shift(2) < informative_1h["open"].shift(2))
    )

    informative_1h["is_downtrend_5"] = (
      (informative_1h["close"] < informative_1h["open"])
      & (informative_1h["close"].shift(1) < informative_1h["open"].shift(1))
      & (informative_1h["close"].shift(2) < informative_1h["open"].shift(2))
      & (informative_1h["close"].shift(3) < informative_1h["open"].shift(3))
      & (informative_1h["close"].shift(4) < informative_1h["open"].shift(4))
    )

    # Wicks
    informative_1h["top_wick_pct"] = (
      informative_1h["high"] - np.maximum(informative_1h["open"], informative_1h["close"])
    ) / np.maximum(informative_1h["open"], informative_1h["close"])

    # Candle change
    informative_1h["change_pct"] = (informative_1h["close"] - informative_1h["open"]) / informative_1h["open"]

    # Max highs
    informative_1h["high_max_3"] = informative_1h["high"].rolling(3).max()
    informative_1h["high_max_6"] = informative_1h["high"].rolling(6).max()
    informative_1h["high_max_12"] = informative_1h["high"].rolling(12).max()
    informative_1h["high_max_24"] = informative_1h["high"].rolling(24).max()
    informative_1h["high_max_36"] = informative_1h["high"].rolling(36).max()
    informative_1h["high_max_48"] = informative_1h["high"].rolling(48).max()

    # Max lows
    informative_1h["low_min_3"] = informative_1h["low"].rolling(3).min()
    informative_1h["low_min_12"] = informative_1h["low"].rolling(12).min()
    informative_1h["low_min_24"] = informative_1h["low"].rolling(24).min()

    # Volume
    informative_1h["volume_mean_factor_12"] = informative_1h["volume"] / informative_1h["volume"].rolling(12).mean()

    # Performance logging
    # -----------------------------------------------------------------------------------------
    tok = time.perf_counter()
    log.debug(f"[{metadata['pair']}] informative_1h_indicators took: {tok - tik:0.4f} seconds.")

    return informative_1h

  def informative_15m_indicators(self, metadata: dict, info_timeframe) -> DataFrame:
    tik = time.perf_counter()
    assert self.dp, "DataProvider is required for multiple timeframes."

    # Get the informative pair
    informative_15m = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=info_timeframe)

    # Indicators
    # -----------------------------------------------------------------------------------------

    # RSI
    informative_15m["rsi_3"] = ta.RSI(informative_15m, timeperiod=3)
    informative_15m["rsi_14"] = ta.RSI(informative_15m, timeperiod=14)

    # EMA
    informative_15m["ema_12"] = ta.EMA(informative_15m, timeperiod=12)
    informative_15m["ema_26"] = ta.EMA(informative_15m, timeperiod=26)
    informative_15m["ema_200"] = ta.EMA(informative_15m, timeperiod=200)

    informative_15m["ema_200_dec_24"] = (informative_15m["ema_200"].isnull()) | (
      informative_15m["ema_200"] <= informative_15m["ema_200"].shift(24)
    )

    # SMA
    informative_15m["sma_200"] = ta.SMA(informative_15m, timeperiod=200)

    # ZL MA
    informative_15m["zlma_50"] = pta.zlma(informative_15m["close"], length=50, matype="linreg", offset=0)

    informative_15m["zlma_50_dec"] = (informative_15m["zlma_50"].isnull()) | (
      informative_15m["zlma_50"] <= informative_15m["zlma_50"].shift(1)
    )

    # BB - 20 STD2
    bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_15m), window=20, stds=2)
    informative_15m["bb20_2_low"] = bollinger["lower"]
    informative_15m["bb20_2_mid"] = bollinger["mid"]
    informative_15m["bb20_2_upp"] = bollinger["upper"]

    # CTI
    informative_15m["cti_20"] = pta.cti(informative_15m["close"], length=20)

    # EWO
    informative_15m["ewo_50_200"] = ewo(informative_15m, 50, 200)

    # Downtrend check
    informative_15m["not_downtrend"] = (
      (informative_15m["close"] > informative_15m["open"])
      | (informative_15m["close"].shift(1) > informative_15m["open"].shift(1))
      | (informative_15m["close"].shift(2) > informative_15m["open"].shift(2))
      | (informative_15m["rsi_14"] > 50.0)
      | (informative_15m["rsi_3"] > 25.0)
    )

    # Volume
    informative_15m["volume_mean_factor_12"] = informative_15m["volume"] / informative_15m["volume"].rolling(12).mean()

    # Performance logging
    # -----------------------------------------------------------------------------------------
    tok = time.perf_counter()
    log.debug(f"[{metadata['pair']}] informative_15m_indicators took: {tok - tik:0.4f} seconds.")

    return informative_15m

  # Coin Pair Base Timeframe Indicators
  # ---------------------------------------------------------------------------------------------
  def base_tf_5m_indicators(self, metadata: dict, dataframe: DataFrame) -> DataFrame:
    tik = time.perf_counter()

    # Indicators
    # -----------------------------------------------------------------------------------------
    # RSI
    dataframe["rsi_3"] = ta.RSI(dataframe, timeperiod=3)
    dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
    dataframe["rsi_20"] = ta.RSI(dataframe, timeperiod=20)

    # EMA
    dataframe["ema_12"] = ta.EMA(dataframe, timeperiod=12)
    dataframe["ema_16"] = ta.EMA(dataframe, timeperiod=16)
    dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
    dataframe["ema_26"] = ta.EMA(dataframe, timeperiod=26)
    dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
    dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
    dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

    dataframe["ema_200_dec_24"] = (dataframe["ema_200"].isnull()) | (
      dataframe["ema_200"] <= dataframe["ema_200"].shift(24)
    )

    dataframe["ema_200_pct_change_144"] = (dataframe["ema_200"] - dataframe["ema_200"].shift(144)) / dataframe[
      "ema_200"
    ].shift(144)
    dataframe["ema_200_pct_change_288"] = (dataframe["ema_200"] - dataframe["ema_200"].shift(288)) / dataframe[
      "ema_200"
    ].shift(288)

    # SMA
    dataframe["sma_16"] = ta.SMA(dataframe, timeperiod=16)
    dataframe["sma_30"] = ta.SMA(dataframe, timeperiod=30)
    dataframe["sma_50"] = ta.SMA(dataframe, timeperiod=50)
    dataframe["sma_75"] = ta.SMA(dataframe, timeperiod=75)
    dataframe["sma_200"] = ta.SMA(dataframe, timeperiod=200)

    # BB 20 - STD2
    bb_20_std2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
    dataframe["bb20_2_low"] = bb_20_std2["lower"]
    dataframe["bb20_2_mid"] = bb_20_std2["mid"]
    dataframe["bb20_2_upp"] = bb_20_std2["upper"]

    # BB 40 - STD2
    bb_40_std2 = qtpylib.bollinger_bands(dataframe["close"], window=40, stds=2)
    dataframe["bb40_2_low"] = bb_40_std2["lower"]
    dataframe["bb40_2_mid"] = bb_40_std2["mid"]
    dataframe["bb40_2_delta"] = (bb_40_std2["mid"] - dataframe["bb40_2_low"]).abs()
    dataframe["bb40_2_tail"] = (dataframe["close"] - dataframe["bb40_2_low"]).abs()

    # Williams %R
    dataframe["r_14"] = williams_r(dataframe, period=14)
    dataframe["r_480"] = williams_r(dataframe, period=480)

    # CTI
    dataframe["cti_20"] = pta.cti(dataframe["close"], length=20)

    # SAR
    dataframe["sar"] = ta.SAR(dataframe)

    # CCI
    dataframe["cci_20"] = ta.CCI(dataframe, source="hlc3", timeperiod=20)

    # TSI
    tsi = pta.tsi(dataframe["close"])
    dataframe["tsi"] = tsi.iloc[:, 0]
    dataframe["tsi_signal"] = tsi.iloc[:, 1]

    # EWO
    dataframe["ewo_50_200"] = ewo(dataframe, 50, 200)

    # Hull Moving Average
    dataframe["hma_55"] = pta.hma(dataframe["close"], length=55)
    dataframe["hma_70"] = pta.hma(dataframe["close"], length=70)

    dataframe["hma_55_buy"] = (dataframe["hma_55"] > dataframe["hma_55"].shift(1)) & (
      dataframe["hma_55"].shift(1) < dataframe["hma_55"].shift(2)
    )
    dataframe["hma_70_buy"] = (dataframe["hma_70"] > dataframe["hma_70"].shift(1)) & (
      dataframe["hma_70"].shift(1) < dataframe["hma_70"].shift(2)
    )

    # ZL MA
    dataframe["zlma_50"] = pta.zlma(dataframe["close"], length=50, matype="linreg", offset=0)

    # EverGet ChandelierExit
    high = dataframe["high"]
    low = dataframe["low"]
    close = dataframe["close"]
    chandelier_atr = ta.ATR(high, low, close, 22) * 3.0
    long_stop = (high.rolling(22).max() if True else high.rolling(22).apply(lambda x: x[:-1].max())) - chandelier_atr
    long_stop_prev = long_stop.shift(1).fillna(long_stop)
    long_stop = close.shift(1).where(close.shift(1) > long_stop_prev, long_stop)
    short_stop = (low.rolling(22).min() if True else low.rolling(22).apply(lambda x: x[:-1].min())) + chandelier_atr
    short_stop_prev = short_stop.shift(1).fillna(short_stop)
    short_stop = close.shift(1).where(close.shift(1) < short_stop_prev, short_stop)
    dataframe["chandelier_dir"] = 1
    dataframe.loc[dataframe["close"] <= long_stop_prev, "chandelier_dir"] = -1
    dataframe.loc[dataframe["close"] > short_stop_prev, "chandelier_dir"] = 1

    # Heiken Ashi
    heikinashi = qtpylib.heikinashi(dataframe)
    dataframe["ha_open"] = heikinashi["open"]
    dataframe["ha_close"] = heikinashi["close"]
    dataframe["ha_high"] = heikinashi["high"]
    dataframe["ha_low"] = heikinashi["low"]

    # Dip protection
    dataframe["tpct_change_0"] = top_percent_change(self, dataframe, 0)
    dataframe["tpct_change_2"] = top_percent_change(self, dataframe, 2)

    # Candle change
    dataframe["change_pct"] = (dataframe["close"] - dataframe["open"]) / dataframe["open"]

    # Close max
    dataframe["close_max_12"] = dataframe["close"].rolling(12).max()
    dataframe["close_max_24"] = dataframe["close"].rolling(24).max()
    dataframe["close_max_48"] = dataframe["close"].rolling(48).max()

    # Open min
    dataframe["open_min_6"] = dataframe["open"].rolling(6).min()
    dataframe["open_min_12"] = dataframe["open"].rolling(12).min()

    # Close min
    dataframe["close_min_12"] = dataframe["close"].rolling(12).min()
    dataframe["close_min_24"] = dataframe["close"].rolling(24).min()

    # Close delta
    dataframe["close_delta"] = (dataframe["close"] - dataframe["close"].shift()).abs()

    # Number of empty candles in the last 288
    dataframe["num_empty_288"] = (dataframe["volume"] <= 0).rolling(window=288, min_periods=288).sum()

    # For sell checks
    dataframe["crossed_below_ema_12_26"] = qtpylib.crossed_below(dataframe["ema_12"], dataframe["ema_26"])

    # Global protections
    # -----------------------------------------------------------------------------------------
    if not self.config["runmode"].value in ("live", "dry_run"):
      # Backtest age filter
      dataframe["bt_agefilter_ok"] = False
      dataframe.loc[dataframe.index > (12 * 24 * self.bt_min_age_days), "bt_agefilter_ok"] = True
    else:
      # Exchange downtime protection
      dataframe["live_data_ok"] = dataframe["volume"].rolling(window=72, min_periods=72).min() > 0

    # Performance logging
    # -----------------------------------------------------------------------------------------
    tok = time.perf_counter()
    log.debug(f"[{metadata['pair']}] base_tf_5m_indicators took: {tok - tik:0.4f} seconds.")

    return dataframe

  # Coin Pair Indicator Switch Case
  # ---------------------------------------------------------------------------------------------
  def info_switcher(self, metadata: dict, info_timeframe) -> DataFrame:
    if info_timeframe == "1d":
      return self.informative_1d_indicators(metadata, info_timeframe)
    elif info_timeframe == "4h":
      return self.informative_4h_indicators(metadata, info_timeframe)
    elif info_timeframe == "1h":
      return self.informative_1h_indicators(metadata, info_timeframe)
    elif info_timeframe == "15m":
      return self.informative_15m_indicators(metadata, info_timeframe)
    else:
      raise RuntimeError(f"{info_timeframe} not supported as informative timeframe for BTC pair.")

  # BTC 1D Indicators
  # ---------------------------------------------------------------------------------------------
  def btc_info_1d_indicators(self, btc_info_pair, btc_info_timeframe, metadata: dict) -> DataFrame:
    tik = time.perf_counter()
    btc_info_1d = self.dp.get_pair_dataframe(btc_info_pair, btc_info_timeframe)
    # Indicators
    # -----------------------------------------------------------------------------------------
    btc_info_1d["rsi_14"] = ta.RSI(btc_info_1d, timeperiod=14)
    # btc_info_1d['pivot'], btc_info_1d['res1'], btc_info_1d['res2'], btc_info_1d['res3'], btc_info_1d['sup1'], btc_info_1d['sup2'], btc_info_1d['sup3'] = pivot_points(btc_info_1d, mode='fibonacci')

    # Add prefix
    # -----------------------------------------------------------------------------------------
    ignore_columns = ["date"]
    btc_info_1d.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

    tok = time.perf_counter()
    log.debug(f"[{metadata['pair']}] btc_info_1d_indicators took: {tok - tik:0.4f} seconds.")

    return btc_info_1d

  # BTC 4h Indicators
  # ---------------------------------------------------------------------------------------------
  def btc_info_4h_indicators(self, btc_info_pair, btc_info_timeframe, metadata: dict) -> DataFrame:
    tik = time.perf_counter()
    btc_info_4h = self.dp.get_pair_dataframe(btc_info_pair, btc_info_timeframe)
    # Indicators
    # -----------------------------------------------------------------------------------------
    # RSI
    btc_info_4h["rsi_14"] = ta.RSI(btc_info_4h, timeperiod=14)

    # SMA
    btc_info_4h["sma_200"] = ta.SMA(btc_info_4h, timeperiod=200)

    # Bull market or not
    btc_info_4h["is_bull"] = btc_info_4h["close"] > btc_info_4h["sma_200"]

    # Add prefix
    # -----------------------------------------------------------------------------------------
    ignore_columns = ["date"]
    btc_info_4h.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

    tok = time.perf_counter()
    log.debug(f"[{metadata['pair']}] btc_info_4h_indicators took: {tok - tik:0.4f} seconds.")

    return btc_info_4h

  # BTC 1h Indicators
  # ---------------------------------------------------------------------------------------------
  def btc_info_1h_indicators(self, btc_info_pair, btc_info_timeframe, metadata: dict) -> DataFrame:
    tik = time.perf_counter()
    btc_info_1h = self.dp.get_pair_dataframe(btc_info_pair, btc_info_timeframe)
    # Indicators
    # -----------------------------------------------------------------------------------------
    # RSI
    btc_info_1h["rsi_14"] = ta.RSI(btc_info_1h, timeperiod=14)

    btc_info_1h["not_downtrend"] = (btc_info_1h["close"] > btc_info_1h["close"].shift(2)) | (
      btc_info_1h["rsi_14"] > 50
    )

    # Add prefix
    # -----------------------------------------------------------------------------------------
    ignore_columns = ["date"]
    btc_info_1h.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

    tok = time.perf_counter()
    log.debug(f"[{metadata['pair']}] btc_info_1h_indicators took: {tok - tik:0.4f} seconds.")

    return btc_info_1h

  # BTC 15m Indicators
  # ---------------------------------------------------------------------------------------------
  def btc_info_15m_indicators(self, btc_info_pair, btc_info_timeframe, metadata: dict) -> DataFrame:
    tik = time.perf_counter()
    btc_info_15m = self.dp.get_pair_dataframe(btc_info_pair, btc_info_timeframe)
    # Indicators
    # -----------------------------------------------------------------------------------------
    btc_info_15m["rsi_14"] = ta.RSI(btc_info_15m, timeperiod=14)

    # Add prefix
    # -----------------------------------------------------------------------------------------
    ignore_columns = ["date"]
    btc_info_15m.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

    tok = time.perf_counter()
    log.debug(f"[{metadata['pair']}] btc_info_15m_indicators took: {tok - tik:0.4f} seconds.")

    return btc_info_15m

  # BTC 5m Indicators
  # ---------------------------------------------------------------------------------------------
  def btc_info_5m_indicators(self, btc_info_pair, btc_info_timeframe, metadata: dict) -> DataFrame:
    tik = time.perf_counter()
    btc_info_5m = self.dp.get_pair_dataframe(btc_info_pair, btc_info_timeframe)
    # Indicators
    # -----------------------------------------------------------------------------------------

    # RSI
    btc_info_5m["rsi_14"] = ta.RSI(btc_info_5m, timeperiod=14)

    # Close max
    btc_info_5m["close_max_24"] = btc_info_5m["close"].rolling(24).max()
    btc_info_5m["close_max_72"] = btc_info_5m["close"].rolling(72).max()

    btc_info_5m["pct_close_max_24"] = (btc_info_5m["close_max_24"] - btc_info_5m["close"]) / btc_info_5m["close"]
    btc_info_5m["pct_close_max_72"] = (btc_info_5m["close_max_72"] - btc_info_5m["close"]) / btc_info_5m["close"]

    # Add prefix
    # -----------------------------------------------------------------------------------------
    ignore_columns = ["date"]
    btc_info_5m.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

    tok = time.perf_counter()
    log.debug(f"[{metadata['pair']}] btc_info_5m_indicators took: {tok - tik:0.4f} seconds.")

    return btc_info_5m

  # BTC Indicator Switch Case
  # ---------------------------------------------------------------------------------------------
  def btc_info_switcher(self, btc_info_pair, btc_info_timeframe, metadata: dict) -> DataFrame:
    if btc_info_timeframe == "1d":
      return self.btc_info_1d_indicators(btc_info_pair, btc_info_timeframe, metadata)
    elif btc_info_timeframe == "4h":
      return self.btc_info_4h_indicators(btc_info_pair, btc_info_timeframe, metadata)
    elif btc_info_timeframe == "1h":
      return self.btc_info_1h_indicators(btc_info_pair, btc_info_timeframe, metadata)
    elif btc_info_timeframe == "15m":
      return self.btc_info_15m_indicators(btc_info_pair, btc_info_timeframe, metadata)
    elif btc_info_timeframe == "5m":
      return self.btc_info_5m_indicators(btc_info_pair, btc_info_timeframe, metadata)
    else:
      raise RuntimeError(f"{btc_info_timeframe} not supported as informative timeframe for BTC pair.")

  def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    tik = time.perf_counter()
    """
        --> BTC informative indicators
        ___________________________________________________________________________________________
        """
    if self.config["stake_currency"] in ["USDT", "BUSD", "USDC", "DAI", "TUSD", "PAX", "USD", "EUR", "GBP"]:
      if ("trading_mode" in self.config) and (self.config["trading_mode"] in ["futures", "margin"]):
        btc_info_pair = f"BTC/{self.config['stake_currency']}:{self.config['stake_currency']}"
      else:
        btc_info_pair = f"BTC/{self.config['stake_currency']}"
    else:
      if ("trading_mode" in self.config) and (self.config["trading_mode"] in ["futures", "margin"]):
        btc_info_pair = "BTC/USDT:USDT"
      else:
        btc_info_pair = "BTC/USDT"

    for btc_info_timeframe in self.btc_info_timeframes:
      btc_informative = self.btc_info_switcher(btc_info_pair, btc_info_timeframe, metadata)
      dataframe = merge_informative_pair(dataframe, btc_informative, self.timeframe, btc_info_timeframe, ffill=True)
      # Customize what we drop - in case we need to maintain some BTC informative ohlcv data
      # Default drop all
      drop_columns = {
        "1d": [f"btc_{s}_{btc_info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]],
        "4h": [f"btc_{s}_{btc_info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]],
        "1h": [f"btc_{s}_{btc_info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]],
        "15m": [f"btc_{s}_{btc_info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]],
        "5m": [f"btc_{s}_{btc_info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]],
      }.get(
        btc_info_timeframe,
        [f"{s}_{btc_info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]],
      )
      drop_columns.append(f"date_{btc_info_timeframe}")
      dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

    """
        --> Indicators on informative timeframes
        ___________________________________________________________________________________________
        """
    for info_timeframe in self.info_timeframes:
      info_indicators = self.info_switcher(metadata, info_timeframe)
      dataframe = merge_informative_pair(dataframe, info_indicators, self.timeframe, info_timeframe, ffill=True)
      # Customize what we drop - in case we need to maintain some informative timeframe ohlcv data
      # Default drop all except base timeframe ohlcv data
      drop_columns = {
        "1d": [f"{s}_{info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]],
        "4h": [f"{s}_{info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]],
        "1h": [f"{s}_{info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]],
        "15m": [f"{s}_{info_timeframe}" for s in ["date", "high", "low", "volume"]],
      }.get(info_timeframe, [f"{s}_{info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]])
      dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

    """
        --> The indicators for the base timeframe  (5m)
        ___________________________________________________________________________________________
        """
    dataframe = self.base_tf_5m_indicators(metadata, dataframe)

    # Global protections
    tok = time.perf_counter()
    log.debug(f"[{metadata['pair']}] Populate indicators took a total of: {tok - tik:0.4f} seconds.")

    return dataframe

  def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    conditions = []
    dataframe.loc[:, "enter_tag"] = ""

    # the number of free slots
    current_free_slots = self.config["max_open_trades"] - len(LocalTrade.get_trades_proxy(is_open=True))
    # if BTC/ETH stake
    is_btc_stake = self.config["stake_currency"] in self.btc_stakes
    allowed_empty_candles = 144 if is_btc_stake else 60

    for buy_enable in self.entry_long_params:
      index = int(buy_enable.split("_")[2])
      item_buy_protection_list = [True]
      if self.entry_long_params[f"{buy_enable}"]:
        # Buy conditions
        # -----------------------------------------------------------------------------------------
        item_sell_logic = []
        item_sell_logic.append(reduce(lambda x, y: x & y, item_buy_protection_list))

        # Condition #1 - Long mode bull. Uptrend.
        if index == 1:
          # Logic
          item_sell_logic.append(dataframe["ema_26"] < dataframe["ema_12"])
          item_sell_logic.append((dataframe["ema_26"] - dataframe["ema_12"]) < -(dataframe["open"] * 0.02))
          item_sell_logic.append(
              (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) < -(dataframe["open"] / 100)
          )
          item_sell_logic.append(dataframe["close"] > (dataframe["bb20_2_low"] * 1.001))


        # Condition #2 - Normal mode bull.
        if index == 2:
          # Logic
          item_sell_logic.append(dataframe["bb40_2_delta"].lt(dataframe["close"] * 0.06))
          item_sell_logic.append(dataframe["close_delta"].lt(dataframe["close"] * 0.02))
          item_sell_logic.append(dataframe["bb40_2_tail"].gt(dataframe["bb40_2_delta"] * 0.2))
          item_sell_logic.append(dataframe["close"].gt(dataframe["bb20_2_low"].shift()))
          item_sell_logic.append(dataframe["close"].ge(dataframe["close"].shift()))

          # Condition #3 - Short mode bear.
          if index == 3:
              # Logic
              item_sell_logic.append(dataframe["rsi_14"] > 64.0)  # Negate condition
              item_sell_logic.append(dataframe["ha_close"] < dataframe["ha_open"])  # Reverse condition
              item_sell_logic.append((dataframe["ema_12"] - dataframe["ema_26"]) > (dataframe["open"] * 0.020))  # Reverse condition

          # Condition #4 - Short mode bear.
          if index == 4:
              # Logic
              
              item_sell_logic.append(dataframe["ema_12"] > dataframe["ema_26"])  # Reverse condition
              item_sell_logic.append((dataframe["ema_12"] - dataframe["ema_26"]) > (dataframe["open"] * 0.018))  # Reverse condition
              item_sell_logic.append(
                  (dataframe["ema_12"].shift() - dataframe["ema_26"].shift()) > (dataframe["open"] / 100)  # Reverse condition
              )
              item_sell_logic.append(dataframe["close"] > (dataframe["bb20_2_low"] * 1.004))  # Reverse condition

          # Condition #5 - Short mode bear.
          if index == 5:
              # Logic
              
              item_sell_logic.append(dataframe["ema_12"] > dataframe["ema_26"])  # Reverse condition
              item_sell_logic.append((dataframe["ema_12"] - dataframe["ema_26"]) > (dataframe["open"] * 0.03))  # Reverse condition
              item_sell_logic.append(
                  (dataframe["ema_12"].shift() - dataframe["ema_26"].shift()) > (dataframe["open"] / 100)  # Reverse condition
              )
              item_sell_logic.append(dataframe["rsi_14"] > 64.0)  # Negate condition

          # Condition #6 - Short mode bear.
          if index == 6:
              # Logic
              
              item_sell_logic.append(dataframe["close"] > (dataframe["ema_26"] * 1.06))  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["bb20_2_low"] * 1.004))  # Reverse condition

          # Condition #7 - Short mode bear.
          if index == 7:
              # Logic
              
              item_sell_logic.append(dataframe["close"] > (dataframe["ema_16"] * 1.026))  # Reverse condition
              item_sell_logic.append(dataframe["ewo_50_200"] < -2.0)  # Reverse condition
              item_sell_logic.append(dataframe["rsi_14"] > 70.0)  # Reverse condition

          # Condition #8 - Short mode bear.
          if index == 8:
              # Logic
              
              item_sell_logic.append(dataframe["close"] > (dataframe["ema_16"] * 1.056))  # Reverse condition
              item_sell_logic.append(dataframe["ewo_50_200"] > 4.0)  # Reverse condition
              item_sell_logic.append(dataframe["rsi_14"] > 70.0)  # Reverse condition

          # Condition #9 - Short mode bear.
          if index == 9:
              # Logic
              
              item_sell_logic.append(dataframe["ema_26_15m"] < dataframe["ema_12_15m"])  # Reverse condition
              item_sell_logic.append((dataframe["ema_26_15m"] - dataframe["ema_12_15m"]) > (dataframe["open_15m"] * 0.020))  # Reverse condition
              item_sell_logic.append(
                  (dataframe["ema_26_15m"].shift(3) - dataframe["ema_12_15m"].shift(3)) > (dataframe["open_15m"] / 100.0)  # Reverse condition
              )
              item_sell_logic.append(dataframe["close_15m"] > (dataframe["bb20_2_low_15m"] * 1.01))  # Reverse condition

          # Condition #10 - Short mode bear.
          if index == 10:
              # Logic
              
              item_sell_logic.append(dataframe["ema_26"] < dataframe["ema_12"])  # Reverse condition
              item_sell_logic.append((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * 0.016))  # Reverse condition
              item_sell_logic.append(
                  (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100)  # Reverse condition
              )
              item_sell_logic.append(dataframe["rsi_14"] > 64.0)  # Reverse condition

          # Condition #11 - Short mode bear.
          if index == 11:
              # Logic
              
              item_sell_logic.append(dataframe["rsi_14"] > self.entry_11_rsi_14_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["cti_20"] > self.entry_11_cti_20_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["ema_26"] < dataframe["ema_12"])  # Reverse condition
              item_sell_logic.append(
                  (dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * self.entry_11_ema_open_offset.value)
              )  # Reverse condition
              item_sell_logic.append(
                  (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100.0)  # Reverse condition
              )
              item_sell_logic.append(dataframe["close"] > (dataframe["sma_30"] * self.entry_11_sma_offset.value))  # Reverse condition

          # Condition #12 - Short mode bear.
          if index == 12:
              # Logic
              
              item_sell_logic.append(dataframe["r_14"] > self.entry_12_r_14_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["bb20_2_low"] * self.entry_12_bb_offset.value))  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["sma_30"] * self.entry_12_sma_offset.value))  # Reverse condition

          # Condition #21 - Short mode bear.
          if index == 21:
              # Logic
              
              item_sell_logic.append(dataframe["ema_26"] < dataframe["ema_12"])  # Reverse condition
              item_sell_logic.append((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * 0.02))  # Reverse condition
              item_sell_logic.append(
                  (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100)  # Reverse condition
              )
              item_sell_logic.append(dataframe["close"] > (dataframe["bb20_2_low"] * 1.001))  # Reverse condition

          # Condition #22 - Short mode bear.
          if index == 22:
              # Logic
              
              item_sell_logic.append(dataframe["close"] > (dataframe["ema_16"] * 1.032))  # Reverse condition
              item_sell_logic.append(dataframe["cti_20"] > 0.9)  # Reverse condition
              item_sell_logic.append(dataframe["rsi_14"] > 50.0)  # Reverse condition

          # Condition #23 - Short mode bear.
          if index == 23:
              # Logic
              
              item_sell_logic.append(dataframe["ewo_50_200_15m"] < -4.2)  # Reverse condition
              item_sell_logic.append(dataframe["rsi_14_15m"].shift(1) > 70.0)  # Reverse condition
              item_sell_logic.append(dataframe["rsi_14_15m"] > 70.0)  # Reverse condition
              item_sell_logic.append(dataframe["rsi_14"] > 65.0)  # Reverse condition
              item_sell_logic.append(dataframe["cti_20"] > 0.8)  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["ema_26_15m"] * 1.043))  # Reverse condition

          # Condition #24 - Short mode bear.
          if index == 24:
              # Logic
              
              item_sell_logic.append(dataframe["rsi_14"] < self.entry_24_rsi_14_min.value)  # Reverse condition
              item_sell_logic.append(dataframe["rsi_14"] > self.entry_24_rsi_14_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["cti_20"] > self.entry_24_cti_20_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["r_14"] > self.entry_24_r_14_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["ewo_50_200"] < self.entry_24_ewo_50_200_min.value)  # Reverse condition
              item_sell_logic.append(dataframe["ewo_50_200"] > self.entry_24_ewo_50_200_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["sma_75"] * self.entry_24_sma_offset.value))  # Reverse condition

          # Condition #25 - Short mode bear.
          if index == 25:
              # Logic
              
              item_sell_logic.append(dataframe["rsi_14"] > self.entry_25_rsi_14_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["cti_20"] > self.entry_25_cti_20_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["ewo_50_200"] < self.entry_25_ewo_50_200_min.value)  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["sma_30"] * self.entry_25_sma_offset.value))  # Reverse condition

          # Condition #26 - Short mode bear.
          if index == 26:
              # Logic
              
              item_sell_logic.append(dataframe["close"] > (dataframe["bb20_2_low"] * self.entry_26_bb_offset.value))  # Reverse condition
              item_sell_logic.append(dataframe["ewo_50_200_1h"] < self.entry_26_ewo_50_200_1h_min.value)  # Reverse condition
              item_sell_logic.append(dataframe["ema_26"] < dataframe["ema_12"])  # Reverse condition
              item_sell_logic.append(
                  (dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * self.entry_26_ema_open_offset.value)
              )  # Reverse condition
              item_sell_logic.append(
                  (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100.0)  # Reverse condition
              )

          # Condition #41 - Short mode bear.
          if index == 41:
              # Logic
              
              item_sell_logic.append(dataframe["bb40_2_delta"].le(dataframe["close"] * 0.964))  # Reverse condition
              item_sell_logic.append(dataframe["close_delta"].le(dataframe["close"] * 0.98))  # Reverse condition
              item_sell_logic.append(dataframe["bb40_2_tail"].gt(dataframe["bb40_2_delta"] * 0.4))  # Reverse condition
              item_sell_logic.append(dataframe["close"].gt(dataframe["bb20_2_low"].shift()))  # Reverse condition
              item_sell_logic.append(dataframe["close"].ge(dataframe["close"].shift()))  # Reverse condition
              item_sell_logic.append(dataframe["rsi_14"] > 64.0)  # Reverse condition

          # Condition #42 - Short mode bear.
          if index == 42:
              # Logic
              
              item_sell_logic.append(dataframe["ema_26"] < dataframe["ema_12"])  # Reverse condition
              item_sell_logic.append((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * 0.018))  # Reverse condition
              item_sell_logic.append(
                  (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100)  # Reverse condition
              )
              item_sell_logic.append(dataframe["close"] > (dataframe["bb20_2_low"] * 1.004))  # Reverse condition
              item_sell_logic.append(dataframe["rsi_14"] > 60.0)  # Reverse condition

          # Condition #43 - Short mode bear.
          if index == 43:
              # Logic
              
              item_sell_logic.append(dataframe["close"] > (dataframe["ema_26"] * 1.066))  # Reverse condition
              item_sell_logic.append(dataframe["cti_20"] > 0.75)  # Reverse condition
              item_sell_logic.append(dataframe["r_14"] > 94.0)  # Reverse condition

          # Condition #44 - Short mode bear.
          if index == 44:
              # Logic
              
              item_sell_logic.append(dataframe["bb20_2_width_1h"] < 0.868)  # Reverse condition
              item_sell_logic.append(dataframe["cti_20"] > 0.8)  # Reverse condition
              item_sell_logic.append(dataframe["r_14"] > 90.0)  # Reverse condition

          # Condition #45 - Short mode bear.
          if index == 45:
              # Logic
              
              item_sell_logic.append(dataframe["rsi_14"] < self.entry_45_rsi_14_min.value)  # Reverse condition
              item_sell_logic.append(dataframe["rsi_14"] > self.entry_45_rsi_14_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["rsi_20"] > dataframe["rsi_20"].shift(1))  # Reverse condition
              item_sell_logic.append(dataframe["cti_20"] > self.entry_45_cti_20_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["sma_16"] * self.entry_45_sma_offset.value))  # Reverse condition

          # Condition #46 - Short mode bear.
          if index == 46:
              # Logic
              
              item_sell_logic.append(dataframe["rsi_14"] > self.entry_46_rsi_14_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["chandelier_dir_1h"].shift(1) > 0)  # Reverse condition
              item_sell_logic.append(dataframe["chandelier_dir_1h"] < 0)  # Reverse condition
              item_sell_logic.append(dataframe["close"] < dataframe["zlma_50_1h"])  # Reverse condition
              item_sell_logic.append(dataframe["ema_12"] > dataframe["ema_26"])  # Reverse condition

          # Condition #47 - Short mode bear.
          if index == 47:
              # Logic
              
              item_sell_logic.append(dataframe["rsi_14"] < self.entry_47_rsi_14_min.value)  # Reverse condition
              item_sell_logic.append(dataframe["rsi_14"] > self.entry_47_rsi_14_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["rsi_20"] < self.entry_47_rsi_20_min.value)  # Reverse condition
              item_sell_logic.append(dataframe["rsi_20"] > self.entry_47_rsi_20_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["cti_20"] > self.entry_47_cti_20_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["chandelier_dir"].shift(1) > 0)  # Reverse condition
              item_sell_logic.append(dataframe["chandelier_dir"] < 0)  # Reverse condition
              item_sell_logic.append(dataframe["ema_12"] > (dataframe["ema_26"] * self.entry_47_ema_offset.value))  # Reverse condition
              item_sell_logic.append(
                  dataframe["close"] > (dataframe["high_max_12_1h"] * (1 - self.entry_47_high_max_12_1h_max.value))  # Reverse condition
              )


          # Condition #48 - Quick mode (Short).
          if index == 48:
              # Logic
              
              item_sell_logic.append(dataframe["rsi_14"] > self.entry_48_rsi_14_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["cci_20"] > self.entry_48_cci_20_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["sma_30"] * self.entry_48_sma_offset.value))  # Reverse condition
              item_sell_logic.append(
                  ((dataframe["close"] - dataframe["open_min_6"]) / dataframe["open_min_6"]) > self.entry_48_inc_min.value  # Reverse condition
              )

          # Condition #49 - Quick mode (Short).
          if index == 49:
              # Logic
              
              item_sell_logic.append(dataframe["rsi_14"] > self.entry_49_rsi_14_max.value)  # Reverse condition
              item_sell_logic.append(
                  ((dataframe["close"] - dataframe["open_min_12"]) / dataframe["open_min_12"]) > self.entry_49_inc_min.value  # Reverse condition
              )

          # Condition #50 - Quick mode (Short)
          if index == 50:
              # Logic
              
              item_sell_logic.append(dataframe["close"] > (dataframe["bb20_2_low"] * (1 + self.entry_50_bb_offset.value)))  # Reverse condition
              item_sell_logic.append(dataframe["ema_26"] < dataframe["ema_12"])  # Reverse condition
              item_sell_logic.append(
                  (dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * self.entry_50_ema_open_offset.value)  # Reverse condition
              )
              item_sell_logic.append(
                  (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100.0)  # Reverse condition
              )

          # Condition #61 - Rebuy mode (Short).
          if index == 61:
              # Logic
              
              item_sell_logic.append(dataframe["rsi_14"] > 60.0)  # Reverse condition
              item_sell_logic.append(dataframe["bb40_2_delta"].lt(dataframe["close"] * 0.97))  # Reverse condition
              item_sell_logic.append(dataframe["close_delta"].lt(dataframe["close"] * 0.982))  # Reverse condition
              item_sell_logic.append(dataframe["bb40_2_tail"].gt(dataframe["bb40_2_delta"] * 0.4))  # Reverse condition
              item_sell_logic.append(dataframe["close"].gt(dataframe["bb20_2_low"].shift()))  # Reverse condition
              item_sell_logic.append(dataframe["close"].ge(dataframe["close"].shift()))  # Reverse condition

          # Condition #81 - Short mode bear.
          if index == 81:
              # Logic
              
              item_sell_logic.append(dataframe["bb40_2_delta"].lt(dataframe["close"] * 0.948))  # Reverse condition
              item_sell_logic.append(dataframe["close_delta"].lt(dataframe["close"] * 0.976))  # Reverse condition
              item_sell_logic.append(dataframe["bb40_2_tail"].gt(dataframe["bb40_2_delta"] * 0.8))  # Reverse condition
              item_sell_logic.append(dataframe["close"].gt(dataframe["bb20_2_low"].shift()))  # Reverse condition
              item_sell_logic.append(dataframe["close"].ge(dataframe["close"].shift()))  # Reverse condition
              item_sell_logic.append(dataframe["rsi_14"] > 70.0)  # Reverse condition

          # Condition #82 - Short mode bear.
          if index == 82:
              # Logic
              
              item_sell_logic.append(dataframe["ema_26"] < dataframe["ema_12"])  # Reverse condition
              item_sell_logic.append((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * 0.015))  # Reverse condition
              item_sell_logic.append(
                  (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100)  # Reverse condition
              )
              item_sell_logic.append(dataframe["cti_20"] > 0.8)  # Reverse condition

          # Condition #101 - Short mode rapid
          if index == 101:
              # Logic
              
              item_sell_logic.append(dataframe["rsi_14"] > 64.0)  # Reverse condition
              item_sell_logic.append(dataframe["rsi_14"] > dataframe["rsi_14"].shift(1))  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["sma_16"] * 1.046))  # Reverse condition
              item_sell_logic.append(dataframe["cti_20_15m"] > -0.5)  # Reverse condition

          # Condition #102 - Short mode rapid
          if index == 102:
              # Logic              
              item_sell_logic.append(dataframe["rsi_14"] < self.entry_46_rsi_14_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["ema_16"] * (1 + self.entry_102_ema_offset.value)))  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["bb20_2_low"] * (1 + self.entry_102_bb_offset.value)))  # Reverse condition

          # Condition #103 - Short mode rapid
          if index == 103:
              # Logic
              
              item_sell_logic.append(dataframe["rsi_14"] < self.entry_103_rsi_14_min.value)  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["sma_16"] * (1 + self.entry_103_sma_offset.value)))  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["bb20_2_mid"] * (1 + self.entry_103_bb_offset.value)))  # Reverse condition

          # Condition #104 - Short mode rapid
          if index == 104:
              # Logic
              
              item_sell_logic.append(dataframe["rsi_14"] < self.entry_104_rsi_14_min.value)  # Reverse condition
              item_sell_logic.append(dataframe["rsi_14"] > self.entry_104_rsi_14_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["sma_16"] * (1 + self.entry_104_sma_offset.value)))  # Reverse condition

          # Condition #105 - Short mode rapid
          if index == 105:
              # Logic
              
              item_sell_logic.append(dataframe["rsi_3"] > 60.0)  # Reverse condition
              item_sell_logic.append(dataframe["rsi_14"] > 46.0)  # Reverse condition
              item_sell_logic.append(dataframe["ema_26"] < dataframe["ema_12"])  # Reverse condition
              item_sell_logic.append((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * 0.015))  # Reverse condition
              item_sell_logic.append(
                  (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100)  # Reverse condition
              )

          # Condition #106 - Rapid mode (Short).
          if index == 106:
              # Logic
              
              item_sell_logic.append(dataframe["cti_20"] > self.entry_106_cti_20_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["ewo_50_200"] > self.entry_106_ewo_50_200_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["sma_30"] * (1 + self.entry_106_sma_offset.value)))  # Reverse condition

          # Condition #107 - Rapid mode (Short)
          if index == 107:
              # Logic
              
              item_sell_logic.append(dataframe["bb40_2_low"].shift().lt(0.0))  # Reverse condition
              item_sell_logic.append(
                  dataframe["bb40_2_delta"].lt(dataframe["close"] * (1 - self.entry_107_bb40_bbdelta_close.value))  # Reverse condition
              )
              item_sell_logic.append(
                  dataframe["close_delta"].lt(dataframe["close"] * (1 - self.entry_107_bb40_closedelta_close.value))  # Reverse condition
              )
              item_sell_logic.append(
                  dataframe["bb40_2_tail"].gt(dataframe["bb40_2_delta"] * (1 - self.entry_107_bb40_tail_bbdelta.value))  # Reverse condition
              )
              item_sell_logic.append(dataframe["close"].gt(dataframe["bb20_2_low"].shift()))  # Reverse condition
              item_sell_logic.append(dataframe["close"].ge(dataframe["close"].shift()))  # Reverse condition
              item_sell_logic.append(dataframe["cti_20"] > self.entry_107_cti_20_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["r_480"] < self.entry_107_r_480_min.value)  # Reverse condition

          # Condition #108 - Rapid mode (Short)
          if index == 108:
              # Logic
              
              item_sell_logic.append(dataframe["rsi_14"] < self.entry_108_rsi_14_min.value)  # Reverse condition
              item_sell_logic.append(dataframe["cti_20"] > self.entry_108_cti_20_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["r_14"] > self.entry_108_r_14_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["r_14"].shift(1) > self.entry_108_r_14_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["bb20_2_low"] * (1 + self.entry_108_bb_offset.value)))  # Reverse condition
              item_sell_logic.append(dataframe["ema_26"] < dataframe["ema_12"])  # Reverse condition
              item_sell_logic.append(
                  (dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * (1 + self.entry_108_ema_open_offset.value))  # Reverse condition
              )
              item_sell_logic.append(
                  (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100.0)  # Reverse condition
              )

          # Condition #109 - Rapid mode (Short)
          if index == 109:
              # Logic
              
              item_sell_logic.append(dataframe["cti_20"] > self.entry_109_cti_20_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["r_14"] > self.entry_109_r_14_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["bb20_2_low"] * (1 + self.entry_109_bb_offset.value)))  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["ema_20"] * (1 + self.entry_109_ema_offset.value)))  # Reverse condition

          # Condition #110 - Rapid mode (Short).
          if index == 110:
              # Logic
              
              item_sell_logic.append(dataframe["cti_20"] > self.entry_110_cti_20_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["ewo_50_200"] > self.entry_110_ewo_50_200_max.value)  # Reverse condition
              item_sell_logic.append(dataframe["close"] > (dataframe["ema_20"] * (1 + self.entry_110_ema_offset.value)))  # Reverse condition

        item_sell_logic.append(dataframe["volume"] > 0)
        item_buy = reduce(lambda x, y: x & y, item_sell_logic)
        dataframe.loc[item_buy, "enter_tag"] += f"{index} "
        conditions.append(item_buy)
        dataframe.loc[:, "enter_short"] = item_buy

    if conditions:
      dataframe.loc[:, "enter_short"] = reduce(lambda x, y: x | y, conditions)
      dataframe.loc[
      (
              (dataframe["rsi_3"] > 60.0)
      ),
      "enter_long"] = 0

    return dataframe

  def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[:, "exit_long"] = 0
    dataframe.loc[:, "exit_short"] = 0

    return dataframe

  def confirm_trade_entry(
    self,
    pair: str,
    order_type: str,
    amount: float,
    rate: float,
    time_in_force: str,
    current_time: datetime,
    entry_tag: Optional[str],
    **kwargs,
  ) -> bool:
    # allow force entries
    if entry_tag == "force_entry":
      return True

    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

    if len(dataframe) < 1:
      return False

    dataframe = dataframe.iloc[-1].squeeze()

    if rate > dataframe["close"]:
      slippage = (rate / dataframe["close"]) - 1.0

      if slippage < self.max_slippage:
        return True
      else:
        log.warning(f"Cancelling buy for {pair} due to slippage {(slippage * 100.0):.2f}%")
        return False

    return True

  def confirm_trade_exit(
    self,
    pair: str,
    trade: Trade,
    order_type: str,
    amount: float,
    rate: float,
    time_in_force: str,
    exit_reason: str,
    current_time: datetime,
    **kwargs,
  ) -> bool:
    # Allow force exits
    if exit_reason != "force_exit":
      if self._should_hold_trade(trade, rate, exit_reason):
        return False
      if exit_reason == "stop_loss":
        return False
      if self.exit_profit_only:
        if self.exit_profit_only:
          profit = 0.0
          if trade.realized_profit != 0.0:
            profit = ((rate - trade.open_rate) / trade.open_rate) * trade.stake_amount * (1 - trade.fee_close)
            profit = profit + trade.realized_profit
            profit = profit / trade.stake_amount
          else:
            profit = trade.calc_profit_ratio(rate)
          if profit < self.exit_profit_offset:
            return False

    self._remove_profit_target(pair)
    return True

  def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
    if self.config["runmode"].value not in ("live", "dry_run"):
      return super().bot_loop_start(datetime, **kwargs)

    if self.hold_support_enabled:
      self.load_hold_trades_config()

    return super().bot_loop_start(current_time, **kwargs)

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
    enter_tags = entry_tag.split()
    if all(c in self.long_rebuy_mode_tags for c in enter_tags):
      return self.futures_mode_leverage_rebuy_mode
    return self.futures_mode_leverage

  def _set_profit_target(
    self, pair: str, sell_reason: str, rate: float, current_profit: float, current_time: datetime
  ):
    self.target_profit_cache.data[pair] = {
      "rate": rate,
      "profit": current_profit,
      "sell_reason": sell_reason,
      "time_profit_reached": current_time.isoformat(),
    }
    self.target_profit_cache.save()

  def _remove_profit_target(self, pair: str):
    if self.target_profit_cache is not None:
      self.target_profit_cache.data.pop(pair, None)
      self.target_profit_cache.save()

  def get_hold_trades_config_file(self):
    proper_holds_file_path = self.config["user_data_dir"].resolve() / "nfi-hold-trades.json"
    if proper_holds_file_path.is_file():
      return proper_holds_file_path

    strat_file_path = pathlib.Path(__file__)
    hold_trades_config_file_resolve = strat_file_path.resolve().parent / "hold-trades.json"
    if hold_trades_config_file_resolve.is_file():
      log.warning(
        "Please move %s to %s which is now the expected path for the holds file",
        hold_trades_config_file_resolve,
        proper_holds_file_path,
      )
      return hold_trades_config_file_resolve

    # The resolved path does not exist, is it a symlink?
    hold_trades_config_file_absolute = strat_file_path.absolute().parent / "hold-trades.json"
    if hold_trades_config_file_absolute.is_file():
      log.warning(
        "Please move %s to %s which is now the expected path for the holds file",
        hold_trades_config_file_absolute,
        proper_holds_file_path,
      )
      return hold_trades_config_file_absolute

  def load_hold_trades_config(self):
    if self.hold_trades_cache is None:
      hold_trades_config_file = self.get_hold_trades_config_file()
      if hold_trades_config_file:
        log.warning("Loading hold support data from %s", hold_trades_config_file)
        self.hold_trades_cache = HoldsCache(hold_trades_config_file)

    if self.hold_trades_cache:
      self.hold_trades_cache.load()

  def _should_hold_trade(self, trade: "Trade", rate: float, sell_reason: str) -> bool:
    if self.config["runmode"].value not in ("live", "dry_run"):
      return False

    if not self.hold_support_enabled:
      return False

    # Just to be sure our hold data is loaded, should be a no-op call after the first bot loop
    self.load_hold_trades_config()

    if not self.hold_trades_cache:
      # Cache hasn't been setup, likely because the corresponding file does not exist, sell
      return False

    if not self.hold_trades_cache.data:
      # We have no pairs we want to hold until profit, sell
      return False

    # By default, no hold should be done
    hold_trade = False

    trade_ids: dict = self.hold_trades_cache.data.get("trade_ids")
    if trade_ids and trade.id in trade_ids:
      trade_profit_ratio = trade_ids[trade.id]
      profit = 0.0
      if trade.realized_profit != 0.0:
        profit = ((rate - trade.open_rate) / trade.open_rate) * trade.stake_amount * (1 - trade.fee_close)
        profit = profit + trade.realized_profit
        profit = profit / trade.stake_amount
      else:
        profit = trade.calc_profit_ratio(rate)
      current_profit_ratio = profit
      if sell_reason == "force_sell":
        formatted_profit_ratio = f"{trade_profit_ratio * 100}%"
        formatted_current_profit_ratio = f"{current_profit_ratio * 100}%"
        log.warning(
          "Force selling %s even though the current profit of %s < %s",
          trade,
          formatted_current_profit_ratio,
          formatted_profit_ratio,
        )
        return False
      elif current_profit_ratio >= trade_profit_ratio:
        # This pair is on the list to hold, and we reached minimum profit, sell
        formatted_profit_ratio = f"{trade_profit_ratio * 100}%"
        formatted_current_profit_ratio = f"{current_profit_ratio * 100}%"
        log.warning(
          "Selling %s because the current profit of %s >= %s",
          trade,
          formatted_current_profit_ratio,
          formatted_profit_ratio,
        )
        return False

      # This pair is on the list to hold, and we haven't reached minimum profit, hold
      hold_trade = True

    trade_pairs: dict = self.hold_trades_cache.data.get("trade_pairs")
    if trade_pairs and trade.pair in trade_pairs:
      trade_profit_ratio = trade_pairs[trade.pair]
      profit = 0.0
      if trade.realized_profit != 0.0:
        profit = ((rate - trade.open_rate) / trade.open_rate) * trade.stake_amount * (1 - trade.fee_close)
        profit = profit + trade.realized_profit
        profit = profit / trade.stake_amount
      else:
        profit = trade.calc_profit_ratio(rate)
      current_profit_ratio = profit
      if sell_reason == "force_sell":
        formatted_profit_ratio = f"{trade_profit_ratio * 100}%"
        formatted_current_profit_ratio = f"{current_profit_ratio * 100}%"
        log.warning(
          "Force selling %s even though the current profit of %s < %s",
          trade,
          formatted_current_profit_ratio,
          formatted_profit_ratio,
        )
        return False
      elif current_profit_ratio >= trade_profit_ratio:
        # This pair is on the list to hold, and we reached minimum profit, sell
        formatted_profit_ratio = f"{trade_profit_ratio * 100}%"
        formatted_current_profit_ratio = f"{current_profit_ratio * 100}%"
        log.warning(
          "Selling %s because the current profit of %s >= %s",
          trade,
          formatted_current_profit_ratio,
          formatted_profit_ratio,
        )
        return False

      # This pair is on the list to hold, and we haven't reached minimum profit, hold
      hold_trade = True

    return hold_trade

  @property
  def protections(self):
      return [
          {
              "method": "StoplossGuard",
              "lookback_period_candles": 12,
              "trade_limit": 3,
              "stop_duration_candles": 4,
              "required_profit": 0.0,
              "only_per_pair": False,
              "only_per_side": False
          }
      ]

# +---------------------------------------------------------------------------+
# |                              Custom Indicators                            |
# +---------------------------------------------------------------------------+


# Range midpoint acts as Support
def is_support(row_data) -> bool:
  conditions = []
  for row in range(len(row_data) - 1):
    if row < len(row_data) // 2:
      conditions.append(row_data[row] > row_data[row + 1])
    else:
      conditions.append(row_data[row] < row_data[row + 1])
  result = reduce(lambda x, y: x & y, conditions)
  return result


# Range midpoint acts as Resistance
def is_resistance(row_data) -> bool:
  conditions = []
  for row in range(len(row_data) - 1):
    if row < len(row_data) // 2:
      conditions.append(row_data[row] < row_data[row + 1])
    else:
      conditions.append(row_data[row] > row_data[row + 1])
  result = reduce(lambda x, y: x & y, conditions)
  return result


# Elliot Wave Oscillator
def ewo(dataframe, ema1_length=5, ema2_length=35):
  ema1 = ta.EMA(dataframe, timeperiod=ema1_length)
  ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
  emadiff = (ema1 - ema2) / dataframe["close"] * 100.0
  return emadiff


# Chaikin Money Flow
def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
  """Chaikin Money Flow (CMF)
  It measures the amount of Money Flow Volume over a specific period.
  http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
  Args:
      dataframe(pandas.Dataframe): dataframe containing ohlcv
      n(int): n period.
      fillna(bool): if True, fill nan values.
  Returns:
      pandas.Series: New feature generated.
  """
  mfv = ((dataframe["close"] - dataframe["low"]) - (dataframe["high"] - dataframe["close"])) / (
    dataframe["high"] - dataframe["low"]
  )
  mfv = mfv.fillna(0.0)  # float division by zero
  mfv *= dataframe["volume"]
  cmf = mfv.rolling(n, min_periods=0).sum() / dataframe["volume"].rolling(n, min_periods=0).sum()
  if fillna:
    cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
  return Series(cmf, name="cmf")


# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
  """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
  of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
  Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
  of its recent trading range.
  The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
  """

  highest_high = dataframe["high"].rolling(center=False, window=period).max()
  lowest_low = dataframe["low"].rolling(center=False, window=period).min()

  WR = Series(
    (highest_high - dataframe["close"]) / (highest_high - lowest_low),
    name=f"{period} Williams %R",
  )

  return WR * -100


def williams_fractals(dataframe: pd.DataFrame, period: int = 2) -> tuple:
  """Williams Fractals implementation

  :param dataframe: OHLC data
  :param period: number of lower (or higher) points on each side of a high (or low)
  :return: tuple of boolean Series (bearish, bullish) where True marks a fractal pattern
  """

  window = 2 * period + 1

  bears = dataframe["high"].rolling(window, center=True).apply(lambda x: x[period] == max(x), raw=True)
  bulls = dataframe["low"].rolling(window, center=True).apply(lambda x: x[period] == min(x), raw=True)

  return bears, bulls


# Volume Weighted Moving Average
def vwma(dataframe: DataFrame, length: int = 10):
  """Indicator: Volume Weighted Moving Average (VWMA)"""
  # Calculate Result
  pv = dataframe["close"] * dataframe["volume"]
  vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe["volume"], timeperiod=length))
  vwma = vwma.fillna(0, inplace=True)
  return vwma


# Exponential moving average of a volume weighted simple moving average
def ema_vwma_osc(dataframe, len_slow_ma):
  slow_ema = Series(ta.EMA(vwma(dataframe, len_slow_ma), len_slow_ma))
  return ((slow_ema - slow_ema.shift(1)) / slow_ema.shift(1)) * 100


def t3_average(dataframe, length=5):
  """
  T3 Average by HPotter on Tradingview
  https://www.tradingview.com/script/qzoC9H1I-T3-Average/
  """
  df = dataframe.copy()

  df["xe1"] = ta.EMA(df["close"], timeperiod=length)
  df["xe1"].fillna(0, inplace=True)
  df["xe2"] = ta.EMA(df["xe1"], timeperiod=length)
  df["xe2"].fillna(0, inplace=True)
  df["xe3"] = ta.EMA(df["xe2"], timeperiod=length)
  df["xe3"].fillna(0, inplace=True)
  df["xe4"] = ta.EMA(df["xe3"], timeperiod=length)
  df["xe4"].fillna(0, inplace=True)
  df["xe5"] = ta.EMA(df["xe4"], timeperiod=length)
  df["xe5"].fillna(0, inplace=True)
  df["xe6"] = ta.EMA(df["xe5"], timeperiod=length)
  df["xe6"].fillna(0, inplace=True)
  b = 0.7
  c1 = -b * b * b
  c2 = 3 * b * b + 3 * b * b * b
  c3 = -6 * b * b - 3 * b - 3 * b * b * b
  c4 = 1 + 3 * b + b * b * b + 3 * b * b
  df["T3Average"] = c1 * df["xe6"] + c2 * df["xe5"] + c3 * df["xe4"] + c4 * df["xe3"]

  return df["T3Average"]


# Pivot Points - 3 variants - daily recommended
def pivot_points(dataframe: DataFrame, mode="fibonacci") -> Series:
  if mode == "simple":
    hlc3_pivot = (dataframe["high"] + dataframe["low"] + dataframe["close"]).shift(1) / 3
    res1 = hlc3_pivot * 2 - dataframe["low"].shift(1)
    sup1 = hlc3_pivot * 2 - dataframe["high"].shift(1)
    res2 = hlc3_pivot + (dataframe["high"] - dataframe["low"]).shift()
    sup2 = hlc3_pivot - (dataframe["high"] - dataframe["low"]).shift()
    res3 = hlc3_pivot * 2 + (dataframe["high"] - 2 * dataframe["low"]).shift()
    sup3 = hlc3_pivot * 2 - (2 * dataframe["high"] - dataframe["low"]).shift()
    return hlc3_pivot, res1, res2, res3, sup1, sup2, sup3
  elif mode == "fibonacci":
    hlc3_pivot = (dataframe["high"] + dataframe["low"] + dataframe["close"]).shift(1) / 3
    hl_range = (dataframe["high"] - dataframe["low"]).shift(1)
    res1 = hlc3_pivot + 0.382 * hl_range
    sup1 = hlc3_pivot - 0.382 * hl_range
    res2 = hlc3_pivot + 0.618 * hl_range
    sup2 = hlc3_pivot - 0.618 * hl_range
    res3 = hlc3_pivot + 1 * hl_range
    sup3 = hlc3_pivot - 1 * hl_range
    return hlc3_pivot, res1, res2, res3, sup1, sup2, sup3
  elif mode == "DeMark":
    demark_pivot_lt = dataframe["low"] * 2 + dataframe["high"] + dataframe["close"]
    demark_pivot_eq = dataframe["close"] * 2 + dataframe["low"] + dataframe["high"]
    demark_pivot_gt = dataframe["high"] * 2 + dataframe["low"] + dataframe["close"]
    demark_pivot = np.where(
      (dataframe["close"] < dataframe["open"]),
      demark_pivot_lt,
      np.where((dataframe["close"] > dataframe["open"]), demark_pivot_gt, demark_pivot_eq),
    )
    dm_pivot = demark_pivot / 4
    dm_res = demark_pivot / 2 - dataframe["low"]
    dm_sup = demark_pivot / 2 - dataframe["high"]
    return dm_pivot, dm_res, dm_sup


# Heikin Ashi candles
def heikin_ashi(dataframe, smooth_inputs=False, smooth_outputs=False, length=10):
  df = dataframe[["open", "close", "high", "low"]].copy().fillna(0)
  if smooth_inputs:
    df["open_s"] = ta.EMA(df["open"], timeframe=length)
    df["high_s"] = ta.EMA(df["high"], timeframe=length)
    df["low_s"] = ta.EMA(df["low"], timeframe=length)
    df["close_s"] = ta.EMA(df["close"], timeframe=length)

    open_ha = (df["open_s"].shift(1) + df["close_s"].shift(1)) / 2
    high_ha = df.loc[:, ["high_s", "open_s", "close_s"]].max(axis=1)
    low_ha = df.loc[:, ["low_s", "open_s", "close_s"]].min(axis=1)
    close_ha = (df["open_s"] + df["high_s"] + df["low_s"] + df["close_s"]) / 4
  else:
    open_ha = (df["open"].shift(1) + df["close"].shift(1)) / 2
    high_ha = df.loc[:, ["high", "open", "close"]].max(axis=1)
    low_ha = df.loc[:, ["low", "open", "close"]].min(axis=1)
    close_ha = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

  open_ha = open_ha.fillna(0)
  high_ha = high_ha.fillna(0)
  low_ha = low_ha.fillna(0)
  close_ha = close_ha.fillna(0)

  if smooth_outputs:
    open_sha = ta.EMA(open_ha, timeframe=length)
    high_sha = ta.EMA(high_ha, timeframe=length)
    low_sha = ta.EMA(low_ha, timeframe=length)
    close_sha = ta.EMA(close_ha, timeframe=length)

    return open_sha, close_sha, low_sha
  else:
    return open_ha, close_ha, low_ha


# Peak Percentage Change
def range_percent_change(self, dataframe: DataFrame, method, length: int) -> float:
  """
  Rolling Percentage Change Maximum across interval.

  :param dataframe: DataFrame The original OHLC dataframe
  :param method: High to Low / Open to Close
  :param length: int The length to look back
  """
  if method == "HL":
    return (dataframe["high"].rolling(length).max() - dataframe["low"].rolling(length).min()) / dataframe[
      "low"
    ].rolling(length).min()
  elif method == "OC":
    return (dataframe["open"].rolling(length).max() - dataframe["close"].rolling(length).min()) / dataframe[
      "close"
    ].rolling(length).min()
  else:
    raise ValueError(f"Method {method} not defined!")


# Percentage distance to top peak
def top_percent_change(self, dataframe: DataFrame, length: int) -> float:
  """
  Percentage change of the current close from the range maximum Open price

  :param dataframe: DataFrame The original OHLC dataframe
  :param length: int The length to look back
  """
  if length == 0:
    return (dataframe["open"] - dataframe["close"]) / dataframe["close"]
  else:
    return (dataframe["open"].rolling(length).max() - dataframe["close"]) / dataframe["close"]


# +---------------------------------------------------------------------------+
# |                              Classes                                      |
# +---------------------------------------------------------------------------+


class Cache:
  def __init__(self, path):
    self.path = path
    self.data = {}
    self._mtime = None
    self._previous_data = {}
    try:
      self.load()
    except FileNotFoundError:
      pass

  @staticmethod
  def rapidjson_load_kwargs():
    return {"number_mode": rapidjson.NM_NATIVE, "parse_mode": rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS}

  @staticmethod
  def rapidjson_dump_kwargs():
    return {"number_mode": rapidjson.NM_NATIVE}

  def load(self):
    if not self._mtime or self.path.stat().st_mtime_ns != self._mtime:
      self._load()

  def save(self):
    if self.data != self._previous_data:
      self._save()

  def process_loaded_data(self, data):
    return data

  def _load(self):
    # This method only exists to simplify unit testing
    with self.path.open("r") as rfh:
      try:
        data = rapidjson.load(rfh, **self.rapidjson_load_kwargs())
      except rapidjson.JSONDecodeError as exc:
        log.error("Failed to load JSON from %s: %s", self.path, exc)
      else:
        self.data = self.process_loaded_data(data)
        self._previous_data = copy.deepcopy(self.data)
        self._mtime = self.path.stat().st_mtime_ns

  def _save(self):
    # This method only exists to simplify unit testing
    rapidjson.dump(self.data, self.path.open("w"), **self.rapidjson_dump_kwargs())
    self._mtime = self.path.stat().st_mtime
    self._previous_data = copy.deepcopy(self.data)


class HoldsCache(Cache):
  @staticmethod
  def rapidjson_load_kwargs():
    return {
      "number_mode": rapidjson.NM_NATIVE,
      "parse_mode": rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS,
      "object_hook": HoldsCache._object_hook,
    }

  @staticmethod
  def rapidjson_dump_kwargs():
    return {
      "number_mode": rapidjson.NM_NATIVE,
      "mapping_mode": rapidjson.MM_COERCE_KEYS_TO_STRINGS,
    }

  def save(self):
    raise RuntimeError("The holds cache does not allow programatical save")

  def process_loaded_data(self, data):
    trade_ids = data.get("trade_ids")
    trade_pairs = data.get("trade_pairs")

    if not trade_ids and not trade_pairs:
      return data

    open_trades = {}
    for trade in Trade.get_trades_proxy(is_open=True):
      open_trades[trade.id] = open_trades[trade.pair] = trade

    r_trade_ids = {}
    if trade_ids:
      if isinstance(trade_ids, dict):
        # New syntax
        for trade_id, profit_ratio in trade_ids.items():
          if not isinstance(trade_id, int):
            log.error("The trade_id(%s) defined under 'trade_ids' in %s is not an integer", trade_id, self.path)
            continue
          if not isinstance(profit_ratio, float):
            log.error(
              "The 'profit_ratio' config value(%s) for trade_id %s in %s is not a float",
              profit_ratio,
              trade_id,
              self.path,
            )
          if trade_id in open_trades:
            formatted_profit_ratio = f"{profit_ratio * 100}%"
            log.warning(
              "The trade %s is configured to HOLD until the profit ratio of %s is met",
              open_trades[trade_id],
              formatted_profit_ratio,
            )
            r_trade_ids[trade_id] = profit_ratio
          else:
            log.warning(
              "The trade_id(%s) is no longer open. Please remove it from 'trade_ids' in %s",
              trade_id,
              self.path,
            )
      else:
        # Initial Syntax
        profit_ratio = data.get("profit_ratio")
        if profit_ratio:
          if not isinstance(profit_ratio, float):
            log.error("The 'profit_ratio' config value(%s) in %s is not a float", profit_ratio, self.path)
        else:
          profit_ratio = 0.005
        formatted_profit_ratio = f"{profit_ratio * 100}%"
        for trade_id in trade_ids:
          if not isinstance(trade_id, int):
            log.error("The trade_id(%s) defined under 'trade_ids' in %s is not an integer", trade_id, self.path)
            continue
          if trade_id in open_trades:
            log.warning(
              "The trade %s is configured to HOLD until the profit ratio of %s is met",
              open_trades[trade_id],
              formatted_profit_ratio,
            )
            r_trade_ids[trade_id] = profit_ratio
          else:
            log.warning(
              "The trade_id(%s) is no longer open. Please remove it from 'trade_ids' in %s",
              trade_id,
              self.path,
            )

    r_trade_pairs = {}
    if trade_pairs:
      for trade_pair, profit_ratio in trade_pairs.items():
        if not isinstance(trade_pair, str):
          log.error("The trade_pair(%s) defined under 'trade_pairs' in %s is not a string", trade_pair, self.path)
          continue
        if "/" not in trade_pair:
          log.error(
            "The trade_pair(%s) defined under 'trade_pairs' in %s does not look like "
            "a valid '<TOKEN_NAME>/<STAKE_CURRENCY>' formatted pair.",
            trade_pair,
            self.path,
          )
          continue
        if not isinstance(profit_ratio, float):
          log.error(
            "The 'profit_ratio' config value(%s) for trade_pair %s in %s is not a float",
            profit_ratio,
            trade_pair,
            self.path,
          )
        formatted_profit_ratio = f"{profit_ratio * 100}%"
        if trade_pair in open_trades:
          log.warning(
            "The trade %s is configured to HOLD until the profit ratio of %s is met",
            open_trades[trade_pair],
            formatted_profit_ratio,
          )
        else:
          log.warning(
            "The trade pair %s is configured to HOLD until the profit ratio of %s is met",
            trade_pair,
            formatted_profit_ratio,
          )
        r_trade_pairs[trade_pair] = profit_ratio

    r_data = {}
    if r_trade_ids:
      r_data["trade_ids"] = r_trade_ids
    if r_trade_pairs:
      r_data["trade_pairs"] = r_trade_pairs
    return r_data

  @staticmethod
  def _object_hook(data):
    _data = {}
    for key, value in data.items():
      try:
        key = int(key)
      except ValueError:
        pass
      _data[key] = value
    return _data
