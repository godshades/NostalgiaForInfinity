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


class NostalgiaForInfinityX4(IStrategy):
  INTERFACE_VERSION = 3

  def version(self) -> str:
    return "v14.1.193"

  stoploss = -0.99

  # Trailing stoploss (not used)
  trailing_stop = False
  trailing_only_offset_is_reached = True
  trailing_stop_positive = 0.01
  trailing_stop_positive_offset = 0.03

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
  stop_threshold_futures_rebuy = 0.9
  stop_threshold_spot_rebuy = 3.9

  # Rebuy mode minimum number of free slots
  rebuy_mode_min_free_slots = 2

  # Position adjust feature
  position_adjustment_enable = True

  # Grinding feature
  grinding_enable = True
  grinding_mode = 2
  stake_grinding_mode_multiplier = 1.0
  stake_grinding_mode_multiplier_alt_1 = 1.0
  stake_grinding_mode_multiplier_alt_2 = 1.0
  # Grinding stop thresholds
  grinding_stop_init = -0.12
  grinding_stop_grinds = -0.16
  # Grinding take profit threshold
  grinding_profit_threshold = 0.012
  # Grinding stakes
  grinding_stakes = [0.25, 0.25, 0.25, 0.25, 0.25]
  grinding_stakes_alt_1 = [0.5, 0.5, 0.5]
  grinding_stakes_alt_2 = [0.75, 0.75]
  # Current total profit
  grinding_thresholds = [-0.04, -0.08, -0.1, -0.12, -0.14]
  grinding_thresholds_alt_1 = [-0.06, -0.12, -0.18]
  grinding_thresholds_alt_2 = [-0.08, -0.18]

  # Grinding mode 1
  grinding_mode_1_stop_grinds = -0.16
  grinding_mode_1_profit_threshold = 0.018
  grinding_mode_1_thresholds = [-0.0, -0.06]
  grinding_mode_1_stakes = [0.2, 0.2, 0.2, 0.2, 0.2]
  grinding_mode_1_sub_thresholds = [-0.06, -0.065, -0.07, -0.075, -0.08]
  grinding_mode_1_stakes_alt_1 = [0.25, 0.25, 0.25, 0.25]
  grinding_mode_1_sub_thresholds_alt_1 = [-0.06, -0.065, -0.07, -0.085]
  grinding_mode_1_stakes_alt_2 = [0.3, 0.3, 0.3, 0.3]
  grinding_mode_1_sub_thresholds_alt_2 = [-0.06, -0.07, -0.09, -0.1]
  grinding_mode_1_stakes_alt_3 = [0.35, 0.35, 0.35]
  grinding_mode_1_sub_thresholds_alt_3 = [-0.06, -0.075, -0.1]
  grinding_mode_1_stakes_alt_4 = [0.45, 0.45, 0.45]
  grinding_mode_1_sub_thresholds_alt_4 = [-0.06, -0.08, -0.11]

  # Grinding mode 2
  grinding_mode_2_derisk_spot = -0.40
  grinding_mode_2_stop_grinds_spot = -0.16
  grinding_mode_2_derisk_futures = -0.50
  grinding_mode_2_stop_grinds_futures = -0.48
  grinding_mode_2_profit_threshold_spot = 0.018
  grinding_mode_2_profit_threshold_futures = 0.018
  grinding_mode_2_stakes_spot = [
    [0.20, 0.20, 0.20, 0.20, 0.20],
    [0.3, 0.3, 0.3, 0.3],
    [0.35, 0.35, 0.35, 0.35],
    [0.4, 0.4, 0.4],
    [0.45, 0.45, 0.45],
    [0.5, 0.5, 0.5],
    [0.75, 0.75],
  ]
  grinding_mode_2_stakes_futures = [
    [0.20, 0.20, 0.20, 0.20, 0.20],
    [0.3, 0.3, 0.3, 0.3],
    [0.35, 0.35, 0.35, 0.35],
    [0.4, 0.4, 0.4],
    [0.45, 0.45, 0.45],
    [0.5, 0.5, 0.5],
    [0.75, 0.75],
  ]
  grinding_mode_2_sub_thresholds_spot = [
    [-0.12, -0.12, -0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12, -0.14],
    [-0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12],
    [-0.12, -0.12],
  ]
  grinding_mode_2_sub_thresholds_futures = [
    [-0.12, -0.12, -0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12, -0.14],
    [-0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12],
    [-0.12, -0.12, -0.12],
    [-0.12, -0.12],
  ]

  # Non rebuy modes
  regular_mode_stake_multiplier_spot = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75]
  regular_mode_rebuy_stakes_spot = [1.0, 1.0, 1.0, 1.0, 1.0]
  regular_mode_grind_1_stakes_spot = [1.0, 1.0, 1.0, 1.0, 1.0]
  regular_mode_rebuy_thresholds_spot = [-0.12, -0.12, -0.12, -0.12, -0.12]
  regular_mode_grind_1_thresholds_spot = [-0.03, -0.12, -0.12, -0.12, -0.12, -0.12]
  regular_mode_grind_1_profit_threshold_spot = 0.018
  regular_mode_derisk_spot = -1.25

  regular_mode_stake_multiplier_futures = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75]
  regular_mode_rebuy_stakes_futures = [1.0, 1.0, 1.0, 1.0, 1.0]
  regular_mode_grind_1_stakes_futures = [1.0, 1.0, 1.0, 1.0, 1.0]
  regular_mode_rebuy_thresholds_futures = [-0.12, -0.12, -0.12, -0.12, -0.12]
  regular_mode_grind_1_thresholds_futures = [-0.03, -0.12, -0.12, -0.12, -0.12, -0.12]
  regular_mode_grind_1_profit_threshold_futures = 0.018
  regular_mode_derisk_futures = -3.75

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
  entry_25_rsi_14_max = DecimalParameter(20.0, 46.0, default=46.0, decimals=0, space="buy", optimize=True)
  entry_25_cti_20_max = DecimalParameter(-0.9, 0.0, default=-0.9, decimals=1, space="buy", optimize=False)
  entry_25_ewo_50_200_min = DecimalParameter(1.0, 8.0, default=2.0, decimals=1, space="buy", optimize=True)
  entry_25_sma_offset = DecimalParameter(0.920, 0.950, default=0.944, decimals=3, space="buy", optimize=True)

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
    if "grinding_mode_2_derisk_spot" in self.config:
      self.grinding_mode_2_derisk_spot = self.config["grinding_mode_2_derisk_spot"]
    if "grinding_mode_2_stop_grinds_spot" in self.config:
      self.grinding_mode_2_stop_grinds_spot = self.config["grinding_mode_2_stop_grinds_spot"]
    if "grinding_profit_threshold" in self.config:
      self.grinding_profit_threshold = self.config["grinding_profit_threshold"]
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

  def exit_normal(
    self,
    pair: str,
    current_rate: float,
    profit_stake: float,
    profit_ratio: float,
    profit_current_stake_ratio: float,
    profit_init_ratio: float,
    max_profit: float,
    max_loss: float,
    filled_entries,
    filled_exits,
    last_candle,
    previous_candle_1,
    previous_candle_2,
    previous_candle_3,
    previous_candle_4,
    previous_candle_5,
    trade: "Trade",
    current_time: "datetime",
    enter_tags,
  ) -> tuple:
    sell = False

    # Original sell signals
    sell, signal_name = self.exit_signals(
      self.normal_mode_name,
      profit_current_stake_ratio,
      max_profit,
      max_loss,
      last_candle,
      previous_candle_1,
      previous_candle_2,
      previous_candle_3,
      previous_candle_4,
      previous_candle_5,
      trade,
      current_time,
      enter_tags,
    )

    # Main sell signals
    if not sell:
      sell, signal_name = self.exit_main(
        self.normal_mode_name,
        profit_current_stake_ratio,
        max_profit,
        max_loss,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Williams %R based sells
    if not sell:
      sell, signal_name = self.exit_r(
        self.normal_mode_name,
        profit_current_stake_ratio,
        max_profit,
        max_loss,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Downtrend/descending based sells
    if not sell:
      sell, signal_name = self.exit_long_dec(
        self.normal_mode_name,
        profit_current_stake_ratio,
        max_profit,
        max_loss,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Stoplosses
    if not sell:
      sell, signal_name = self.exit_stoploss(
        self.normal_mode_name,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        max_profit,
        max_loss,
        filled_entries,
        filled_exits,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Profit Target Signal
    # Check if pair exist on target_profit_cache
    if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
      previous_rate = self.target_profit_cache.data[pair]["rate"]
      previous_profit = self.target_profit_cache.data[pair]["profit"]
      previous_sell_reason = self.target_profit_cache.data[pair]["sell_reason"]
      previous_time_profit_reached = datetime.fromisoformat(self.target_profit_cache.data[pair]["time_profit_reached"])

      sell_max, signal_name_max = self.exit_profit_target(
        self.normal_mode_name,
        pair,
        trade,
        current_time,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        last_candle,
        previous_candle_1,
        previous_rate,
        previous_profit,
        previous_sell_reason,
        previous_time_profit_reached,
        enter_tags,
      )
      if sell_max and signal_name_max is not None:
        return True, f"{signal_name_max}_m"
      if previous_sell_reason in [f"exit_{self.normal_mode_name}_stoploss_u_e"]:
        if profit_ratio > (previous_profit + 0.005):
          mark_pair, mark_signal = self.mark_profit_target(
            self.normal_mode_name,
            pair,
            True,
            previous_sell_reason,
            trade,
            current_time,
            current_rate,
            profit_ratio,
            last_candle,
            previous_candle_1,
          )
          if mark_pair:
            self._set_profit_target(pair, mark_signal, current_rate, profit_ratio, current_time)
      elif (profit_current_stake_ratio > (previous_profit + 0.001)) and (
        previous_sell_reason not in [f"exit_{self.normal_mode_name}_stoploss_doom"]
      ):
        # Update the target, raise it.
        mark_pair, mark_signal = self.mark_profit_target(
          self.normal_mode_name,
          pair,
          True,
          previous_sell_reason,
          trade,
          current_time,
          current_rate,
          profit_current_stake_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)

    # Add the pair to the list, if a sell triggered and conditions met
    if sell and signal_name is not None:
      previous_profit = None
      if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
        previous_profit = self.target_profit_cache.data[pair]["profit"]
      if signal_name in [
        f"exit_{self.normal_mode_name}_stoploss_doom",
        f"exit_{self.normal_mode_name}_stoploss_u_e",
      ]:
        mark_pair, mark_signal = self.mark_profit_target(
          self.normal_mode_name,
          pair,
          sell,
          signal_name,
          trade,
          current_time,
          current_rate,
          profit_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_ratio, current_time)
        else:
          # Just sell it, without maximize
          return True, f"{signal_name}"
      elif (previous_profit is None) or (previous_profit < profit_current_stake_ratio):
        mark_pair, mark_signal = self.mark_profit_target(
          self.normal_mode_name,
          pair,
          sell,
          signal_name,
          trade,
          current_time,
          current_rate,
          profit_current_stake_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)
        else:
          # Just sell it, without maximize
          return True, f"{signal_name}"
    else:
      if profit_current_stake_ratio >= self.profit_max_thresholds[0]:
        previous_profit = None
        if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
          previous_profit = self.target_profit_cache.data[pair]["profit"]
        if (previous_profit is None) or (previous_profit < profit_current_stake_ratio):
          mark_signal = f"exit_profit_{self.normal_mode_name}_max"
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)

    if signal_name not in [
      f"exit_profit_{self.normal_mode_name}_max",
      f"exit_{self.normal_mode_name}_stoploss_doom",
      f"exit_{self.normal_mode_name}_stoploss_u_e",
    ]:
      if sell and (signal_name is not None):
        return True, f"{signal_name}"

    return False, None

  def exit_pump(
    self,
    pair: str,
    current_rate: float,
    profit_stake: float,
    profit_ratio: float,
    profit_current_stake_ratio: float,
    profit_init_ratio: float,
    max_profit: float,
    max_loss: float,
    filled_entries,
    filled_exits,
    last_candle,
    previous_candle_1,
    previous_candle_2,
    previous_candle_3,
    previous_candle_4,
    previous_candle_5,
    trade: "Trade",
    current_time: "datetime",
    enter_tags,
  ) -> tuple:
    sell = False

    # Original sell signals
    sell, signal_name = self.exit_signals(
      self.pump_mode_name,
      profit_current_stake_ratio,
      max_profit,
      max_loss,
      last_candle,
      previous_candle_1,
      previous_candle_2,
      previous_candle_3,
      previous_candle_4,
      previous_candle_5,
      trade,
      current_time,
      enter_tags,
    )

    # Main sell signals
    if not sell:
      sell, signal_name = self.exit_main(
        self.pump_mode_name,
        profit_current_stake_ratio,
        max_profit,
        max_loss,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Williams %R based sells
    if not sell:
      sell, signal_name = self.exit_r(
        self.pump_mode_name,
        profit_current_stake_ratio,
        max_profit,
        max_loss,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Downtrend/descending based sells
    if not sell:
      sell, signal_name = self.exit_long_dec(
        self.pump_mode_name,
        profit_current_stake_ratio,
        max_profit,
        max_loss,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Stoplosses
    if not sell:
      sell, signal_name = self.exit_stoploss(
        self.pump_mode_name,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        max_profit,
        max_loss,
        filled_entries,
        filled_exits,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Profit Target Signal
    # Check if pair exist on target_profit_cache
    if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
      previous_rate = self.target_profit_cache.data[pair]["rate"]
      previous_profit = self.target_profit_cache.data[pair]["profit"]
      previous_sell_reason = self.target_profit_cache.data[pair]["sell_reason"]
      previous_time_profit_reached = datetime.fromisoformat(self.target_profit_cache.data[pair]["time_profit_reached"])

      sell_max, signal_name_max = self.exit_profit_target(
        self.pump_mode_name,
        pair,
        trade,
        current_time,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        last_candle,
        previous_candle_1,
        previous_rate,
        previous_profit,
        previous_sell_reason,
        previous_time_profit_reached,
        enter_tags,
      )
      if sell_max and signal_name_max is not None:
        return True, f"{signal_name_max}_m"
      if previous_sell_reason in [f"exit_{self.pump_mode_name}_stoploss_u_e"]:
        if profit_ratio > (previous_profit + 0.005):
          mark_pair, mark_signal = self.mark_profit_target(
            self.pump_mode_name,
            pair,
            True,
            previous_sell_reason,
            trade,
            current_time,
            current_rate,
            profit_ratio,
            last_candle,
            previous_candle_1,
          )
          if mark_pair:
            self._set_profit_target(pair, mark_signal, current_rate, profit_ratio, current_time)
      elif (profit_current_stake_ratio > (previous_profit + 0.001)) and (
        previous_sell_reason not in [f"exit_{self.pump_mode_name}_stoploss_doom"]
      ):
        # Update the target, raise it.
        mark_pair, mark_signal = self.mark_profit_target(
          self.pump_mode_name,
          pair,
          True,
          previous_sell_reason,
          trade,
          current_time,
          current_rate,
          profit_current_stake_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)

    # Add the pair to the list, if a sell triggered and conditions met
    if sell and signal_name is not None:
      previous_profit = None
      if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
        previous_profit = self.target_profit_cache.data[pair]["profit"]
      if signal_name in [
        f"exit_{self.pump_mode_name}_stoploss_doom",
        f"exit_{self.pump_mode_name}_stoploss_u_e",
      ]:
        mark_pair, mark_signal = self.mark_profit_target(
          self.pump_mode_name,
          pair,
          sell,
          signal_name,
          trade,
          current_time,
          current_rate,
          profit_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_ratio, current_time)
        else:
          # Just sell it, without maximize
          return True, f"{signal_name}"
      elif (previous_profit is None) or (previous_profit < profit_current_stake_ratio):
        mark_pair, mark_signal = self.mark_profit_target(
          self.pump_mode_name,
          pair,
          sell,
          signal_name,
          trade,
          current_time,
          current_rate,
          profit_current_stake_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)
        else:
          # Just sell it, without maximize
          return True, f"{signal_name}"
    else:
      if profit_current_stake_ratio >= self.profit_max_thresholds[2]:
        previous_profit = None
        if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
          previous_profit = self.target_profit_cache.data[pair]["profit"]
        if (previous_profit is None) or (previous_profit < profit_current_stake_ratio):
          mark_signal = f"exit_profit_{self.pump_mode_name}_max"
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)

    if signal_name not in [
      f"exit_profit_{self.pump_mode_name}_max",
      f"exit_{self.pump_mode_name}_stoploss_doom",
      f"exit_{self.pump_mode_name}_stoploss_u_e",
    ]:
      if sell and (signal_name is not None):
        return True, f"{signal_name}"

    return False, None

  def exit_quick(
    self,
    pair: str,
    current_rate: float,
    profit_stake: float,
    profit_ratio: float,
    profit_current_stake_ratio: float,
    profit_init_ratio: float,
    max_profit: float,
    max_loss: float,
    filled_entries,
    filled_exits,
    last_candle,
    previous_candle_1,
    previous_candle_2,
    previous_candle_3,
    previous_candle_4,
    previous_candle_5,
    trade: "Trade",
    current_time: "datetime",
    enter_tags,
  ) -> tuple:
    sell = False

    # Original sell signals
    sell, signal_name = self.exit_signals(
      self.quick_mode_name,
      profit_current_stake_ratio,
      max_profit,
      max_loss,
      last_candle,
      previous_candle_1,
      previous_candle_2,
      previous_candle_3,
      previous_candle_4,
      previous_candle_5,
      trade,
      current_time,
      enter_tags,
    )

    # Main sell signals
    if not sell:
      sell, signal_name = self.exit_main(
        self.quick_mode_name,
        profit_current_stake_ratio,
        max_profit,
        max_loss,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Williams %R based sells
    if not sell:
      sell, signal_name = self.exit_r(
        self.quick_mode_name,
        profit_current_stake_ratio,
        max_profit,
        max_loss,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Downtrend/descending based sells
    if not sell:
      sell, signal_name = self.exit_long_dec(
        self.quick_mode_name,
        profit_current_stake_ratio,
        max_profit,
        max_loss,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Stoplosses
    if not sell:
      sell, signal_name = self.exit_stoploss(
        self.quick_mode_name,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        max_profit,
        max_loss,
        filled_entries,
        filled_exits,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Extra sell logic
    if not sell:
      if (0.09 >= profit_current_stake_ratio > 0.02) and (last_candle["rsi_14"] > 78.0):
        sell, signal_name = True, f"exit_{self.quick_mode_name}_q_1"

      if (0.09 >= profit_current_stake_ratio > 0.02) and (last_candle["cti_20"] > 0.95):
        sell, signal_name = True, f"exit_{self.quick_mode_name}_q_2"

      if (0.09 >= profit_current_stake_ratio > 0.02) and (last_candle["r_14"] >= -0.1):
        sell, signal_name = True, f"exit_{self.quick_mode_name}_q_3"

    # Profit Target Signal
    # Check if pair exist on target_profit_cache
    if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
      previous_rate = self.target_profit_cache.data[pair]["rate"]
      previous_profit = self.target_profit_cache.data[pair]["profit"]
      previous_sell_reason = self.target_profit_cache.data[pair]["sell_reason"]
      previous_time_profit_reached = datetime.fromisoformat(self.target_profit_cache.data[pair]["time_profit_reached"])

      sell_max, signal_name_max = self.exit_profit_target(
        self.quick_mode_name,
        pair,
        trade,
        current_time,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        last_candle,
        previous_candle_1,
        previous_rate,
        previous_profit,
        previous_sell_reason,
        previous_time_profit_reached,
        enter_tags,
      )
      if sell_max and signal_name_max is not None:
        return True, f"{signal_name_max}_m"
      if previous_sell_reason in [f"exit_{self.quick_mode_name}_stoploss_u_e"]:
        if profit_ratio > (previous_profit + 0.005):
          mark_pair, mark_signal = self.mark_profit_target(
            self.quick_mode_name,
            pair,
            True,
            previous_sell_reason,
            trade,
            current_time,
            current_rate,
            profit_ratio,
            last_candle,
            previous_candle_1,
          )
          if mark_pair:
            self._set_profit_target(pair, mark_signal, current_rate, profit_ratio, current_time)
      elif (profit_current_stake_ratio > (previous_profit + 0.001)) and (
        previous_sell_reason not in [f"exit_{self.quick_mode_name}_stoploss_doom"]
      ):
        # Update the target, raise it.
        mark_pair, mark_signal = self.mark_profit_target(
          self.quick_mode_name,
          pair,
          True,
          previous_sell_reason,
          trade,
          current_time,
          current_rate,
          profit_current_stake_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)

    # Add the pair to the list, if a sell triggered and conditions met
    if sell and signal_name is not None:
      previous_profit = None
      if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
        previous_profit = self.target_profit_cache.data[pair]["profit"]
      if signal_name in [
        f"exit_{self.quick_mode_name}_stoploss_doom",
        f"exit_{self.quick_mode_name}_stoploss_u_e",
      ]:
        mark_pair, mark_signal = self.mark_profit_target(
          self.quick_mode_name,
          pair,
          sell,
          signal_name,
          trade,
          current_time,
          current_rate,
          profit_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_ratio, current_time)
        else:
          # Just sell it, without maximize
          return True, f"{signal_name}"
      elif (previous_profit is None) or (previous_profit < profit_current_stake_ratio):
        mark_pair, mark_signal = self.mark_profit_target(
          self.quick_mode_name,
          pair,
          sell,
          signal_name,
          trade,
          current_time,
          current_rate,
          profit_current_stake_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)
        else:
          # Just sell it, without maximize
          return True, f"{signal_name}"
    else:
      if profit_current_stake_ratio >= self.profit_max_thresholds[4]:
        previous_profit = None
        if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
          previous_profit = self.target_profit_cache.data[pair]["profit"]
        if (previous_profit is None) or (previous_profit < profit_current_stake_ratio):
          mark_signal = f"exit_profit_{self.quick_mode_name}_max"
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)

    if signal_name not in [
      f"exit_profit_{self.quick_mode_name}_max",
      f"exit_{self.quick_mode_name}_stoploss_doom",
      f"exit_{self.quick_mode_name}_stoploss_u_e",
    ]:
      if sell and (signal_name is not None):
        return True, f"{signal_name}"

    return False, None

  def long_exit_rebuy(
    self,
    pair: str,
    current_rate: float,
    profit_stake: float,
    profit_ratio: float,
    profit_current_stake_ratio: float,
    profit_init_ratio: float,
    max_profit: float,
    max_loss: float,
    filled_entries,
    filled_exits,
    last_candle,
    previous_candle_1,
    previous_candle_2,
    previous_candle_3,
    previous_candle_4,
    previous_candle_5,
    trade: "Trade",
    current_time: "datetime",
    enter_tags,
  ) -> tuple:
    sell = False

    # Original sell signals
    sell, signal_name = self.exit_signals(
      self.long_rebuy_mode_name,
      profit_current_stake_ratio,
      max_profit,
      max_loss,
      last_candle,
      previous_candle_1,
      previous_candle_2,
      previous_candle_3,
      previous_candle_4,
      previous_candle_5,
      trade,
      current_time,
      enter_tags,
    )

    # Main sell signals
    if not sell:
      sell, signal_name = self.exit_main(
        self.long_rebuy_mode_name,
        profit_current_stake_ratio,
        max_profit,
        max_loss,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Williams %R based sells
    if not sell:
      sell, signal_name = self.exit_r(
        self.long_rebuy_mode_name,
        profit_current_stake_ratio,
        max_profit,
        max_loss,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Downtrend/descending based sells
    if not sell:
      sell, signal_name = self.exit_long_dec(
        self.long_rebuy_mode_name,
        profit_current_stake_ratio,
        max_profit,
        max_loss,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Stoplosses
    if not sell:
      if profit_stake < -(
        filled_entries[0].cost
        * (self.stop_threshold_futures_rebuy if self.is_futures_mode else self.stop_threshold_spot_rebuy)
        / (trade.leverage if self.is_futures_mode else 1.0)
      ):
        sell, signal_name = True, f"exit_{self.long_rebuy_mode_name}_stoploss_doom"

    # Profit Target Signal
    # Check if pair exist on target_profit_cache
    if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
      previous_rate = self.target_profit_cache.data[pair]["rate"]
      previous_profit = self.target_profit_cache.data[pair]["profit"]
      previous_sell_reason = self.target_profit_cache.data[pair]["sell_reason"]
      previous_time_profit_reached = datetime.fromisoformat(self.target_profit_cache.data[pair]["time_profit_reached"])

      sell_max, signal_name_max = self.exit_profit_target(
        self.long_rebuy_mode_name,
        pair,
        trade,
        current_time,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        last_candle,
        previous_candle_1,
        previous_rate,
        previous_profit,
        previous_sell_reason,
        previous_time_profit_reached,
        enter_tags,
      )
      if sell_max and signal_name_max is not None:
        return True, f"{signal_name_max}_m"
      if previous_sell_reason in [f"exit_{self.long_rebuy_mode_name}_stoploss_u_e"]:
        if profit_ratio > (previous_profit + 0.005):
          mark_pair, mark_signal = self.mark_profit_target(
            self.long_rebuy_mode_name,
            pair,
            True,
            previous_sell_reason,
            trade,
            current_time,
            current_rate,
            profit_ratio,
            last_candle,
            previous_candle_1,
          )
          if mark_pair:
            self._set_profit_target(pair, mark_signal, current_rate, profit_ratio, current_time)
      elif (profit_current_stake_ratio > (previous_profit + 0.001)) and (
        previous_sell_reason not in [f"exit_{self.long_rebuy_mode_name}_stoploss_doom"]
      ):
        # Update the target, raise it.
        mark_pair, mark_signal = self.mark_profit_target(
          self.long_rebuy_mode_name,
          pair,
          True,
          previous_sell_reason,
          trade,
          current_time,
          current_rate,
          profit_current_stake_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)

    # Add the pair to the list, if a sell triggered and conditions met
    if sell and signal_name is not None:
      previous_profit = None
      if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
        previous_profit = self.target_profit_cache.data[pair]["profit"]
      if signal_name in [
        f"exit_{self.long_rebuy_mode_name}_stoploss_doom",
        f"exit_{self.long_rebuy_mode_name}_stoploss_u_e",
      ]:
        mark_pair, mark_signal = self.mark_profit_target(
          self.long_rebuy_mode_name,
          pair,
          sell,
          signal_name,
          trade,
          current_time,
          current_rate,
          profit_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_ratio, current_time)
        else:
          # Just sell it, without maximize
          return True, f"{signal_name}"
      elif (previous_profit is None) or (previous_profit < profit_current_stake_ratio):
        mark_pair, mark_signal = self.mark_profit_target(
          self.long_rebuy_mode_name,
          pair,
          sell,
          signal_name,
          trade,
          current_time,
          current_rate,
          profit_current_stake_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)
        else:
          # Just sell it, without maximize
          return True, f"{signal_name}"
    else:
      if profit_current_stake_ratio >= self.profit_max_thresholds[6]:
        previous_profit = None
        if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
          previous_profit = self.target_profit_cache.data[pair]["profit"]
        if (previous_profit is None) or (previous_profit < profit_current_stake_ratio):
          mark_signal = f"exit_profit_{self.long_rebuy_mode_name}_max"
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)

    if signal_name not in [f"exit_profit_{self.long_rebuy_mode_name}_max"]:
      if sell and (signal_name is not None):
        return True, f"{signal_name}"

    return False, None

  def exit_long(
    self,
    pair: str,
    current_rate: float,
    profit_stake: float,
    profit_ratio: float,
    profit_current_stake_ratio: float,
    profit_init_ratio: float,
    max_profit: float,
    max_loss: float,
    filled_entries,
    filled_exits,
    last_candle,
    previous_candle_1,
    previous_candle_2,
    previous_candle_3,
    previous_candle_4,
    previous_candle_5,
    trade: "Trade",
    current_time: "datetime",
    enter_tags,
  ) -> tuple:
    sell = False

    # Original sell signals
    sell, signal_name = self.exit_signals(
      self.long_mode_name,
      profit_current_stake_ratio,
      max_profit,
      max_loss,
      last_candle,
      previous_candle_1,
      previous_candle_2,
      previous_candle_3,
      previous_candle_4,
      previous_candle_5,
      trade,
      current_time,
      enter_tags,
    )

    # Main sell signals
    if not sell:
      sell, signal_name = self.exit_main(
        self.long_mode_name,
        profit_current_stake_ratio,
        max_profit,
        max_loss,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Williams %R based sells
    if not sell:
      sell, signal_name = self.exit_r(
        self.long_mode_name,
        profit_current_stake_ratio,
        max_profit,
        max_loss,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Stoplosses
    if not sell:
      sell, signal_name = self.exit_stoploss(
        self.long_mode_name,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        max_profit,
        max_loss,
        filled_entries,
        filled_exits,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )
    # Profit Target Signal
    # Check if pair exist on target_profit_cache
    if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
      previous_rate = self.target_profit_cache.data[pair]["rate"]
      previous_profit = self.target_profit_cache.data[pair]["profit"]
      previous_sell_reason = self.target_profit_cache.data[pair]["sell_reason"]
      previous_time_profit_reached = datetime.fromisoformat(self.target_profit_cache.data[pair]["time_profit_reached"])

      sell_max, signal_name_max = self.exit_profit_target(
        self.long_mode_name,
        pair,
        trade,
        current_time,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        last_candle,
        previous_candle_1,
        previous_rate,
        previous_profit,
        previous_sell_reason,
        previous_time_profit_reached,
        enter_tags,
      )
      if sell_max and signal_name_max is not None:
        return True, f"{signal_name_max}_m"
      if previous_sell_reason in [f"exit_{self.long_mode_name}_stoploss_u_e"]:
        if profit_ratio > (previous_profit + 0.005):
          mark_pair, mark_signal = self.mark_profit_target(
            self.long_mode_name,
            pair,
            True,
            previous_sell_reason,
            trade,
            current_time,
            current_rate,
            profit_ratio,
            last_candle,
            previous_candle_1,
          )
          if mark_pair:
            self._set_profit_target(pair, mark_signal, current_rate, profit_ratio, current_time)
      elif (profit_current_stake_ratio > (previous_profit + 0.001)) and (
        previous_sell_reason not in [f"exit_{self.long_mode_name}_stoploss_doom"]
      ):
        # Update the target, raise it.
        mark_pair, mark_signal = self.mark_profit_target(
          self.long_mode_name,
          pair,
          True,
          previous_sell_reason,
          trade,
          current_time,
          current_rate,
          profit_current_stake_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)

    # Add the pair to the list, if a sell triggered and conditions met
    if sell and signal_name is not None:
      previous_profit = None
      if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
        previous_profit = self.target_profit_cache.data[pair]["profit"]
      if signal_name in [
        f"exit_{self.long_mode_name}_stoploss_doom",
        f"exit_{self.long_mode_name}_stoploss_u_e",
      ]:
        mark_pair, mark_signal = self.mark_profit_target(
          self.long_mode_name,
          pair,
          sell,
          signal_name,
          trade,
          current_time,
          current_rate,
          profit_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_ratio, current_time)
        else:
          # Just sell it, without maximize
          return True, f"{signal_name}"
      elif (previous_profit is None) or (previous_profit < profit_current_stake_ratio):
        mark_pair, mark_signal = self.mark_profit_target(
          self.long_mode_name,
          pair,
          sell,
          signal_name,
          trade,
          current_time,
          current_rate,
          profit_current_stake_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)
        else:
          # Just sell it, without maximize
          return True, f"{signal_name}"
    else:
      if profit_current_stake_ratio >= self.profit_max_thresholds[8]:
        previous_profit = None
        if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
          previous_profit = self.target_profit_cache.data[pair]["profit"]
        if (previous_profit is None) or (previous_profit < profit_current_stake_ratio):
          mark_signal = f"exit_profit_{self.long_mode_name}_max"
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)

    if signal_name not in [
      f"exit_profit_{self.long_mode_name}_max",
      f"exit_{self.long_mode_name}_stoploss_doom",
      f"exit_{self.long_mode_name}_stoploss_u_e",
    ]:
      if sell and (signal_name is not None):
        return True, f"{signal_name}"

    return False, None

  def long_exit_rapid(
    self,
    pair: str,
    current_rate: float,
    profit_stake: float,
    profit_ratio: float,
    profit_current_stake_ratio: float,
    profit_init_ratio: float,
    max_profit: float,
    max_loss: float,
    filled_entries,
    filled_exits,
    last_candle,
    previous_candle_1,
    previous_candle_2,
    previous_candle_3,
    previous_candle_4,
    previous_candle_5,
    trade: "Trade",
    current_time: "datetime",
    enter_tags,
  ) -> tuple:
    sell = False

    # Original sell signals
    sell, signal_name = self.exit_signals(
      self.long_rapid_mode_name,
      profit_current_stake_ratio,
      max_profit,
      max_loss,
      last_candle,
      previous_candle_1,
      previous_candle_2,
      previous_candle_3,
      previous_candle_4,
      previous_candle_5,
      trade,
      current_time,
      enter_tags,
    )

    # Main sell signals
    if not sell:
      sell, signal_name = self.exit_main(
        self.long_rapid_mode_name,
        profit_current_stake_ratio,
        max_profit,
        max_loss,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Williams %R based sells
    if not sell:
      sell, signal_name = self.exit_r(
        self.long_rapid_mode_name,
        profit_current_stake_ratio,
        max_profit,
        max_loss,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Downtrend/descending based sells
    if not sell:
      sell, signal_name = self.exit_long_dec(
        self.long_rapid_mode_name,
        profit_current_stake_ratio,
        max_profit,
        max_loss,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Stoplosses
    if not sell:
      sell, signal_name = self.exit_stoploss(
        self.long_rapid_mode_name,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        max_profit,
        max_loss,
        filled_entries,
        filled_exits,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )

    # Extra sell logic
    if not sell:
      if (0.09 >= profit_current_stake_ratio > 0.01) and (last_candle["rsi_14"] > 78.0):
        sell, signal_name = True, f"exit_{self.long_rapid_mode_name}_rpd_1"

      if (0.09 >= profit_current_stake_ratio > 0.01) and (last_candle["cti_20"] > 0.95):
        sell, signal_name = True, f"exit_{self.long_rapid_mode_name}_rpd_2"

      if (0.09 >= profit_current_stake_ratio > 0.01) and (last_candle["r_14"] >= -0.1):
        sell, signal_name = True, f"exit_{self.long_rapid_mode_name}_rpd_3"

      # Stoplosses
      if profit_stake < -(
        filled_entries[0].cost
        * (self.stop_threshold_futures_rapid if self.is_futures_mode else self.stop_threshold_spot_rapid)
        / (trade.leverage if self.is_futures_mode else 1.0)
      ):
        sell, signal_name = True, f"exit_{self.long_rapid_mode_name}_stoploss_doom"

    # Profit Target Signal
    # Check if pair exist on target_profit_cache
    if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
      previous_rate = self.target_profit_cache.data[pair]["rate"]
      previous_profit = self.target_profit_cache.data[pair]["profit"]
      previous_sell_reason = self.target_profit_cache.data[pair]["sell_reason"]
      previous_time_profit_reached = datetime.fromisoformat(self.target_profit_cache.data[pair]["time_profit_reached"])

      sell_max, signal_name_max = self.exit_profit_target(
        self.long_rapid_mode_name,
        pair,
        trade,
        current_time,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        last_candle,
        previous_candle_1,
        previous_rate,
        previous_profit,
        previous_sell_reason,
        previous_time_profit_reached,
        enter_tags,
      )
      if sell_max and signal_name_max is not None:
        return True, f"{signal_name_max}_m"
      if previous_sell_reason in [f"exit_{self.long_rapid_mode_name}_stoploss_u_e"]:
        if profit_ratio > (previous_profit + 0.005):
          mark_pair, mark_signal = self.mark_profit_target(
            self.long_rapid_mode_name,
            pair,
            True,
            previous_sell_reason,
            trade,
            current_time,
            current_rate,
            profit_ratio,
            last_candle,
            previous_candle_1,
          )
          if mark_pair:
            self._set_profit_target(pair, mark_signal, current_rate, profit_ratio, current_time)
      elif (profit_current_stake_ratio > (previous_profit + 0.001)) and (
        previous_sell_reason not in [f"exit_{self.long_rapid_mode_name}_stoploss_doom"]
      ):
        # Update the target, raise it.
        mark_pair, mark_signal = self.mark_profit_target(
          self.long_rapid_mode_name,
          pair,
          True,
          previous_sell_reason,
          trade,
          current_time,
          current_rate,
          profit_current_stake_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)

    # Add the pair to the list, if a sell triggered and conditions met
    if sell and signal_name is not None:
      previous_profit = None
      if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
        previous_profit = self.target_profit_cache.data[pair]["profit"]
      if signal_name in [
        f"exit_{self.long_rapid_mode_name}_stoploss_doom",
        f"exit_{self.long_rapid_mode_name}_stoploss_u_e",
      ]:
        mark_pair, mark_signal = self.mark_profit_target(
          self.long_rapid_mode_name,
          pair,
          sell,
          signal_name,
          trade,
          current_time,
          current_rate,
          profit_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_ratio, current_time)
        else:
          # Just sell it, without maximize
          return True, f"{signal_name}"
      elif (previous_profit is None) or (previous_profit < profit_current_stake_ratio):
        mark_pair, mark_signal = self.mark_profit_target(
          self.long_rapid_mode_name,
          pair,
          sell,
          signal_name,
          trade,
          current_time,
          current_rate,
          profit_current_stake_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)
        else:
          # Just sell it, without maximize
          return True, f"{signal_name}"
    else:
      if profit_current_stake_ratio >= 0.01:
        previous_profit = None
        if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
          previous_profit = self.target_profit_cache.data[pair]["profit"]
        if (previous_profit is None) or (previous_profit < profit_current_stake_ratio):
          mark_signal = f"exit_profit_{self.long_rapid_mode_name}_max"
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)

    if signal_name not in [f"exit_profit_{self.long_rapid_mode_name}_max"]:
      if sell and (signal_name is not None):
        return True, f"{signal_name}"

    return False, None

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

  def exit_signals(
    self,
    mode_name: str,
    current_profit: float,
    max_profit: float,
    max_loss: float,
    last_candle,
    previous_candle_1,
    previous_candle_2,
    previous_candle_3,
    previous_candle_4,
    previous_candle_5,
    trade: "Trade",
    current_time: "datetime",
    buy_tag,
  ) -> tuple:
    # Sell signal 1
    if (
      (last_candle["rsi_14"] > 79.0)
      and (last_candle["close"] > last_candle["bb20_2_upp"])
      and (previous_candle_1["close"] > previous_candle_1["bb20_2_upp"])
      and (previous_candle_2["close"] > previous_candle_2["bb20_2_upp"])
      and (previous_candle_3["close"] > previous_candle_3["bb20_2_upp"])
      and (previous_candle_4["close"] > previous_candle_4["bb20_2_upp"])
    ):
      if last_candle["close"] > last_candle["ema_200"]:
        if current_profit > 0.01:
          return True, f"exit_{mode_name}_1_1_1"
      else:
        if current_profit > 0.01:
          return True, f"exit_{mode_name}_1_2_1"

    # Sell signal 2
    elif (
      (last_candle["rsi_14"] > 80.0)
      and (last_candle["close"] > last_candle["bb20_2_upp"])
      and (previous_candle_1["close"] > previous_candle_1["bb20_2_upp"])
      and (previous_candle_2["close"] > previous_candle_2["bb20_2_upp"])
    ):
      if last_candle["close"] > last_candle["ema_200"]:
        if current_profit > 0.01:
          return True, f"exit_{mode_name}_2_1_1"
      else:
        if current_profit > 0.01:
          return True, f"exit_{mode_name}_2_2_1"

    # Sell signal 3
    elif last_candle["rsi_14"] > 85.0:
      if last_candle["close"] > last_candle["ema_200"]:
        if current_profit > 0.01:
          return True, f"exit_{mode_name}_3_1_1"
      else:
        if current_profit > 0.01:
          return True, f"exit_{mode_name}_3_2_1"

    # Sell signal 4
    elif (last_candle["rsi_14"] > 80.0) and (last_candle["rsi_14_1h"] > 78.0):
      if last_candle["close"] > last_candle["ema_200"]:
        if current_profit > 0.01:
          return True, f"exit_{mode_name}_4_1_1"
      else:
        if current_profit > 0.01:
          return True, f"exit_{mode_name}_4_2_1"

    # Sell signal 6
    elif (
      (last_candle["close"] < last_candle["ema_200"])
      and (last_candle["close"] > last_candle["ema_50"])
      and (last_candle["rsi_14"] > 79.0)
    ):
      if current_profit > 0.01:
        return True, f"exit_{mode_name}_6_1"

    # Sell signal 7
    elif (last_candle["rsi_14_1h"] > 79.0) and (last_candle["crossed_below_ema_12_26"]):
      if last_candle["close"] > last_candle["ema_200"]:
        if current_profit > 0.01:
          return True, f"exit_{mode_name}_7_1_1"
      else:
        if current_profit > 0.01:
          return True, f"exit_{mode_name}_7_2_1"

    # Sell signal 8
    elif last_candle["close"] > last_candle["bb20_2_upp_1h"] * 1.08:
      if last_candle["close"] > last_candle["ema_200"]:
        if current_profit > 0.01:
          return True, f"exit_{mode_name}_8_1_1"
      else:
        if current_profit > 0.01:
          return True, f"exit_{mode_name}_8_2_1"

    return False, None

  def exit_main(
    self,
    mode_name: str,
    current_profit: float,
    max_profit: float,
    max_loss: float,
    last_candle,
    previous_candle_1,
    previous_candle_2,
    previous_candle_3,
    previous_candle_4,
    previous_candle_5,
    trade: "Trade",
    current_time: "datetime",
    buy_tag,
  ) -> tuple:
    if last_candle["close"] > last_candle["sma_200_1h"]:
      if 0.01 > current_profit >= 0.001:
        if last_candle["rsi_14"] < 10.0:
          return True, f"exit_{mode_name}_o_0"
      elif 0.02 > current_profit >= 0.01:
        if last_candle["rsi_14"] < 28.0:
          return True, f"exit_{mode_name}_o_1"
      elif 0.03 > current_profit >= 0.02:
        if last_candle["rsi_14"] < 30.0:
          return True, f"exit_{mode_name}_o_2"
      elif 0.04 > current_profit >= 0.03:
        if last_candle["rsi_14"] < 32.0:
          return True, f"exit_{mode_name}_o_3"
      elif 0.05 > current_profit >= 0.04:
        if last_candle["rsi_14"] < 34.0:
          return True, f"exit_{mode_name}_o_4"
      elif 0.06 > current_profit >= 0.05:
        if last_candle["rsi_14"] < 36.0:
          return True, f"exit_{mode_name}_o_5"
      elif 0.07 > current_profit >= 0.06:
        if last_candle["rsi_14"] < 38.0:
          return True, f"exit_{mode_name}_o_6"
      elif 0.08 > current_profit >= 0.07:
        if last_candle["rsi_14"] < 40.0:
          return True, f"exit_{mode_name}_o_7"
      elif 0.09 > current_profit >= 0.08:
        if last_candle["rsi_14"] < 42.0:
          return True, f"exit_{mode_name}_o_8"
      elif 0.1 > current_profit >= 0.09:
        if last_candle["rsi_14"] < 44.0:
          return True, f"exit_{mode_name}_o_9"
      elif 0.12 > current_profit >= 0.1:
        if last_candle["rsi_14"] < 46.0:
          return True, f"exit_{mode_name}_o_10"
      elif 0.2 > current_profit >= 0.12:
        if last_candle["rsi_14"] < 44.0:
          return True, f"exit_{mode_name}_o_11"
      elif current_profit >= 0.2:
        if last_candle["rsi_14"] < 42.0:
          return True, f"exit_{mode_name}_o_12"
    elif last_candle["close"] < last_candle["sma_200_1h"]:
      if 0.01 > current_profit >= 0.001:
        if last_candle["rsi_14"] < 12.0:
          return True, f"exit_{mode_name}_u_0"
      elif 0.02 > current_profit >= 0.01:
        if last_candle["rsi_14"] < 30.0:
          return True, f"exit_{mode_name}_u_1"
      elif 0.03 > current_profit >= 0.02:
        if last_candle["rsi_14"] < 32.0:
          return True, f"exit_{mode_name}_u_2"
      elif 0.04 > current_profit >= 0.03:
        if last_candle["rsi_14"] < 34.0:
          return True, f"exit_{mode_name}_u_3"
      elif 0.05 > current_profit >= 0.04:
        if last_candle["rsi_14"] < 36.0:
          return True, f"exit_{mode_name}_u_4"
      elif 0.06 > current_profit >= 0.05:
        if last_candle["rsi_14"] < 38.0:
          return True, f"exit_{mode_name}_u_5"
      elif 0.07 > current_profit >= 0.06:
        if last_candle["rsi_14"] < 40.0:
          return True, f"exit_{mode_name}_u_6"
      elif 0.08 > current_profit >= 0.07:
        if last_candle["rsi_14"] < 42.0:
          return True, f"exit_{mode_name}_u_7"
      elif 0.09 > current_profit >= 0.08:
        if last_candle["rsi_14"] < 44.0:
          return True, f"exit_{mode_name}_u_8"
      elif 0.1 > current_profit >= 0.09:
        if last_candle["rsi_14"] < 46.0:
          return True, f"exit_{mode_name}_u_9"
      elif 0.12 > current_profit >= 0.1:
        if last_candle["rsi_14"] < 48.0:
          return True, f"exit_{mode_name}_u_10"
      elif 0.2 > current_profit >= 0.12:
        if last_candle["rsi_14"] < 46.0:
          return True, f"exit_{mode_name}_u_11"
      elif current_profit >= 0.2:
        if last_candle["rsi_14"] < 44.0:
          return True, f"exit_{mode_name}_u_12"

    return False, None

  def exit_r(
    self,
    mode_name: str,
    current_profit: float,
    max_profit: float,
    max_loss: float,
    last_candle,
    previous_candle_1,
    previous_candle_2,
    previous_candle_3,
    previous_candle_4,
    previous_candle_5,
    trade: "Trade",
    current_time: "datetime",
    buy_tag,
  ) -> tuple:
    if 0.01 > current_profit >= 0.001:
      if (last_candle["r_480"] > -0.1) and (last_candle["r_14"] >= -1.0) and (last_candle["rsi_14"] > 75.0):
        return True, f"exit_{mode_name}_w_0_1"
      elif (last_candle["r_14"] >= -1.0) and (last_candle["rsi_14"] > 84.0):
        return True, f"exit_{mode_name}_w_0_2"
      elif (last_candle["r_14"] >= -1.0) and (last_candle["rsi_14"] < 40.0):
        return True, f"exit_{mode_name}_w_0_3"
      elif (last_candle["r_14"] >= -1.0) and (last_candle["rsi_14"] > 78.0) and (last_candle["r_480_1h"] > -20.0):
        return True, f"exit_{mode_name}_w_0_4"
      elif (last_candle["r_14"] >= -1.0) and (last_candle["rsi_14"] > 75.0) and (last_candle["cti_20"] > 0.97):
        return True, f"exit_{mode_name}_w_0_5"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] > 75.0)
        and (last_candle["r_480_1h"] > -5.0)
        and (last_candle["r_480_4h"] > -5.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 60.0)
        and (last_candle["cti_20_1d"] > 0.80)
      ):
        return True, f"exit_{mode_name}_w_0_6"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] > 75.0)
        and (last_candle["r_480_1h"] > -25.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_4h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 50.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_0_7"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["cti_20_4h"] >= 0.70)
        and (last_candle["rsi_14_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.02)
      ):
        return True, f"exit_{mode_name}_w_0_8"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 82.0)
        and (last_candle["rsi_14_15m"] >= 72.0)
        and (last_candle["cti_20_4h"] <= -0.50)
        and (last_candle["cti_20_1d"] >= 0.70)
      ):
        return True, f"exit_{mode_name}_w_0_9"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_0_10"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 74.0)
        and (last_candle["rsi_14_15m"] >= 74.0)
        and (last_candle["rsi_14_1h"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["r_480_1h"] > -30.0)
      ):
        return True, f"exit_{mode_name}_w_0_11"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 78.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] > -30.0)
        and (last_candle["change_pct_1d"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_0_12"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.80)
        and (last_candle["rsi_14_4h"] >= 65.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_0_13"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_1h"] >= 70.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
      ):
        return True, f"exit_{mode_name}_w_0_14"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
        and (last_candle["r_480_4h"] > -15.0)
        and (last_candle["change_pct_1h"] < -0.00)
      ):
        return True, f"exit_{mode_name}_w_0_15"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["top_wick_pct_1d"] > 0.16)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.80))
        and (last_candle["hl_pct_change_6_1d"] > 0.75)
      ):
        return True, f"exit_{mode_name}_w_0_16"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["cti_20_1d"] >= 0.80)
        and (last_candle["r_480_1h"] > -30.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["top_wick_pct_1d"] > 0.08)
      ):
        return True, f"exit_{mode_name}_w_0_17"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.90)
      ):
        return True, f"exit_{mode_name}_w_0_18"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 80.0)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.50)
      ):
        return True, f"exit_{mode_name}_w_0_19"
    elif 0.02 > current_profit >= 0.01:
      if last_candle["r_480"] > -0.2:
        return True, f"exit_{mode_name}_w_1_1"
      elif (last_candle["r_14"] >= -1.0) and (last_candle["rsi_14"] > 78.0):
        return True, f"exit_{mode_name}_w_1_2"
      elif (last_candle["r_14"] >= -2.0) and (last_candle["rsi_14"] < 46.0):
        return True, f"exit_{mode_name}_w_1_3"
      elif (last_candle["r_14"] >= -5.0) and (last_candle["rsi_14"] > 74.0) and (last_candle["r_480_1h"] > -25.0):
        return True, f"exit_{mode_name}_w_1_4"
      elif (last_candle["r_14"] >= -2.0) and (last_candle["cti_20"] > 0.95):
        return True, f"exit_{mode_name}_w_1_5"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] > 70.0)
        and (last_candle["r_480_1h"] > -10.0)
        and (last_candle["r_480_4h"] > -15.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 60.0)
        and (last_candle["cti_20_1d"] > 0.80)
      ):
        return True, f"exit_{mode_name}_w_1_6"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] > 60.0)
        and (last_candle["r_480_1h"] > -25.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_4h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 50.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_1_7"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["cti_20_4h"] >= 0.70)
        and (last_candle["rsi_14_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.02)
      ):
        return True, f"exit_{mode_name}_w_1_8"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 76.0)
        and (last_candle["rsi_14_15m"] >= 70.0)
        and (last_candle["cti_20_4h"] <= -0.50)
        and (last_candle["cti_20_1d"] >= 0.70)
      ):
        return True, f"exit_{mode_name}_w_1_9"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_1_10"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] >= 70.0)
        and (last_candle["rsi_14_1h"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["r_480_1h"] > -30.0)
      ):
        return True, f"exit_{mode_name}_w_1_11"
      elif (
        (last_candle["r_14"] >= -24.0)
        and (last_candle["rsi_14"] >= 74.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] > -30.0)
        and (last_candle["change_pct_1d"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_1_12"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.80)
        and (last_candle["rsi_14_4h"] >= 65.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_1_13"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_1h"] >= 70.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
      ):
        return True, f"exit_{mode_name}_w_1_14"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
        and (last_candle["r_480_4h"] > -15.0)
        and (last_candle["change_pct_1h"] < -0.00)
      ):
        return True, f"exit_{mode_name}_w_1_15"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["top_wick_pct_1d"] > 0.16)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.80))
        and (last_candle["hl_pct_change_6_1d"] > 0.75)
      ):
        return True, f"exit_{mode_name}_w_1_16"
      elif (
        (last_candle["r_14"] >= -16.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.80)
        and (last_candle["r_480_1h"] > -30.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["top_wick_pct_1d"] > 0.08)
      ):
        return True, f"exit_{mode_name}_w_1_17"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.90)
      ):
        return True, f"exit_{mode_name}_w_1_18"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 80.0)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.50)
      ):
        return True, f"exit_{mode_name}_w_1_19"
    elif 0.03 > current_profit >= 0.02:
      if last_candle["r_480"] > -0.3:
        return True, f"exit_{mode_name}_w_2_1"
      elif (last_candle["r_14"] >= -1.0) and (last_candle["rsi_14"] > 77.0):
        return True, f"exit_{mode_name}_w_2_2"
      elif (last_candle["r_14"] >= -2.0) and (last_candle["rsi_14"] < 48.0):
        return True, f"exit_{mode_name}_w_2_3"
      elif (last_candle["r_14"] >= -5.0) and (last_candle["rsi_14"] > 73.0) and (last_candle["r_480_1h"] > -25.0):
        return True, f"exit_{mode_name}_w_2_4"
      elif (last_candle["r_14"] >= -3.0) and (last_candle["cti_20"] > 0.95):
        return True, f"exit_{mode_name}_w_2_5"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] > 60.0)
        and (last_candle["r_480_1h"] > -25.0)
        and (last_candle["r_480_4h"] > -20.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 60.0)
        and (last_candle["cti_20_1d"] > 0.80)
      ):
        return True, f"exit_{mode_name}_w_2_6"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] > 60.0)
        and (last_candle["r_480_1h"] > -25.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_4h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 50.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_2_7"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["cti_20_4h"] >= 0.70)
        and (last_candle["rsi_14_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.02)
      ):
        return True, f"exit_{mode_name}_w_2_8"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 74.0)
        and (last_candle["rsi_14_15m"] >= 68.0)
        and (last_candle["cti_20_4h"] <= -0.50)
        and (last_candle["cti_20_1d"] >= 0.70)
      ):
        return True, f"exit_{mode_name}_w_2_9"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_2_10"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] >= 70.0)
        and (last_candle["rsi_14_1h"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["r_480_1h"] > -30.0)
      ):
        return True, f"exit_{mode_name}_w_2_11"
      elif (
        (last_candle["r_14"] >= -24.0)
        and (last_candle["rsi_14"] >= 72.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] > -30.0)
        and (last_candle["change_pct_1d"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_2_12"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.80)
        and (last_candle["rsi_14_4h"] >= 65.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_2_13"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_1h"] >= 70.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
      ):
        return True, f"exit_{mode_name}_w_2_14"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
        and (last_candle["r_480_4h"] > -15.0)
        and (last_candle["change_pct_1h"] < -0.00)
      ):
        return True, f"exit_{mode_name}_w_2_15"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["top_wick_pct_1d"] > 0.16)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.80))
        and (last_candle["hl_pct_change_6_1d"] > 0.75)
      ):
        return True, f"exit_{mode_name}_w_2_16"
      elif (
        (last_candle["r_14"] >= -18.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.80)
        and (last_candle["r_480_1h"] > -30.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["top_wick_pct_1d"] > 0.08)
      ):
        return True, f"exit_{mode_name}_w_2_17"
      elif (
        (last_candle["r_14"] >= -5.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.90)
      ):
        return True, f"exit_{mode_name}_w_2_18"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 80.0)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.50)
      ):
        return True, f"exit_{mode_name}_w_2_19"
    elif 0.04 > current_profit >= 0.03:
      if last_candle["r_480"] > -0.4:
        return True, f"exit_{mode_name}_w_3_1"
      elif (last_candle["r_14"] >= -1.0) and (last_candle["rsi_14"] > 76.0):
        return True, f"exit_{mode_name}_w_3_2"
      elif (last_candle["r_14"] >= -2.0) and (last_candle["rsi_14"] < 50.0):
        return True, f"exit_{mode_name}_w_3_3"
      elif (last_candle["r_14"] >= -5.0) and (last_candle["rsi_14"] > 72.0) and (last_candle["r_480_1h"] > -25.0):
        return True, f"exit_{mode_name}_w_3_4"
      elif (last_candle["r_14"] >= -4.0) and (last_candle["cti_20"] > 0.95):
        return True, f"exit_{mode_name}_w_3_5"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] > 60.0)
        and (last_candle["r_480_1h"] > -25.0)
        and (last_candle["r_480_4h"] > -20.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 60.0)
        and (last_candle["cti_20_1d"] > 0.80)
      ):
        return True, f"exit_{mode_name}_w_3_6"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] > 60.0)
        and (last_candle["r_480_1h"] > -25.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_4h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 50.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_3_7"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["cti_20_4h"] >= 0.70)
        and (last_candle["rsi_14_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.02)
      ):
        return True, f"exit_{mode_name}_w_3_8"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 72.0)
        and (last_candle["rsi_14_15m"] >= 66.0)
        and (last_candle["cti_20_4h"] <= -0.50)
        and (last_candle["cti_20_1d"] >= 0.70)
      ):
        return True, f"exit_{mode_name}_w_3_9"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_3_10"
      elif (
        (last_candle["r_14"] >= -3.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] >= 70.0)
        and (last_candle["rsi_14_1h"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["r_480_1h"] > -30.0)
      ):
        return True, f"exit_{mode_name}_w_3_11"
      elif (
        (last_candle["r_14"] >= -24.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] > -30.0)
        and (last_candle["change_pct_1d"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_3_12"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.80)
        and (last_candle["rsi_14_4h"] >= 65.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_3_13"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_1h"] >= 70.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
      ):
        return True, f"exit_{mode_name}_w_3_14"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
        and (last_candle["r_480_4h"] > -15.0)
        and (last_candle["change_pct_1h"] < -0.00)
      ):
        return True, f"exit_{mode_name}_w_3_15"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["top_wick_pct_1d"] > 0.16)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.80))
        and (last_candle["hl_pct_change_6_1d"] > 0.75)
      ):
        return True, f"exit_{mode_name}_w_3_16"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.80)
        and (last_candle["r_480_1h"] > -30.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["top_wick_pct_1d"] > 0.08)
      ):
        return True, f"exit_{mode_name}_w_3_17"
      elif (
        (last_candle["r_14"] >= -35.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.90)
      ):
        return True, f"exit_{mode_name}_w_3_18"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 80.0)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.50)
      ):
        return True, f"exit_{mode_name}_w_3_19"
    elif 0.05 > current_profit >= 0.04:
      if last_candle["r_480"] > -0.5:
        return True, f"exit_{mode_name}_w_4_1"
      elif (last_candle["r_14"] >= -1.0) and (last_candle["rsi_14"] > 75.0):
        return True, f"exit_{mode_name}_w_4_2"
      elif (last_candle["r_14"] >= -2.0) and (last_candle["rsi_14"] < 52.0):
        return True, f"exit_{mode_name}_w_4_3"
      elif (last_candle["r_14"] >= -5.0) and (last_candle["rsi_14"] > 71.0) and (last_candle["r_480_1h"] > -25.0):
        return True, f"exit_{mode_name}_w_4_4"
      elif (last_candle["r_14"] >= -5.0) and (last_candle["cti_20"] > 0.95):
        return True, f"exit_{mode_name}_w_4_5"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] > 60.0)
        and (last_candle["r_480_1h"] > -25.0)
        and (last_candle["r_480_4h"] > -20.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 60.0)
        and (last_candle["cti_20_1d"] > 0.80)
      ):
        return True, f"exit_{mode_name}_w_4_6"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] > 60.0)
        and (last_candle["r_480_1h"] > -25.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_4h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 50.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_4_7"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["cti_20_4h"] >= 0.70)
        and (last_candle["rsi_14_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.02)
      ):
        return True, f"exit_{mode_name}_w_4_8"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] >= 64.0)
        and (last_candle["cti_20_4h"] <= -0.50)
        and (last_candle["cti_20_1d"] >= 0.70)
      ):
        return True, f"exit_{mode_name}_w_4_9"
      elif (
        (last_candle["r_14"] >= -14.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_4_10"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] >= 70.0)
        and (last_candle["rsi_14_1h"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["r_480_1h"] > -30.0)
      ):
        return True, f"exit_{mode_name}_w_4_11"
      elif (
        (last_candle["r_14"] >= -24.0)
        and (last_candle["rsi_14"] >= 68.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] > -30.0)
        and (last_candle["change_pct_1d"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_4_12"
      elif (
        (last_candle["r_14"] >= -14.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.80)
        and (last_candle["rsi_14_4h"] >= 65.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_4_13"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_1h"] >= 70.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
      ):
        return True, f"exit_{mode_name}_w_4_14"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
        and (last_candle["r_480_4h"] > -15.0)
        and (last_candle["change_pct_1h"] < -0.00)
      ):
        return True, f"exit_{mode_name}_w_4_15"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["top_wick_pct_1d"] > 0.16)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.80))
        and (last_candle["hl_pct_change_6_1d"] > 0.75)
      ):
        return True, f"exit_{mode_name}_w_4_16"
      elif (
        (last_candle["r_14"] >= -22.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.80)
        and (last_candle["r_480_1h"] > -30.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["top_wick_pct_1d"] > 0.08)
      ):
        return True, f"exit_{mode_name}_w_4_17"
      elif (
        (last_candle["r_14"] >= -40.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.90)
      ):
        return True, f"exit_{mode_name}_w_4_18"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 80.0)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.50)
      ):
        return True, f"exit_{mode_name}_w_4_19"
    elif 0.06 > current_profit >= 0.05:
      if last_candle["r_480"] > -0.6:
        return True, f"exit_{mode_name}_w_5_1"
      elif (last_candle["r_14"] >= -1.0) and (last_candle["rsi_14"] > 74.0):
        return True, f"exit_{mode_name}_w_5_2"
      elif (last_candle["r_14"] >= -2.0) and (last_candle["rsi_14"] < 54.0):
        return True, f"exit_{mode_name}_w_5_3"
      elif (last_candle["r_14"] >= -5.0) and (last_candle["rsi_14"] > 70.0) and (last_candle["r_480_1h"] > -25.0):
        return True, f"exit_{mode_name}_w_5_4"
      elif (last_candle["r_14"] >= -6.0) and (last_candle["cti_20"] > 0.95):
        return True, f"exit_{mode_name}_w_5_5"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] > 60.0)
        and (last_candle["r_480_1h"] > -25.0)
        and (last_candle["r_480_4h"] > -20.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 60.0)
        and (last_candle["cti_20_1d"] > 0.80)
      ):
        return True, f"exit_{mode_name}_w_5_6"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] > 60.0)
        and (last_candle["r_480_1h"] > -25.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_4h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 50.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_5_7"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["cti_20_4h"] >= 0.70)
        and (last_candle["rsi_14_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.02)
      ):
        return True, f"exit_{mode_name}_w_5_8"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 68.0)
        and (last_candle["rsi_14_15m"] >= 62.0)
        and (last_candle["cti_20_4h"] <= -0.50)
        and (last_candle["cti_20_1d"] >= 0.70)
      ):
        return True, f"exit_{mode_name}_w_5_9"
      elif (
        (last_candle["r_14"] >= -15.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_5_10"
      elif (
        (last_candle["r_14"] >= -5.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] >= 70.0)
        and (last_candle["rsi_14_1h"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["r_480_1h"] > -30.0)
      ):
        return True, f"exit_{mode_name}_w_5_11"
      elif (
        (last_candle["r_14"] >= -24.0)
        and (last_candle["rsi_14"] >= 66.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] > -30.0)
        and (last_candle["change_pct_1d"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_5_12"
      elif (
        (last_candle["r_14"] >= -16.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.80)
        and (last_candle["rsi_14_4h"] >= 65.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_5_13"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_1h"] >= 70.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
      ):
        return True, f"exit_{mode_name}_w_5_14"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
        and (last_candle["r_480_4h"] > -15.0)
        and (last_candle["change_pct_1h"] < -0.00)
      ):
        return True, f"exit_{mode_name}_w_5_15"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["top_wick_pct_1d"] > 0.16)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.80))
        and (last_candle["hl_pct_change_6_1d"] > 0.75)
      ):
        return True, f"exit_{mode_name}_w_5_16"
      elif (
        (last_candle["r_14"] >= -24.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.80)
        and (last_candle["r_480_1h"] > -30.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["top_wick_pct_1d"] > 0.08)
      ):
        return True, f"exit_{mode_name}_w_5_17"
      elif (
        (last_candle["r_14"] >= -45.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.90)
      ):
        return True, f"exit_{mode_name}_w_5_18"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 80.0)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.50)
      ):
        return True, f"exit_{mode_name}_w_5_19"
    elif 0.07 > current_profit >= 0.06:
      if last_candle["r_480"] > -0.7:
        return True, f"exit_{mode_name}_w_6_1"
      elif (last_candle["r_14"] >= -1.0) and (last_candle["rsi_14"] > 75.0):
        return True, f"exit_{mode_name}_w_6_2"
      elif (last_candle["r_14"] >= -2.0) and (last_candle["rsi_14"] < 52.0):
        return True, f"exit_{mode_name}_w_6_3"
      elif (last_candle["r_14"] >= -5.0) and (last_candle["rsi_14"] > 71.0) and (last_candle["r_480_1h"] > -25.0):
        return True, f"exit_{mode_name}_w_6_4"
      elif (last_candle["r_14"] >= -5.0) and (last_candle["cti_20"] > 0.95):
        return True, f"exit_{mode_name}_w_6_5"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] > 60.0)
        and (last_candle["r_480_1h"] > -20.0)
        and (last_candle["r_480_4h"] > -15.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 60.0)
        and (last_candle["cti_20_1d"] > 0.80)
      ):
        return True, f"exit_{mode_name}_w_6_6"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] > 60.0)
        and (last_candle["r_480_1h"] > -25.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_4h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 50.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_6_7"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["cti_20_4h"] >= 0.70)
        and (last_candle["rsi_14_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.02)
      ):
        return True, f"exit_{mode_name}_w_6_8"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] >= 64.0)
        and (last_candle["cti_20_4h"] <= -0.50)
        and (last_candle["cti_20_1d"] >= 0.70)
      ):
        return True, f"exit_{mode_name}_w_6_9"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_6_10"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] >= 70.0)
        and (last_candle["rsi_14_1h"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["r_480_1h"] > -30.0)
      ):
        return True, f"exit_{mode_name}_w_6_11"
      elif (
        (last_candle["r_14"] >= -14.0)
        and (last_candle["rsi_14"] >= 68.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] > -30.0)
        and (last_candle["change_pct_1d"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_6_12"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.80)
        and (last_candle["rsi_14_4h"] >= 65.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_6_13"
      elif (
        (last_candle["r_14"] >= -18.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_1h"] >= 70.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
      ):
        return True, f"exit_{mode_name}_w_6_14"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
        and (last_candle["r_480_4h"] > -15.0)
        and (last_candle["change_pct_1h"] < -0.00)
      ):
        return True, f"exit_{mode_name}_w_6_15"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["top_wick_pct_1d"] > 0.16)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.80))
        and (last_candle["hl_pct_change_6_1d"] > 0.75)
      ):
        return True, f"exit_{mode_name}_w_6_16"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.80)
        and (last_candle["r_480_1h"] > -30.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["top_wick_pct_1d"] > 0.08)
      ):
        return True, f"exit_{mode_name}_w_6_17"
      elif (
        (last_candle["r_14"] >= -35.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.90)
      ):
        return True, f"exit_{mode_name}_w_6_18"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 80.0)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.50)
      ):
        return True, f"exit_{mode_name}_w_6_19"
    elif 0.08 > current_profit >= 0.07:
      if last_candle["r_480"] > -0.8:
        return True, f"exit_{mode_name}_w_7_1"
      elif (last_candle["r_14"] >= -1.0) and (last_candle["rsi_14"] > 76.0):
        return True, f"exit_{mode_name}_w_7_2"
      elif (last_candle["r_14"] >= -2.0) and (last_candle["rsi_14"] < 50.0):
        return True, f"exit_{mode_name}_w_7_3"
      elif (last_candle["r_14"] >= -5.0) and (last_candle["rsi_14"] > 72.0) and (last_candle["r_480_1h"] > -25.0):
        return True, f"exit_{mode_name}_w_7_4"
      elif (last_candle["r_14"] >= -4.0) and (last_candle["cti_20"] > 0.95):
        return True, f"exit_{mode_name}_w_7_5"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] > 60.0)
        and (last_candle["r_480_1h"] > -15.0)
        and (last_candle["r_480_4h"] > -10.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 60.0)
        and (last_candle["cti_20_1d"] > 0.80)
      ):
        return True, f"exit_{mode_name}_w_7_6"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] > 60.0)
        and (last_candle["r_480_1h"] > -25.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_4h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 50.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_7_7"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["cti_20_4h"] >= 0.70)
        and (last_candle["rsi_14_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.02)
      ):
        return True, f"exit_{mode_name}_w_7_8"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 72.0)
        and (last_candle["rsi_14_15m"] >= 66.0)
        and (last_candle["cti_20_4h"] <= -0.50)
        and (last_candle["cti_20_1d"] >= 0.70)
      ):
        return True, f"exit_{mode_name}_w_7_9"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 64.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_7_10"
      elif (
        (last_candle["r_14"] >= -3.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] >= 70.0)
        and (last_candle["rsi_14_1h"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["r_480_1h"] > -30.0)
      ):
        return True, f"exit_{mode_name}_w_7_11"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] > -30.0)
        and (last_candle["change_pct_1d"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_7_12"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.80)
        and (last_candle["rsi_14_4h"] >= 65.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_7_13"
      elif (
        (last_candle["r_14"] >= -16.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_1h"] >= 70.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
      ):
        return True, f"exit_{mode_name}_w_7_14"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
        and (last_candle["r_480_4h"] > -15.0)
        and (last_candle["change_pct_1h"] < -0.00)
      ):
        return True, f"exit_{mode_name}_w_7_15"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["top_wick_pct_1d"] > 0.16)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.80))
        and (last_candle["hl_pct_change_6_1d"] > 0.75)
      ):
        return True, f"exit_{mode_name}_w_7_16"
      elif (
        (last_candle["r_14"] >= -16.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.80)
        and (last_candle["r_480_1h"] > -30.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["top_wick_pct_1d"] > 0.08)
      ):
        return True, f"exit_{mode_name}_w_7_17"
      elif (
        (last_candle["r_14"] >= -25.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.90)
      ):
        return True, f"exit_{mode_name}_w_7_18"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 80.0)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.50)
      ):
        return True, f"exit_{mode_name}_w_7_19"
    elif 0.09 > current_profit >= 0.08:
      if last_candle["r_480"] > -0.9:
        return True, f"exit_{mode_name}_w_8_1"
      elif (last_candle["r_14"] >= -1.0) and (last_candle["rsi_14"] > 77.0):
        return True, f"exit_{mode_name}_w_8_2"
      elif (last_candle["r_14"] >= -2.0) and (last_candle["rsi_14"] < 48.0):
        return True, f"exit_{mode_name}_w_8_3"
      elif (last_candle["r_14"] >= -5.0) and (last_candle["rsi_14"] > 73.0) and (last_candle["r_480_1h"] > -25.0):
        return True, f"exit_{mode_name}_w_8_4"
      elif (last_candle["r_14"] >= -3.0) and (last_candle["cti_20"] > 0.95):
        return True, f"exit_{mode_name}_w_8_5"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] > 60.0)
        and (last_candle["r_480_1h"] > -15.0)
        and (last_candle["r_480_4h"] > -10.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 60.0)
        and (last_candle["cti_20_1d"] > 0.80)
      ):
        return True, f"exit_{mode_name}_w_8_6"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] > 60.0)
        and (last_candle["r_480_1h"] > -25.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_4h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 50.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_8_7"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["cti_20_4h"] >= 0.70)
        and (last_candle["rsi_14_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.02)
      ):
        return True, f"exit_{mode_name}_w_8_8"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 74.0)
        and (last_candle["rsi_14_15m"] >= 68.0)
        and (last_candle["cti_20_4h"] <= -0.50)
        and (last_candle["cti_20_1d"] >= 0.70)
      ):
        return True, f"exit_{mode_name}_w_8_9"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 66.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_8_10"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] >= 70.0)
        and (last_candle["rsi_14_1h"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["r_480_1h"] > -30.0)
      ):
        return True, f"exit_{mode_name}_w_8_11"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 72.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] > -30.0)
        and (last_candle["change_pct_1d"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_8_12"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.80)
        and (last_candle["rsi_14_4h"] >= 65.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_8_13"
      elif (
        (last_candle["r_14"] >= -14.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_1h"] >= 70.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
      ):
        return True, f"exit_{mode_name}_w_8_14"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
        and (last_candle["r_480_4h"] > -15.0)
        and (last_candle["change_pct_1h"] < -0.00)
      ):
        return True, f"exit_{mode_name}_w_8_15"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["top_wick_pct_1d"] > 0.16)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.80))
        and (last_candle["hl_pct_change_6_1d"] > 0.75)
      ):
        return True, f"exit_{mode_name}_w_8_16"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.80)
        and (last_candle["r_480_1h"] > -30.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["top_wick_pct_1d"] > 0.08)
      ):
        return True, f"exit_{mode_name}_w_8_17"
      elif (
        (last_candle["r_14"] >= -15.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.90)
      ):
        return True, f"exit_{mode_name}_w_8_18"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 80.0)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.50)
      ):
        return True, f"exit_{mode_name}_w_8_19"
    elif 0.1 > current_profit >= 0.09:
      if last_candle["r_480"] > -1.0:
        return True, f"exit_{mode_name}_w_9_1"
      elif (last_candle["r_14"] >= -1.0) and (last_candle["rsi_14"] > 78.0):
        return True, f"exit_{mode_name}_w_9_2"
      elif (last_candle["r_14"] >= -2.0) and (last_candle["rsi_14"] < 46.0):
        return True, f"exit_{mode_name}_w_9_3"
      elif (last_candle["r_14"] >= -5.0) and (last_candle["rsi_14"] > 74.0) and (last_candle["r_480_1h"] > -25.0):
        return True, f"exit_{mode_name}_w_9_4"
      elif (last_candle["r_14"] >= -2.0) and (last_candle["cti_20"] > 0.95):
        return True, f"exit_{mode_name}_w_9_5"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] > 60.0)
        and (last_candle["r_480_1h"] > -15.0)
        and (last_candle["r_480_4h"] > -10.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 60.0)
        and (last_candle["cti_20_1d"] > 0.80)
      ):
        return True, f"exit_{mode_name}_w_9_6"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] > 60.0)
        and (last_candle["r_480_1h"] > -25.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_4h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 50.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_9_7"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["cti_20_4h"] >= 0.70)
        and (last_candle["rsi_14_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.02)
      ):
        return True, f"exit_{mode_name}_w_9_8"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 76.0)
        and (last_candle["rsi_14_15m"] >= 70.0)
        and (last_candle["cti_20_4h"] <= -0.50)
        and (last_candle["cti_20_1d"] >= 0.70)
      ):
        return True, f"exit_{mode_name}_w_9_9"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 68.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_9_10"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] >= 70.0)
        and (last_candle["rsi_14_1h"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["r_480_1h"] > -30.0)
      ):
        return True, f"exit_{mode_name}_w_9_11"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 74.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] > -30.0)
        and (last_candle["change_pct_1d"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_9_12"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.80)
        and (last_candle["rsi_14_4h"] >= 65.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_9_13"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_1h"] >= 70.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
      ):
        return True, f"exit_{mode_name}_w_9_14"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
        and (last_candle["r_480_4h"] > -15.0)
        and (last_candle["change_pct_1h"] < -0.00)
      ):
        return True, f"exit_{mode_name}_w_9_15"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["top_wick_pct_1d"] > 0.16)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.80))
        and (last_candle["hl_pct_change_6_1d"] > 0.75)
      ):
        return True, f"exit_{mode_name}_w_9_16"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.80)
        and (last_candle["r_480_1h"] > -30.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["top_wick_pct_1d"] > 0.08)
      ):
        return True, f"exit_{mode_name}_w_9_17"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.90)
      ):
        return True, f"exit_{mode_name}_w_9_18"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 80.0)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.50)
      ):
        return True, f"exit_{mode_name}_w_9_19"
    elif 0.12 > current_profit >= 0.1:
      if last_candle["r_480"] > -1.1:
        return True, f"exit_{mode_name}_w_10_1"
      elif (last_candle["r_14"] >= -1.0) and (last_candle["rsi_14"] > 79.0):
        return True, f"exit_{mode_name}_w_10_2"
      elif (last_candle["r_14"] >= -2.0) and (last_candle["rsi_14"] < 44.0):
        return True, f"exit_{mode_name}_w_10_3"
      elif (last_candle["r_14"] >= -5.0) and (last_candle["rsi_14"] > 75.0) and (last_candle["r_480_1h"] > -25.0):
        return True, f"exit_{mode_name}_w_10_4"
      elif (last_candle["r_14"] >= -1.0) and (last_candle["cti_20"] > 0.95):
        return True, f"exit_{mode_name}_w_10_5"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] > 65.0)
        and (last_candle["r_480_1h"] > -10.0)
        and (last_candle["r_480_4h"] > -5.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 60.0)
        and (last_candle["cti_20_1d"] > 0.80)
      ):
        return True, f"exit_{mode_name}_w_10_6"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] > 65.0)
        and (last_candle["r_480_1h"] > -25.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_4h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 50.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_10_7"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 75.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["cti_20_4h"] >= 0.70)
        and (last_candle["rsi_14_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.02)
      ):
        return True, f"exit_{mode_name}_w_10_8"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 78.0)
        and (last_candle["rsi_14_15m"] >= 72.0)
        and (last_candle["cti_20_4h"] <= -0.50)
        and (last_candle["cti_20_1d"] >= 0.70)
      ):
        return True, f"exit_{mode_name}_w_10_9"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_10_10"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 72.0)
        and (last_candle["rsi_14_15m"] >= 72.0)
        and (last_candle["rsi_14_1h"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["r_480_1h"] > -30.0)
      ):
        return True, f"exit_{mode_name}_w_10_11"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 76.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] > -30.0)
        and (last_candle["change_pct_1d"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_10_12"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.80)
        and (last_candle["rsi_14_4h"] >= 65.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_10_13"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_1h"] >= 70.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
      ):
        return True, f"exit_{mode_name}_w_10_14"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
        and (last_candle["r_480_4h"] > -15.0)
        and (last_candle["change_pct_1h"] < -0.00)
      ):
        return True, f"exit_{mode_name}_w_10_15"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["top_wick_pct_1d"] > 0.16)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.80))
        and (last_candle["hl_pct_change_6_1d"] > 0.75)
      ):
        return True, f"exit_{mode_name}_w_10_16"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.80)
        and (last_candle["r_480_1h"] > -30.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["top_wick_pct_1d"] > 0.08)
      ):
        return True, f"exit_{mode_name}_w_10_17"
      elif (
        (last_candle["r_14"] >= -5.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.90)
      ):
        return True, f"exit_{mode_name}_w_10_18"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 80.0)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.50)
      ):
        return True, f"exit_{mode_name}_w_10_19"
    elif 0.2 > current_profit >= 0.12:
      if last_candle["r_480"] > -0.4:
        return True, f"exit_{mode_name}_w_11_1"
      elif (last_candle["r_14"] >= -1.0) and (last_candle["rsi_14"] > 80.0):
        return True, f"exit_{mode_name}_w_11_2"
      elif (last_candle["r_14"] >= -2.0) and (last_candle["rsi_14"] < 42.0):
        return True, f"exit_{mode_name}_w_11_3"
      elif (last_candle["r_14"] >= -5.0) and (last_candle["rsi_14"] > 76.0) and (last_candle["r_480_1h"] > -25.0):
        return True, f"exit_{mode_name}_w_11_4"
      elif (last_candle["r_14"] >= -0.5) and (last_candle["cti_20"] > 0.95):
        return True, f"exit_{mode_name}_w_11_5"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] > 70.0)
        and (last_candle["r_480_1h"] > -10.0)
        and (last_candle["r_480_4h"] > -5.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 60.0)
        and (last_candle["cti_20_1d"] > 0.80)
      ):
        return True, f"exit_{mode_name}_w_11_6"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] > 70.0)
        and (last_candle["r_480_1h"] > -25.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_4h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 50.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_11_7"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 76.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["cti_20_4h"] >= 0.70)
        and (last_candle["rsi_14_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.02)
      ):
        return True, f"exit_{mode_name}_w_11_8"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 78.0)
        and (last_candle["rsi_14_15m"] >= 74.0)
        and (last_candle["cti_20_4h"] <= -0.50)
        and (last_candle["cti_20_1d"] >= 0.70)
      ):
        return True, f"exit_{mode_name}_w_11_9"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 72.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_11_10"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 74.0)
        and (last_candle["rsi_14_15m"] >= 74.0)
        and (last_candle["rsi_14_1h"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["r_480_1h"] > -30.0)
      ):
        return True, f"exit_{mode_name}_w_11_11"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 78.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] > -30.0)
        and (last_candle["change_pct_1d"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_11_12"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.80)
        and (last_candle["rsi_14_4h"] >= 65.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_11_13"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_1h"] >= 70.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
      ):
        return True, f"exit_{mode_name}_w_11_14"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
        and (last_candle["r_480_4h"] > -15.0)
        and (last_candle["change_pct_1h"] < -0.00)
      ):
        return True, f"exit_{mode_name}_w_11_15"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["top_wick_pct_1d"] > 0.16)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.80))
        and (last_candle["hl_pct_change_6_1d"] > 0.75)
      ):
        return True, f"exit_{mode_name}_w_11_16"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.80)
        and (last_candle["r_480_1h"] > -30.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["top_wick_pct_1d"] > 0.08)
      ):
        return True, f"exit_{mode_name}_w_11_17"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.90)
      ):
        return True, f"exit_{mode_name}_w_11_18"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 65.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 80.0)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.50)
      ):
        return True, f"exit_{mode_name}_w_11_19"
    elif current_profit >= 0.2:
      if last_candle["r_480"] > -0.2:
        return True, f"exit_{mode_name}_w_12_1"
      elif (last_candle["r_14"] >= -1.0) and (last_candle["rsi_14"] > 81.0):
        return True, f"exit_{mode_name}_w_12_2"
      elif (last_candle["r_14"] >= -2.0) and (last_candle["rsi_14"] < 40.0):
        return True, f"exit_{mode_name}_w_12_3"
      elif (last_candle["r_14"] >= -5.0) and (last_candle["rsi_14"] > 77.0) and (last_candle["r_480_1h"] > -25.0):
        return True, f"exit_{mode_name}_w_12_4"
      elif (last_candle["r_14"] >= -0.1) and (last_candle["cti_20"] > 0.95):
        return True, f"exit_{mode_name}_w_12_5"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] > 75.0)
        and (last_candle["r_480_1h"] > -5.0)
        and (last_candle["r_480_4h"] > -5.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 60.0)
        and (last_candle["cti_20_1d"] > 0.80)
      ):
        return True, f"exit_{mode_name}_w_12_6"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] > 75.0)
        and (last_candle["r_480_1h"] > -25.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["rsi_14_1h"] > 50.0)
        and (last_candle["rsi_14_4h"] > 50.0)
        and (last_candle["rsi_14_1d"] > 50.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_12_7"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 78.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["cti_20_4h"] >= 0.70)
        and (last_candle["rsi_14_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.02)
      ):
        return True, f"exit_{mode_name}_w_12_8"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 80.0)
        and (last_candle["rsi_14_15m"] >= 76.0)
        and (last_candle["cti_20_4h"] <= -0.50)
        and (last_candle["cti_20_1d"] >= 0.70)
      ):
        return True, f"exit_{mode_name}_w_12_9"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 74.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_12_10"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 76.0)
        and (last_candle["rsi_14_15m"] >= 76.0)
        and (last_candle["rsi_14_1h"] >= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["r_480_1h"] > -30.0)
      ):
        return True, f"exit_{mode_name}_w_12_11"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 80.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] > -30.0)
        and (last_candle["change_pct_1d"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_12_12"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.80)
        and (last_candle["rsi_14_4h"] >= 65.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.01)
      ):
        return True, f"exit_{mode_name}_w_12_13"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_1h"] >= 70.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
      ):
        return True, f"exit_{mode_name}_w_12_14"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_4h"] >= 75.0)
        and (last_candle["r_480_4h"] > -15.0)
        and (last_candle["change_pct_1h"] < -0.00)
      ):
        return True, f"exit_{mode_name}_w_12_15"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["top_wick_pct_1d"] > 0.16)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.80))
        and (last_candle["hl_pct_change_6_1d"] > 0.75)
      ):
        return True, f"exit_{mode_name}_w_12_16"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["cti_20_1d"] >= 0.80)
        and (last_candle["r_480_1h"] > -30.0)
        and (last_candle["r_480_4h"] > -25.0)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["top_wick_pct_1d"] > 0.08)
      ):
        return True, f"exit_{mode_name}_w_12_17"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.90)
      ):
        return True, f"exit_{mode_name}_w_12_18"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 80.0)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["hl_pct_change_48_1h"] > 0.50)
      ):
        return True, f"exit_{mode_name}_w_12_19"

    return False, None

  def exit_long_dec(
    self,
    mode_name: str,
    current_profit: float,
    max_profit: float,
    max_loss: float,
    last_candle,
    previous_candle_1,
    previous_candle_2,
    previous_candle_3,
    previous_candle_4,
    previous_candle_5,
    trade: "Trade",
    current_time: "datetime",
    buy_tag,
  ) -> tuple:
    if 0.01 > current_profit >= 0.001:
      if (
        (last_candle["r_14"] > -1.0)
        and (last_candle["rsi_14"] > 70.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_0_1"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_0_2"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["cti_20_1d"] >= 0.50)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["change_pct_4h"] <= -0.02)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_0_3"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 90.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] < -75.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1h"] < -0.03)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_0_4"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 98.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["change_pct_1d"] < -0.02)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_0_5"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 95.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.7)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_0_6"
      elif (
        (last_candle["rsi_3"] >= 98.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.5)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_24"] == True)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_0_7"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["r_480_4h"] < -90.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_0_8"
      elif (
        (last_candle["rsi_14"] <= 20.0)
        and (last_candle["rsi_3_1h"] <= 10.0)
        and (last_candle["rsi_3_4h"] <= 6.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_0_9"
      elif (
        (last_candle["rsi_14"] <= 20.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_1h"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_0_10"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1d"] < -0.10)
        and (last_candle["not_downtrend_4h"] == False)
      ):
        return True, f"exit_{mode_name}_d_0_11"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_0_12"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["change_pct_4h"] < -0.03)
        and (last_candle["top_wick_pct_4h"] > 0.03)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_0_13"
      elif (
        (last_candle["rsi_14"] <= 30.0)
        and (last_candle["rsi_14_15m"] <= 40.0)
        and (last_candle["rsi_3_1d"] <= 6.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_0_14"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 74.0)
        and (last_candle["rsi_3_1h"] <= 20.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_0_15"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] >= 0.70)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_0_16"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] >= 60.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] < -70.0)
        and (last_candle["change_pct_1d"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_0_17"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_0_18"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_max_6_1d"] >= 85.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.06)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_0_19"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_max_6_1d"] >= 70.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_0_20"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["close"] < (last_candle["high_max_48_1h"] * 0.75))
      ):
        return True, f"exit_{mode_name}_d_0_21"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.01)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.70))
      ):
        return True, f"exit_{mode_name}_d_0_22"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_0_23"
    elif 0.02 > current_profit >= 0.01:
      if (
        (last_candle["r_14"] > -10.0)
        and (last_candle["rsi_14"] > 66.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_1_1"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_1_2"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["cti_20_1d"] >= 0.50)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["change_pct_4h"] <= -0.02)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_1_3"
      elif (
        (last_candle["r_14"] >= -40.0)
        and (last_candle["rsi_3"] >= 80.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] < -75.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1h"] < -0.03)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_1_4"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_3"] >= 90.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["change_pct_1d"] < -0.02)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_1_5"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_3"] >= 80.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.7)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_1_6"
      elif (
        (last_candle["rsi_3"] >= 60.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.5)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_24"] == True)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_1_7"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 68.0)
        and (last_candle["r_480_4h"] < -90.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_1_8"
      elif (
        (last_candle["rsi_14"] <= 30.0)
        and (last_candle["rsi_3_1h"] <= 10.0)
        and (last_candle["rsi_3_4h"] <= 6.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_1_9"
      elif (
        (last_candle["rsi_14"] <= 46.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_1h"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_1_10"
      elif (
        (last_candle["r_14"] >= -5.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1d"] < -0.10)
        and (last_candle["not_downtrend_4h"] == False)
      ):
        return True, f"exit_{mode_name}_d_1_11"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_1_12"
      elif (
        (last_candle["r_14"] >= -15.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["change_pct_4h"] < -0.03)
        and (last_candle["top_wick_pct_4h"] > 0.03)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_1_13"
      elif (
        (last_candle["rsi_14"] <= 46.0)
        and (last_candle["rsi_14_15m"] <= 40.0)
        and (last_candle["rsi_3_1d"] <= 6.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_1_14"
      elif (
        (last_candle["r_14"] >= -5.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_3_1h"] <= 20.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_1_15"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] >= 0.70)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_1_16"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] >= 60.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] < -70.0)
        and (last_candle["change_pct_1d"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_1_17"
      elif (
        (last_candle["r_14"] >= -60.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_1_18"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_max_6_1d"] >= 85.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.06)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_1_19"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_max_6_1d"] >= 70.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_1_20"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["close"] < (last_candle["high_max_48_1h"] * 0.75))
      ):
        return True, f"exit_{mode_name}_d_1_21"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.01)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.70))
      ):
        return True, f"exit_{mode_name}_d_1_22"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_1_23"
    elif 0.03 > current_profit >= 0.02:
      if (
        (last_candle["r_14"] > -16.0)
        and (last_candle["rsi_14"] > 56.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_2_1"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_2_2"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["cti_20_1d"] >= 0.50)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["change_pct_4h"] <= -0.02)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_2_3"
      elif (
        (last_candle["r_14"] >= -40.0)
        and (last_candle["rsi_3"] >= 80.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] < -75.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1h"] < -0.03)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_2_4"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_3"] >= 90.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["change_pct_1d"] < -0.02)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_2_5"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_3"] >= 70.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.7)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_2_6"
      elif (
        (last_candle["rsi_3"] >= 60.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.5)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_24"] == True)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_2_7"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 66.0)
        and (last_candle["r_480_4h"] < -90.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_2_8"
      elif (
        (last_candle["rsi_14"] <= 40.0)
        and (last_candle["rsi_3_1h"] <= 10.0)
        and (last_candle["rsi_3_4h"] <= 6.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_2_9"
      elif (
        (last_candle["rsi_14"] <= 48.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_1h"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_2_10"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1d"] < -0.10)
        and (last_candle["not_downtrend_4h"] == False)
      ):
        return True, f"exit_{mode_name}_d_2_11"
      elif (
        (last_candle["r_14"] >= -9.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_2_12"
      elif (
        (last_candle["r_14"] >= -15.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["change_pct_4h"] < -0.03)
        and (last_candle["top_wick_pct_4h"] > 0.03)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_2_13"
      elif (
        (last_candle["rsi_14"] <= 48.0)
        and (last_candle["rsi_14_15m"] <= 40.0)
        and (last_candle["rsi_3_1d"] <= 6.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_2_14"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 68.0)
        and (last_candle["rsi_3_1h"] <= 20.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_2_15"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] >= 0.70)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_2_16"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] >= 60.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] < -70.0)
        and (last_candle["change_pct_1d"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_2_17"
      elif (
        (last_candle["r_14"] >= -60.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_2_18"
      elif (
        (last_candle["r_14"] >= -40.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_max_6_1d"] >= 85.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.06)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_2_19"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_max_6_1d"] >= 70.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_2_20"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["close"] < (last_candle["high_max_48_1h"] * 0.75))
      ):
        return True, f"exit_{mode_name}_d_2_21"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.01)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.70))
      ):
        return True, f"exit_{mode_name}_d_2_22"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_2_23"
    elif 0.04 > current_profit >= 0.03:
      if (
        (last_candle["r_14"] > -16.0)
        and (last_candle["rsi_14"] > 54.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_3_1"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_3_2"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["cti_20_1d"] >= 0.50)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["change_pct_4h"] <= -0.02)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_3_3"
      elif (
        (last_candle["r_14"] >= -40.0)
        and (last_candle["rsi_3"] >= 80.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] < -75.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1h"] < -0.03)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_3_4"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_3"] >= 90.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["change_pct_1d"] < -0.02)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_3_5"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_3"] >= 70.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.7)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_3_6"
      elif (
        (last_candle["rsi_3"] >= 60.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.5)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_24"] == True)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_3_7"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 64.0)
        and (last_candle["r_480_4h"] < -90.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_3_8"
      elif (
        (last_candle["rsi_14"] <= 42.0)
        and (last_candle["rsi_3_1h"] <= 10.0)
        and (last_candle["rsi_3_4h"] <= 6.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_3_9"
      elif (
        (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_1h"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_3_10"
      elif (
        (last_candle["r_14"] >= -7.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1d"] < -0.10)
        and (last_candle["not_downtrend_4h"] == False)
      ):
        return True, f"exit_{mode_name}_d_3_11"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_3_12"
      elif (
        (last_candle["r_14"] >= -15.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["change_pct_4h"] < -0.03)
        and (last_candle["top_wick_pct_4h"] > 0.03)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_3_13"
      elif (
        (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 40.0)
        and (last_candle["rsi_3_1d"] <= 6.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_3_14"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_14"] >= 66.0)
        and (last_candle["rsi_3_1h"] <= 20.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_3_15"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] >= 0.70)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_3_16"
      elif (
        (last_candle["r_14"] >= -16.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] >= 60.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] < -70.0)
        and (last_candle["change_pct_1d"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_3_17"
      elif (
        (last_candle["r_14"] >= -60.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_3_18"
      elif (
        (last_candle["r_14"] >= -40.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_max_6_1d"] >= 85.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.06)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_3_19"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_max_6_1d"] >= 70.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_3_20"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["close"] < (last_candle["high_max_48_1h"] * 0.75))
      ):
        return True, f"exit_{mode_name}_d_3_21"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.01)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.70))
      ):
        return True, f"exit_{mode_name}_d_3_22"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_3_23"
    elif 0.05 > current_profit >= 0.04:
      if (
        (last_candle["r_14"] > -16.0)
        and (last_candle["rsi_14"] > 52.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_4_1"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_4_2"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["cti_20_1d"] >= 0.50)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["change_pct_4h"] <= -0.02)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_4_3"
      elif (
        (last_candle["r_14"] >= -40.0)
        and (last_candle["rsi_3"] >= 80.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] < -75.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1h"] < -0.03)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_4_4"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_3"] >= 90.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["change_pct_1d"] < -0.02)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_4_5"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_3"] >= 70.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.7)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_4_6"
      elif (
        (last_candle["rsi_3"] >= 60.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.5)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_24"] == True)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_4_7"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 62.0)
        and (last_candle["r_480_4h"] < -90.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_4_8"
      elif (
        (last_candle["rsi_14"] <= 44.0)
        and (last_candle["rsi_3_1h"] <= 10.0)
        and (last_candle["rsi_3_4h"] <= 6.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_4_9"
      elif (
        (last_candle["rsi_14"] <= 52.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_1h"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_4_10"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1d"] < -0.10)
        and (last_candle["not_downtrend_4h"] == False)
      ):
        return True, f"exit_{mode_name}_d_4_11"
      elif (
        (last_candle["r_14"] >= -11.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_4_12"
      elif (
        (last_candle["r_14"] >= -15.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["change_pct_4h"] < -0.03)
        and (last_candle["top_wick_pct_4h"] > 0.03)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_4_13"
      elif (
        (last_candle["rsi_14"] <= 52.0)
        and (last_candle["rsi_14_15m"] <= 40.0)
        and (last_candle["rsi_3_1d"] <= 6.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_4_14"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14"] >= 64.0)
        and (last_candle["rsi_3_1h"] <= 20.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_4_15"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] >= 0.70)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_4_16"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] >= 60.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] < -70.0)
        and (last_candle["change_pct_1d"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_4_17"
      elif (
        (last_candle["r_14"] >= -60.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_4_18"
      elif (
        (last_candle["r_14"] >= -40.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_max_6_1d"] >= 85.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.06)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_4_19"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_max_6_1d"] >= 70.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_4_20"
      elif (
        (last_candle["r_14"] >= -14.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["close"] < (last_candle["high_max_48_1h"] * 0.75))
      ):
        return True, f"exit_{mode_name}_d_4_21"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.01)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.70))
      ):
        return True, f"exit_{mode_name}_d_4_22"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_4_23"
    elif 0.06 > current_profit >= 0.05:
      if (
        (last_candle["r_14"] > -16.0)
        and (last_candle["rsi_14"] > 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_5_1"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_5_2"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["cti_20_1d"] >= 0.50)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["change_pct_4h"] <= -0.02)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_5_3"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 90.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] < -75.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1h"] < -0.03)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_5_4"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_3"] >= 90.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["change_pct_1d"] < -0.02)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_5_5"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_3"] >= 70.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.7)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_5_6"
      elif (
        (last_candle["rsi_3"] >= 60.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.5)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_24"] == True)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_5_7"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["r_480_4h"] < -90.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_5_8"
      elif (
        (last_candle["rsi_14"] <= 46.0)
        and (last_candle["rsi_3_1h"] <= 10.0)
        and (last_candle["rsi_3_4h"] <= 6.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_5_9"
      elif (
        (last_candle["rsi_14"] <= 54.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_1h"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_5_10"
      elif (
        (last_candle["r_14"] >= -9.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1d"] < -0.10)
        and (last_candle["not_downtrend_4h"] == False)
      ):
        return True, f"exit_{mode_name}_d_5_11"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_5_12"
      elif (
        (last_candle["r_14"] >= -15.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["change_pct_4h"] < -0.03)
        and (last_candle["top_wick_pct_4h"] > 0.03)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_5_13"
      elif (
        (last_candle["rsi_14"] <= 54.0)
        and (last_candle["rsi_14_15m"] <= 40.0)
        and (last_candle["rsi_3_1d"] <= 6.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_5_14"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_3_1h"] <= 20.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_5_15"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] >= 0.70)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_5_16"
      elif (
        (last_candle["r_14"] >= -24.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] >= 60.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] < -70.0)
        and (last_candle["change_pct_1d"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_5_17"
      elif (
        (last_candle["r_14"] >= -60.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_5_18"
      elif (
        (last_candle["r_14"] >= -40.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_max_6_1d"] >= 85.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.06)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_5_19"
      elif (
        (last_candle["r_14"] >= -30.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_max_6_1d"] >= 70.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_5_20"
      elif (
        (last_candle["r_14"] >= -16.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["close"] < (last_candle["high_max_48_1h"] * 0.75))
      ):
        return True, f"exit_{mode_name}_d_5_21"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.01)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.70))
      ):
        return True, f"exit_{mode_name}_d_5_22"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_5_23"
    elif 0.07 > current_profit >= 0.06:
      if (
        (last_candle["r_14"] > -16.0)
        and (last_candle["rsi_14"] > 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_6_1"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_6_2"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["cti_20_1d"] >= 0.50)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["change_pct_4h"] <= -0.02)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_6_3"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 90.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] < -75.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1h"] < -0.03)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_6_4"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_3"] >= 94.0)
        and (last_candle["rsi_14"] >= 64.0)
        and (last_candle["change_pct_1d"] < -0.02)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_6_5"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_3"] >= 80.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.7)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_6_6"
      elif (
        (last_candle["rsi_3"] >= 90.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.5)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_24"] == True)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_6_7"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 62.0)
        and (last_candle["r_480_4h"] < -90.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_6_8"
      elif (
        (last_candle["rsi_14"] <= 44.0)
        and (last_candle["rsi_3_1h"] <= 10.0)
        and (last_candle["rsi_3_4h"] <= 6.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_6_9"
      elif (
        (last_candle["rsi_14"] <= 52.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_1h"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_6_10"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1d"] < -0.10)
        and (last_candle["not_downtrend_4h"] == False)
      ):
        return True, f"exit_{mode_name}_d_6_11"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 66.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_6_12"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["change_pct_4h"] < -0.03)
        and (last_candle["top_wick_pct_4h"] > 0.03)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_6_13"
      elif (
        (last_candle["rsi_14"] <= 52.0)
        and (last_candle["rsi_14_15m"] <= 40.0)
        and (last_candle["rsi_3_1d"] <= 6.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_6_14"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_14"] >= 64.0)
        and (last_candle["rsi_3_1h"] <= 20.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_6_15"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 66.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] >= 0.70)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_6_16"
      elif (
        (last_candle["r_14"] >= -22.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] >= 60.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] < -70.0)
        and (last_candle["change_pct_1d"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_6_17"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_6_18"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_max_6_1d"] >= 85.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.06)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_6_19"
      elif (
        (last_candle["r_14"] >= -20.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_max_6_1d"] >= 70.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_6_20"
      elif (
        (last_candle["r_14"] >= -14.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["close"] < (last_candle["high_max_48_1h"] * 0.75))
      ):
        return True, f"exit_{mode_name}_d_6_21"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.01)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.70))
      ):
        return True, f"exit_{mode_name}_d_6_22"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_6_23"
    elif 0.08 > current_profit >= 0.07:
      if (
        (last_candle["r_14"] > -16.0)
        and (last_candle["rsi_14"] > 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_7_1"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_7_2"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["cti_20_1d"] >= 0.50)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["change_pct_4h"] <= -0.02)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_7_3"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 90.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] < -75.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1h"] < -0.03)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_7_4"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_3"] >= 96.0)
        and (last_candle["rsi_14"] >= 68.0)
        and (last_candle["change_pct_1d"] < -0.02)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_7_5"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_3"] >= 90.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.7)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_7_6"
      elif (
        (last_candle["rsi_3"] >= 95.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.5)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_24"] == True)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_7_7"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 64.0)
        and (last_candle["r_480_4h"] < -90.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_7_8"
      elif (
        (last_candle["rsi_14"] <= 42.0)
        and (last_candle["rsi_3_1h"] <= 10.0)
        and (last_candle["rsi_3_4h"] <= 6.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_7_9"
      elif (
        (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_1h"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_7_10"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1d"] < -0.10)
        and (last_candle["not_downtrend_4h"] == False)
      ):
        return True, f"exit_{mode_name}_d_7_11"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 68.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_7_12"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["change_pct_4h"] < -0.03)
        and (last_candle["top_wick_pct_4h"] > 0.03)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_7_13"
      elif (
        (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 40.0)
        and (last_candle["rsi_3_1d"] <= 6.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_7_14"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 66.0)
        and (last_candle["rsi_3_1h"] <= 20.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_7_15"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 68.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] >= 0.70)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_7_16"
      elif (
        (last_candle["r_14"] >= -14.0)
        and (last_candle["rsi_14"] >= 62.0)
        and (last_candle["rsi_14_15m"] >= 60.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] < -70.0)
        and (last_candle["change_pct_1d"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_7_17"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_7_18"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_max_6_1d"] >= 85.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.06)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_7_19"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_max_6_1d"] >= 70.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_7_20"
      elif (
        (last_candle["r_14"] >= -12.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["close"] < (last_candle["high_max_48_1h"] * 0.75))
      ):
        return True, f"exit_{mode_name}_d_7_21"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.01)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.70))
      ):
        return True, f"exit_{mode_name}_d_7_22"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_7_23"
    elif 0.09 > current_profit >= 0.08:
      if (
        (last_candle["r_14"] > -16.0)
        and (last_candle["rsi_14"] > 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_8_1"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_8_2"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["cti_20_1d"] >= 0.50)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["change_pct_4h"] <= -0.02)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_8_3"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 90.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] < -75.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1h"] < -0.03)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_8_4"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 98.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["change_pct_1d"] < -0.02)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_8_5"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_3"] >= 95.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.7)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_8_6"
      elif (
        (last_candle["rsi_3"] >= 98.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.5)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_24"] == True)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_8_7"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 66.0)
        and (last_candle["r_480_4h"] < -90.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_8_8"
      elif (
        (last_candle["rsi_14"] <= 40.0)
        and (last_candle["rsi_3_1h"] <= 10.0)
        and (last_candle["rsi_3_4h"] <= 6.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_8_9"
      elif (
        (last_candle["rsi_14"] <= 48.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_1h"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_8_10"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 64.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1d"] < -0.10)
        and (last_candle["not_downtrend_4h"] == False)
      ):
        return True, f"exit_{mode_name}_d_8_11"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_8_12"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["change_pct_4h"] < -0.03)
        and (last_candle["top_wick_pct_4h"] > 0.03)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_8_13"
      elif (
        (last_candle["rsi_14"] <= 48.0)
        and (last_candle["rsi_14_15m"] <= 40.0)
        and (last_candle["rsi_3_1d"] <= 6.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_8_14"
      elif (
        (last_candle["r_14"] >= -5.0)
        and (last_candle["rsi_14"] >= 68.0)
        and (last_candle["rsi_3_1h"] <= 20.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_8_15"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] >= 0.70)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_8_16"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 64.0)
        and (last_candle["rsi_14_15m"] >= 60.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] < -70.0)
        and (last_candle["change_pct_1d"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_8_17"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_8_18"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_max_6_1d"] >= 85.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.06)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_8_19"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_max_6_1d"] >= 70.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_8_20"
      elif (
        (last_candle["r_14"] >= -10.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["close"] < (last_candle["high_max_48_1h"] * 0.75))
      ):
        return True, f"exit_{mode_name}_d_8_21"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.01)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.70))
      ):
        return True, f"exit_{mode_name}_d_8_22"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_8_23"
    elif 0.1 > current_profit >= 0.09:
      if (
        (last_candle["r_14"] > -16.0)
        and (last_candle["rsi_14"] > 52.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_9_1"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_9_2"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["cti_20_1d"] >= 0.50)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["change_pct_4h"] <= -0.02)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_9_3"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 90.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] < -75.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1h"] < -0.03)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_9_4"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 98.0)
        and (last_candle["rsi_14"] >= 74.0)
        and (last_candle["change_pct_1d"] < -0.02)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_9_5"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 98.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.7)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_9_6"
      elif (
        (last_candle["rsi_3"] >= 99.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.5)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_24"] == True)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_9_7"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 68.0)
        and (last_candle["r_480_4h"] < -90.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_9_8"
      elif (
        (last_candle["rsi_14"] <= 38.0)
        and (last_candle["rsi_3_1h"] <= 10.0)
        and (last_candle["rsi_3_4h"] <= 6.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_9_9"
      elif (
        (last_candle["rsi_14"] <= 46.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_1h"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_9_10"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 64.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1d"] < -0.10)
        and (last_candle["not_downtrend_4h"] == False)
      ):
        return True, f"exit_{mode_name}_d_9_11"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 74.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_9_12"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["change_pct_4h"] < -0.03)
        and (last_candle["top_wick_pct_4h"] > 0.03)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_9_13"
      elif (
        (last_candle["rsi_14"] <= 46.0)
        and (last_candle["rsi_14_15m"] <= 40.0)
        and (last_candle["rsi_3_1d"] <= 6.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_9_14"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_3_1h"] <= 20.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_9_15"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 72.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] >= 0.70)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_9_16"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 66.0)
        and (last_candle["rsi_14_15m"] >= 60.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] < -70.0)
        and (last_candle["change_pct_1d"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_9_17"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_9_18"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_max_6_1d"] >= 85.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.06)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_9_19"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_max_6_1d"] >= 70.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_9_20"
      elif (
        (last_candle["r_14"] >= -8.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["close"] < (last_candle["high_max_48_1h"] * 0.75))
      ):
        return True, f"exit_{mode_name}_d_9_21"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.01)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.70))
      ):
        return True, f"exit_{mode_name}_d_9_22"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_9_23"
    elif 0.12 > current_profit >= 0.1:
      if (
        (last_candle["r_14"] > -16.0)
        and (last_candle["rsi_14"] > 54.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_10_1"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_10_2"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["cti_20_1d"] >= 0.50)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["change_pct_4h"] <= -0.02)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_10_3"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 90.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] < -75.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1h"] < -0.03)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_10_4"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 98.0)
        and (last_candle["rsi_14"] >= 76.0)
        and (last_candle["change_pct_1d"] < -0.02)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_10_5"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 98.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.8)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_10_6"
      elif (
        (last_candle["rsi_3"] >= 99.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.5)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_24"] == True)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_10_7"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["r_480_4h"] < -90.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_10_8"
      elif (
        (last_candle["rsi_14"] <= 36.0)
        and (last_candle["rsi_3_1h"] <= 10.0)
        and (last_candle["rsi_3_4h"] <= 6.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_10_9"
      elif (
        (last_candle["rsi_14"] <= 44.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_1h"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_10_10"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 68.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1d"] < -0.10)
        and (last_candle["not_downtrend_4h"] == False)
      ):
        return True, f"exit_{mode_name}_d_10_11"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 76.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_10_12"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["change_pct_4h"] < -0.03)
        and (last_candle["top_wick_pct_4h"] > 0.03)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_10_13"
      elif (
        (last_candle["rsi_14"] <= 44.0)
        and (last_candle["rsi_14_15m"] <= 40.0)
        and (last_candle["rsi_3_1d"] <= 6.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_10_14"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 72.0)
        and (last_candle["rsi_3_1h"] <= 20.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_10_15"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 74.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] >= 0.70)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_10_16"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 68.0)
        and (last_candle["rsi_14_15m"] >= 60.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] < -70.0)
        and (last_candle["change_pct_1d"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_10_17"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_10_18"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_max_6_1d"] >= 85.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.06)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_10_19"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_max_6_1d"] >= 70.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_10_20"
      elif (
        (last_candle["r_14"] >= -6.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["close"] < (last_candle["high_max_48_1h"] * 0.75))
      ):
        return True, f"exit_{mode_name}_d_10_21"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.01)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.70))
      ):
        return True, f"exit_{mode_name}_d_10_22"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_10_23"
    elif 0.2 > current_profit >= 0.12:
      if (
        (last_candle["r_14"] > -16.0)
        and (last_candle["rsi_14"] > 56.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_11_1"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_11_2"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["cti_20_1d"] >= 0.50)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["change_pct_4h"] <= -0.02)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_11_3"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 90.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] < -75.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 60.0)
        and (last_candle["change_pct_1h"] < -0.03)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_11_4"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 98.0)
        and (last_candle["rsi_14"] >= 78.0)
        and (last_candle["change_pct_1d"] < -0.02)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_11_5"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 98.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.8)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_11_6"
      elif (
        (last_candle["rsi_3"] >= 99.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.5)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_24"] == True)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_11_7"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 74.0)
        and (last_candle["r_480_4h"] < -90.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_11_8"
      elif (
        (last_candle["rsi_14"] <= 34.0)
        and (last_candle["rsi_3_1h"] <= 10.0)
        and (last_candle["rsi_3_4h"] <= 6.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_11_9"
      elif (
        (last_candle["rsi_14"] <= 40.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_1h"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_11_10"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1d"] < -0.10)
        and (last_candle["not_downtrend_4h"] == False)
      ):
        return True, f"exit_{mode_name}_d_11_11"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 78.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_11_12"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["change_pct_4h"] < -0.03)
        and (last_candle["top_wick_pct_4h"] > 0.03)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_11_13"
      elif (
        (last_candle["rsi_14"] <= 42.0)
        and (last_candle["rsi_14_15m"] <= 40.0)
        and (last_candle["rsi_3_1d"] <= 6.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_11_14"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 74.0)
        and (last_candle["rsi_3_1h"] <= 20.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_11_15"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 76.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] >= 0.70)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_11_16"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["rsi_14_15m"] >= 60.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] < -70.0)
        and (last_candle["change_pct_1d"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_11_17"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_11_18"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_max_6_1d"] >= 85.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.06)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_11_19"
      elif (
        (last_candle["r_14"] >= -2.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_max_6_1d"] >= 70.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_11_20"
      elif (
        (last_candle["r_14"] >= -4.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["close"] < (last_candle["high_max_48_1h"] * 0.75))
      ):
        return True, f"exit_{mode_name}_d_11_21"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.01)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.70))
      ):
        return True, f"exit_{mode_name}_d_11_22"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_11_23"
    elif current_profit >= 0.2:
      if (
        (last_candle["r_14"] > -10.0)
        and (last_candle["rsi_14"] > 66.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_12_1"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["ema_200_dec_4_1d"] == True)
        and (last_candle["change_pct_4h"] < -0.03)
      ):
        return True, f"exit_{mode_name}_d_12_2"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["rsi_14_4h"] >= 50.0)
        and (last_candle["cti_20_1d"] >= 0.50)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["change_pct_4h"] <= -0.02)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_12_3"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 90.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] < -75.0)
        and (last_candle["cti_20_1d"] > 0.5)
        and (last_candle["rsi_14_1d"] >= 70.0)
        and (last_candle["change_pct_1h"] < -0.03)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_12_4"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 98.0)
        and (last_candle["rsi_14"] >= 80.0)
        and (last_candle["change_pct_1d"] < -0.02)
        and (last_candle["change_pct_4h"] < -0.02)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_12_5"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_3"] >= 98.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.8)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_12_6"
      elif (
        (last_candle["rsi_3"] >= 99.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_4h"] >= 0.5)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_24"] == True)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_12_7"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 76.0)
        and (last_candle["r_480_4h"] < -90.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_12_8"
      elif (
        (last_candle["rsi_14"] <= 30.0)
        and (last_candle["rsi_3_1h"] <= 10.0)
        and (last_candle["rsi_3_4h"] <= 6.0)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_12_9"
      elif (
        (last_candle["rsi_14"] <= 36.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["rsi_14_4h"] >= 60.0)
        and (last_candle["change_pct_1h"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_12_10"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 74.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["change_pct_1d"] < -0.10)
        and (last_candle["not_downtrend_4h"] == False)
      ):
        return True, f"exit_{mode_name}_d_12_11"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 80.0)
        and (last_candle["cti_20_1d"] >= 0.8)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_12_12"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 74.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["change_pct_4h"] < -0.03)
        and (last_candle["top_wick_pct_4h"] > 0.03)
        and (last_candle["ema_200_dec_4_1d"] == True)
      ):
        return True, f"exit_{mode_name}_d_12_13"
      elif (
        (last_candle["rsi_14"] <= 40.0)
        and (last_candle["rsi_14_15m"] <= 40.0)
        and (last_candle["rsi_3_1d"] <= 6.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
      ):
        return True, f"exit_{mode_name}_d_12_14"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 76.0)
        and (last_candle["rsi_3_1h"] <= 20.0)
        and (last_candle["r_480_4h"] >= -30.0)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_12_15"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 78.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["cti_20_1d"] >= 0.70)
        and (last_candle["change_pct_1d"] < -0.04)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_12_16"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 72.0)
        and (last_candle["rsi_14_15m"] >= 60.0)
        and (last_candle["rsi_14_1h"] >= 60.0)
        and (last_candle["rsi_14_1d"] >= 50.0)
        and (last_candle["r_480_4h"] < -70.0)
        and (last_candle["change_pct_1d"] < -0.04)
      ):
        return True, f"exit_{mode_name}_d_12_17"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] <= 50.0)
        and (last_candle["rsi_14_15m"] <= 50.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_24_15m"] == True)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["ema_200_dec_24_4h"] == True)
      ):
        return True, f"exit_{mode_name}_d_12_18"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["rsi_14_15m"] >= 50.0)
        and (last_candle["rsi_14_max_6_1d"] >= 85.0)
        and (last_candle["change_pct_1h"] < -0.01)
        and (last_candle["change_pct_4h"] < -0.06)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_12_19"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_max_6_1d"] >= 70.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
      ):
        return True, f"exit_{mode_name}_d_12_20"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 50.0)
        and (last_candle["cti_20_dec_3_1d"] == True)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["ema_200_dec_48_1h"] == True)
        and (last_candle["close"] < (last_candle["high_max_48_1h"] * 0.75))
      ):
        return True, f"exit_{mode_name}_d_12_21"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 60.0)
        and (last_candle["rsi_14_max_6_4h"] >= 70.0)
        and (last_candle["change_pct_4h"] < -0.01)
        and (last_candle["not_downtrend_4h"] == False)
        and (last_candle["close"] < (last_candle["high_max_24_1h"] * 0.70))
      ):
        return True, f"exit_{mode_name}_d_12_22"
      elif (
        (last_candle["r_14"] >= -1.0)
        and (last_candle["rsi_14"] >= 70.0)
        and (last_candle["not_downtrend_1h"] == False)
        and (last_candle["not_downtrend_1d"] == False)
        and (last_candle["close"] < (last_candle["high_max_6_1d"] * 0.80))
      ):
        return True, f"exit_{mode_name}_d_12_23"

    return False, None

  def exit_stoploss(
    self,
    mode_name: str,
    current_rate: float,
    profit_stake: float,
    profit_ratio: float,
    profit_current_stake_ratio: float,
    profit_init_ratio: float,
    max_profit: float,
    max_loss: float,
    filled_entries,
    filled_exits,
    last_candle,
    previous_candle_1,
    previous_candle_2,
    previous_candle_3,
    previous_candle_4,
    previous_candle_5,
    trade: "Trade",
    current_time: "datetime",
    buy_tag,
  ) -> tuple:
    is_backtest = self.dp.runmode.value in ["backtest", "hyperopt"]
    # Stoploss doom
    if (
      self.is_futures_mode is False
      and profit_stake
      < -(filled_entries[0].cost * self.stop_threshold / (trade.leverage if self.is_futures_mode else 1.0))
      # temporary
      and (trade.open_date_utc.replace(tzinfo=None) >= datetime(2023, 6, 13) or is_backtest)
    ):
      return True, f"exit_{mode_name}_stoploss"

    if (
      self.is_futures_mode is True
      and profit_stake
      < -(filled_entries[0].cost * self.stop_threshold_futures / (trade.leverage if self.is_futures_mode else 1.0))
      # temporary
      and (trade.open_date_utc.replace(tzinfo=None) >= datetime(2023, 10, 17) or is_backtest)
    ):
      return True, f"exit_{mode_name}_stoploss"

    return False, None

  def exit_short_normal(
    self,
    pair: str,
    current_rate: float,
    profit_stake: float,
    profit_ratio: float,
    profit_current_stake_ratio: float,
    profit_init_ratio: float,
    max_profit: float,
    max_loss: float,
    filled_entries,
    filled_exits,
    last_candle,
    previous_candle_1,
    previous_candle_2,
    previous_candle_3,
    previous_candle_4,
    previous_candle_5,
    trade: "Trade",
    current_time: "datetime",
    enter_tags,
  ) -> tuple:
    sell = False

    # Original sell signals
    sell, signal_name = self.exit_short_signals(
      self.short_normal_mode_name,
      profit_current_stake_ratio,
      max_profit,
      max_loss,
      last_candle,
      previous_candle_1,
      previous_candle_2,
      previous_candle_3,
      previous_candle_4,
      previous_candle_5,
      trade,
      current_time,
      enter_tags,
    )

    # # Main sell signals
    # if not sell:
    #     sell, signal_name = self.exit_short_main(self.short_normal_mode_name, profit_current_stake_ratio, max_profit, max_loss, last_candle, previous_candle_1, previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5, trade, current_time, enter_tags)

    # # Williams %R based sells
    # if not sell:
    #     sell, signal_name = self.exit_short_r(self.short_normal_mode_name, profit_current_stake_ratio, max_profit, max_loss, last_candle, previous_candle_1, previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5, trade, current_time, enter_tags)

    # # Stoplosses
    # if not sell:
    #     sell, signal_name = self.exit_short_stoploss(self.short_normal_mode_name, current_rate, profit_stake, profit_ratio, profit_current_stake_ratio, profit_init_ratio, max_profit, max_loss, filled_entries, filled_exits, last_candle, previous_candle_1, previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5, trade, current_time, enter_tags)

    # Profit Target Signal
    # Check if pair exist on target_profit_cache
    if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
      previous_rate = self.target_profit_cache.data[pair]["rate"]
      previous_profit = self.target_profit_cache.data[pair]["profit"]
      previous_sell_reason = self.target_profit_cache.data[pair]["sell_reason"]
      previous_time_profit_reached = datetime.fromisoformat(self.target_profit_cache.data[pair]["time_profit_reached"])

      sell_max, signal_name_max = self.exit_profit_target(
        self.short_normal_mode_name,
        pair,
        trade,
        current_time,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        last_candle,
        previous_candle_1,
        previous_rate,
        previous_profit,
        previous_sell_reason,
        previous_time_profit_reached,
        enter_tags,
      )
      if sell_max and signal_name_max is not None:
        return True, f"{signal_name_max}_m"
      if previous_sell_reason in [f"exit_{self.short_normal_mode_name}_stoploss_u_e"]:
        if profit_ratio > (previous_profit + 0.005):
          mark_pair, mark_signal = self.mark_profit_target(
            self.short_normal_mode_name,
            pair,
            True,
            previous_sell_reason,
            trade,
            current_time,
            current_rate,
            profit_ratio,
            last_candle,
            previous_candle_1,
          )
          if mark_pair:
            self._set_profit_target(pair, mark_signal, current_rate, profit_ratio, current_time)
      elif (profit_current_stake_ratio > (previous_profit + 0.001)) and (
        previous_sell_reason not in [f"exit_{self.short_normal_mode_name}_stoploss_doom"]
      ):
        # Update the target, raise it.
        mark_pair, mark_signal = self.mark_profit_target(
          self.short_normal_mode_name,
          pair,
          True,
          previous_sell_reason,
          trade,
          current_time,
          current_rate,
          profit_current_stake_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)

    # Add the pair to the list, if a sell triggered and conditions met
    if sell and signal_name is not None:
      previous_profit = None
      if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
        previous_profit = self.target_profit_cache.data[pair]["profit"]
      if signal_name in [
        f"exit_{self.short_normal_mode_name}_stoploss_doom",
        f"exit_{self.short_normal_mode_name}_stoploss_u_e",
      ]:
        mark_pair, mark_signal = self.mark_profit_target(
          self.short_normal_mode_name,
          pair,
          sell,
          signal_name,
          trade,
          current_time,
          current_rate,
          profit_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_ratio, current_time)
        else:
          # Just sell it, without maximize
          return True, f"{signal_name}"
      elif (previous_profit is None) or (previous_profit < profit_current_stake_ratio):
        mark_pair, mark_signal = self.mark_profit_target(
          self.short_normal_mode_name,
          pair,
          sell,
          signal_name,
          trade,
          current_time,
          current_rate,
          profit_current_stake_ratio,
          last_candle,
          previous_candle_1,
        )
        if mark_pair:
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)
        else:
          # Just sell it, without maximize
          return True, f"{signal_name}"
    else:
      if profit_current_stake_ratio >= self.profit_max_thresholds[0]:
        previous_profit = None
        if self.target_profit_cache is not None and pair in self.target_profit_cache.data:
          previous_profit = self.target_profit_cache.data[pair]["profit"]
        if (previous_profit is None) or (previous_profit < profit_current_stake_ratio):
          mark_signal = f"exit_profit_{self.short_normal_mode_name}_max"
          self._set_profit_target(pair, mark_signal, current_rate, profit_current_stake_ratio, current_time)

    if signal_name not in [
      f"exit_profit_{self.short_normal_mode_name}_max",
      f"exit_{self.short_normal_mode_name}_stoploss_doom",
      f"exit_{self.short_normal_mode_name}_stoploss_u_e",
    ]:
      if sell and (signal_name is not None):
        return True, f"{signal_name}"

    return False, None

  def exit_short_signals(
    self,
    mode_name: str,
    current_profit: float,
    max_profit: float,
    max_loss: float,
    last_candle,
    previous_candle_1,
    previous_candle_2,
    previous_candle_3,
    previous_candle_4,
    previous_candle_5,
    trade: "Trade",
    current_time: "datetime",
    buy_tag,
  ) -> tuple:
    # Sell signal 1
    if (
      (last_candle["rsi_14"] < 30.0)
      and (last_candle["close"] < last_candle["bb20_2_low"])
      and (previous_candle_1["close"] < previous_candle_1["bb20_2_low"])
      and (previous_candle_2["close"] < previous_candle_2["bb20_2_low"])
      and (previous_candle_3["close"] < previous_candle_3["bb20_2_low"])
      and (previous_candle_4["close"] < previous_candle_4["bb20_2_low"])
    ):
      if last_candle["close"] < last_candle["ema_200"]:
        if current_profit > 0.01:
          return True, f"exit_{mode_name}_1_1_1"
      else:
        if current_profit > 0.01:
          return True, f"exit_{mode_name}_1_2_1"

    # Sell signal 2
    elif (
      (last_candle["rsi_14"] < 26.0)
      and (last_candle["close"] < last_candle["bb20_2_low"])
      and (previous_candle_1["close"] < previous_candle_1["bb20_2_low"])
      and (previous_candle_2["close"] < previous_candle_2["bb20_2_low"])
    ):
      if last_candle["close"] < last_candle["ema_200"]:
        if current_profit > 0.01:
          return True, f"exit_{mode_name}_2_1_1"
      else:
        if current_profit > 0.01:
          return True, f"exit_{mode_name}_2_2_1"

    # Sell signal 3
    elif last_candle["rsi_14"] < 16.0:
      if last_candle["close"] < last_candle["ema_200"]:
        if current_profit > 0.01:
          return True, f"exit_{mode_name}_3_1_1"
      else:
        if current_profit > 0.01:
          return True, f"exit_{mode_name}_3_2_1"

    # # Sell signal 4
    # elif (last_candle['rsi_14'] > 80.0) and (last_candle['rsi_14_1h'] > 78.0):
    #     if (last_candle['close'] > last_candle['ema_200']):
    #         if (current_profit > 0.01):
    #             return True, f'exit_{mode_name}_4_1_1'
    #     else:
    #         if (current_profit > 0.01):
    #             return True, f'exit_{mode_name}_4_2_1'

    # # Sell signal 6
    # elif (last_candle['close'] < last_candle['ema_200']) and (last_candle['close'] > last_candle['ema_50']) and (last_candle['rsi_14'] > 79.0):
    #     if (current_profit > 0.01):
    #         return True, f'exit_{mode_name}_6_1'

    # # Sell signal 7
    # elif (last_candle['rsi_14_1h'] > 79.0) and (last_candle['crossed_below_ema_12_26']):
    #     if (last_candle['close'] > last_candle['ema_200']):
    #         if (current_profit > 0.01):
    #             return True, f'exit_{mode_name}_7_1_1'
    #     else:
    #         if (current_profit > 0.01):
    #             return True, f'exit_{mode_name}_7_2_1'

    # # Sell signal 8
    # elif (last_candle['close'] > last_candle['bb20_2_upp_1h'] * 1.08):
    #     if (last_candle['close'] > last_candle['ema_200']):
    #         if (current_profit > 0.01):
    #             return True, f'exit_{mode_name}_8_1_1'
    #     else:
    #         if (current_profit > 0.01):
    #             return True, f'exit_{mode_name}_8_2_1'

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

  def custom_exit(
    self, pair: str, trade: "Trade", current_time: "datetime", current_rate: float, current_profit: float, **kwargs
  ):
    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    last_candle = dataframe.iloc[-1].squeeze()
    previous_candle_1 = dataframe.iloc[-2].squeeze()
    previous_candle_2 = dataframe.iloc[-3].squeeze()
    previous_candle_3 = dataframe.iloc[-4].squeeze()
    previous_candle_4 = dataframe.iloc[-5].squeeze()
    previous_candle_5 = dataframe.iloc[-6].squeeze()

    enter_tag = "empty"
    if hasattr(trade, "enter_tag") and trade.enter_tag is not None:
      enter_tag = trade.enter_tag
    enter_tags = enter_tag.split()

    filled_entries = trade.select_filled_orders(trade.entry_side)
    filled_exits = trade.select_filled_orders(trade.exit_side)

    profit_stake = 0.0
    profit_ratio = 0.0
    profit_current_stake_ratio = 0.0
    profit_init_ratio = 0.0
    profit_stake, profit_ratio, profit_current_stake_ratio, profit_init_ratio = self.calc_total_profit(
      trade, filled_entries, filled_exits, current_rate
    )

    max_profit = (trade.max_rate - trade.open_rate) / trade.open_rate
    max_loss = (trade.open_rate - trade.min_rate) / trade.min_rate

    count_of_entries = len(filled_entries)
    if count_of_entries > 1:
      initial_entry = filled_entries[0]
      if initial_entry is not None and initial_entry.average is not None:
        max_profit = (trade.max_rate - initial_entry.average) / initial_entry.average
        max_loss = (initial_entry.average - trade.min_rate) / trade.min_rate

    # Normal mode
    if any(c in self.normal_mode_tags for c in enter_tags):
      sell, signal_name = self.exit_normal(
        pair,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        max_profit,
        max_loss,
        filled_entries,
        filled_exits,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )
      if sell and (signal_name is not None):
        return f"{signal_name} ( {enter_tag})"

    # Pump mode
    if any(c in self.pump_mode_tags for c in enter_tags):
      sell, signal_name = self.exit_pump(
        pair,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        max_profit,
        max_loss,
        filled_entries,
        filled_exits,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )
      if sell and (signal_name is not None):
        return f"{signal_name} ( {enter_tag})"

    # Quick mode
    if any(c in self.quick_mode_tags for c in enter_tags):
      sell, signal_name = self.exit_quick(
        pair,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        max_profit,
        max_loss,
        filled_entries,
        filled_exits,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )
      if sell and (signal_name is not None):
        return f"{signal_name} ( {enter_tag})"

    # Long Rebuy mode
    if all(c in self.long_rebuy_mode_tags for c in enter_tags):
      sell, signal_name = self.long_exit_rebuy(
        pair,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        max_profit,
        max_loss,
        filled_entries,
        filled_exits,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )
      if sell and (signal_name is not None):
        return f"{signal_name} ( {enter_tag})"

    # Long mode
    if any(c in self.long_mode_tags for c in enter_tags):
      sell, signal_name = self.exit_long(
        pair,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        max_profit,
        max_loss,
        filled_entries,
        filled_exits,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )
      if sell and (signal_name is not None):
        return f"{signal_name} ( {enter_tag})"

    # Long rapid mode
    if any(c in self.long_rapid_mode_tags for c in enter_tags):
      sell, signal_name = self.long_exit_rapid(
        pair,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        max_profit,
        max_loss,
        filled_entries,
        filled_exits,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )
      if sell and (signal_name is not None):
        return f"{signal_name} ( {enter_tag})"

    # Short normal mode
    if any(c in self.short_normal_mode_tags for c in enter_tags):
      sell, signal_name = self.exit_short_normal(
        pair,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        max_profit,
        max_loss,
        filled_entries,
        filled_exits,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )
      if sell and (signal_name is not None):
        return f"{signal_name} ( {enter_tag})"

    # Trades not opened by X4
    if not any(
      c
      in (
        self.normal_mode_tags
        + self.pump_mode_tags
        + self.quick_mode_tags
        + self.long_rebuy_mode_tags
        + self.long_mode_tags
        + self.long_rapid_mode_tags
        + self.short_normal_mode_tags
      )
      for c in enter_tags
    ):
      # use normal mode for such trades
      sell, signal_name = self.exit_normal(
        pair,
        current_rate,
        profit_stake,
        profit_ratio,
        profit_current_stake_ratio,
        profit_init_ratio,
        max_profit,
        max_loss,
        filled_entries,
        filled_exits,
        last_candle,
        previous_candle_1,
        previous_candle_2,
        previous_candle_3,
        previous_candle_4,
        previous_candle_5,
        trade,
        current_time,
        enter_tags,
      )
      if sell and (signal_name is not None):
        return f"{signal_name} ( {enter_tag})"

    return None

  def custom_stake_amount(
    self,
    pair: str,
    current_time: datetime,
    current_rate: float,
    proposed_stake: float,
    min_stake: Optional[float],
    max_stake: float,
    leverage: float,
    entry_tag: Optional[str],
    side: str,
    **kwargs,
  ) -> float:
    if self.position_adjustment_enable == True:
      enter_tags = entry_tag.split()
      # Rebuy mode
      if all(c in self.long_rebuy_mode_tags for c in enter_tags):
        stake_multiplier = self.rebuy_mode_stake_multiplier
        # Low stakes, on Binance mostly
        if (proposed_stake * self.rebuy_mode_stake_multiplier) < min_stake:
          stake_multiplier = self.rebuy_mode_stake_multiplier_alt
        return proposed_stake * stake_multiplier
      else:
        for i, item in enumerate(
          self.regular_mode_stake_multiplier_futures
          if self.is_futures_mode
          else self.regular_mode_stake_multiplier_spot
        ):
          if (proposed_stake * item) > min_stake:
            stake_multiplier = item
            return proposed_stake * stake_multiplier

    return proposed_stake

  def adjust_trade_position(
    self,
    trade: Trade,
    current_time: datetime,
    current_rate: float,
    current_profit: float,
    min_stake: Optional[float],
    max_stake: float,
    current_entry_rate: float,
    current_exit_rate: float,
    current_entry_profit: float,
    current_exit_profit: float,
    **kwargs,
  ):
    if self.position_adjustment_enable == False:
      return None

    enter_tag = "empty"
    if hasattr(trade, "enter_tag") and trade.enter_tag is not None:
      enter_tag = trade.enter_tag
    enter_tags = enter_tag.split()

    # Grinding
    if any(
      c
      in (
        self.normal_mode_tags
        + self.pump_mode_tags
        + self.quick_mode_tags
        + self.long_mode_tags
        + self.long_rapid_mode_tags
      )
      for c in enter_tags
    ) or not any(
      c
      in (
        self.normal_mode_tags
        + self.pump_mode_tags
        + self.quick_mode_tags
        + self.long_rebuy_mode_tags
        + self.long_mode_tags
        + self.long_rapid_mode_tags
      )
      for c in enter_tags
    ):
      return self.long_grind_adjust_trade_position(
        trade,
        enter_tags,
        current_time,
        current_rate,
        current_profit,
        min_stake,
        max_stake,
        current_entry_rate,
        current_exit_rate,
        current_entry_profit,
        current_exit_profit,
      )

    # Rebuy mode
    if all(c in self.long_rebuy_mode_tags for c in enter_tags):
      return self.long_rebuy_adjust_trade_position(
        trade,
        enter_tags,
        current_time,
        current_rate,
        current_profit,
        min_stake,
        max_stake,
        current_entry_rate,
        current_exit_rate,
        current_entry_profit,
        current_exit_profit,
      )

    return None

  def long_grind_adjust_trade_position(
    self,
    trade: Trade,
    enter_tags,
    current_time: datetime,
    current_rate: float,
    current_profit: float,
    min_stake: Optional[float],
    max_stake: float,
    current_entry_rate: float,
    current_exit_rate: float,
    current_entry_profit: float,
    current_exit_profit: float,
    **kwargs,
  ):
    is_backtest = self.dp.runmode.value in ["backtest", "hyperopt"]
    if self.grinding_enable:
      dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
      if len(dataframe) < 2:
        return None
      last_candle = dataframe.iloc[-1].squeeze()
      previous_candle = dataframe.iloc[-2].squeeze()

      filled_orders = trade.select_filled_orders()
      filled_entries = trade.select_filled_orders(trade.entry_side)
      filled_exits = trade.select_filled_orders(trade.exit_side)
      count_of_entries = trade.nr_of_successful_entries
      count_of_exits = trade.nr_of_successful_exits

      if count_of_entries == 0:
        return None

      if len(filled_orders) < 1:
        return None
      has_order_tags = False
      if hasattr(filled_orders[0], "ft_order_tag"):
        has_order_tags = True

      exit_rate = current_rate
      if self.dp.runmode.value in ("live", "dry_run"):
        ticker = self.dp.ticker(trade.pair)
        if ("bid" in ticker) and ("ask" in ticker):
          if trade.is_short:
            if self.config["exit_pricing"]["price_side"] in ["ask", "other"]:
              if ticker["ask"] is not None:
                exit_rate = ticker["ask"]
          else:
            if self.config["exit_pricing"]["price_side"] in ["bid", "other"]:
              if ticker["bid"] is not None:
                exit_rate = ticker["bid"]

      profit_stake, profit_ratio, profit_current_stake_ratio, profit_init_ratio = self.calc_total_profit(
        trade, filled_entries, filled_exits, exit_rate
      )

      slice_amount = filled_entries[0].cost
      slice_profit = (exit_rate - filled_orders[-1].safe_price) / filled_orders[-1].safe_price
      slice_profit_entry = (exit_rate - filled_entries[-1].safe_price) / filled_entries[-1].safe_price
      slice_profit_exit = (
        ((exit_rate - filled_exits[-1].safe_price) / filled_exits[-1].safe_price) if count_of_exits > 0 else 0.0
      )

      current_stake_amount = trade.amount * current_rate
      is_derisk = trade.amount < filled_entries[0].safe_filled
      is_derisk_calc = False

      # Rebuy mode
      if all(c in self.long_rebuy_mode_tags for c in enter_tags):
        slice_amount /= self.rebuy_mode_stake_multiplier
      elif not is_derisk and (trade.open_date_utc.replace(tzinfo=None) >= datetime(2024, 2, 5) or is_backtest):
        rebuy_stake, order_tag, is_derisk_calc = self.long_adjust_trade_position_no_derisk(
          trade,
          enter_tags,
          current_time,
          current_rate,
          current_profit,
          min_stake,
          max_stake,
          current_entry_rate,
          current_exit_rate,
          current_entry_profit,
          current_exit_profit,
          last_candle,
          previous_candle,
          filled_orders,
          filled_entries,
          filled_exits,
          exit_rate,
          slice_amount,
          slice_profit_entry,
          slice_profit,
          profit_ratio,
          profit_stake,
          profit_init_ratio,
          current_stake_amount,
          has_order_tags,
        )
        if rebuy_stake is not None:
          if has_order_tags:
            return rebuy_stake, order_tag
          else:
            return rebuy_stake
        elif count_of_exits == 0:
          return None
        elif not is_derisk_calc:
          return None

      if not all(c in self.long_rebuy_mode_tags for c in enter_tags):
        # First entry is lower now, therefore the grinds must adjust
        if trade.open_date_utc.replace(tzinfo=None) >= datetime(2024, 2, 5) or is_backtest:
          slice_amount /= (
            self.regular_mode_stake_multiplier_futures[0]
            if self.is_futures_mode
            else self.regular_mode_stake_multiplier_spot[0]
          )

      max_sub_grinds = 0
      grinding_mode_2_stakes = []
      grinding_mode_2_sub_thresholds = []
      for i, item in enumerate(
        self.grinding_mode_2_stakes_futures if self.is_futures_mode else self.grinding_mode_2_stakes_spot
      ):
        if (slice_amount * item[0] / (trade.leverage if self.is_futures_mode else 1.0)) > min_stake:
          grinding_mode_2_stakes = item
          grinding_mode_2_sub_thresholds = (
            self.grinding_mode_2_sub_thresholds_futures[i]
            if self.is_futures_mode
            else self.grinding_mode_2_sub_thresholds_spot[i]
          )
          max_sub_grinds = len(grinding_mode_2_stakes)
          break
      grinding_mode_2_derisk = (
        self.grinding_mode_2_derisk_futures if self.is_futures_mode else self.grinding_mode_2_derisk_spot
      )
      grinding_mode_2_stop_grinds = (
        self.grinding_mode_2_stop_grinds_futures if self.is_futures_mode else self.grinding_mode_2_stop_grinds_spot
      )
      grinding_mode_2_profit_threshold = (
        self.grinding_mode_2_profit_threshold_futures
        if self.is_futures_mode
        else self.grinding_mode_2_profit_threshold_spot
      )
      partial_sell = False
      is_sell_found = False
      sub_grind_count = 0
      total_amount = 0.0
      total_cost = 0.0
      current_open_rate = 0.0
      current_grind_stake = 0.0
      current_grind_stake_profit = 0.0
      for order in reversed(filled_orders):
        if (order.ft_order_side == "buy") and (order is not filled_orders[0]):
          sub_grind_count += 1
          total_amount += order.safe_filled
          total_cost += order.safe_filled * order.safe_price
        elif order.ft_order_side == "sell":
          is_sell_found = True
          if (order.safe_remaining * exit_rate / (trade.leverage if self.is_futures_mode else 1.0)) > min_stake:
            partial_sell = True
          break
      if sub_grind_count > 0:
        current_open_rate = total_cost / total_amount
        current_grind_stake = total_amount * exit_rate * (1 - trade.fee_close)
        current_grind_stake_profit = current_grind_stake - total_cost

      # Buy
      if (not partial_sell) and (sub_grind_count < max_sub_grinds):
        if (
          (
            (slice_profit_entry if (sub_grind_count > 0) else profit_init_ratio)
            < (0.0 if (is_derisk and sub_grind_count == 0) else grinding_mode_2_sub_thresholds[sub_grind_count])
          )
          and (current_time - timedelta(minutes=10) > filled_entries[-1].order_filled_utc)
          and ((current_time - timedelta(hours=12) > filled_orders[-1].order_filled_utc) or (slice_profit < -0.06))
          and self.long_grind_buy(last_candle, previous_candle, slice_profit)
        ):
          buy_amount = (
            slice_amount * grinding_mode_2_stakes[sub_grind_count] / (trade.leverage if self.is_futures_mode else 1.0)
          )
          if buy_amount < (min_stake * 1.5):
            buy_amount = min_stake * 1.5
          if buy_amount > max_stake:
            return None
          self.dp.send_msg(
            f"Grinding entry [{trade.pair}] | Rate: {current_rate} | Stake amount: {buy_amount} | Profit (stake): {profit_stake} | Profit: {(profit_ratio * 100.0):.2f}%"
          )
          return buy_amount

      # Sell remaining if partial fill on exit
      if partial_sell:
        order = filled_exits[-1]
        sell_amount = order.safe_remaining * exit_rate / (trade.leverage if self.is_futures_mode else 1.0)
        if (current_stake_amount - sell_amount) < (min_stake * 1.5):
          sell_amount = (trade.amount * exit_rate / (trade.leverage if self.is_futures_mode else 1.0)) - (
            min_stake * 1.5
          )
        grind_profit = (exit_rate - order.safe_price) / order.safe_price
        if sell_amount > min_stake:
          # Test if it's the last exit. Normal exit with partial fill
          if (trade.stake_amount - sell_amount) > min_stake:
            self.dp.send_msg(
              f"Grinding exit (remaining) [{trade.pair}] | Rate: {exit_rate} | Stake amount: {sell_amount} | Coin amount: {order.safe_remaining} | Profit (stake): {profit_stake} | Profit: {(profit_ratio * 100.0):.2f}% | Grind profit: {(grind_profit * 100.0):.2f}% ({grind_profit * sell_amount} {self.config['stake_currency']})"
            )
            return -sell_amount

      # Sell
      elif sub_grind_count > 0:
        grind_profit = (exit_rate - current_open_rate) / current_open_rate
        if grind_profit > grinding_mode_2_profit_threshold:
          sell_amount = total_amount * exit_rate / (trade.leverage if self.is_futures_mode else 1.0)
          if (current_stake_amount - sell_amount) < (min_stake * 1.5):
            sell_amount = (trade.amount * exit_rate / (trade.leverage if self.is_futures_mode else 1.0)) - (
              min_stake * 1.5
            )
          if sell_amount > min_stake:
            self.dp.send_msg(
              f"Grinding exit [{trade.pair}] | Rate: {exit_rate} | Stake amount: {sell_amount} | Coin amount: {total_amount} | Profit (stake): {profit_stake} | Profit: {(profit_ratio * 100.0):.2f}% | Grind profit: {(grind_profit * 100.0):.2f}% ({grind_profit * sell_amount * trade.leverage} {self.config['stake_currency']})"
            )
            return -sell_amount

      # Grind stop
      if (
        (
          (
            (
              current_grind_stake_profit
              < (slice_amount * grinding_mode_2_stop_grinds / (trade.leverage if self.is_futures_mode else 1.0))
            )
            if is_sell_found
            else (
              profit_stake
              < (slice_amount * grinding_mode_2_derisk / (trade.leverage if self.is_futures_mode else 1.0))
            )
          )
          or (
            (
              profit_stake
              < (slice_amount * grinding_mode_2_derisk / (trade.leverage if self.is_futures_mode else 1.0))
            )
            and (
              (
                (trade.amount * exit_rate / (trade.leverage if self.is_futures_mode else 1.0))
                - (total_amount * exit_rate / (trade.leverage if self.is_futures_mode else 1.0))
              )
              > (min_stake * 3.0)
            )
            # temporary
            and (trade.open_date_utc.replace(tzinfo=None) >= datetime(2023, 12, 19) or is_backtest)
          )
        )
        # temporary
        and (
          (trade.open_date_utc.replace(tzinfo=None) >= datetime(2023, 8, 28) or is_backtest)
          or (filled_entries[-1].order_date_utc.replace(tzinfo=None) >= datetime(2023, 8, 28) or is_backtest)
        )
      ):
        sell_amount = trade.amount * exit_rate / (trade.leverage if self.is_futures_mode else 1.0) * 0.999
        if (current_stake_amount / (trade.leverage if self.is_futures_mode else 1.0) - sell_amount) < (
          min_stake * 1.5
        ):
          sell_amount = (trade.amount * exit_rate / (trade.leverage if self.is_futures_mode else 1.0)) - (
            min_stake * 1.5
          )
        if sell_amount > min_stake:
          grind_profit = 0.0
          if current_open_rate > 0.0:
            grind_profit = ((exit_rate - current_open_rate) / current_open_rate) if is_sell_found else profit_ratio
          self.dp.send_msg(
            f"Grinding stop exit [{trade.pair}] | Rate: {exit_rate} | Stake amount: {sell_amount} | Coin amount: {total_amount} | Profit (stake): {profit_stake} | Profit: {(profit_ratio * 100.0):.2f}% | Grind profit: {(grind_profit * 100.0):.2f}%"
          )
          return -sell_amount

    return None

  def long_grind_buy(self, last_candle: Series, previous_candle: Series, slice_profit: float) -> float:
    if (
      (
        (last_candle["close"] > (last_candle["close_max_12"] * 0.88))
        and (last_candle["close"] > (last_candle["close_max_24"] * 0.82))
        and (last_candle["close"] > (last_candle["close_max_48"] * 0.76))
        and (last_candle["btc_pct_close_max_72_5m"] < 0.03)
        and (last_candle["btc_pct_close_max_24_5m"] < 0.03)
      )
      and (
        (last_candle["enter_long"] == True)
        or (
          (last_candle["rsi_3"] > 10.0)
          and (last_candle["rsi_3_15m"] > 20.0)
          and (last_candle["rsi_3_1h"] > 20.0)
          and (last_candle["rsi_3_4h"] > 20.0)
          and (last_candle["rsi_14"] < 46.0)
          and (last_candle["ha_close"] > last_candle["ha_open"])
          and (last_candle["ema_12"] < (last_candle["ema_26"] * 0.990))
          and (last_candle["cti_20_1h"] < 0.8)
          and (last_candle["rsi_14_1h"] < 80.0)
        )
        or (
          (last_candle["rsi_14"] < 36.0)
          and (last_candle["close"] < (last_candle["sma_16"] * 0.998))
          and (last_candle["rsi_3"] > 16.0)
          and (last_candle["rsi_3_15m"] > 30.0)
          and (last_candle["rsi_3_1h"] > 30.0)
          and (last_candle["rsi_3_4h"] > 30.0)
        )
        or (
          (last_candle["rsi_14"] < 36.0)
          and (previous_candle["rsi_3"] > 6.0)
          and (last_candle["ema_26"] > last_candle["ema_12"])
          and ((last_candle["ema_26"] - last_candle["ema_12"]) > (last_candle["open"] * 0.010))
          and ((previous_candle["ema_26"] - previous_candle["ema_12"]) > (last_candle["open"] / 100.0))
          and (last_candle["rsi_3_1h"] > 20.0)
          and (last_candle["rsi_3_4h"] > 20.0)
          and (last_candle["cti_20_1h"] < 0.8)
          and (last_candle["rsi_14_1h"] < 80.0)
        )
        or (
          (last_candle["rsi_14"] > 30.0)
          and (last_candle["rsi_14"] < 60.0)
          and (last_candle["hma_70_buy"])
          and (last_candle["close"] < (last_candle["high_max_12_1h"] * 0.90))
          and (last_candle["cti_20_15m"] < 0.5)
          and (last_candle["rsi_14_15m"] < 50.0)
          and (last_candle["cti_20_1h"] < 0.8)
          and (last_candle["rsi_14_1h"] < 80.0)
        )
        or (
          (last_candle["rsi_3"] > 10.0)
          and (last_candle["rsi_3_15m"] > 20.0)
          and (last_candle["rsi_3_1h"] > 20.0)
          and (last_candle["rsi_3_4h"] > 20.0)
          and (last_candle["rsi_14"] < 36.0)
          and (last_candle["zlma_50_dec_15m"] == False)
          and (last_candle["zlma_50_dec_1h"] == False)
        )
        or (
          (last_candle["rsi_14"] < 40.0)
          and (last_candle["rsi_14_15m"] < 40.0)
          and (last_candle["rsi_3"] > 6.0)
          and (last_candle["ema_26_15m"] > last_candle["ema_12_15m"])
          and ((last_candle["ema_26_15m"] - last_candle["ema_12_15m"]) > (last_candle["open_15m"] * 0.006))
          and ((previous_candle["ema_26_15m"] - previous_candle["ema_12_15m"]) > (last_candle["open_15m"] / 100.0))
          and (last_candle["rsi_3_15m"] > 10.0)
          and (last_candle["rsi_3_1h"] > 26.0)
          and (last_candle["rsi_3_4h"] > 26.0)
          and (last_candle["cti_20_1h"] < 0.8)
          and (last_candle["rsi_14_1h"] < 80.0)
        )
        or (
          (last_candle["rsi_14"] > 35.0)
          and (last_candle["rsi_3"] > 4.0)
          and (last_candle["rsi_3"] < 46.0)
          and (last_candle["rsi_14"] < previous_candle["rsi_14"])
          and (last_candle["close"] < (last_candle["sma_16"] * 0.982))
          and (last_candle["cti_20"] < -0.6)
          and (last_candle["rsi_3_1h"] > 20.0)
          and (last_candle["rsi_3_4h"] > 20.0)
          and (last_candle["not_downtrend_1d"] == True)
        )
        or (
          (last_candle["rsi_3"] > 12.0)
          and (last_candle["rsi_3_15m"] > 26.0)
          and (last_candle["rsi_3_1h"] > 26.0)
          and (last_candle["rsi_3_4h"] > 26.0)
          and (last_candle["rsi_14"] < 40.0)
          and (last_candle["ema_12"] < (last_candle["ema_26"] * 0.994))
          and (last_candle["cti_20_1h"] < 0.8)
          and (last_candle["rsi_14_1h"] < 80.0)
          and (last_candle["cti_20_4h"] < 0.8)
          and (last_candle["rsi_14_4h"] < 80.0)
          and (last_candle["ema_200_dec_48_1h"] == False)
        )
        or (
          (last_candle["rsi_14"] < 60.0)
          and (last_candle["hma_55_buy"])
          and (last_candle["rsi_3_1h"] > 4.0)
          and (last_candle["rsi_3_4h"] > 4.0)
          and (last_candle["cti_20_15m"] < 0.8)
          and (last_candle["cti_20_1h"] < 0.8)
          and (last_candle["cti_20_4h"] < 0.8)
          and (last_candle["rsi_14_1h"] < 80.0)
          and (last_candle["close"] < (last_candle["high_max_12_1h"] * 0.90))
        )
        or (
          (last_candle["rsi_3"] > 12.0)
          and (last_candle["rsi_3_15m"] > 30.0)
          and (last_candle["rsi_3_1h"] > 30.0)
          and (last_candle["rsi_3_4h"] > 30.0)
          and (last_candle["rsi_14"] < 40.0)
          and (last_candle["cti_20_15m"] < 0.8)
          and (last_candle["rsi_14_15m"] < 70.0)
          and (last_candle["cti_20_1h"] < 0.8)
          and (last_candle["rsi_14_1h"] < 80.0)
          and (last_candle["cti_20_4h"] < 0.8)
          and (last_candle["rsi_14_4h"] < 80.0)
          and (last_candle["r_14_1h"] > -80.0)
          and (last_candle["ema_12"] < (last_candle["ema_26"] * 0.995))
        )
        or (
          (last_candle["rsi_3"] > 12.0)
          and (last_candle["rsi_3_15m"] > 30.0)
          and (last_candle["rsi_3_1h"] > 30.0)
          and (last_candle["rsi_3_4h"] > 30.0)
          and (last_candle["rsi_14"] < 46.0)
          and (last_candle["rsi_14_1d"] < 70.0)
          and (last_candle["not_downtrend_1h"] == True)
          and (last_candle["not_downtrend_4h"] == True)
          and (last_candle["not_downtrend_1d"] == True)
        )
        or (
          (last_candle["rsi_3"] > 12.0)
          and (last_candle["rsi_3_15m"] > 30.0)
          and (last_candle["rsi_3_1h"] > 30.0)
          and (last_candle["rsi_3_4h"] > 30.0)
          and (last_candle["rsi_14"] < 46.0)
          and (last_candle["ha_close"] > last_candle["ha_open"])
          and (last_candle["close"] < last_candle["res_hlevel_4h"])
          and (last_candle["close"] > last_candle["sup_level_4h"])
          and (last_candle["close"] < last_candle["res_hlevel_1d"])
          and (last_candle["close"] > last_candle["sup_level_1d"])
          and (last_candle["close"] < last_candle["res3_1d"])
          and (last_candle["close"] > (last_candle["high_max_24_1h"] * 0.85))
          and (last_candle["hl_pct_change_24_1h"] < 0.35)
        )
      )
    ):
      return True

    return False

  def long_adjust_trade_position_no_derisk(
    self,
    trade: Trade,
    enter_tags,
    current_time: datetime,
    current_rate: float,
    current_profit: float,
    min_stake: Optional[float],
    max_stake: float,
    current_entry_rate: float,
    current_exit_rate: float,
    current_entry_profit: float,
    current_exit_profit: float,
    last_candle: Series,
    previous_candle: Series,
    filled_orders: "Orders",
    filled_entries: "Orders",
    filled_exits: "Orders",
    exit_rate: float,
    slice_amount: float,
    slice_profit_entry: float,
    slice_profit: float,
    profit_ratio: float,
    profit_stake: float,
    profit_init_ratio: float,
    current_stake_amount: float,
    has_order_tags: bool,
    **kwargs,
  ) -> tuple[Optional[float], str, bool]:
    regular_mode_rebuy_stakes = (
      self.regular_mode_rebuy_stakes_futures if self.is_futures_mode else self.regular_mode_rebuy_stakes_spot
    )
    max_rebuy_sub_grinds = len(regular_mode_rebuy_stakes)
    regular_mode_rebuy_sub_thresholds = (
      self.regular_mode_rebuy_thresholds_futures if self.is_futures_mode else self.regular_mode_rebuy_thresholds_spot
    )
    regular_mode_grind_1_stakes = (
      self.regular_mode_grind_1_stakes_futures if self.is_futures_mode else self.regular_mode_grind_1_stakes_spot
    )
    max_grind_1_sub_grinds = len(regular_mode_grind_1_stakes)
    regular_mode_grind_1_sub_thresholds = (
      self.regular_mode_grind_1_thresholds_futures
      if self.is_futures_mode
      else self.regular_mode_grind_1_thresholds_spot
    )
    regular_mode_grind_1_profit_threshold = (
      self.regular_mode_grind_1_profit_threshold_futures
      if self.is_futures_mode
      else self.regular_mode_grind_1_profit_threshold_spot
    )

    partial_sell = False
    is_derisk = False
    rebuy_sub_grind_count = 0
    rebuy_total_amount = 0.0
    rebuy_total_cost = 0.0
    rebuy_current_open_rate = 0.0
    rebuy_current_grind_stake = 0.0
    rebuy_current_grind_stake_profit = 0.0
    rebuy_is_sell_found = False
    rebuy_found = False
    rebuy_buy_orders = []
    rebuy_distance_ratio = 0.0
    grind_1_sub_grind_count = 0
    grind_1_total_amount = 0.0
    grind_1_total_cost = 0.0
    grind_1_current_open_rate = 0.0
    grind_1_current_grind_stake = 0.0
    grind_1_current_grind_stake_profit = 0.0
    grind_1_is_sell_found = False
    grind_1_found = False
    grind_1_buy_orders = []
    grind_1_distance_ratio = 0.0
    for order in reversed(filled_orders):
      if (order.ft_order_side == "buy") and (order is not filled_orders[0]):
        order_tag = ""
        if has_order_tags:
          if order.ft_order_tag is not None:
            order_tag = order.ft_order_tag
            if order.ft_order_tag == "g1":
              order_tag = "g1"
            elif order.ft_order_tag == "r":
              order_tag = "r"
        if not grind_1_is_sell_found and order_tag == "g1":
          grind_1_sub_grind_count += 1
          grind_1_total_amount += order.safe_filled
          grind_1_total_cost += order.safe_filled * order.safe_price
          grind_1_buy_orders.append(order.id)
          if not grind_1_found:
            grind_1_distance_ratio = (exit_rate - order.safe_price) / order.safe_price
            grind_1_found = True
        elif not rebuy_is_sell_found:
          rebuy_sub_grind_count += 1
          rebuy_total_amount += order.safe_filled
          rebuy_total_cost += order.safe_filled * order.safe_price
          rebuy_buy_orders.append(order.id)
          if not rebuy_found:
            rebuy_distance_ratio = (exit_rate - order.safe_price) / order.safe_price
            rebuy_found = True
      elif order.ft_order_side == "sell":
        if (order.safe_remaining * exit_rate / (trade.leverage if self.is_futures_mode else 1.0)) > min_stake:
          partial_sell = True
          break
        order_tag = ""
        if has_order_tags:
          if order.ft_order_tag is not None:
            sell_order_tag = order.ft_order_tag
            order_mode = sell_order_tag.split(" ", 1)
            if len(order_mode) > 0:
              if order_mode[0] == "g1":
                order_tag = "g1"
              elif order_mode[0] == "d":
                order_tag = "d"
              elif order_mode[0] == "r":
                order_tag = "r"
        if order_tag == "g1":
          grind_1_is_sell_found = True
        elif order_tag == "d":
          is_derisk = True
          grind_1_is_sell_found = True
          rebuy_is_sell_found = True
        else:
          rebuy_is_sell_found = True
        if not is_derisk:
          start_amount = filled_orders[0].safe_filled
          current_amount = 0.0
          for order2 in filled_orders:
            if order2.ft_order_side == "buy":
              current_amount += order2.safe_filled
            if order2.ft_order_side == "sell":
              current_amount -= order2.safe_filled
            if order2 is order:
              if current_amount < start_amount:
                is_derisk = True
        # found sells for all modes
        if rebuy_is_sell_found and grind_1_is_sell_found:
          break

    # The trade already de-risked
    if is_derisk:
      return None, "", is_derisk
    if not has_order_tags and len(filled_exits) > 0:
      return None, "", is_derisk

    if rebuy_sub_grind_count > 0:
      rebuy_current_open_rate = rebuy_total_cost / rebuy_total_amount
      rebuy_current_grind_stake = rebuy_total_amount * exit_rate * (1 - trade.fee_close)
      rebuy_current_grind_stake_profit = rebuy_current_grind_stake - rebuy_total_cost
    if grind_1_sub_grind_count > 0:
      grind_1_current_open_rate = grind_1_total_cost / grind_1_total_amount
      grind_1_current_grind_stake = grind_1_total_amount * exit_rate * (1 - trade.fee_close)
      grind_1_current_grind_stake_profit = grind_1_current_grind_stake - grind_1_total_cost

    # Sell remaining if partial fill on exit
    if partial_sell:
      order = filled_exits[-1]
      sell_amount = order.safe_remaining * exit_rate / (trade.leverage if self.is_futures_mode else 1.0)
      if (current_stake_amount - sell_amount) < (min_stake * 1.5):
        sell_amount = (trade.amount * exit_rate / (trade.leverage if self.is_futures_mode else 1.0)) - (
          min_stake * 1.5
        )
      if sell_amount > min_stake:
        # Test if it's the last exit. Normal exit with partial fill
        if (trade.stake_amount - sell_amount) > min_stake:
          self.dp.send_msg(
            f"Exit (remaining) [{trade.pair}] | Rate: {exit_rate} | Stake amount: {sell_amount} | Coin amount: {order.safe_remaining} | Profit (stake): {profit_stake} | Profit: {(profit_ratio * 100.0):.2f}%"
          )
          order_tag = "p"
          if has_order_tags:
            if order.ft_order_tag is not None:
              order_tag = order.ft_order_tag
          return -sell_amount, order_tag, is_derisk

    # Rebuy
    if (not partial_sell) and (not rebuy_is_sell_found) and (rebuy_sub_grind_count < max_rebuy_sub_grinds):
      if (
        (0 <= rebuy_sub_grind_count < max_rebuy_sub_grinds)
        and (slice_profit_entry < regular_mode_rebuy_sub_thresholds[rebuy_sub_grind_count])
        and (
          (last_candle["close"] > (last_candle["close_max_12"] * 0.92))
          and (last_candle["close"] > (last_candle["close_max_24"] * 0.88))
          and (last_candle["close"] > (last_candle["close_max_48"] * 0.84))
          and (last_candle["btc_pct_close_max_72_5m"] < 0.03)
          and (last_candle["btc_pct_close_max_24_5m"] < 0.03)
        )
        and (
          (last_candle["rsi_3"] > 16.0)
          and (last_candle["rsi_3_15m"] > 16.0)
          and (last_candle["rsi_3_1h"] > 4.0)
          and (last_candle["rsi_3_4h"] > 4.0)
          and (last_candle["rsi_14"] < 50.0)
        )
      ):
        buy_amount = (
          slice_amount
          * regular_mode_rebuy_stakes[rebuy_sub_grind_count]
          / (trade.leverage if self.is_futures_mode else 1.0)
        )
        if buy_amount > max_stake:
          buy_amount = max_stake
        if buy_amount < (min_stake * 1.5):
          buy_amount = min_stake * 1.5
        if buy_amount > max_stake:
          return None, "", is_derisk
        self.dp.send_msg(
          f"Rebuy (r) [{trade.pair}] | Rate: {current_rate} | Stake amount: {buy_amount} | Profit (stake): {profit_stake} | Profit: {(profit_ratio * 100.0):.2f}%"
        )
        log.info(
          f"Rebuy (r) [{trade.pair}] | Rate: {current_rate} | Stake amount: {buy_amount} | Profit (stake): {profit_stake} | Profit: {(profit_ratio * 100.0):.2f}%"
        )
        order_tag = "r"
        return buy_amount, order_tag, is_derisk

    # Gringing g1
    # Grinding entry
    if has_order_tags and (not partial_sell) and (grind_1_sub_grind_count < max_grind_1_sub_grinds):
      if (
        (
          (grind_1_distance_ratio if (grind_1_sub_grind_count > 0) else profit_init_ratio)
          < (regular_mode_grind_1_sub_thresholds[grind_1_sub_grind_count])
        )
        and (current_time - timedelta(minutes=10) > filled_entries[-1].order_filled_utc)
        and ((current_time - timedelta(hours=12) > filled_orders[-1].order_filled_utc) or (slice_profit < -0.06))
        and self.long_grind_buy(last_candle, previous_candle, slice_profit)
      ):
        buy_amount = (
          slice_amount
          * regular_mode_grind_1_stakes[grind_1_sub_grind_count]
          / (trade.leverage if self.is_futures_mode else 1.0)
        )
        if buy_amount < (min_stake * 1.5):
          buy_amount = min_stake * 1.5
        if buy_amount > max_stake:
          return None, "", is_derisk
        self.dp.send_msg(
          f"Grinding entry (g1) [{trade.pair}] | Rate: {current_rate} | Stake amount: {buy_amount} | Profit (stake): {profit_stake} | Profit: {(profit_ratio * 100.0):.2f}%"
        )
        log.info(
          f"Grinding entry (g1) [{trade.pair}] | Rate: {current_rate} | Stake amount: {buy_amount} | Profit (stake): {profit_stake} | Profit: {(profit_ratio * 100.0):.2f}%"
        )
        order_tag = "g1"
        return buy_amount, order_tag, is_derisk

    # Grinding Exit
    if has_order_tags and grind_1_sub_grind_count > 0:
      grind_profit = (exit_rate - grind_1_current_open_rate) / grind_1_current_open_rate
      if grind_profit > regular_mode_grind_1_profit_threshold:
        sell_amount = grind_1_total_amount * exit_rate / (trade.leverage if self.is_futures_mode else 1.0)
        if (current_stake_amount - sell_amount) < (min_stake * 1.5):
          sell_amount = (trade.amount * exit_rate / (trade.leverage if self.is_futures_mode else 1.0)) - (
            min_stake * 1.5
          )
        if sell_amount > min_stake:
          self.dp.send_msg(
            f"Grinding exit (g1) [{trade.pair}] | Rate: {exit_rate} | Stake amount: {sell_amount} | Coin amount: {grind_1_total_amount} | Profit (stake): {profit_stake} | Profit: {(profit_ratio * 100.0):.2f}% | Grind profit: {(grind_profit * 100.0):.2f}% ({grind_profit * sell_amount * trade.leverage} {self.config['stake_currency']})"
          )
          log.info(
            f"Grinding exit (g1) [{trade.pair}] | Rate: {exit_rate} | Stake amount: {sell_amount} | Coin amount: {grind_1_total_amount} | Profit (stake): {profit_stake} | Profit: {(profit_ratio * 100.0):.2f}% | Grind profit: {(grind_profit * 100.0):.2f}% ({grind_profit * sell_amount * trade.leverage} {self.config['stake_currency']})"
          )
          order_tag = "g1"
          for grind_entry_id in grind_1_buy_orders:
            order_tag += " " + str(grind_entry_id)
          return -sell_amount, order_tag, is_derisk

    # De-risk
    if profit_stake < (
      slice_amount
      * (self.regular_mode_derisk_futures if self.is_futures_mode else self.regular_mode_derisk_spot)
      / (trade.leverage if self.is_futures_mode else 1.0)
    ):
      sell_amount = trade.amount * exit_rate / (trade.leverage if self.is_futures_mode else 1.0) * 0.999
      if (current_stake_amount / (trade.leverage if self.is_futures_mode else 1.0) - sell_amount) < (min_stake * 1.5):
        sell_amount = (trade.amount * exit_rate / (trade.leverage if self.is_futures_mode else 1.0)) - (
          min_stake * 1.5
        )
      if sell_amount > min_stake:
        grind_profit = 0.0
        self.dp.send_msg(
          f"Rebuy de-risk [{trade.pair}] | Rate: {exit_rate} | Stake amount: {sell_amount} | Profit (stake): {profit_stake} | Profit: {(profit_ratio * 100.0):.2f}%"
        )
        log.info(
          f"Rebuy de-risk [{trade.pair}] | Rate: {exit_rate} | Stake amount: {sell_amount} | Profit (stake): {profit_stake} | Profit: {(profit_ratio * 100.0):.2f}%"
        )
        return -sell_amount, "d", is_derisk

    return None, "", is_derisk

  def long_rebuy_adjust_trade_position(
    self,
    trade: Trade,
    enter_tags,
    current_time: datetime,
    current_rate: float,
    current_profit: float,
    min_stake: Optional[float],
    max_stake: float,
    current_entry_rate: float,
    current_exit_rate: float,
    current_entry_profit: float,
    current_exit_profit: float,
    **kwargs,
  ) -> Optional[float]:
    dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
    if len(dataframe) < 2:
      return None
    last_candle = dataframe.iloc[-1].squeeze()
    previous_candle = dataframe.iloc[-2].squeeze()

    filled_orders = trade.select_filled_orders()
    filled_entries = trade.select_filled_orders(trade.entry_side)
    filled_exits = trade.select_filled_orders(trade.exit_side)
    count_of_entries = trade.nr_of_successful_entries
    count_of_exits = trade.nr_of_successful_exits

    if count_of_entries == 0:
      return None

    # The first exit is de-risk (providing the trade is still open)
    if count_of_exits > 0:
      return self.long_grind_adjust_trade_position(
        trade,
        enter_tags,
        current_time,
        current_rate,
        current_profit,
        min_stake,
        max_stake,
        current_entry_rate,
        current_exit_rate,
        current_entry_profit,
        current_exit_profit,
      )

    exit_rate = current_rate
    if self.dp.runmode.value in ("live", "dry_run"):
      ticker = self.dp.ticker(trade.pair)
      if ("bid" in ticker) and ("ask" in ticker):
        if trade.is_short:
          if self.config["exit_pricing"]["price_side"] in ["ask", "other"]:
            if ticker["ask"] is not None:
              exit_rate = ticker["ask"]
        else:
          if self.config["exit_pricing"]["price_side"] in ["bid", "other"]:
            if ticker["bid"] is not None:
              exit_rate = ticker["bid"]

    profit_stake, profit_ratio, profit_current_stake_ratio, profit_init_ratio = self.calc_total_profit(
      trade, filled_entries, filled_exits, exit_rate
    )

    slice_amount = filled_entries[0].cost
    slice_profit = (exit_rate - filled_orders[-1].safe_price) / filled_orders[-1].safe_price
    slice_profit_entry = (exit_rate - filled_entries[-1].safe_price) / filled_entries[-1].safe_price
    slice_profit_exit = (
      ((exit_rate - filled_exits[-1].safe_price) / filled_exits[-1].safe_price) if count_of_exits > 0 else 0.0
    )

    current_stake_amount = trade.amount * current_rate

    is_rebuy = False

    rebuy_mode_stakes = self.rebuy_mode_stakes_futures if self.is_futures_mode else self.rebuy_mode_stakes_spot
    max_sub_grinds = len(rebuy_mode_stakes)
    rebuy_mode_sub_thresholds = (
      self.rebuy_mode_thresholds_futures if self.is_futures_mode else self.rebuy_mode_thresholds_spot
    )
    partial_sell = False
    sub_grind_count = 0
    total_amount = 0.0
    total_cost = 0.0
    current_open_rate = 0.0
    current_grind_stake = 0.0
    current_grind_stake_profit = 0.0
    for order in reversed(filled_orders):
      if (order.ft_order_side == "buy") and (order is not filled_orders[0]):
        sub_grind_count += 1
        total_amount += order.safe_filled
        total_cost += order.safe_filled * order.safe_price
      elif order.ft_order_side == "sell":
        if (order.safe_remaining * exit_rate / (trade.leverage if self.is_futures_mode else 1.0)) > min_stake:
          partial_sell = True
        break
    if sub_grind_count > 0:
      current_open_rate = total_cost / total_amount
      current_grind_stake = total_amount * exit_rate * (1 - trade.fee_close)
      current_grind_stake_profit = current_grind_stake - total_cost

    if (not partial_sell) and (sub_grind_count < max_sub_grinds):
      if (
        ((0 <= sub_grind_count < max_sub_grinds) and (slice_profit_entry < rebuy_mode_sub_thresholds[sub_grind_count]))
        and (
          (last_candle["close_max_12"] < (last_candle["close"] * 1.14))
          and (last_candle["close_max_24"] < (last_candle["close"] * 1.20))
          and (last_candle["close_max_48"] < (last_candle["close"] * 1.26))
          and (last_candle["btc_pct_close_max_72_5m"] < 0.03)
          and (last_candle["btc_pct_close_max_24_5m"] < 0.03)
        )
        and (
          (last_candle["rsi_3"] > 10.0)
          and (last_candle["rsi_3_15m"] > 10.0)
          and (last_candle["rsi_3_1h"] > 10.0)
          and (last_candle["rsi_3_4h"] > 10.0)
          and (last_candle["rsi_14"] < 46.0)
        )
      ):
        buy_amount = (
          slice_amount * rebuy_mode_stakes[sub_grind_count] / (trade.leverage if self.is_futures_mode else 1.0)
        )
        if buy_amount > max_stake:
          buy_amount = max_stake
        if buy_amount < (min_stake * 1.5):
          buy_amount = min_stake * 1.5
        self.dp.send_msg(
          f"Rebuy [{trade.pair}] | Rate: {current_rate} | Stake amount: {buy_amount} | Profit (stake): {profit_stake} | Profit: {(profit_ratio * 100.0):.2f}%"
        )
        return buy_amount

      if profit_stake < (
        slice_amount
        * (self.rebuy_mode_derisk_futures if self.is_futures_mode else self.rebuy_mode_derisk_spot)
        / (trade.leverage if self.is_futures_mode else 1.0)
      ):
        sell_amount = trade.amount * exit_rate / (trade.leverage if self.is_futures_mode else 1.0) * 0.999
        if (current_stake_amount / (trade.leverage if self.is_futures_mode else 1.0) - sell_amount) < (
          min_stake * 1.5
        ):
          sell_amount = (trade.amount * exit_rate / (trade.leverage if self.is_futures_mode else 1.0)) - (
            min_stake * 1.5
          )
        if sell_amount > min_stake:
          grind_profit = 0.0
          self.dp.send_msg(
            f"Rebuy de-risk [{trade.pair}] | Rate: {exit_rate} | Stake amount: {sell_amount} | Profit (stake): {profit_stake} | Profit: {(profit_ratio * 100.0):.2f}%"
          )
          return -sell_amount

    return None

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
        item_buy_logic = []
        item_buy_logic.append(reduce(lambda x, y: x & y, item_buy_protection_list))

        # Condition #1 - Long mode bull. Uptrend.
        if index == 1:
          # Logic
          item_buy_logic.append(dataframe["ema_26"] > dataframe["ema_12"])
          item_buy_logic.append((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * 0.02))
          item_buy_logic.append(
            (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100)
          )
          item_buy_logic.append(dataframe["close"] < (dataframe["bb20_2_low"] * 0.999))

        # Condition #2 - Normal mode bull.
        if index == 2:
          # Logic
          item_buy_logic.append(dataframe["bb40_2_delta"].gt(dataframe["close"] * 0.06))
          item_buy_logic.append(dataframe["close_delta"].gt(dataframe["close"] * 0.02))
          item_buy_logic.append(dataframe["bb40_2_tail"].lt(dataframe["bb40_2_delta"] * 0.2))
          item_buy_logic.append(dataframe["close"].lt(dataframe["bb40_2_low"].shift()))
          item_buy_logic.append(dataframe["close"].le(dataframe["close"].shift()))

        # Condition #3 - Normal mode bull.
        if index == 3:
          # Logic
          item_buy_logic.append(dataframe["rsi_14"] < 36.0)
          item_buy_logic.append(dataframe["ha_close"] > dataframe["ha_open"])
          item_buy_logic.append((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * 0.020))

        # Condition #4 - Normal mode bull.
        if index == 4:
          # Logic
          item_buy_logic.append(dataframe["ema_26"] > dataframe["ema_12"])
          item_buy_logic.append((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * 0.018))
          item_buy_logic.append(
            (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100)
          )
          item_buy_logic.append(dataframe["close"] < (dataframe["bb20_2_low"] * 0.996))

        # Condition #5 - Normal mode bull.
        if index == 5:
          # Logic
          item_buy_logic.append(dataframe["ema_26"] > dataframe["ema_12"])
          item_buy_logic.append((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * 0.03))
          item_buy_logic.append(
            (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100)
          )
          item_buy_logic.append(dataframe["rsi_14"] < 36.0)

        # Condition #6 - Normal mode bull.
        if index == 6:
          # Logic
          item_buy_logic.append(dataframe["close"] < (dataframe["ema_26"] * 0.94))
          item_buy_logic.append(dataframe["close"] < (dataframe["bb20_2_low"] * 0.996))

        # Condition #7 Normal mode.
        if index == 7:
          # Logic
          item_buy_logic.append(dataframe["close"] < (dataframe["ema_16"] * 0.974))
          item_buy_logic.append(dataframe["ewo_50_200"] > 2.0)
          item_buy_logic.append(dataframe["rsi_14"] < 30.0)

        # Condition #8 Normal mode.
        if index == 8:
         # Logic
          item_buy_logic.append(dataframe["close"] < (dataframe["ema_16"] * 0.944))
          item_buy_logic.append(dataframe["ewo_50_200"] < -4.0)
          item_buy_logic.append(dataframe["rsi_14"] < 30.0)

        # Condition #9 - Normal mode.
        if index == 9:
          # Logic
          item_buy_logic.append(dataframe["ema_26_15m"] > dataframe["ema_12_15m"])
          item_buy_logic.append((dataframe["ema_26_15m"] - dataframe["ema_12_15m"]) > (dataframe["open_15m"] * 0.020))
          item_buy_logic.append(
            (dataframe["ema_26_15m"].shift(3) - dataframe["ema_12_15m"].shift(3)) > (dataframe["open_15m"] / 100.0)
          )
          item_buy_logic.append(dataframe["close_15m"] < (dataframe["bb20_2_low_15m"] * 0.99))

        # Condition #10 - Normal mode (Long)
        if index == 10:
          # Logic
          item_buy_logic.append(dataframe["ema_26"] > dataframe["ema_12"])
          item_buy_logic.append((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * 0.016))
          item_buy_logic.append(
            (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100)
          )
          item_buy_logic.append(dataframe["rsi_14"] < 36.0)

        if index == 11:
          # Logic
          item_buy_logic.append(dataframe["rsi_14"] < self.entry_11_rsi_14_max.value)
          item_buy_logic.append(dataframe["cti_20"] < self.entry_11_cti_20_max.value)
          item_buy_logic.append(dataframe["ema_26"] > dataframe["ema_12"])
          item_buy_logic.append(
            (dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * self.entry_11_ema_open_offset.value)
          )
          item_buy_logic.append(
            (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100.0)
          )
          item_buy_logic.append(dataframe["close"] < (dataframe["sma_30"] * self.entry_11_sma_offset.value))

        # Condition #12 - Normal mode (Long)
        if index == 12:
          # Logic
          item_buy_logic.append(dataframe["r_14"] < self.entry_12_r_14_max.value)
          item_buy_logic.append(dataframe["close"] < (dataframe["bb20_2_low"] * self.entry_12_bb_offset.value))
          item_buy_logic.append(dataframe["close"] < (dataframe["sma_30"] * self.entry_12_sma_offset.value))

        # Condition #21 - Pump mode bull.
        if index == 21:
          # Logic
          item_buy_logic.append(dataframe["ema_26"] > dataframe["ema_12"])
          item_buy_logic.append((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * 0.02))
          item_buy_logic.append(
            (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100)
          )
          item_buy_logic.append(dataframe["close"] < (dataframe["bb20_2_low"] * 0.999))

        # Condition #22 - Pump mode bull.
        if index == 22:
          # Logic
          item_buy_logic.append(dataframe["close"] < (dataframe["ema_16"] * 0.968))
          item_buy_logic.append(dataframe["cti_20"] < -0.9)
          item_buy_logic.append(dataframe["rsi_14"] < 50.0)

        # Condition #23 - Pump mode.
        if index == 23:
          # Logic
          item_buy_logic.append(dataframe["ewo_50_200_15m"] > 4.2)
          item_buy_logic.append(dataframe["rsi_14_15m"].shift(1) < 30.0)
          item_buy_logic.append(dataframe["rsi_14_15m"] < 30.0)
          item_buy_logic.append(dataframe["rsi_14"] < 35.0)
          item_buy_logic.append(dataframe["cti_20"] < -0.8)
          item_buy_logic.append(dataframe["close"] < (dataframe["ema_26_15m"] * 0.958))

        # Condition #24 - Pump mode (Long)
        if index == 24:
          # Logic
          item_buy_logic.append(dataframe["rsi_14"] > self.entry_24_rsi_14_min.value)
          item_buy_logic.append(dataframe["rsi_14"] < self.entry_24_rsi_14_max.value)
          item_buy_logic.append(dataframe["cti_20"] < self.entry_24_cti_20_max.value)
          item_buy_logic.append(dataframe["r_14"] < self.entry_24_r_14_max.value)
          item_buy_logic.append(dataframe["ewo_50_200"] > self.entry_24_ewo_50_200_min.value)
          item_buy_logic.append(dataframe["ewo_50_200"] < self.entry_24_ewo_50_200_max.value)
          item_buy_logic.append(dataframe["close"] < (dataframe["sma_75"] * self.entry_24_sma_offset.value))

        # Condition #25 - Pump mode (Long).
        if index == 25:
          # Logic
          item_buy_logic.append(dataframe["rsi_14"] < self.entry_25_rsi_14_max.value)
          item_buy_logic.append(dataframe["cti_20"] < self.entry_25_cti_20_max.value)
          item_buy_logic.append(dataframe["ewo_50_200"] > self.entry_25_ewo_50_200_min.value)
          item_buy_logic.append(dataframe["close"] < (dataframe["sma_30"] * self.entry_25_sma_offset.value))

        # Condition #26 - Pump mode (Long).
        if index == 26:
          # Logic
          item_buy_logic.append(dataframe["close"] < (dataframe["bb20_2_low"] * self.entry_26_bb_offset.value))
          item_buy_logic.append(dataframe["ewo_50_200_1h"] > self.entry_26_ewo_50_200_1h_min.value)
          item_buy_logic.append(dataframe["ema_26"] > dataframe["ema_12"])
          item_buy_logic.append(
            (dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * self.entry_26_ema_open_offset.value)
          )
          item_buy_logic.append(
            (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100.0)
          )

        # Condition #41 - Quick mode bull.
        if index == 41:
          # Logic
          item_buy_logic.append(dataframe["bb40_2_delta"].gt(dataframe["close"] * 0.036))
          item_buy_logic.append(dataframe["close_delta"].gt(dataframe["close"] * 0.02))
          item_buy_logic.append(dataframe["bb40_2_tail"].lt(dataframe["bb40_2_delta"] * 0.4))
          item_buy_logic.append(dataframe["close"].lt(dataframe["bb40_2_low"].shift()))
          item_buy_logic.append(dataframe["close"].le(dataframe["close"].shift()))
          item_buy_logic.append(dataframe["rsi_14"] < 36.0)

        # Condition #42 - Quick mode bull.
        if index == 42:
          # Logic
          item_buy_logic.append(dataframe["ema_26"] > dataframe["ema_12"])
          item_buy_logic.append((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * 0.018))
          item_buy_logic.append(
            (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100)
          )
          item_buy_logic.append(dataframe["close"] < (dataframe["bb20_2_low"] * 0.996))
          item_buy_logic.append(dataframe["rsi_14"] < 40.0)

        # Condition #43 - Quick mode bull.
        if index == 43:
          # Logic
          item_buy_logic.append(dataframe["close"] < (dataframe["ema_26"] * 0.938))
          item_buy_logic.append(dataframe["cti_20"] < -0.75)
          item_buy_logic.append(dataframe["r_14"] < -94.0)

        # Condition #44 - Quick mode bull.
        if index == 44:
          # Logic
          item_buy_logic.append(dataframe["bb20_2_width_1h"] > 0.132)
          item_buy_logic.append(dataframe["cti_20"] < -0.8)
          item_buy_logic.append(dataframe["r_14"] < -90.0)

        # Condition #45 - Quick mode (Long).
        if index == 45:
         # Logic
          item_buy_logic.append(dataframe["rsi_14"] > self.entry_45_rsi_14_min.value)
          item_buy_logic.append(dataframe["rsi_14"] < self.entry_45_rsi_14_max.value)
          item_buy_logic.append(dataframe["rsi_20"] < dataframe["rsi_20"].shift(1))
          item_buy_logic.append(dataframe["cti_20"] < self.entry_45_cti_20_max.value)
          item_buy_logic.append(dataframe["close"] < (dataframe["sma_16"] * self.entry_45_sma_offset.value))

        # Condition #46 - Quick mode (Long).
        if index == 46:
          # Logic
          item_buy_logic.append(dataframe["rsi_14"] < self.entry_46_rsi_14_max.value)
          item_buy_logic.append(dataframe["chandelier_dir_1h"].shift(1) < -0)
          item_buy_logic.append(dataframe["chandelier_dir_1h"] > 0)
          item_buy_logic.append(dataframe["close"] > dataframe["zlma_50_1h"])
          item_buy_logic.append(dataframe["ema_12"] < dataframe["ema_26"])

        # Condition #47 - Quick mode (Long).
        if index == 47:
          # Logic
          item_buy_logic.append(dataframe["rsi_14"] > self.entry_47_rsi_14_min.value)
          item_buy_logic.append(dataframe["rsi_14"] < self.entry_47_rsi_14_max.value)
          item_buy_logic.append(dataframe["rsi_20"] > self.entry_47_rsi_20_min.value)
          item_buy_logic.append(dataframe["rsi_20"] < self.entry_47_rsi_20_max.value)
          item_buy_logic.append(dataframe["cti_20"] < self.entry_47_cti_20_max.value)
          item_buy_logic.append(dataframe["chandelier_dir"].shift(1) < -0)
          item_buy_logic.append(dataframe["chandelier_dir"] > 0)
          item_buy_logic.append(dataframe["ema_12"] < (dataframe["ema_26"] * self.entry_47_ema_offset.value))
          item_buy_logic.append(
            dataframe["close"] < (dataframe["high_max_12_1h"] * self.entry_47_high_max_12_1h_max.value)
          )

        # Condition #48 - Quick mode (Long).
        if index == 48:
          # Logic
          item_buy_logic.append(dataframe["rsi_14"] < self.entry_48_rsi_14_max.value)
          item_buy_logic.append(dataframe["cci_20"] < self.entry_48_cci_20_max.value)
          item_buy_logic.append(dataframe["close"] < (dataframe["sma_30"] * self.entry_48_sma_offset.value))
          item_buy_logic.append(
            ((dataframe["close"] - dataframe["open_min_6"]) / dataframe["open_min_6"]) > self.entry_48_inc_min.value
          )

        # Condition #49 - Quick mode (Long).
        if index == 49:
          # Logic
          item_buy_logic.append(dataframe["rsi_14"] < self.entry_49_rsi_14_max.value)
          item_buy_logic.append(
            ((dataframe["close"] - dataframe["open_min_12"]) / dataframe["open_min_12"]) > self.entry_49_inc_min.value
          )

        # Condition #50 - Quick mode (Long)
        if index == 50:
          # Logic
          item_buy_logic.append(dataframe["close"] < (dataframe["bb20_2_low"] * self.entry_50_bb_offset.value))
          item_buy_logic.append(dataframe["ema_26"] > dataframe["ema_12"])
          item_buy_logic.append(
            (dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * self.entry_50_ema_open_offset.value)
          )
          item_buy_logic.append(
            (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100.0)
          )

        # Condition #61 - Rebuy mode (Long).
        if index == 61:
          # Logic
          item_buy_logic.append(dataframe["rsi_14"] < 40.0)
          item_buy_logic.append(dataframe["bb40_2_delta"].gt(dataframe["close"] * 0.03))
          item_buy_logic.append(dataframe["close_delta"].gt(dataframe["close"] * 0.018))
          item_buy_logic.append(dataframe["bb40_2_tail"].lt(dataframe["bb40_2_delta"] * 0.4))
          item_buy_logic.append(dataframe["close"].lt(dataframe["bb40_2_low"].shift()))
          item_buy_logic.append(dataframe["close"].le(dataframe["close"].shift()))

        # Condition #81 - Long mode bull.
        if index == 81:
          # Logic
          item_buy_logic.append(dataframe["bb40_2_delta"].gt(dataframe["close"] * 0.052))
          item_buy_logic.append(dataframe["close_delta"].gt(dataframe["close"] * 0.024))
          item_buy_logic.append(dataframe["bb40_2_tail"].lt(dataframe["bb40_2_delta"] * 0.2))
          item_buy_logic.append(dataframe["close"].lt(dataframe["bb40_2_low"].shift()))
          item_buy_logic.append(dataframe["close"].le(dataframe["close"].shift()))
          item_buy_logic.append(dataframe["rsi_14"] < 30.0)

        # Condition #82 - Long mode bull.
        if index == 82:
          # Logic
          item_buy_logic.append(dataframe["ema_26"] > dataframe["ema_12"])
          item_buy_logic.append((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * 0.03))
          item_buy_logic.append(
            (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100)
          )
          item_buy_logic.append(dataframe["cti_20"] < -0.8)

        # Condition #101 - Long mode rapid
        if index == 101:
          # Logic
          item_buy_logic.append(dataframe["rsi_14"] < 36.0)
          item_buy_logic.append(dataframe["rsi_14"] < dataframe["rsi_14"].shift(1))
          item_buy_logic.append(dataframe["close"] < (dataframe["sma_16"] * 0.956))
          item_buy_logic.append(dataframe["cti_20_15m"] < -0.5)

        # Condition #102 - Long mode rapid
        if index == 102:
          # Logic
          item_buy_logic.append(dataframe["rsi_14"] < self.entry_46_rsi_14_max.value)
          item_buy_logic.append(dataframe["close"] < (dataframe["ema_16"] * self.entry_102_ema_offset.value))
          item_buy_logic.append(dataframe["close"] < (dataframe["bb20_2_low"] * self.entry_102_bb_offset.value))

        # Condition #103 - Long mode rapid
        if index == 103:
          # Logic
          item_buy_logic.append(dataframe["rsi_14"] > self.entry_103_rsi_14_min.value)
          item_buy_logic.append(dataframe["close"] < (dataframe["sma_16"] * self.entry_103_sma_offset.value))
          item_buy_logic.append(dataframe["close"] < (dataframe["bb20_2_mid"] * self.entry_103_bb_offset.value))

        # Condition #104 - Long mode rapid
        if index == 104:
          # Logic
          item_buy_logic.append(dataframe["rsi_14"] > self.entry_104_rsi_14_min.value)
          item_buy_logic.append(dataframe["rsi_14"] < self.entry_104_rsi_14_max.value)
          item_buy_logic.append(dataframe["close"] < (dataframe["sma_16"] * self.entry_104_sma_offset.value))

        # Condition #105 - Long mode rapid
        if index == 105:
          # Logic
          item_buy_logic.append(dataframe["rsi_3"] < 60.0)
          item_buy_logic.append(dataframe["rsi_14"] < 46.0)
          item_buy_logic.append(dataframe["ema_26"] > dataframe["ema_12"])
          item_buy_logic.append((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * 0.018))
          item_buy_logic.append(
            (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100)
          )

        # Condition #106 - Rapid mode (Long).
        if index == 106:
          # Logic
          item_buy_logic.append(dataframe["cti_20"] < self.entry_106_cti_20_max.value)
          item_buy_logic.append(dataframe["ewo_50_200"] < self.entry_106_ewo_50_200_max.value)
          item_buy_logic.append(dataframe["close"] < (dataframe["sma_30"] * self.entry_106_sma_offset.value))

        # Condition #107 - Rapid mode (Long)
        if index == 107:
          # Logic
          item_buy_logic.append(dataframe["bb40_2_low"].shift().gt(0.0))
          item_buy_logic.append(
            dataframe["bb40_2_delta"].gt(dataframe["close"] * self.entry_107_bb40_bbdelta_close.value)
          )
          item_buy_logic.append(
            dataframe["close_delta"].gt(dataframe["close"] * self.entry_107_bb40_closedelta_close.value)
          )
          item_buy_logic.append(
            dataframe["bb40_2_tail"].lt(dataframe["bb40_2_delta"] * self.entry_107_bb40_tail_bbdelta.value)
          )
          item_buy_logic.append(dataframe["close"].lt(dataframe["bb40_2_low"].shift()))
          item_buy_logic.append(dataframe["close"].le(dataframe["close"].shift()))
          item_buy_logic.append(dataframe["cti_20"] < self.entry_107_cti_20_max.value)
          item_buy_logic.append(dataframe["r_480"] > self.entry_107_r_480_min.value)

        # Condition #108 - Rapid mode (Long)
        if index == 108:
          # Logic
          item_buy_logic.append(dataframe["rsi_14"] > self.entry_108_rsi_14_min.value)
          item_buy_logic.append(dataframe["cti_20"] < self.entry_108_cti_20_max.value)
          item_buy_logic.append(dataframe["r_14"] < self.entry_108_r_14_max.value)
          item_buy_logic.append(dataframe["r_14"].shift(1) < self.entry_108_r_14_max.value)
          item_buy_logic.append(dataframe["close"] < (dataframe["bb20_2_low"] * self.entry_108_bb_offset.value))
          item_buy_logic.append(dataframe["ema_26"] > dataframe["ema_12"])
          item_buy_logic.append(
            (dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * self.entry_108_ema_open_offset.value)
          )
          item_buy_logic.append(
            (dataframe["ema_26"].shift() - dataframe["ema_12"].shift()) > (dataframe["open"] / 100.0)
          )

        # Condition #109 - Rapid mode (Long)
        if index == 109:
          # Logic
          item_buy_logic.append(dataframe["cti_20"] < self.entry_109_cti_20_max.value)
          item_buy_logic.append(dataframe["r_14"] < self.entry_109_r_14_max.value)
          item_buy_logic.append(dataframe["close"] < (dataframe["bb20_2_low"] * self.entry_109_bb_offset.value))
          item_buy_logic.append(dataframe["close"] < (dataframe["ema_20"] * self.entry_109_ema_offset.value))

        # Condition #110 - Rapid mode (Long).
        if index == 110:
          # Logic
          item_buy_logic.append(dataframe["cti_20"] < self.entry_110_cti_20_max.value)
          item_buy_logic.append(dataframe["ewo_50_200"] < self.entry_110_ewo_50_200_max.value)
          item_buy_logic.append(dataframe["close"] < (dataframe["ema_20"] * self.entry_110_ema_offset.value))

        item_buy_logic.append(dataframe["volume"] > 0)
        item_buy = reduce(lambda x, y: x & y, item_buy_logic)
        dataframe.loc[item_buy, "enter_tag"] += f"{index} "
        conditions.append(item_buy)
        dataframe.loc[:, "enter_long"] = item_buy

    if conditions:
      dataframe.loc[:, "enter_long"] = reduce(lambda x, y: x | y, conditions)

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
