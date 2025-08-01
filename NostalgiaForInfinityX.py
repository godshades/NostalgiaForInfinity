import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
from functools import reduce
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, DecimalParameter, stoploss_from_open, RealParameter,IntParameter,informative
from pandas import DataFrame, Series
from datetime import datetime
import math
import logging
from freqtrade.persistence import Trade
import pandas_ta as pta
from technical.indicators import RMI

logger = logging.getLogger(__name__)

def ewo(dataframe, sma1_length=5, sma2_length=35):
    sma1 = ta.EMA(dataframe, timeperiod=sma1_length)
    sma2 = ta.EMA(dataframe, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / dataframe['close'] * 100
    return smadif



def top_percent_change_dca(dataframe: DataFrame, length: int) -> float:
        """
        Percentage change of the current close from the range maximum Open price

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        if length == 0:
            return (dataframe['open'] - dataframe['close']) / dataframe['close']
        else:
            return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']


#############################################################################################################
##                NostalgiaForInfinityX by iterativ                                                        ##
##           https://github.com/iterativv/NostalgiaForInfinity                                             ##
##                                                                                                         ##
##    Strategy for Freqtrade https://github.com/freqtrade/freqtrade                                        ##
##                                                                                                         ##
#############################################################################################################
##               GENERAL RECOMMENDATIONS                                                                   ##
##                                                                                                         ##
##   For optimal performance, suggested to use between 4 and 6 open trades, with unlimited stake.          ##
##   A pairlist with 40 to 80 pairs. Volume pairlist works well.                                           ##
##   Prefer stable coin (USDT, USDC etc) pairs, instead of BTC or ETH pairs.                              ##
##   Highly recommended to blacklist leveraged tokens (*BULL, *BEAR, *UP, *DOWN etc).                      ##
##   Ensure that you don't override any variables in you config.json. Especially                           ##
##   the timeframe (must be 5m).                                                                           ##
##     use_exit_signal must set to true (or not set at all).                                               ##
##     exit_profit_only must set to false (or not set at all).                                             ##
##     ignore_roi_if_entry_signal must set to true (or not set at all).                                    ##
##                                                                                                         ##
#############################################################################################################
##               HOLD SUPPORT                                                                              ##
##                                                                                                         ##
## -------- SPECIFIC TRADES ------------------------------------------------------------------------------ ##
##   In case you want to have SOME of the trades to only be sold when on profit, add a file named          ##
##   "nfi-hold-trades.json" in the user_data directory                                                     ##
##                                                                                                         ##
##   The contents should be similar to:                                                                    ##
##                                                                                                         ##
##   {"trade_ids": [1, 3, 7], "profit_ratio": 0.005}                                                       ##
##                                                                                                         ##
##   Or, for individual profit ratios(Notice the trade ID's as strings:                                    ##
##                                                                                                         ##
##   {"trade_ids": {"1": 0.001, "3": -0.005, "7": 0.05}}                                                   ##
##                                                                                                         ##
##   NOTE:                                                                                                 ##
##    * `trade_ids` is a list of integers, the trade ID's, which you can get from the logs or from the     ##
##      output of the telegram status command.                                                             ##
##    * Regardless of the defined profit ratio(s), the strategy MUST still produce a SELL signal for the   ##
##      HOLD support logic to run                                                                          ##
##    * This feature can be completely disabled with the holdSupportEnabled class attribute                ##
##                                                                                                         ##
## -------- SPECIFIC PAIRS ------------------------------------------------------------------------------- ##
##   In case you want to have some pairs to always be on held until a specific profit, using the same      ##
##   "nfi-hold-trades.json" file add something like:                                                       ##
##                                                                                                         ##
##   {"trade_pairs": {"BTC/USDT": 0.001, "ETH/USDT": -0.005}}                                              ##
##                                                                                                         ##
## -------- SPECIFIC TRADES AND PAIRS -------------------------------------------------------------------- ##
##   It is also valid to include specific trades and pairs on the holds file, for example:                 ##
##                                                                                                         ##
##   {"trade_ids": {"1": 0.001}, "trade_pairs": {"BTC/USDT": 0.001}}                                       ##
#############################################################################################################
##               DONATIONS                                                                                 ##
##                                                                                                         ##
##   BTC: bc1qvflsvddkmxh7eqhc4jyu5z5k6xcw3ay8jl49sk                                                       ##
##   ETH (ERC20): 0x83D3cFb8001BDC5d2211cBeBB8cB3461E5f7Ec91                                               ##
##   BEP20/BSC (USDT, ETH, BNB, ...): 0x86A0B21a20b39d16424B7c8003E4A7e12d78ABEe                           ##
##   TRC20/TRON (USDT, TRON, ...): TTAa9MX6zMLXNgWMhg7tkNormVHWCoq8Xk                                      ##
##                                                                                                         ##
##  Patreon: https://www.patreon.com/iterativ                                                              ##
##                                                                                                         ##
##               REFERRAL LINKS                                                                            ##
##                                                                                                         ##
##  Binance: https://www.binance.com/join?ref=C68K26A9 (20% discount on trading fees)                      ##
##  Kucoin: https://www.kucoin.com/r/af/QBSSS5J2 (20% lifetime discount on trading fees)                   ##
##  Gate: https://www.gate.io/share/nfinfinity (20% lifetime discount on trading fees)                     ##
##  OKX: https://www.okx.com/join/11749725931 (20% discount on trading fees)                               ##
##  MEXC: https://promote.mexc.com/a/luA6Xclb (10% discount on trading fees)                               ##
##  ByBit: https://partner.bybit.com/b/nfi                                                                 ##
##  Bitget: https://bonus.bitget.com/fdqe83481698435803831 (lifetime 20% +10% extra spot rebate)           ##
##  BitMart: https://www.bitmart.com/invite/nfinfinity/en-US (20% lifetime discount on trading fees)       ##
##  HTX: https://www.htx.com/invite/en-us/1f?invite_code=ubpt2223                                          ##
##         (Welcome Bonus worth 241 USDT upon completion of a deposit and trade)                           ##
##  Bitvavo: https://bitvavo.com/invite?a=D22103A4BC (no fees for the first € 10000)                       ##
#############################################################################################################


def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)


class Gemini(IStrategy):
    """
    PASTE OUTPUT FROM HYPEROPT HERE
    Can be overridden for specific sub-strategies (stake currencies) at the bottom.
    """

    buy_params = {
       "bbdelta_close": 0.01568,
       "bbdelta_tail": 0.75301,
       "close_bblower": 0.01195,
       "closedelta_close": 0.0092,
       "base_nb_candles_buy": 12,
       "rsi_buy": 58,
       "low_offset": 0.985,
       "rocr_1h": 0.57032,
       "rocr1_1h": 0.7210406300824859,

        
        "buy_clucha_bbdelta_close": 0.049,
        "buy_clucha_bbdelta_tail": 1.146,
        "buy_clucha_close_bblower": 0.018,
        "buy_clucha_closedelta_close": 0.017,
        "buy_clucha_rocr_1h": 0.526,

        "buy_cci": -116,
        "buy_cci_length": 25,
        "buy_rmi": 49,
        "buy_rmi_length": 17,
        "buy_srsi_fk": 32,
        "buy_bb_width_1h": 1.074,

    }   

    sell_params = {

      "pHSL": -0.397,
      "pPF_1": 0.012,
      "pPF_2": 0.07,
      "pSL_1": 0.015,
      "pSL_2": 0.068,
      "sell_bbmiddle_close": 1.0909210168690215,
      "sell_fisher": 0.46405736994786184,
      
      "base_nb_candles_sell": 22,
      "high_offset": 1.014,
      "high_offset_2": 1.01,

      "sell_u_e_2_cmf": -0.0,
      "sell_u_e_2_ema_close_delta": 0.016,
      "sell_u_e_2_rsi": 10,

      "sell_deadfish_profit": -0.063,
      "sell_deadfish_bb_factor": 0.954,
      "sell_deadfish_bb_width": 0.043,
      "sell_deadfish_volume_factor": 2.37
    }

    INTERFACE_VERSION = 3
    exit_profit_only = True
    
    minimal_roi = {
        "0": 100
    }

    position_adjustment_enable = True

    stoploss = -0.296  # use custom stoploss

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = True

    """
    END HYPEROPT
    """

    timeframe = '5m'

    use_exit_signal = True
    ignore_roi_if_entry_signal = False

    use_custom_stoploss = False

    process_only_new_candles = True
    startup_candle_count = 168

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        if ("trading_mode" in self.config) and (self.config["trading_mode"] in ["futures", "margin"]):
            self.can_short = True
    
    def is_support(self, row_data) -> bool:
        conditions = []
        for row in range(len(row_data)-1):
            if row < len(row_data)/2:
                conditions.append(row_data[row] > row_data[row+1])
            else:
                conditions.append(row_data[row] < row_data[row+1])
        return reduce(lambda x, y: x & y, conditions)

    fast_ewo = 50
    slow_ewo = 200

    
    buy_44_ma_offset = 0.982
    buy_44_ewo = -18.143
    buy_44_cti = -0.8
    buy_44_r_1h = -75.0

    buy_37_ma_offset = 0.98
    buy_37_ewo = 9.8
    buy_37_rsi = 56.0
    buy_37_cti = -0.7

    buy_ema_open_mult_7 = 0.030
    buy_cti_7 = -0.89

    is_optimize_dip = False
    buy_rmi = IntParameter(30, 50, default=35, optimize= is_optimize_dip)
    buy_cci = IntParameter(-135, -90, default=-133, optimize= is_optimize_dip)
    buy_srsi_fk = IntParameter(30, 50, default=25, optimize= is_optimize_dip)
    buy_cci_length = IntParameter(25, 45, default=25, optimize = is_optimize_dip)
    buy_rmi_length = IntParameter(8, 20, default=8, optimize = is_optimize_dip)

    is_optimize_break = False
    buy_bb_width = DecimalParameter(0.065, 0.135, default=0.095, optimize = is_optimize_break)
    buy_bb_delta = DecimalParameter(0.018, 0.035, default=0.025, optimize = is_optimize_break)
    
    is_optimize_check = False
    buy_roc_1h = IntParameter(-25, 200, default=10, optimize = is_optimize_check)
    buy_bb_width_1h = DecimalParameter(0.3, 2.0, default=0.3, optimize = is_optimize_check)

    is_optimize_clucha = False
    buy_clucha_bbdelta_close = DecimalParameter(0.01,0.05, default=0.02206, optimize=is_optimize_clucha)
    buy_clucha_bbdelta_tail = DecimalParameter(0.7, 1.2, default=1.02515, optimize=is_optimize_clucha)
    buy_clucha_close_bblower = DecimalParameter(0.001, 0.05, default=0.03669, optimize=is_optimize_clucha)
    buy_clucha_closedelta_close = DecimalParameter(0.001, 0.05, default=0.04401, optimize=is_optimize_clucha)
    buy_clucha_rocr_1h = DecimalParameter(0.1, 1.0, default=0.47782, optimize=is_optimize_clucha)

    
    
    is_optimize_local_uptrend = False
    buy_ema_diff = DecimalParameter(0.022, 0.027, default=0.025, optimize = is_optimize_local_uptrend)
    buy_bb_factor = DecimalParameter(0.990, 0.999, default=0.995, optimize = False)
    buy_closedelta = DecimalParameter(12.0, 18.0, default=15.0, optimize = is_optimize_local_uptrend)

    rocr_1h = RealParameter(0.5, 1.0, default=0.54904, space='buy', optimize=True)
    rocr1_1h = RealParameter(0.5, 1.0, default=0.72, space='buy', optimize=True)
    bbdelta_close = RealParameter(0.0005, 0.02, default=0.01965, space='buy', optimize=True)
    closedelta_close = RealParameter(0.0005, 0.02, default=0.00556, space='buy', optimize=True)
    bbdelta_tail = RealParameter(0.7, 1.0, default=0.95089, space='buy', optimize=True)
    close_bblower = RealParameter(0.0005, 0.02, default=0.00799, space='buy', optimize=True)

    sell_fisher = RealParameter(0.1, 0.5, default=0.38414, space='sell', optimize=False)
    sell_bbmiddle_close = RealParameter(0.97, 1.1, default=1.07634, space='sell', optimize=False)

    is_optimize_deadfish = True
    sell_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05 , space='sell', optimize = is_optimize_deadfish)
    sell_deadfish_profit = DecimalParameter(-0.15, -0.05, default=-0.08 , space='sell', optimize = is_optimize_deadfish)
    sell_deadfish_bb_factor = DecimalParameter(0.90, 1.20, default=1.0 , space='sell', optimize = is_optimize_deadfish)
    sell_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.5 ,space='sell', optimize = is_optimize_deadfish)

    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)
    low_offset = DecimalParameter(0.985, 0.995, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(1.005, 1.015, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.010, 1.020, default=sell_params['high_offset_2'], space='sell', optimize=True)
    
    sell_trail_profit_min_1 = DecimalParameter(0.1, 0.25, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_1 = DecimalParameter(0.3, 0.5, default=0.4, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_1 = DecimalParameter(0.04, 0.1, default=0.03, space='sell', decimals=3, optimize=False, load=True)

    sell_trail_profit_min_2 = DecimalParameter(0.04, 0.1, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_2 = DecimalParameter(0.08, 0.25, default=0.11, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_2 = DecimalParameter(0.04, 0.2, default=0.015, space='sell', decimals=3, optimize=False, load=True)

    pHSL = DecimalParameter(-0.500, -0.040, default=-0.08, decimals=3, space='sell', optimize=False, load=True)

    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', optimize=False, load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', optimize=False, load=True)

    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell',optimize=False, load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', optimize=False,load=True)
    


    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]

        informative_pairs += [("BTC/USDT", "5m"),
                             ]
        return informative_pairs
    
    
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = len(filled_buys)

        if (last_candle is not None):
            if (current_profit > self.sell_trail_profit_min_1.value) & (current_profit < self.sell_trail_profit_max_1.value) & (((trade.max_rate - trade.open_rate) / 100) > (current_profit + self.sell_trail_down_1.value)):
                return 'trail_target_1'
            elif (current_profit > self.sell_trail_profit_min_2.value) & (current_profit < self.sell_trail_profit_max_2.value) & (((trade.max_rate - trade.open_rate) / 100) > (current_profit + self.sell_trail_down_2.value)):
                return 'trail_target_2'
            elif (current_profit > 3) & (last_candle['rsi'] > 85):
                 return 'RSI-85 target'            
            if (current_profit > 0) & (count_of_buys < 4) & (last_candle['close'] > last_candle['hma_50']) & (last_candle['close'] > (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) & (last_candle['rsi']>50) & (last_candle['volume'] > 0) & (last_candle['rsi_fast'] > last_candle['rsi_slow']):
                return 'sell signal1'
            if (current_profit > 0) & (count_of_buys >= 4) & (last_candle['close'] > last_candle['hma_50'] * 1.01) & (last_candle['close'] > (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) & (last_candle['rsi']>50) & (last_candle['volume'] > 0) & (last_candle['rsi_fast'] > last_candle['rsi_slow']):
                return 'sell signal1 * 1.01'
            if (current_profit > 0) & (last_candle['close'] > last_candle['hma_50']) & (last_candle['close'] > (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &  (last_candle['volume'] > 0) & (last_candle['rsi_fast'] > last_candle['rsi_slow']):
                return 'sell signal2'           
            if (    (current_profit < self.sell_deadfish_profit.value)
                and (last_candle['close'] < last_candle['ema_200'])
                and (last_candle['bb_width'] < self.sell_deadfish_bb_width.value)
                and (last_candle['close'] > last_candle['bb_middleband2'] * self.sell_deadfish_bb_factor.value)
                and (last_candle['volume_mean_12'] < last_candle['volume_mean_24'] * self.sell_deadfish_volume_factor.value)
                and (last_candle['cmf'] < 0.0)
            ):
                return f"sell_stoploss_deadfish"

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        
        info_tf = '5m'

        informative = self.dp.get_pair_dataframe('BTC/USDT', timeframe=info_tf)
        informative_btc = informative.copy().shift(1)



        dataframe['btc_close'] = informative_btc['close']
        dataframe['btc_ema_fast'] = ta.EMA(informative_btc, timeperiod=20)
        dataframe['btc_ema_slow'] = ta.EMA(informative_btc, timeperiod=25)
        dataframe['down'] = (dataframe['btc_ema_fast'] < dataframe['btc_ema_slow']).astype('int')

        for val in self.base_nb_candles_sell.range:
             dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)
        
        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)
        
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

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
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        dataframe['bb_delta_cluc'] = (dataframe['bb_middleband2_40'] - dataframe['bb_lowerband2_40']).abs()
        dataframe['ha_closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()

        stoch = ta.STOCHRSI(dataframe, 15, 20, 2, 2)
        dataframe['srsi_fk'] = stoch['fastk']
        dataframe['srsi_fd'] = stoch['fastd']

        mid, lower = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['mid'] = mid

        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()

        dataframe['bb_lowerband'] = dataframe['lower']
        dataframe['bb_middleband'] = dataframe['mid']

        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']
        dataframe['bb_delta'] = ((dataframe['bb_lowerband2'] - dataframe['bb_lowerband3']) / dataframe['bb_lowerband2'])

        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3)
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        
        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['vwap_low'] = vwap_low
        
        dataframe['vwap_upperband'] = vwap_high
        dataframe['vwap_middleband'] = vwap
        dataframe['vwap_lowerband'] = vwap_low
        dataframe['vwap_width'] = ( (dataframe['vwap_upperband'] - dataframe['vwap_lowerband']) / dataframe['vwap_middleband'] ) * 100

        dataframe['ema_vwap_diff_50'] = ( ( dataframe['ema_50'] - dataframe['vwap_lowerband'] ) / dataframe['ema_50'] )

        dataframe['tpct_change_0']   = top_percent_change_dca(dataframe,0)
        dataframe['tpct_change_1']   = top_percent_change_dca(dataframe,1)
        dataframe['tcp_percent_4'] =   top_percent_change_dca(dataframe , 4)

        dataframe['ewo'] = ewo(dataframe, 50, 200)

        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['sma_30'] = ta.SMA(dataframe, timeperiod=30)

        for val in self.buy_rmi_length.range:
            dataframe[f'rmi_length_{val}'] = RMI(dataframe, length=val, mom=4)

        for val in self.buy_cci_length.range:
            dataframe[f'cci_length_{val}'] = ta.CCI(dataframe, val)

        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        dataframe['bb_delta_cluc'] = (dataframe['bb_middleband2_40'] - dataframe['bb_lowerband2_40']).abs()

        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)

            
        
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)

        dataframe['r_14'] = williams_r(dataframe, period=14)

        dataframe['ema_5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema_10'] = ta.EMA(dataframe, timeperiod=10)

        dataframe['pm'], dataframe['pmx'] = pmax(heikinashi, MAtype=1, length=9, multiplier=27, period=10, src=3)
        dataframe['source'] = (dataframe['high'] + dataframe['low'] + dataframe['open'] + dataframe['close'])/4
        dataframe['pmax_thresh'] = ta.EMA(dataframe['source'], timeperiod=9)
        dataframe['sma_75'] = ta.SMA(dataframe, timeperiod=75)

        rsi = ta.RSI(dataframe)
        dataframe["rsi"] = rsi
        rsi = 0.1 * (rsi - 50)
        dataframe["fisher"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        inf_tf = '1h'

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        inf_heikinashi = qtpylib.heikinashi(informative)

        informative['ha_close'] = inf_heikinashi['close']
        informative['rocr'] = ta.ROCR(informative['ha_close'], timeperiod=168)
        informative['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        informative['cmf'] = chaikin_money_flow(dataframe, 20)
        sup_series = informative['low'].rolling(window = 5, center=True).apply(lambda row: self.is_support(row), raw=True).shift(2)
        informative['sup_level'] = Series(np.where(sup_series, np.where(informative['close'] < informative['open'], informative['close'], informative['open']), float('NaN'))).ffill()
        informative['roc'] = ta.ROC(informative, timeperiod=9)

        informative['r_480'] = williams_r(informative, period=480)

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
        informative['bb_lowerband2'] = bollinger2['lower']
        informative['bb_middleband2'] = bollinger2['mid']
        informative['bb_upperband2'] = bollinger2['upper']
        informative['bb_width'] = ((informative['bb_upperband2'] - informative['bb_lowerband2']) / informative['bb_middleband2'])
        
        informative['r_84'] = williams_r(informative, period=84)
        informative['cti_40'] = pta.cti(informative["close"], length=40)
        
        
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)
        
        dataframe['dx']  = ta.DX(dataframe)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['pdi'] = ta.PLUS_DI(dataframe)
        dataframe['mdi'] = ta.MINUS_DI(dataframe)
        dataframe[['bbl', 'bbm', 'bbu']] = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)[['lower', 'mid', 'upper']]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        btc_dump = (
                (dataframe['btc_close'].rolling(24).max() >= (dataframe['btc_close'] * 1.03 ))
            )  
        rsi_check = (
                (dataframe['rsi_84'] < 60) &
                (dataframe['rsi_112'] < 60)
            ) 

        dataframe.loc[
                ((dataframe[f'rmi_length_{self.buy_rmi_length.value}'] < self.buy_rmi.value) &
                (dataframe[f'cci_length_{self.buy_cci_length.value}'] <= self.buy_cci.value) &
                (dataframe['srsi_fk'] < self.buy_srsi_fk.value) &
                (dataframe['bb_delta'] > self.buy_bb_delta.value) &
                (dataframe['bb_width'] > self.buy_bb_width.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000 ) &    # from BinH
                (dataframe['close'] < dataframe['bb_lowerband3'] * self.buy_bb_factor.value)&
                (dataframe['roc_1h'] < self.buy_roc_1h.value) &
                (dataframe['bb_width_1h'] < self.buy_bb_width_1h.value)
            ),
        ['enter_long', 'enter_tag']] = (1, 'DIP signal')     

        dataframe.loc[

                ((dataframe['bb_delta'] > self.buy_bb_delta.value) &
                (dataframe['bb_width'] > self.buy_bb_width.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000 ) &    # from BinH
                (dataframe['close'] < dataframe['bb_lowerband3'] * self.buy_bb_factor.value)&
                (dataframe['roc_1h'] < self.buy_roc_1h.value) &
                (dataframe['bb_width_1h'] < self.buy_bb_width_1h.value)

            ),
        ['enter_long', 'enter_tag']] = (1, 'Break signal')    
        
        
        
        
        
        
        dataframe.loc[
                        
                    ((dataframe['rocr_1h'] > self.buy_clucha_rocr_1h.value ) &
                
                        (dataframe['bb_lowerband2_40'].shift() > 0) &
                        (dataframe['bb_delta_cluc'] > dataframe['ha_close'] * self.buy_clucha_bbdelta_close.value) &
                        (dataframe['ha_closedelta'] > dataframe['ha_close'] * self.buy_clucha_closedelta_close.value) &
                        (dataframe['tail'] < dataframe['bb_delta_cluc'] * self.buy_clucha_bbdelta_tail.value) &
                        (dataframe['ha_close'] < dataframe['bb_lowerband2_40'].shift()) &
                        (dataframe['close'] > (dataframe['sup_level_1h'] * 0.88)) &
                        (dataframe['ha_close'] < dataframe['ha_close'].shift()) 
            
                    ),
        ['enter_long', 'enter_tag']] = (1, 'cluc_HA')    
        
         
        dataframe.loc[    
                ((dataframe['ema_200'] > (dataframe['ema_200'].shift(12) * 1.01)) &
                (dataframe['ema_200'] > (dataframe['ema_200'].shift(48) * 1.07)) &
                (dataframe['bb_lowerband2_40'].shift().gt(0)) &
                (dataframe['bb_delta_cluc'].gt(dataframe['close'] * 0.056)) &
                (dataframe['closedelta'].gt(dataframe['close'] * 0.01)) &
                (dataframe['tail'].lt(dataframe['bb_delta_cluc'] * 0.5)) &
                (dataframe['close'].lt(dataframe['bb_lowerband2_40'].shift())) &
                (dataframe['close'].le(dataframe['close'].shift())) &
                (dataframe['close'] > dataframe['ema_50'] * 0.912)
            
            ),
        ['enter_long', 'enter_tag']] = (1, 'NFIX39')
        
        dataframe.loc[
                ((dataframe['close'] > (dataframe['sup_level_1h'] * 0.72)) &
                (dataframe['close'] < (dataframe['ema_16'] * 0.982)) &
                (dataframe['EWO'] < -10.0) &
                (dataframe['cti'] < -0.9)

            ),
        ['enter_long', 'enter_tag']] = (1, 'NFIX29')
        
        dataframe.loc[
                ((dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['ema_26'] - dataframe['ema_12'] > dataframe['open'] * self.buy_ema_diff.value) &
                (dataframe['ema_26'].shift() - dataframe['ema_12'].shift() > dataframe['open'] / 100) &
                (dataframe['close'] < dataframe['bb_lowerband2'] * self.buy_bb_factor.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000 ) 
                
             ),
        ['enter_long', 'enter_tag']] = (1, 'local_uptrend')
        
        dataframe.loc[
                (
                
                (dataframe['close'] < dataframe['vwap_low']) &
                (dataframe['tcp_percent_4'] > 0.053) & # 0.053)
                (dataframe['cti'] < -0.8) & # -0.8)
                (dataframe['rsi'] < 35) &
                (dataframe['rsi_84'] < 60) &
                (dataframe['rsi_112'] < 60) &

                (dataframe['volume'] > 0)
           ),
        ['enter_long', 'enter_tag']] = (1, 'vwap')
        
        dataframe.loc[
                ((dataframe['bb_width_1h'] > 0.131) &
                (dataframe['r_14'] < -51) &
                (dataframe['r_84_1h'] < -70) &
                (dataframe['cti'] < -0.845) &
                (dataframe['cti_40_1h'] < -0.735)
                &
                ( (dataframe['close'].rolling(48).max() >= (dataframe['close'] * 1.1 )) ) &


                (dataframe['btc_close'].rolling(24).max() >= (dataframe['btc_close'] * 1.03 ))
          ),
        ['enter_long', 'enter_tag']] = (1, 'insta_signal') 

        dataframe.loc[
            ((dataframe['close'] < (dataframe['ema_16'] * self.buy_44_ma_offset))&
            (dataframe['ewo'] < self.buy_44_ewo)&
            (dataframe['cti'] < self.buy_44_cti)&
            (dataframe['r_480_1h'] < self.buy_44_r_1h)&

            (dataframe['volume'] > 0)
          ),
        ['enter_long', 'enter_tag']] = (1, 'NFINext44') 


        dataframe.loc[  
            ((dataframe['pm'] > dataframe['pmax_thresh'])&
            (dataframe['close'] < dataframe['sma_75'] * self.buy_37_ma_offset)&
            (dataframe['ewo'] > self.buy_37_ewo)&
            (dataframe['rsi'] < self.buy_37_rsi)&
            (dataframe['cti'] < self.buy_37_cti)

        ),
        ['enter_long', 'enter_tag']] = (1, 'NFINext37')   

        dataframe.loc[ 
            ((dataframe['ema_26'] > dataframe['ema_12'])&
            ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_7))&
            ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100))&
            (dataframe['cti'] < self.buy_cti_7)      

        ),        
        ['enter_long', 'enter_tag']] = (1, 'NFINext7')   

        dataframe.loc[
                ((dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < 46) &
                (dataframe['rsi'] > 19) &
                (dataframe['close'] < dataframe['sma_15'] * 0.942) &
                (dataframe['cti'] < -0.86)
        ),        
        ['enter_long', 'enter_tag']] = (1, 'NFINext32')

        dataframe.loc[
                ((dataframe['bb_lowerband2_40'].shift() > 0) &
                (dataframe['bb_delta_cluc'] > dataframe['close'] * 0.059) &
                (dataframe['ha_closedelta'] > dataframe['close'] * 0.023) &
                (dataframe['tail'] < dataframe['bb_delta_cluc'] * 0.24) &
                (dataframe['close'] < dataframe['bb_lowerband2_40'].shift()) &
                (dataframe['close'] < dataframe['close'].shift()) &
                (btc_dump == 0)
        ),        
        ['enter_long', 'enter_tag']] = (1, 'sma_3')

        dataframe.loc[
                ((dataframe['close'] < dataframe['vwap_lowerband']) &
                (dataframe['tpct_change_1'] > 0.04) &
                (dataframe['cti'] < -0.8) &
                (dataframe['rsi'] < 35) &
                (rsi_check) &
                (btc_dump == 0)
        ),        
        ['enter_long', 'enter_tag']] = (1, 'WVAP')

        dataframe.loc[
            (
                (dataframe['dx']  > dataframe['mdi']) &
                (dataframe['adx'] > dataframe['mdi']) &
                (dataframe['pdi'] > dataframe['mdi'])
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Long DI enter')

        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['close'], dataframe['bbu'])
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Long Bollinger enter')

        dataframe.loc[
            (
                (dataframe['dx']  > dataframe['mdi']) &
                (dataframe['adx'] > dataframe['pdi']) &
                (dataframe['mdi'] > dataframe['pdi'])
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Short DI enter')

        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe['close'], dataframe['bbl'])
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Short Bollinger enter')
      
        return dataframe



    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (dataframe['fisher'] > self.sell_fisher.value) &
            (dataframe['ha_high'].le(dataframe['ha_high'].shift(1))) &
            (dataframe['ha_high'].shift(1).le(dataframe['ha_high'].shift(2))) &
            (dataframe['ha_close'].le(dataframe['ha_close'].shift(1))) &
            (dataframe['ema_fast'] > dataframe['ha_close']) &
            ((dataframe['ha_close'] * self.sell_bbmiddle_close.value) > dataframe['bb_middleband']) &
            (dataframe['volume'] > 0),
            'exit_long'
        ] = 0
        
        dataframe.loc[
            (dataframe['ha_close'] > dataframe['ha_open']) &
            (dataframe['ha_close'].shift(1) > dataframe['ha_open'].shift(1)) & # Previous HA was also green
            (dataframe['ha_open'].shift(1) > dataframe['ha_close'].shift(2)) & # Previous HA opened above prior HA close (gap up or strong green)
            (dataframe['volume'] > 0),
            'exit_short'
        ] = 0

        return dataframe
    
    initial_safety_order_trigger = -0.018
    max_safety_orders = 8
    safety_order_step_scale = 1.2
    safety_order_volume_scale = 1.4
      
    def top_percent_change_dca(self, dataframe: DataFrame, length: int) -> float:
        """
        Percentage change of the current close from the range maximum Open price

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        if length == 0:
            return (dataframe['open'] - dataframe['close']) / dataframe['close']
        else:
            return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']
        
    def leverage(self, pair: str, current_time: "datetime", current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return 6

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        if current_profit > self.initial_safety_order_trigger:
            return None


        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()





        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = len(filled_buys)
        if count_of_buys == 1 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] < last_candle['open']) :
            
                return None
        elif count_of_buys == 2 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] < last_candle['open']) and (last_candle['ema_vwap_diff_50'] < 0.215):
            
                return None
        elif count_of_buys == 3 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] < last_candle['open'])and (last_candle['ema_vwap_diff_50'] < 0.215) :
           
                
                return None
        elif count_of_buys == 4 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] < last_candle['open'])and (last_candle['ema_vwap_diff_50'] < 0.215) and (last_candle['ema_5']) >= (last_candle['ema_10']):
            
                
                return None
        elif count_of_buys == 5 and (last_candle['cmf_1h'] < 0.00) and (last_candle['close'] < last_candle['open']) and (last_candle['rsi_14_1h'] < 30) and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] < last_candle['open']) and (last_candle['ema_vwap_diff_50'] < 0.215) and (last_candle['ema_5']) >= (last_candle['ema_10']):
           
                logger.info(f"DCA for {trade.pair} waiting for cmf_1h ({last_candle['cmf_1h']}) to rise above 0. Waiting for rsi_1h ({last_candle['rsi_14_1h']})to rise above 30")
                return None
        elif count_of_buys == 6 and (last_candle['cmf_1h'] < 0.00) and (last_candle['close'] < last_candle['open']) and (last_candle['rsi_14_1h'] < 30) and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] < last_candle['open'] and (last_candle['ema_vwap_diff_50'] < 0.215)) and (last_candle['ema_5']) >= (last_candle['ema_10']): 
            
                logger.info(f"DCA for {trade.pair} waiting for cmf_1h ({last_candle['cmf_1h']}) to rise above 0. Waiting for rsi_1h ({last_candle['rsi_14_1h']})to rise above 30")
                return None
        elif count_of_buys == 7 and (last_candle['cmf_1h'] < 0.00) and (last_candle['close'] < last_candle['open']) and (last_candle['rsi_14_1h'] < 30) and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] < last_candle['open'] and (last_candle['ema_vwap_diff_50'] < 0.215)) and (last_candle['ema_5']) >= (last_candle['ema_10']):
            
                logger.info(f"DCA for {trade.pair} waiting for cmf_1h ({last_candle['cmf_1h']}) to rise above 0. Waiting for rsi_1h ({last_candle['rsi_14_1h']})to rise above 30")
                return None
        elif count_of_buys == 8 and (last_candle['cmf_1h'] < 0.00) and (last_candle['close'] < last_candle['open']) and (last_candle['rsi_14_1h'] < 30) and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] < last_candle['open'] and (last_candle['ema_vwap_diff_50'] < 0.215)) and (last_candle['ema_5']) >= (last_candle['ema_10']):
            
                logger.info(f"DCA for {trade.pair} waiting for cmf_1h ({last_candle['cmf_1h']}) to rise above 0. Waiting for rsi_1h ({last_candle['rsi_14_1h']})to rise above 30")
                return None
        
        if 1 <= count_of_buys <= self.max_safety_orders:
            safety_order_trigger = (abs(self.initial_safety_order_trigger) * count_of_buys)
            if (self.safety_order_step_scale > 1):
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (math.pow(self.safety_order_step_scale,(count_of_buys - 1)) - 1) / (self.safety_order_step_scale - 1))
            elif (self.safety_order_step_scale < 1):
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (1 - math.pow(self.safety_order_step_scale,(count_of_buys - 1))) / (1 - self.safety_order_step_scale))

            if current_profit <= (-1 * abs(safety_order_trigger)):
                try:

                    stake_amount = filled_buys[0].cost

                    stake_amount = stake_amount * math.pow(self.safety_order_volume_scale,(count_of_buys - 1))
                    amount = stake_amount / current_rate
                    logger.info(f"Initiating safety order buy #{count_of_buys} for {trade.pair} with stake amount of {stake_amount} which equals {amount}")
                    return stake_amount
                except Exception as exception:
                    logger.info(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}') 
                    return None

        return None

def pmax(df, period, multiplier, length, MAtype, src):

    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)

    mavalue = 'MA_' + str(MAtype) + '_' + str(length)
    atr = 'ATR_' + str(period)
    pm = 'pm_' + str(period) + '_' + str(multiplier) + '_' + str(length) + '_' + str(MAtype)
    pmx = 'pmX_' + str(period) + '_' + str(multiplier) + '_' + str(length) + '_' + str(MAtype)

    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4

    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        mavalue = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        mavalue = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        mavalue = VIDYA(df, length=length)
    elif MAtype == 6:
        mavalue = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        mavalue = ta.WMA(df, timeperiod=length)
    elif MAtype == 8:
        mavalue = vwma(df, length)
    elif MAtype == 9:
        mavalue = zema(df, period=length)

    df[atr] = ta.ATR(df, timeperiod=period)
    df['basic_ub'] = mavalue + ((multiplier/10) * df[atr])
    df['basic_lb'] = mavalue - ((multiplier/10) * df[atr])


    basic_ub = df['basic_ub'].values
    final_ub = np.full(len(df), 0.00)
    basic_lb = df['basic_lb'].values
    final_lb = np.full(len(df), 0.00)

    for i in range(period, len(df)):
        final_ub[i] = basic_ub[i] if (
            basic_ub[i] < final_ub[i - 1]
            or mavalue[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = basic_lb[i] if (
            basic_lb[i] > final_lb[i - 1]
            or mavalue[i - 1] < final_lb[i - 1]) else final_lb[i - 1]

    df['final_ub'] = final_ub
    df['final_lb'] = final_lb

    pm_arr = np.full(len(df), 0.00)
    for i in range(period, len(df)):
        pm_arr[i] = (
            final_ub[i] if (pm_arr[i - 1] == final_ub[i - 1]
                                    and mavalue[i] <= final_ub[i])
        else final_lb[i] if (
            pm_arr[i - 1] == final_ub[i - 1]
            and mavalue[i] > final_ub[i]) else final_lb[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] >= final_lb[i]) else final_ub[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] < final_lb[i]) else 0.00)

    pm = Series(pm_arr)

    pmx = np.where((pm_arr > 0.00), np.where((mavalue < pm_arr), 'down',  'up'), np.NaN)

    return pm, pmx
    


