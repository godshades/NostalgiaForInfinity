import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from strategy_constants import *
from trade_state_manager import TradeState  # Add this import

logger = logging.getLogger(__name__)

class MMLExitSystem:
    """Advanced Murrey Math Level based exit system"""
    
    def __init__(self, use_emergency_exits: bool = True):
        self.use_emergency_exits = use_emergency_exits
        
    def calculate_exits(self, df: pd.DataFrame, can_short: bool = True) -> pd.DataFrame:
        """
        Main entry point for MML exit calculations
        This is your _populate_custom_exits_advanced method, but enhanced
        """
        
        # Initialize exit columns
        df["exit_long"] = 0
        df["exit_short"] = 0
        df["exit_tag"] = ""
        
        # Calculate MML market structure
        mml_structure = self._calculate_mml_structure(df)
        
        # Calculate long exits
        long_exits = self._calculate_long_exits(df, mml_structure)
        
        # Calculate short exits if enabled
        if can_short:
            short_exits = self._calculate_short_exits(df, mml_structure)
        else:
            short_exits = pd.DataFrame(index=df.index)
            short_exits['any_exit'] = False
        
        # Apply exits with priority system
        df = self._apply_exits_with_priority(df, long_exits, short_exits)
        
        return df

    def calculate_exits_with_state(self, df: pd.DataFrame, state_manager, pair: str, 
                                can_short: bool = True) -> pd.DataFrame:
        """Calculate exits with state manager integration"""
        
        # First calculate normal exits
        df = self.calculate_exits(df, can_short)
        
        # Check state for emergency conditions
        current_state = state_manager.get_state(pair)
        
        # Force exit if in emergency state
        if current_state == TradeState.EMERGENCY_EXIT:
            df.iloc[-1, df.columns.get_loc('exit_long')] = 1
            df.iloc[-1, df.columns.get_loc('exit_short')] = 1
            df.iloc[-1, df.columns.get_loc('exit_tag')] = 'emergency_state_exit'
        
        # Update state based on exit signals
        if df['exit_long'].any() or df['exit_short'].any():
            # Find the type of exit
            last_exit_idx = df[(df['exit_long'] == 1) | (df['exit_short'] == 1)].index[-1]
            exit_tag = df.loc[last_exit_idx, 'exit_tag']
            
            if 'Emergency' in exit_tag:
                state_manager.transition(pair, TradeState.EMERGENCY_EXIT)
            else:
                state_manager.transition(pair, TradeState.EXITING)
        
        return df
    
    def _calculate_mml_structure(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MML market structure indicators"""
        
        structure = {}
        
        # Bullish/Bearish structure (from your original code)
        structure['bullish_mml'] = (
            (df["close"] > df["[6/8]P"]) |
            ((df["close"] > df["[4/8]P"]) & (df["close"].shift(5) < df["[4/8]P"].shift(5)))
        )
        
        structure['bearish_mml'] = (
            (df["close"] < df["[2/8]P"]) |
            ((df["close"] < df["[4/8]P"]) & (df["close"].shift(5) > df["[4/8]P"].shift(5)))
        )
        
        # MML resistance/support levels
        structure['at_resistance'] = (
            (df["high"] >= df["[6/8]P"]) |  # At 75%
            (df["high"] >= df["[7/8]P"]) |  # At 87.5%
            (df["high"] >= df["[8/8]P"])    # At 100%
        )
        
        structure['at_support'] = (
            (df["low"] <= df["[2/8]P"]) |   # At 25%
            (df["low"] <= df["[1/8]P"]) |   # At 12.5%
            (df["low"] <= df["[0/8]P"])     # At 0%
        )
        
        return structure
    
    def _calculate_long_exits(self, df: pd.DataFrame, mml_structure: Dict) -> pd.DataFrame:
        """Calculate all long exit signals (your original logic, organized)"""
        
        exits = pd.DataFrame(index=df.index)
        
        # 1. Profit-Taking Exits
        exits['resistance_profit'] = (
            mml_structure['at_resistance'] &
            (df["close"] < df["high"]) &
            (df["rsi"] > 65) &
            (df["maxima"] == 1) &
            (df["volume"] > df["volume"].rolling(10).mean())
        )
        
        exits['extreme_overbought'] = (
            (df["close"] > df["[7/8]P"]) &
            (df["rsi"] > RSI_EXTREME_OVERBOUGHT - 5) &  # 75
            (df["close"] < df["close"].shift(1)) &
            (df["maxima"] == 1)
        )
        
        exits['volume_exhaustion'] = (
            mml_structure['at_resistance'] &
            (df["volume"] < df["volume"].rolling(20).mean() * VOLUME_EXHAUSTION_MULTIPLIER) &
            (df["rsi"] > RSI_OVERBOUGHT) &
            (df["close"] < df["close"].shift(1)) &
            (df["close"] < df["close"].rolling(3).mean())
        )
        
        # 2. Structure Breakdown (your enhanced version)
        exits['structure_breakdown'] = (
            (df["close"] < df["[4/8]P"]) &
            (df["close"].shift(1) >= df["[4/8]P"].shift(1)) &
            mml_structure['bullish_mml'].shift(1) &
            (df["close"] < df["[4/8]P"] * 0.995) &
            (df["close"] < df["close"].shift(1)) &
            (df["close"] < df["close"].shift(2)) &
            (df["rsi"] < 45) &
            (df["volume"] > df["volume"].rolling(15).mean() * 2.0) &
            (df["close"] < df["open"]) &
            (df["low"] < df["low"].shift(1)) &
            (df["close"] < df["close"].rolling(3).mean()) &
            (df.get("momentum_quality", 0) < 0)
        )
        
        # 3. Momentum Divergence
        exits['momentum_divergence'] = (
            mml_structure['at_resistance'] &
            (df["rsi"] < df["rsi"].shift(1)) &
            (df["rsi"].shift(1) < df["rsi"].shift(2)) &
            (df["rsi"] < df["rsi"].shift(3)) &
            (df["close"] >= df["close"].shift(1)) &
            (df["maxima"] == 1) &
            (df["rsi"] > 60)
        )
        
        # 4. Range Exit
        exits['range_exit'] = (
            (df["close"] >= df["[2/8]P"]) &
            (df["close"] <= df["[6/8]P"]) &
            (df["high"] >= df["[6/8]P"]) &
            (df["close"] < df["[6/8]P"] * 0.995) &
            (df["rsi"] > 65) &
            (df["maxima"] == 1) &
            (df["volume"] > df["volume"].rolling(10).mean() * 1.2)
        )
        
        # 5. Emergency Exit
        if self.use_emergency_exits:
            exits['emergency'] = (
                (df["close"] < df["[0/8]P"]) &
                (df["rsi"] < RSI_EXTREME_OVERSOLD) &
                (df["volume"] > df["volume"].rolling(20).mean() * 2.5) &
                (df["close"] < df["close"].shift(1)) &
                (df["close"] < df["close"].shift(2)) &
                (df["close"] < df["open"])
            )
        else:
            exits['emergency'] = False
        
        # Combine all exits
        exits['any_exit'] = exits[
            ['resistance_profit', 'extreme_overbought', 'volume_exhaustion',
             'structure_breakdown', 'momentum_divergence', 'range_exit', 'emergency']
        ].any(axis=1)
        
        return exits
    
    def _calculate_short_exits(self, df: pd.DataFrame, mml_structure: Dict) -> pd.DataFrame:
        """Calculate all short exit signals (your original logic, organized)"""
        
        exits = pd.DataFrame(index=df.index)
        
        # 1. Profit-Taking Exits
        exits['support_profit'] = (
            mml_structure['at_support'] &
            (df["close"] > df["low"]) &
            (df["rsi"] < 35) &
            (df["minima"] == 1) &
            (df["volume"] > df["volume"].rolling(10).mean())
        )
        
        exits['extreme_oversold'] = (
            (df["close"] < df["[1/8]P"]) &
            (df["rsi"] < 25) &
            (df["close"] > df["close"].shift(1)) &
            (df["minima"] == 1)
        )
        
        exits['volume_exhaustion'] = (
            mml_structure['at_support'] &
            (df["volume"] < df["volume"].rolling(20).mean() * VOLUME_EXHAUSTION_MULTIPLIER) &
            (df["rsi"] < RSI_OVERSOLD) &
            (df["close"] > df["close"].shift(1)) &
            (df["close"] > df["close"].rolling(3).mean())
        )
        
        # 2. Structure Breakout
        exits['structure_breakout'] = (
            (df["close"] > df["[4/8]P"]) &
            (df["close"].shift(1) <= df["[4/8]P"].shift(1)) &
            mml_structure['bearish_mml'].shift(1) &
            (df["close"] > df["[4/8]P"] * 1.005) &
            (df["close"] > df["close"].shift(1)) &
            (df["close"] > df["close"].shift(2)) &
            (df["rsi"] > 55) &
            (df["volume"] > df["volume"].rolling(15).mean() * 2.0) &
            (df["close"] > df["open"]) &
            (df["high"] > df["high"].shift(1)) &
            (df.get("momentum_quality", 0) > 0)
        )
        
        # 3. Momentum Divergence
        exits['momentum_divergence'] = (
            mml_structure['at_support'] &
            (df["rsi"] > df["rsi"].shift(1)) &
            (df["rsi"].shift(1) > df["rsi"].shift(2)) &
            (df["rsi"] > df["rsi"].shift(3)) &
            (df["close"] <= df["close"].shift(1)) &
            (df["minima"] == 1) &
            (df["rsi"] < 40)
        )
        
        # 4. Range Exit
        exits['range_exit'] = (
            (df["close"] >= df["[2/8]P"]) &
            (df["close"] <= df["[6/8]P"]) &
            (df["low"] <= df["[2/8]P"]) &
            (df["close"] > df["[2/8]P"] * 1.005) &
            (df["rsi"] < 35) &
            (df["minima"] == 1) &
            (df["volume"] > df["volume"].rolling(10).mean() * 1.2)
        )
        
        # 5. Emergency Exit
        if self.use_emergency_exits:
            exits['emergency'] = (
                (df["close"] > df["[8/8]P"]) &
                (df["rsi"] > RSI_EXTREME_OVERBOUGHT) &
                (df["volume"] > df["volume"].rolling(20).mean() * 2.5) &
                (df["close"] > df["close"].shift(1)) &
                (df["close"] > df["close"].shift(2)) &
                (df["close"] > df["open"])
            )
        else:
            exits['emergency'] = False
        
        # Combine all exits
        exits['any_exit'] = exits[
            ['support_profit', 'extreme_oversold', 'volume_exhaustion',
             'structure_breakout', 'momentum_divergence', 'range_exit', 'emergency']
        ].any(axis=1)
        
        return exits
    
    def _apply_exits_with_priority(self, df: pd.DataFrame, 
                                   long_exits: pd.DataFrame, 
                                   short_exits: pd.DataFrame) -> pd.DataFrame:
        """Apply exit signals with proper priority and coordination"""
        
        # Coordinate with entry signals if they exist
        has_long_entry = "enter_long" in df.columns and (df["enter_long"] == 1).any()
        has_short_entry = "enter_short" in df.columns and (df["enter_short"] == 1).any()
        
        # Don't exit if we have a new entry signal at the same candle
        if has_long_entry:
            long_entry_mask = df["enter_long"] == 1
            long_exits['any_exit'] = long_exits['any_exit'] & (~long_entry_mask)
        
        if has_short_entry:
            short_entry_mask = df["enter_short"] == 1
            short_exits['any_exit'] = short_exits['any_exit'] & (~short_entry_mask)
        
        # Apply Long Exits with priority
        df.loc[long_exits['any_exit'], "exit_long"] = 1
        
        # Priority system for exit tags (Emergency > Structure > Profit)
        for idx in df[df["exit_long"] == 1].index:
            if long_exits.loc[idx, 'emergency']:
                df.loc[idx, "exit_tag"] = "MML_Emergency_Long"
            elif long_exits.loc[idx, 'structure_breakdown']:
                df.loc[idx, "exit_tag"] = "MML_Structure_Breakdown"
            elif long_exits.loc[idx, 'resistance_profit']:
                df.loc[idx, "exit_tag"] = "MML_Resistance_Profit"
            elif long_exits.loc[idx, 'extreme_overbought']:
                df.loc[idx, "exit_tag"] = "MML_Extreme_Overbought"
            elif long_exits.loc[idx, 'volume_exhaustion']:
                df.loc[idx, "exit_tag"] = "MML_Volume_Exhaustion"
            elif long_exits.loc[idx, 'momentum_divergence']:
                df.loc[idx, "exit_tag"] = "MML_Momentum_Divergence"
            elif long_exits.loc[idx, 'range_exit']:
                df.loc[idx, "exit_tag"] = "MML_Range_Exit"
        
        # Apply Short Exits with priority
        if not short_exits.empty and 'any_exit' in short_exits.columns:
            df.loc[short_exits['any_exit'], "exit_short"] = 1
            
            for idx in df[df["exit_short"] == 1].index:
                if short_exits.loc[idx, 'emergency']:
                    df.loc[idx, "exit_tag"] = "MML_Emergency_Short"
                elif short_exits.loc[idx, 'structure_breakout']:
                    df.loc[idx, "exit_tag"] = "MML_Structure_Breakout"
                elif short_exits.loc[idx, 'support_profit']:
                    df.loc[idx, "exit_tag"] = "MML_Support_Profit"
                elif short_exits.loc[idx, 'extreme_oversold']:
                    df.loc[idx, "exit_tag"] = "MML_Extreme_Oversold"
                elif short_exits.loc[idx, 'volume_exhaustion']:
                    df.loc[idx, "exit_tag"] = "MML_Volume_Exhaustion_Short"
                elif short_exits.loc[idx, 'momentum_divergence']:
                    df.loc[idx, "exit_tag"] = "MML_Momentum_Divergence_Short"
                elif short_exits.loc[idx, 'range_exit']:
                    df.loc[idx, "exit_tag"] = "MML_Range_Exit_Short"
        
        return df
    
    def get_exit_reason(self, df: pd.DataFrame, idx: int, side: str = 'long') -> str:
        """Get detailed exit reason for logging"""
        
        reasons = []
        
        if side == 'long':
            if df.loc[idx, 'exit_long'] == 1:
                tag = df.loc[idx, 'exit_tag']
                
                # Add detailed context
                reasons.append(f"{tag}")
                reasons.append(f"RSI: {df.loc[idx, 'rsi']:.1f}")
                reasons.append(f"Close: {df.loc[idx, 'close']:.4f}")
                
                if 'Structure' in tag:
                    reasons.append(f"MML [4/8]P: {df.loc[idx, '[4/8]P']:.4f}")
                elif 'Resistance' in tag:
                    reasons.append(f"MML [6/8]P: {df.loc[idx, '[6/8]P']:.4f}")
        
        return " | ".join(reasons)