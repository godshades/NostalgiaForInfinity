import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    """Reusable market analysis functions"""
    
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate dynamic support and resistance levels"""
        support = df['low'].rolling(window=window).min()
        resistance = df['high'].rolling(window=window).max()
        return support, resistance
    
    @staticmethod
    def detect_trend_change(df: pd.DataFrame, sensitivity: float = 0.02) -> pd.Series:
        """Detect trend changes using multiple indicators"""
        ema_short = df['close'].ewm(span=10).mean()
        ema_long = df['close'].ewm(span=30).mean()
        
        trend_change = (
            ((ema_short > ema_long) & (ema_short.shift(1) <= ema_long.shift(1))) |  # Bullish cross
            ((ema_short < ema_long) & (ema_short.shift(1) >= ema_long.shift(1)))     # Bearish cross
        )
        
        return trend_change
    
    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
        """Calculate volume profile for support/resistance identification"""
        price_range = df['high'] - df['low']
        price_bins = pd.cut(df['close'], bins=bins)
        volume_profile = df.groupby(price_bins)['volume'].sum()
        
        # Find high volume nodes (potential support/resistance)
        high_volume_threshold = volume_profile.quantile(0.7)
        high_volume_prices = volume_profile[volume_profile > high_volume_threshold]
        
        return high_volume_prices

class SignalGenerator:
    """Reusable signal generation logic"""
    
    @staticmethod
    def calculate_entry_score(df: pd.DataFrame, row_idx: int, 
                             weights: dict = None) -> float:
        """Calculate unified entry score for both long and short"""
        if weights is None:
            weights = {
                'momentum': 0.25,
                'volume': 0.20,
                'structure': 0.20,
                'confluence': 0.25,
                'risk': 0.10
            }
        
        row = df.iloc[row_idx]
        
        score = 0
        score += weights['momentum'] * row.get('momentum_quality', 0) / 5
        score += weights['volume'] * row.get('volume_pressure', 0) / 5
        score += weights['structure'] * row.get('structure_score', 0) / 10
        score += weights['confluence'] * row.get('confluence_score', 0) / 5
        score += weights['risk'] * (1 - row.get('atr', 0) / row.get('close', 1))
        
        return score
    
    @staticmethod
    def generate_exit_conditions(df: pd.DataFrame, trade_side: str = 'long') -> pd.DataFrame:
        """Generate exit conditions for both long and short trades"""
        
        exit_conditions = pd.DataFrame(index=df.index)
        
        if trade_side == 'long':
            # Profit taking
            exit_conditions['profit_target'] = (
                (df['rsi'] > 70) &
                (df['close'] > df['open']) &
                (df.get('near_resistance', 0) == 1)
            )
            
            # Stop loss
            exit_conditions['stop_loss'] = (
                (df['close'] < df.get('minima_sort_threshold', df['close'])) &
                (df['rsi'] < 30)
            )
            
            # Trend reversal
            exit_conditions['trend_reversal'] = (
                (df.get('structure_break_down', 0) == 1) &
                (df['volume'] > df['avg_volume'] * 1.5)
            )
            
        else:  # short
            # Profit taking
            exit_conditions['profit_target'] = (
                (df['rsi'] < 30) &
                (df['close'] < df['open']) &
                (df.get('near_support', 0) == 1)
            )
            
            # Stop loss
            exit_conditions['stop_loss'] = (
                (df['close'] > df.get('maxima_sort_threshold', df['close'])) &
                (df['rsi'] > 70)
            )
            
            # Trend reversal
            exit_conditions['trend_reversal'] = (
                (df.get('structure_break_up', 0) == 1) &
                (df['volume'] > df['avg_volume'] * 1.5)
            )
        
        # Combine all exit conditions
        exit_conditions['exit_signal'] = exit_conditions.any(axis=1)
        
        return exit_conditions

class RiskManager:
    """Risk management utilities"""
    
    @staticmethod
    def calculate_position_size(
        portfolio_value: float,
        risk_per_trade: float,
        stop_loss_pct: float,
        max_position_pct: float = 0.1
    ) -> float:
        """Calculate position size based on risk"""
        
        # Kelly Criterion inspired sizing
        risk_amount = portfolio_value * risk_per_trade
        position_size = risk_amount / abs(stop_loss_pct)
        
        # Apply maximum position limit
        max_position = portfolio_value * max_position_pct
        position_size = min(position_size, max_position)
        
        return position_size
    
    @staticmethod
    def calculate_dynamic_stop_loss(
        df: pd.DataFrame,
        row_idx: int,
        base_stop: float = 0.02,
        use_atr: bool = True
    ) -> float:
        """Calculate dynamic stop loss based on market conditions"""
        
        row = df.iloc[row_idx]
        
        if use_atr and 'atr' in df.columns:
            atr = row['atr']
            close = row['close']
            atr_stop = (atr / close) * 2  # 2x ATR
            
            # Adjust based on volatility
            if 'volatility' in df.columns:
                volatility = row['volatility']
                if volatility > 0.05:  # High volatility
                    atr_stop *= 1.5
            
            return min(atr_stop, 0.15)  # Cap at 15%
        
        return base_stop
    
    @staticmethod
    def check_correlation_risk(
        pairs: list,
        correlation_threshold: float = 0.7
    ) -> dict:
        """Check correlation between pairs to avoid concentrated risk"""
        # This would need price data for all pairs
        # Simplified example structure
        correlation_matrix = {}
        high_correlation_pairs = []
        
        # In practice, calculate actual correlations
        # For now, return empty risk assessment
        return {
            'high_correlation_pairs': high_correlation_pairs,
            'risk_level': 'low'
        }

class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    @staticmethod
    def batch_calculate_indicators(df: pd.DataFrame, indicators: list) -> pd.DataFrame:
        """Batch calculate multiple indicators efficiently"""
        
        # Pre-calculate common base values
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Calculate all indicators in one pass
        for indicator in indicators:
            if indicator == 'sma':
                for period in [10, 20, 50]:
                    df[f'sma_{period}'] = df['close'].rolling(period).mean()
            
            elif indicator == 'rsi':
                # Vectorized RSI calculation
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
            
            elif indicator == 'volume_profile':
                df['volume_ma'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df