# AlexNexusForgeV7NS Strategy Documentation

## Overview
Advanced multi-factor confluence trading strategy with state-based trade management.

## Key Features
- **State Machine**: Tracks trade lifecycle
- **Risk Management**: Dynamic position sizing and stop losses
- **Performance Optimized**: Caching and batch calculations
- **Modular Design**: Reusable components

## Components

### 1. Market Analyzer
- Support/Resistance detection
- Trend change identification
- Volume profile analysis

### 2. Signal Generator
- Unified entry scoring
- Exit condition generation
- Multi-timeframe analysis

### 3. Risk Manager
- Position sizing
- Dynamic stop losses
- Correlation risk assessment

### 4. Trade State Manager
- State machine implementation
- Trade lifecycle tracking
- Metadata management

## Configuration

### Risk Parameters
- `risk_per_trade`: 2% default
- `max_correlation`: 0.7
- `max_position_pct`: 10%

### Performance Settings
- Cache enabled by default
- Batch indicator calculation
- Parallel processing for independent calculations

## Testing
Run unit tests:
```bash
python -m pytest test_strategy.py