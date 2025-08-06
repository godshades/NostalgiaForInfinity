from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TradeState(Enum):
    """Trade states for state machine"""
    IDLE = "idle"
    ENTERING = "entering"
    MANAGING = "managing"
    SCALING_IN = "scaling_in"
    SCALING_OUT = "scaling_out"
    EXITING = "exiting"
    EMERGENCY_EXIT = "emergency_exit"

class TradeStateManager:
    """Manage trade lifecycle with state machine"""
    
    def __init__(self):
        self.states: Dict[str, TradeState] = {}
        self.trade_metadata: Dict[str, Dict[str, Any]] = {}
    
    def get_state(self, pair: str) -> TradeState:
        """Get current state for a pair"""
        return self.states.get(pair, TradeState.IDLE)
    
    def transition(self, pair: str, new_state: TradeState, metadata: Dict[str, Any] = None):
        """Transition to a new state"""
        old_state = self.get_state(pair)
        
        # Validate transition
        if not self._is_valid_transition(old_state, new_state):
            logger.warning(f"{pair}: Invalid transition from {old_state} to {new_state}")
            return False
        
        # Update state
        self.states[pair] = new_state
        
        # Update metadata
        if metadata:
            if pair not in self.trade_metadata:
                self.trade_metadata[pair] = {}
            self.trade_metadata[pair].update(metadata)
            self.trade_metadata[pair]['last_transition'] = datetime.now()
        
        logger.info(f"{pair}: State transition {old_state} -> {new_state}")
        return True
    
    def _is_valid_transition(self, from_state: TradeState, to_state: TradeState) -> bool:
        """Check if state transition is valid"""
        
        valid_transitions = {
            TradeState.IDLE: [TradeState.ENTERING],
            TradeState.ENTERING: [TradeState.MANAGING, TradeState.EMERGENCY_EXIT],
            TradeState.MANAGING: [TradeState.SCALING_IN, TradeState.SCALING_OUT, 
                                 TradeState.EXITING, TradeState.EMERGENCY_EXIT],
            TradeState.SCALING_IN: [TradeState.MANAGING, TradeState.EMERGENCY_EXIT],
            TradeState.SCALING_OUT: [TradeState.MANAGING, TradeState.EXITING],
            TradeState.EXITING: [TradeState.IDLE],
            TradeState.EMERGENCY_EXIT: [TradeState.IDLE]
        }
        
        return to_state in valid_transitions.get(from_state, [])
    
    def should_allow_entry(self, pair: str) -> bool:
        """Check if entry is allowed based on state"""
        return self.get_state(pair) == TradeState.IDLE
    
    def should_allow_dca(self, pair: str) -> bool:
        """Check if DCA is allowed based on state"""
        return self.get_state(pair) in [TradeState.MANAGING, TradeState.SCALING_IN]
    
    def should_force_exit(self, pair: str) -> bool:
        """Check if exit should be forced"""
        state = self.get_state(pair)
        
        # Check for emergency conditions
        if state == TradeState.EMERGENCY_EXIT:
            return True
        
        # Check for timeout
        if pair in self.trade_metadata:
            last_transition = self.trade_metadata[pair].get('last_transition')
            if last_transition and (datetime.now() - last_transition) > timedelta(hours=48):
                logger.warning(f"{pair}: Trade timeout - forcing exit")
                return True
        
        return False
    
    def get_trade_quality(self, pair: str) -> str:
        """Get trade quality from metadata"""
        if pair in self.trade_metadata:
            return self.trade_metadata[pair].get('quality', 'unknown')
        return 'unknown'
    
    def reset_state(self, pair: str):
        """Reset state for a pair"""
        if pair in self.states:
            del self.states[pair]
        if pair in self.trade_metadata:
            del self.trade_metadata[pair]