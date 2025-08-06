# trade_state_manager.py - Single File Version
import json
import os
from pathlib import Path
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from threading import Lock
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
    def __init__(self, bot_name: str = "default", state_file_path: str = "user_data/state_data"):
        self.states: Dict[str, TradeState] = {}
        self.trade_metadata: Dict[str, Dict[str, Any]] = {}
        self.bot_name = bot_name
        
        # Single file for all states
        self.state_file_path = Path(state_file_path)
        self.state_file_path.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_file_path / f"{self.bot_name}_states.json"
        
        # Thread safety for file operations
        self._file_lock = Lock()
        
        # Auto-save interval
        self._last_save = datetime.now()
        self._auto_save_interval = 300  # seconds
        
        logger.info(f"TradeStateManager initialized for bot: {self.bot_name}")
        
        # Load existing states
        self.load_all_states()
        
    def get_state(self, pair: str) -> TradeState:
        """Get current state for a pair"""
        return self.states.get(pair, TradeState.IDLE)
    
    def _serialize_state_data(self) -> Dict:
        """Serialize all states to dictionary"""
        data = {
            'bot_name': self.bot_name,
            'version': '2.0',
            'last_updated': datetime.now().isoformat(),
            'pairs': {}
        }
        
        for pair in self.states.keys():
            pair_data = {
                'state': self.states[pair].value,
                'metadata': self.trade_metadata.get(pair, {}).copy()
            }
            
            # Convert datetime objects
            for key, value in pair_data['metadata'].items():
                if isinstance(value, datetime):
                    pair_data['metadata'][key] = value.isoformat()
            
            data['pairs'][pair] = pair_data
        
        return data
    
    def save_all_states(self, force: bool = False):
        """Save all states to single file"""
        # Check if we should auto-save
        if not force and (datetime.now() - self._last_save).total_seconds() < self._auto_save_interval:
            return
        
        with self._file_lock:
            try:
                data = self._serialize_state_data()
                
                # Write to temp file first (atomic operation)
                temp_file = self.state_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Rename to actual file
                temp_file.replace(self.state_file)
                
                self._last_save = datetime.now()
                logger.debug(f"[{self.bot_name}] Saved {len(self.states)} states")
                
            except Exception as e:
                logger.error(f"[{self.bot_name}] Failed to save states: {e}")
    
    def load_all_states(self):
        """Load all states from single file"""
        if not self.state_file.exists():
            logger.info(f"[{self.bot_name}] No existing state file found")
            return
        
        with self._file_lock:
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                # Verify bot name
                if data.get('bot_name') != self.bot_name:
                    logger.warning(f"State file belongs to different bot: {data.get('bot_name')}")
                    return
                
                # Check file age
                last_updated = datetime.fromisoformat(data['last_updated'])
                file_age_hours = (datetime.now() - last_updated).total_seconds() / 3600
                
                loaded_count = 0
                stale_count = 0
                
                for pair, pair_data in data.get('pairs', {}).items():
                    # Restore state
                    self.states[pair] = TradeState(pair_data['state'])
                    
                    # Restore metadata
                    metadata = pair_data.get('metadata', {})
                    for key, value in metadata.items():
                        if key.endswith('_time') or key == 'last_transition':
                            try:
                                metadata[key] = datetime.fromisoformat(value)
                            except:
                                pass
                    
                    self.trade_metadata[pair] = metadata
                    
                    # Check if individual state is stale
                    entry_time = metadata.get('entry_time')
                    if isinstance(entry_time, datetime):
                        state_age = (datetime.now() - entry_time).total_seconds() / 3600
                        if state_age > 24:
                            self.states[pair] = TradeState.IDLE
                            self.trade_metadata[pair] = {}
                            stale_count += 1
                    
                    loaded_count += 1
                
                logger.info(f"[{self.bot_name}] Loaded {loaded_count} states "
                          f"(file age: {file_age_hours:.1f}h, stale: {stale_count})")
                
            except Exception as e:
                logger.error(f"[{self.bot_name}] Failed to load states: {e}")
    
    def save_state(self, pair: str):
        """Save single state (triggers full save with auto-save logic)"""
        self.save_all_states(force=False)
    
    def transition(self, pair: str, new_state: TradeState, metadata: Dict[str, Any] = None):
        """Transition with auto-save"""
        old_state = self.get_state(pair)
        
        if not self._is_valid_transition(old_state, new_state):
            logger.warning(f"{pair}: Invalid transition from {old_state} to {new_state}")
            return False
        
        self.states[pair] = new_state
        
        if metadata:
            if pair not in self.trade_metadata:
                self.trade_metadata[pair] = {}
            self.trade_metadata[pair].update(metadata)
            self.trade_metadata[pair]['last_transition'] = datetime.now()
        
        logger.info(f"{pair}: State transition {old_state} -> {new_state}")
        
        # Auto-save on state changes
        self.save_all_states(force=False)
        
        return True
    
    def reset_state(self, pair: str):
        """Reset state for a pair"""
        if pair in self.states:
            del self.states[pair]
        if pair in self.trade_metadata:
            del self.trade_metadata[pair]
        
        # Save after reset
        self.save_all_states(force=True)
    
    def cleanup_stale_states(self, hours: int = 24):
        """Remove stale states from memory and file"""
        cleaned = 0
        for pair in list(self.states.keys()):
            if pair in self.trade_metadata:
                last_transition = self.trade_metadata[pair].get('last_transition')
                if isinstance(last_transition, datetime):
                    age = (datetime.now() - last_transition).total_seconds() / 3600
                    if age > hours:
                        self.reset_state(pair)
                        cleaned += 1
        
        if cleaned > 0:
            logger.info(f"[{self.bot_name}] Cleaned {cleaned} stale states")
            self.save_all_states(force=True)
            
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