"""
Configuration management for the trading bot
"""

# Default configuration
DEFAULT_CONFIG = {
    'initial_balance': 10000.0,
    'symbols': ['BTC', 'ETH', 'SOL'],
    'default_quantity': 0.1,
}


class Config:
    """Configuration handler for the trading bot."""
    
    def __init__(self, **kwargs):
        """
        Initialize configuration.
        
        Args:
            **kwargs: Configuration parameters to override defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value."""
        self.config[key] = value
    
    def get_all(self) -> dict:
        """Get all configuration."""
        return self.config.copy()
