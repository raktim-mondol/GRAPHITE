from utils.imports import *
class ConfigManager:
    def __init__(self, config_path=None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = config_path
        self.config = self._load_config() if config_path else self._default_config()
        
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.load(f, Loader=SafeLoader)
            return self._merge_with_default(config)
        except Exception as e:
            logging.error(f"Error loading config from {self.config_path}: {str(e)}")
            logging.info("Using default configuration")
            return self._default_config()
    
    def _default_config(self):
        """Default configuration"""
        return {
            'data_params': {
                'train_dir': 'data/train',
                'val_dir': 'data/val',
                'patch_size': 224,
                'overlap_ratio': 0.5,
                'min_tissue_percentage': 0.5
            },
            'model_params': {
                'input_dim': 512,
                'hidden_dim': 256,
                'num_levels': 3,
                'num_gat_layers': 3,
                'num_heads': 4,
                'dropout': 0.1,
                'spatial_threshold': 1.5,
                'scale_threshold': 2.0
            },
            'training_params': {
                'batch_size': 8,
                'learning_rate': 0.0001,
                'weight_decay': 1e-5,
                'num_epochs': 100,
                'early_stopping_patience': 10
            },
            'paths': {
                'save_dir': 'results/hiergat_ssl',
                'log_dir': 'logs/hiergat_ssl'
            }
        }
    
    def _merge_with_default(self, config):
        """Merge loaded config with default config"""
        default_config = self._default_config()
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
            elif isinstance(default_config[key], dict):
                for subkey in default_config[key]:
                    if subkey not in config[key]:
                        config[key][subkey] = default_config[key][subkey]
        return config
    
    def update_from_args(self, args):
        """Update config with command line arguments"""
        self.config['data_params']['train_dir'] = args.train_dir
        self.config['data_params']['val_dir'] = args.val_dir
        self.config['paths']['save_dir'] = args.output_dir
        
    def save_config(self, save_path):
        """Save current configuration to YAML file"""
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)