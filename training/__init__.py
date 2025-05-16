# Default configuration file for the RL agent in the autonomous driving environment
from .meta_trainer import MetaPPOTrainer
from .curriculum import CurriculumManager
from .callbacks import AdaptiveDetectionCallback, VideoRecordCallback

__all__ = [
    'MetaPPOTrainer',
    'CurriculumManager',
    'AdaptiveDetectionCallback',
    'VideoRecordCallback'
]

# Version information
__version__ = "0.1.0"