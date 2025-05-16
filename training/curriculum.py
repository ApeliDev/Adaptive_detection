import numpy as np
from typing import Dict, List

class CurriculumManager:
    def __init__(self, scenarios: List[Dict]):
        self.scenarios = scenarios
        self.current_scenario = 0
        self.metrics_history = []
        self.completion_threshold = 0.8  
    
    def update_metrics(self, episode_metrics: Dict):
        """Update performance metrics"""
        self.metrics_history.append(episode_metrics)
        
        # Check if should advance curriculum
        if len(self.metrics_history) > 10:
            recent_success = np.mean([m['success'] for m in self.metrics_history[-10:]])
            if recent_success > self.completion_threshold:
                self.current_scenario = min(self.current_scenario + 1, len(self.scenarios)-1)
    
    def get_current_scenario(self) -> Dict:
        """Get current scenario configuration"""
        return self.scenarios[self.current_scenario]
    
    def get_difficulty(self) -> float:
        """Get current difficulty level (0-1)"""
        return self.current_scenario / (len(self.scenarios) - 1)
    