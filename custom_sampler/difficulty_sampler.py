from verl.experimental.dataset.sampler import AbstractSampler
from collections.abc import Sized
import numpy as np
from omegaconf import DictConfig

class DifficultySampler(AbstractSampler):
    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        self.data_config = data_config
        dataframe = self.data_source.dataframe

        assert "avg_reward" in dataframe.column_names, "avg_reward column not found in data source"
        offline_rewards = dataframe.select_columns("avg_reward").to_list()
        offline_rewards = [a['avg_reward'] for a in offline_rewards]
        offline_rewards = np.array(offline_rewards)
        offline_rewards = offline_rewards.argsort(stable=True)
        # reverse the offline_rewards, so we start with easiest samples
        offline_rewards = offline_rewards[::-1]

        self.offline_rewards = offline_rewards

    def __iter__(self):
        for idx in self.offline_rewards:
            yield idx

    def __len__(self):
        return len(self.offline_rewards)


class DifficultyDynamicSampler(AbstractSampler):
    """
    Dynamic sampler implementing curriculum learning with difficulty-based progression.
    
    Strategy:
    - Each sample is yielded exactly once per epoch (without replacement)
    - Curriculum progress can be configured to control the difficulty distribution
    - Initially prioritizes easier samples, can shift towards harder samples
    """
    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        self.data_config = data_config
        dataframe = self.data_source.dataframe

        assert "avg_reward" in dataframe.column_names, "avg_reward column not found in data source"
        offline_rewards = dataframe.select_columns("avg_reward").to_list()
        offline_rewards = [a['avg_reward'] for a in offline_rewards]
        offline_rewards = np.array(offline_rewards)
        
        # Sort indices by difficulty: ascending order (easiest first)
        self.sorted_indices = np.argsort(offline_rewards)
        self.offline_rewards = offline_rewards[self.sorted_indices]
        self.n_samples = len(self.sorted_indices)
        
        # Curriculum parameters (can be customized via data_config)
        self.temperature = data_config.get("curriculum_temperature", 5.0)
        self.curriculum_progress = data_config.get("curriculum_progress", 0.0)

    def __iter__(self):
        """
        Yield indices ensuring each sample is sampled exactly once.
        
        Uses curriculum weighting to determine the order:
        - curriculum_progress=0.0: samples mostly ordered from easy to hard
        - curriculum_progress=1.0: samples mostly ordered from hard to easy
        """
        # Create position array (0 = easiest, 1 = hardest)
        positions = np.arange(self.n_samples) / max(1, self.n_samples - 1)
        
        # Create curriculum weights: Gaussian centered at curriculum_progress position
        logits = -self.temperature * (positions - self.curriculum_progress) ** 2
        weights = np.exp(logits)
        weights = weights / np.sum(weights)
        
        # Create a permutation weighted by curriculum probabilities
        # This ensures each sample appears exactly once, ordered by curriculum
        order = np.random.choice(
            self.n_samples, 
            size=self.n_samples, 
            replace=False, 
            p=weights
        )
        
        # Yield samples in curriculum-ordered sequence
        for position in order:
            yield self.sorted_indices[position]

    def __len__(self):
        return self.n_samples