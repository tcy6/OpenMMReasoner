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

