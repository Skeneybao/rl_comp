import abc

import torch


class FeatureEngine(abc.ABC):
    @abc.abstractmethod
    def get_input_shape(self):
        pass

    @abc.abstractmethod
    def get_feature(self, observation) -> torch.Tensor:
        pass


class FeatureEngineExample(FeatureEngine):

    def __init__(self, feature_to_use=None):
        pass

    def get_input_shape(self):
        return 3

    def get_feature(self, observation):
        feature_array = torch.tensor([
            self.feature1(observation),
            self.feature2(observation),
            self.feature3(observation),
        ])

    def feature1(self, observation):
        return 1

    def feature2(self, observation):
        return 2

    def feature3(self, observation):
        return 3


class FeatureEngineDummy(FeatureEngine):

    def get_input_shape(self):
        return 34

    def get_feature(self, observation) -> torch.Tensor:
        return torch.tensor(list(observation.values()))
