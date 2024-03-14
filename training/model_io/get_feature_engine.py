from typing import Type

from training.model_io.featureEngine import *


def get_feature_engine_type(name: str) -> Type[FeatureEngine]:
    if name == 'dummy':
        return FeatureEngineDummy
    elif name == 'example':
        return FeatureEngineExample
    elif name == 'version1':
        return FeatureEngineVersion1
    elif name == 'version2':
        return FeatureEngineVersion2
    elif name == 'version3':
        return FeatureEngineVersion3
    elif name == 'version3Simple':
        return FeatureEngineVersion3_Simple
    elif name == 'version4':
        return FeatureEngineVersion4
    elif name == 'single600TMod':
        return FeatureEngine_single600T_Mod
    else:
        raise ValueError(f'Unknown feature engine name: {name}')
