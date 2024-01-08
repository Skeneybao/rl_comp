from training.model_io.featureEngine import *


def get_feature_engine_type(name: str) -> Type[FeatureEngine]:
    if name == 'dummy':
        return FeatureEngineDummy
    elif name == 'example':
        return FeatureEngineExample
    elif name == 'version1':
        return FeatureEngineVersion1
    else:
        raise ValueError(f'Unknown feature engine name: {name}')
