from src.features import FeatureConfig, extract

def extract_one(args):
    path, label, cfg_dict = args
    feature_cfg = FeatureConfig(**cfg_dict)
    try:
        mel, prosody = extract(path, feature_cfg)
        return (mel, prosody, label, None)
    except Exception as e:
        return (None, None, label, str(e))
