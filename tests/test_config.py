from pd_model import config

def test_every_data_path_is_distinct():
    # two outputs sharing a path means one silently overwrites the other
    paths = [
        config.LOANS_PATH,
        config.PERFORMANCE_PATH,
        config.LABELS_PATH,
        config.FEATURES_PATH,
        config.MODEL_TABLE_PATH,
    ]
    assert len(set(paths)) == len(paths)