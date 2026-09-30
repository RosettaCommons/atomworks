from atomworks.ml.preprocessing.preprocess import PreprocessConfig

# Test configuration with lower polymer limit for speed
TEST_CONFIG = PreprocessConfig(
    polymer_pn_unit_limit=50,
)
