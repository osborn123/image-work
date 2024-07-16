from .abstract_load_test import LoadTest

class APIEmbeddingLoadTest(LoadTest):
    def __init__(self, args, task_name, feature_dir):
        super().__init__(args, task_name, feature_dir)
    
    def load_features(self, feature1_path, feature2_path):
        pass
    
    def test(self, args, feature1, feature2, feature1_transformed=None, only_1_index=False, only_2_index=False, overlap_index=None):
        pass