


class random_forest:
    def __init__(self, n_trees, max_depth, min_sample_split, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_sample_split
        self.n_features = n_features
        self.trees = []
         
    
         
    