
from src.adaptive_epsilon import adaptive_epsilon

def adaptive_robust_training(X, Y, target_epsilon, batch_size):
    batches = adaptive_epsilon(X, Y, target_epsilon, batch_size)