def build_hmm(n_states, covariance_type="full", n_iter=200, random_state=0):
    """Construct an untrained Gaussian HMM model"""
    pass


def fit_hmm(model, X_train):
    """Fit HMM parameters with EM on training data"""
    pass


def score_hmm(model, X):
    """Compute log-likelihood of observations under the model"""
    pass


def decode_regimes(model, X):
    """Decode most likely hidden state sequence with Viterbi"""
    pass


def summarize_regimes(model):
    """Summarize regime statistics such as means and transition matrix"""
    pass


def fit_and_evaluate_hmms(X_train, X_test, state_list):
    """Fit HMMs for each state count and collect train/test scores"""
    pass
