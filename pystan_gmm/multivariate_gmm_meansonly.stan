data {
    int<lower=1> D;     // number of dimensions
    int<lower=1> N;     // number of samples
    int<lower=1> M;     // number of mixture components
    vector[D] X[N];     // data to train
}
parameters {
    vector[D] mu[M];                // means
    simplex[M] weights;             // mixture weights
    vector<lower=0.0>[D] sigma[M];  // standard deviation
}
model {
    real ps[M];
    for(n in 1:N){
        for(m in 1:M){
            ps[m] <- log(weights[m]);
            ps[m] <- ps[m] + normal_log(X[n], mu[m], sigma[m]);
        }
        increment_log_prob(log_sum_exp(ps));
    }
}