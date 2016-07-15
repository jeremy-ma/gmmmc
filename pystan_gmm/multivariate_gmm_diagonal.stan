data {
    int<lower=1> D;     // number of dimensions
    int<lower=1> N;     // number of samples
    int<lower=1> M;     // number of mixture components
    vector[D] X[N];     // data to train
}
parameters {
    simplex[M] weights;             // mixture weights
    vector[D] mu[M];                // means
    vector<lower=0.0>[D] sigma[M];  // standard deviation
}
model {
    real ps[M];
    for(n in 1:N){
        for(m in 1:M){
            ps[m] <- log(weights[m]);
            for(d in 1:D){
                ps[m] <- ps[m] + normal_log(X[n, d], mu[m, d], sigma[m, d]);
            }

        }
        increment_log_prob(log_sum_exp(ps));
    }
}