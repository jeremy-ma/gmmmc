#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <float.h>
#define _USE_MATH_DEFINES
#include <math.h>


//DAXPY - y = a*x + y
// DDOT - dot product


double mahal_dist(double * __restrict data, double * __restrict mean, double * __restrict icov, int n_features)
{
	double temp,mdist=0;
	for(int i=0;i < n_features; i++){
		temp = (data[i]-mean[i]);
		mdist += (temp * temp) * icov[i];
	}
	return mdist;
}

double vec_logprob(double * __restrict data, double * __restrict means, double * __restrict icov,
                   double * __restrict weights, int n_mixtures, int n_features,
                   double * __restrict tarray, double * __restrict gconst)
{
	double tmax = -DBL_MAX;

	for(int i=0;i<n_mixtures;i++){
		tarray[i] = (-0.5) * mahal_dist(data, means + i*n_features, icov + (i*n_features), n_features);
		if(tarray[i]>tmax){
		    tmax = tarray[i];
		}
	}

	double temp = 0;
	//logsumexp
	for(int i=0;i<n_mixtures;i++){
	    temp += gconst[i]*exp(tarray[i]-tmax);
	}

	return (tmax + log(temp));

}

double data_logprob(double * __restrict data, double * __restrict means, double * __restrict covars,
                    double * __restrict weights, int n_samples, int n_mixtures, int n_features)
{
	double * __restrict gconst; //pointer to store mixture normalising constants -  (2pi)^-D/2|mix cov|^-1/2
	double * __restrict icov; //ponter to store inverse covariance values to enable division to be replace by multiplication in mahal_dist() function
	double * __restrict tarray; //pointer to temporary array to store P(x|mixture component) values

	//allocate memory for icov (ndim*nmix)
	if(!(icov = static_cast<double*>(malloc(n_features * n_mixtures * sizeof(double))))){
		printf("icov memory allocation failed\n");
		return 0;
	}

	//allocate memory for gconst (one per mixture)
	if(!(gconst = static_cast<double*>(malloc(n_mixtures * sizeof(double))))){
		printf("gconst memory allocation failed\n");
		free(static_cast<void*>(icov)); //free memory already allocated to icov before exiting function
		return 0;
	}

	//allocate memory for tarray (one array per thread)
	if(!(tarray = static_cast<double*>(malloc(n_mixtures *sizeof(double))))){
		printf("tarray memory allocation failed\n");
		free(static_cast<void*>(icov)); //free memory already allocated to icov before exiting function
		free(static_cast<void*>(gconst)); //free memory allocated to gconst before exiting function
		return 0;
	}

	double lprob = 0; //total log probability for all data vectors
	long i,j;
	double temp;

    for(i=0; i<n_mixtures; i++){
        for (j=0; j<n_features; j++){
            icov[i*n_features + j] = 1.0/covars[i*n_features + j];
        }
    }

    for(i=0;i<n_mixtures;i++){
        temp = 1;
        for(int j=0;j<n_features;j++){
            temp *= icov[i*n_features+j];
        }
        gconst[i] = weights[i]*sqrt(pow((M_1_PI/2), n_features)*temp);
    }

    for(i=0; i < n_samples; i++){
        lprob += vec_logprob(data + i * n_features, means, icov, weights, n_mixtures, n_features, tarray, gconst); //compute log prob per vector
    }

	free(static_cast<void*>(tarray)); //free memory allocated to tarray
	free(static_cast<void*>(gconst)); //free memory allocated to gconst
	free(static_cast<void*>(icov)); //free memory allocated to icov

	return lprob;
}