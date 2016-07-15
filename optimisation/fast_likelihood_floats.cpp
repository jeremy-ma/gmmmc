#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <float.h>
#define _USE_MATH_DEFINES
#include <math.h>

float mahal_dist(float * __restrict data, float * __restrict mean, float * __restrict icov, int n_features)
{
	float temp,mdist=0;
	for(int i=0;i < n_features; i++){
		temp = (data[i]-mean[i]);
		mdist += (temp * temp) * icov[i];
	}
	return mdist;
}

float vec_logprob(float * __restrict data, float * __restrict means, float * __restrict icov,
                   float * __restrict weights, int n_mixtures, int n_features,
                   float * __restrict tarray, float * __restrict gconst)
{
	float tmax = -DBL_MAX;

	for(int i=0;i<n_mixtures;i++){
		tarray[i] = (-0.5) * mahal_dist(data, means + i*n_features, icov + (i*n_features), n_features);
		if(tarray[i]>tmax){
		    tmax = tarray[i];
		}
	}

	float temp = 0;
	//logsumexp
	for(int i=0;i<n_mixtures;i++){
	    temp += gconst[i]*exp(tarray[i]-tmax);
	}

	return (tmax + log(temp));

}

float data_logprob_float(float * __restrict data, float * __restrict means, float * __restrict covars,
                    float * __restrict weights, int n_samples, int n_mixtures, int n_features)
{
	float * __restrict gconst; //pointer to store mixture normalising constants -  (2pi)^-D/2|mix cov|^-1/2
	float * __restrict icov; //ponter to store inverse covariance values to enable division to be replace by multiplication in mahal_dist() function
	float * __restrict tarray; //pointer to temporary array to store P(x|mixture component) values

	//allocate memory for icov (ndim*nmix)
	if(!(icov = static_cast<float*>(malloc(n_features * n_mixtures * sizeof(float))))){
		printf("icov memory allocation failed\n");
		return 0;
	}

	//allocate memory for gconst (one per mixture)
	if(!(gconst = static_cast<float*>(malloc(n_mixtures * sizeof(float))))){
		printf("gconst memory allocation failed\n");
		free(static_cast<void*>(icov)); //free memory already allocated to icov before exiting function
		return 0;
	}

	//allocate memory for tarray (one array per thread)
	if(!(tarray = static_cast<float*>(malloc(n_mixtures *sizeof(float))))){
		printf("tarray memory allocation failed\n");
		free(static_cast<void*>(icov)); //free memory already allocated to icov before exiting function
		free(static_cast<void*>(gconst)); //free memory allocated to gconst before exiting function
		return 0;
	}

	float lprob = 0; //total log probability for all data vectors
	long i,j;
	float temp;

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