﻿# from https://refractiveindex.info (Schinke et al. 2015: n,k 0.25-1.45 µm)
# wavelength[um], n , k
ri = [
0.25,	1.637,	3.5889,
0.26,	1.737,	3.9932,
0.27,	2.03,	4.5958,
0.28,	2.84,	5.1961,
0.29,	4.185,	5.3124,
0.3,	5.049,	4.29,
0.31,	5.091,	3.6239,
0.32,	5.085,	3.2824,
0.33,	5.135,	3.0935,
0.34,	5.245,	2.9573,
0.35,	5.423,	2.9078,
0.36,	5.914,	2.9135,
0.37,	6.82,	2.1403,
0.38,	6.587,	0.98399,
0.39,	6.025,	0.50308,
0.4,	5.623,	0.32627,
0.41,	5.341,	0.24127,
0.42,	5.11,	0.17694,
0.43,	4.932,	0.13766,
0.44,	4.79,	0.11201,
0.45,	4.673,	0.095362,
0.46,	4.572,	0.079105,
0.47,	4.485,	0.07024,
0.48,	4.412,	0.059817,
0.49,	4.349,	0.05381,
0.5,	4.289,	0.048542,
0.51,	4.235,	0.043831,
0.52,	4.187,	0.039531,
0.53,	4.145,	0.034804,
0.54,	4.103,	0.029896,
0.55,	4.073,	0.028038,
0.56,	4.038,	0.026551,
0.57,	4.006,	0.023746,
0.58,	3.977,	0.021896,
0.59,	3.954,	0.020076,
0.6,	3.931,	0.018521,
0.61,	3.908,	0.017257,
0.62,	3.888,	0.016809,
0.63,	3.869,	0.016268,
0.64,	3.851,	0.014693,
0.65,	3.835,	0.014447,
0.66,	3.817,	0.013608,
0.67,	3.805,	0.012807,
0.68,	3.791,	0.012045,
0.69,	3.776,	0.011317,
0.7,	3.765,	0.010623,
0.71,	3.753,	0.009961,
0.72,	3.741,	0.0093335,
0.73,	3.73,	0.0087312,
0.74,	3.719,	0.0081618,
0.75,	3.712,	0.0076156,
0.76,	3.701,	0.0070942,
0.77,	3.693,	0.0066054,
0.78,	3.684,	0.0061338,
0.79,	3.677,	0.0056888,
0.8,	3.669,	0.0052655,
0.81,	3.662,	0.004864,
0.82,	3.655,	0.0044836,
0.83,	3.646,	0.0041235,
0.84,	3.641,	0.0037828,
0.85,	3.636,	0.0034605,
0.86,	3.628,	0.0031563,
0.87,	3.622,	0.0028697,
0.88,	3.617,	0.0026001,
0.89,	3.613,	0.0023464,
0.9,	3.61,	0.0021092,
0.91,	3.604,	0.0018864,
0.92,	3.598,	0.0016787,
0.93,	3.597,	0.0014757,
0.94,	3.59,	0.0013061,
0.95,	3.584,	0.0011393,
0.96,	3.584,	0.00098243,
0.97,	3.578,	0.0008406,
0.98,	3.582,	0.00071334,
0.99,	3.579,	0.00059638,
1,	3.575,	0.0004902,
1.01,	3.572,	0.00039616,
1.02,	3.568,	0.00031437,
1.03,	3.565,	0.00024048,
1.04,	3.562,	0.00017959,
1.05,	3.559,	0.00013043,
1.06,	3.556,	0.00009245,
1.07,	3.553,	0.00006782,
1.08,	3.549,	0.000052168,
1.09,	3.547,	0.00003977,
1.1,	3.545,	0.000030217,
1.11,	3.542,	0.000022913,
1.12,	3.54,	0.000017068,
1.13,	3.537,	0.000012382,
1.14,	3.534,	0.000008621,
1.15,	3.533,	0.0000056876,
1.16,	3.53,	0.0000034275,
1.17,	3.527,	0.0000017653,
1.18,	3.526,	5.5561e-7,
1.19,	3.524,	2.3153e-7,
1.2,	3.522,	1.3904e-7,
1.21,	3.52,	8.0863e-8,
1.22,	3.518,	4.794e-8,
1.23,	3.517,	2.7132e-8,
1.24,	3.515,	1.4318e-8,
1.25,	3.513,	5.8798e-9,
1.26,	3.512,	2.3352e-9,
1.27,	3.509,	1.2714e-9,
1.28,	3.509,	7.5284e-10,
1.29,	3.506,	4.4799e-10,
1.3,	3.505,	2.7228e-10,
1.31,	3.503,	1.5856e-10,
1.32,	3.502,	8.7196e-11,
1.33,	3.501,	4.2039e-11,
1.34,	3.5,	1.8128e-11,
1.35,	3.499,	1.0428e-11,
1.36,	3.497,	6.2911e-12,
1.37,	3.496,	3.903e-12,
1.38,	3.496,	2.6367e-12,
1.39,	3.496,	1.7377e-12,
1.4,	3.493,	1.0428e-12,
1.41,	3.492,	6.0422e-13,
1.42,	3.492,	4.2895e-13,
1.43,	3.49,	2.0381e-13,
1.44,	3.488,	1.3785e-13,
1.45,	3.487,	1.0901e-13,
]