#pragma once
typedef struct net {
	int n_layers;
	int* sizes;
	int** biases;
	int** weights;
}net;

net* init_net(int* sizes, int n_layers);
double* neurons_output(double* input, net* const the_net);