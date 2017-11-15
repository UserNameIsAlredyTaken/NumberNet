#include <stdlib.h>

typedef struct net {
	int n_layers;
	int* sizes;
	int** biases;
	int** weights;
}net;

net* init_net(int* sizes,int n_layers){
	net* the_net = malloc(sizeof(net));
	the_net->sizes = sizes;
	the_net->n_layers = n_layers;
	th
	return the_net;
}