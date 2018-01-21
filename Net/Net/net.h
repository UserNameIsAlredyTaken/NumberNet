#pragma once
struct net {
	size_t n_layers;
	size_t* sizes;
	int** biases;
	int*** weights;
};

struct net* init_net(int* sizes, int n_layers);
double* neurons_output(double* input, struct net* const the_net);
struct net gradient_descent(struct data* train_d, int const train_d_length, int const n_epohs, int const mini_batch_size, double const learning_rate, struct net* the_net, struct data* test_d, int const test_d_length);
void shuffle(struct data* old_d, int const size);
struct net* init_zero_net(struct net* net_template);
double sigmoid_func(double const weighed_inp);