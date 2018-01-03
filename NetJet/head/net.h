#pragma once
typedef struct net {
    int n_layers;
    int* sizes;
    int** biases;
    int*** weights;
}net;

net* init_net(int* sizes, int n_layers);
double* neurons_output(double* input, net* const the_net);
void gradient_descent(data* train_d, int train_d_length, int n_epohs, int mini_batch_size, double learning_rate, net* the_net, data* test_d, int test_d_length);