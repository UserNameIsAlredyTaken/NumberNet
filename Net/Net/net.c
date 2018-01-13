#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "get_data.h"

#pragma pack(push,1)
typedef struct net {
	int n_layers;
	int* sizes;
	double** biases;
	double*** weights;
}net;
#pragma pack(pop)

static void clean_all(net* net){
	for (int i = 1; i < net->n_layers; i++){
		for (int j = 0; j < net->sizes[i]; j++){
			free(net->weights[i][j]);
		}
		free(net->weights[i]);
		free(net->biases[i]);
	}
	free(net->weights);
	free(net->biases);
	free(net);
}

net* init_net(int* sizes, int const n_layers) {
	srand(time(NULL));
	net* the_net = malloc(sizeof(net));
	the_net->sizes = sizes;
	the_net->n_layers = n_layers;
	the_net->biases = (double**)malloc(sizeof(double*)*(n_layers));
	the_net->weights = (double***)malloc(sizeof(double**)*(n_layers));
	for (int i = 1; i < n_layers; i++) {///fill biases/weights for each neural layer except first
		the_net->biases[i] = (double*)malloc(sizeof(double)*sizes[i]);
		the_net->weights[i] = (double**)malloc(sizeof(double*)*sizes[i]);
		for (int j = 0; j < sizes[i]; j++) {///fill biases/weights for each neuron
			
			the_net->biases[i][j] = (double)rand() / RAND_MAX*2.0 - 1.0;

			//printf("b[%d][%d]:%f", i, j, the_net->biases[i][j]);
			the_net->weights[i][j] = (double*)malloc(sizeof(double)*sizes[i - 1]); //printf("{");
			for (int k = 0; k < sizes[i - 1]; k++) {///fill weights for each neuron (in the count of neurons in previous layer)

				the_net->weights[i][j][k] = (double)rand() / RAND_MAX*2.0 - 1.0;

				//printf("w[%d][%d][%d]:%f ", i, j, k, the_net->weights[i][j][k]);
			} //printf("} ");
		} //printf("\n");
	}
	return the_net;
}

double sigmoid_func(double const weighed_inp) {
	return 1 / (1 + exp(-1 * weighed_inp));
}

static double sigmoid_deriv(double const weighed_inp) {
	return sigmoid_func(weighed_inp)*(1 - sigmoid_func(weighed_inp));
}

static double cost_deriv(double const actual_output, uint8_t const expecting_output) {
	return actual_output - expecting_output;
}

static double* softmax(double* last_layer, int length){
	double denominator = 0;
	for (int i = 0; i < length; i++) {
		denominator += exp(last_layer[i]);
	}
	for (int i = 0; i < length;i++) {
		last_layer[i] = exp(last_layer[i]) / denominator;
	}
	return last_layer;
}

/*INPUT is the vector of data, and pointer to the net structure
*OUTPUT is the pointer to result vector
*(lenth of the vector should be the same as the count of neurons in the input layer of the net(just a convention)),*/
double* neurons_output(uint8_t* input, net* const the_net) {
	double* input_prev = (double*)malloc(sizeof(double)*the_net->sizes[0]);
	double* input_next = (double*)malloc(sizeof(double)*the_net->sizes[1]);
	for (int i = 0; i<the_net->sizes[0]; i++) {///converting input uint8_t data to double
		input_prev[i] = input[i];
	}
	for (int i = 1; i<the_net->n_layers; i++) {///iterate for each neuron layer except the last one 
		for (int j = 0; j < the_net->sizes[i]; j++) {///iterate for each neuron
			double weighed_sum = 0;
			for (int k = 0; k < the_net->sizes[i - 1]; k++) {///summing the input neuron's values multiplying by weights
				weighed_sum += input_prev[k] * the_net->weights[i][j][k];
			}
			input_next[j] = sigmoid_func(weighed_sum + the_net->biases[i][j]);		
			

			/*printf("%f:", weighed_sum + the_net->biases[i][j]);*/
			//printf("%f ", input_next[j]);
			//printf("\n");*/

		}//printf("\n");
		free(input_prev);
		input_prev = input_next;
		input_next = (double*)malloc(sizeof(double)*the_net->sizes[i + 1]);
	}
	free(input_next);
	//printf("\n\n");	
	return softmax(input_prev, the_net->sizes[the_net->n_layers - 1]);///implemtnting softmax funnction
	//return input_prev;
}

void shuffle(data* old_d, int size) {
	int j;
	for (int i = 0; i<size; i++) {
		j = rand() % size;
		data tmp = old_d[i];
		old_d[i] = old_d[j];
		old_d[j] = tmp;
	}
}

static net* init_zero_net(net* net_template) {
	net* zero_net = malloc(sizeof(net));///initializing copy of the obtaind net and fill all biases\weights by zeros
	zero_net->sizes = net_template->sizes;
	zero_net->n_layers = net_template->n_layers;
	zero_net->biases = (double**)malloc(sizeof(double*)*(net_template->n_layers));
	zero_net->weights = (double***)malloc(sizeof(double**)*(net_template->n_layers));
	for (int i = 1; i < net_template->n_layers; i++) {
		zero_net->biases[i] = (double*)malloc(sizeof(double)*net_template->sizes[i]);
		zero_net->weights[i] = (double**)malloc(sizeof(double*)*net_template->sizes[i]);
		for (int j = 0; j < net_template->sizes[i]; j++) {
			zero_net->biases[i][j] = 0;
			//printf("b[%d][%d]:%f", i, j, zero_net->biases[i][j]);
			zero_net->weights[i][j] = (double*)malloc(sizeof(double)*net_template->sizes[i - 1]); //printf("{");
			for (int k = 0; k < net_template->sizes[i - 1]; k++) {
				zero_net->weights[i][j][k] = 0;

				//printf("w[%d][%d][%d]:%f ", i, j, k, zero_net->weights[i][j][k]);
			}//printf("} ");
		}//printf("\n");
	}
	return zero_net;
}

static net* backpropagation(data* x_y_tuple, net* the_net) {
	net* delta_gradient_net = init_zero_net(the_net);
	double** activation_value = (double**)malloc(sizeof(double*)*the_net->n_layers);///array with all activations vectors
	activation_value[0] = (double*)malloc(sizeof(double)*NUMBER_OF_PIXELS);
	for (int i = 0; i<NUMBER_OF_PIXELS; i++) {///initializing first layer of activation_value by x values		
		activation_value[0][i] = (double)x_y_tuple->x[i];
	}
	double** weighted_input = (double**)malloc(sizeof(double*)*the_net->n_layers);///array with all weighted inputs vrctors
	int last_index = the_net->n_layers - 1;///initializing the index of the last layer


	///moving forvard
	double weighed_sum;
	for (int i = 1; i<the_net->n_layers; i++) {
		activation_value[i] = (double*)malloc(sizeof(double)*the_net->sizes[i]);
		weighted_input[i] = (double*)malloc(sizeof(double)*the_net->sizes[i]);///first for weighted_input has index 1(not 0)
		for (int j = 0; j < the_net->sizes[i]; j++) {
			weighed_sum = 0;
			for (int k = 0; k < the_net->sizes[i - 1]; k++) {///summing the input neuron's values multiplying by weights
				weighed_sum += activation_value[i - 1][k] * the_net->weights[i][j][k];
			}
			weighted_input[i][j] = weighed_sum + the_net->biases[i][j];
			activation_value[i][j] = sigmoid_func(weighted_input[i][j]);
		}
	}
	activation_value[last_index] = softmax(activation_value[last_index], the_net->sizes[last_index]);

	///computing output error	
	double* forward_layer_error = (double*)malloc(sizeof(double)*the_net->sizes[last_index]);//initializing output error
	for (int j = 0; j<the_net->sizes[last_index]; j++) {///iterate for the last layer
		forward_layer_error[j] = cost_deriv(activation_value[last_index][j], x_y_tuple->y[j])*sigmoid_deriv(weighted_input[last_index][j]);

		/*printf("cost_d: %f ", cost_deriv(activation_value[last_index][j], x_y_tuple->y[j]));
		printf("sig_d: %f ", cost_deriv(activation_value[last_index][j], x_y_tuple->y[j]));
		printf("fle: %f\n", forward_layer_error[j]);*/

		delta_gradient_net->biases[last_index][j] = forward_layer_error[j];
		for (int k = 0; k<the_net->sizes[last_index - 1]; k++) {
			delta_gradient_net->weights[last_index][j][k] = activation_value[last_index - 1][k] * forward_layer_error[j];
		}
	}//printf("\n");

	///moving backward
	for (int i = last_index - 1; i>0; i--) {
		double* layer_error = (double*)malloc(sizeof(double)*the_net->sizes[i]);
		for (int j = 0; j<the_net->sizes[i]; j++) {
			///finding layer_error
			double weighted_error_sum = 0;
			for (int k = 0; k<the_net->sizes[i + 1]; k++) {
				weighted_error_sum += forward_layer_error[k] * the_net->weights[i + 1][k][j];///k - index for next layer neurons, j - index for this layer neurons
			}
			layer_error[j] = weighted_error_sum*sigmoid_deriv(weighted_input[i][j]);

			//printf("%f\n", layer_error[j]);

			///finding delta weights\biases
			delta_gradient_net->biases[i][j] = layer_error[j];
			for (int k = 0; k<the_net->sizes[i - 1]; k++) {
				delta_gradient_net->weights[i][j][k] = activation_value[i - 1][k] * layer_error[j];
			}
		}
		free(forward_layer_error);
		forward_layer_error = (double*)malloc(sizeof(double)*the_net->sizes[i]);
		for (int i = 0; i<the_net->sizes[i]; i++) {///initializing first layer of activation_value by x values		
			forward_layer_error[i] = layer_error[i];
		}
		free(layer_error);
	}
	free(activation_value[0]);
	for (int i = 1; i < the_net->n_layers; i++) {
		free(activation_value[i]);
		free(weighted_input[i]);
	}
	free(forward_layer_error);
	free(activation_value);
	free(weighted_input);
	 
	/*for (int i = 0; i < 15; i++) {
		printf("%f ", delta_gradient_net->biases[1][i]);
	}printf("\n\n");

	for (int i = 0; i < 10; i++) {
		printf("%f ", delta_gradient_net->biases[2][i]);
	}printf("\n\n");*/

	return delta_gradient_net;
}

static void update_mb(data* mb, int mb_size, double lr, net* the_net) {
	net* gradients_net = init_zero_net(the_net);
	for (int l = 0; l<mb_size; l++) {
		net* delta_gradient_net = backpropagation(&mb[l], the_net);///applying backpropagation method
		for (int i = 1; i < the_net->n_layers; i++) { ///summing delta_gradient_net with gradient net
			for (int j = 0; j < the_net->sizes[i]; j++) {
				gradients_net->biases[i][j] += delta_gradient_net->biases[i][j];

				//printf("dgb[%d][%d]:%f ",i,j, delta_gradient_net->biases[i][j]);

				for (int k = 0; k < the_net->sizes[i - 1]; k++) {
					gradients_net->weights[i][j][k] += delta_gradient_net->weights[i][j][k];
					//printf("dgw[%d][%d][%d]:%f ", i, j,k, delta_gradient_net->weights[i][j][k]);
				}
			}//printf("\n");
		}//printf("\n\n\n");
		clean_all(delta_gradient_net);///freeing memory
	}
	for (int i = 1; i < the_net->n_layers; i++) { //updating weights\biases
		for (int j = 0; j < the_net->sizes[i]; j++) {
			the_net->biases[i][j] -= (lr / mb_size)*gradients_net->biases[i][j];

			//printf("b[%d][%d]:%f ", i, j, the_net->biases[i][j]);

			for (int k = 0; k < the_net->sizes[i - 1]; k++) {
				gradients_net->weights[i][j][k] -= (lr / mb_size)*gradients_net->weights[i][j][k];
			}
		}//printf("\n");
	}	
	clean_all(gradients_net);
}

static int check(data* test_d, int test_d_lenght, net* the_net) {
	int counter = 0;
	for (int i = 0; i < test_d_lenght; i++) {
		double max = 0;
		int max_i = -1;
		int y = -2;
		double* actual_output = neurons_output(test_d[i].x, the_net);
		for (int j = 0; j<10; j++) {
			if (actual_output[j]>max) {
				max = actual_output[j];
				max_i = j;
			}
			if (test_d[i].y[j] == 1) {
				y = j;
			}
		}
		if (y == max_i) {counter++;}
	}	
	return counter;
}

static double cost_func(data* test_d, int test_d_lenght, net* the_net) {
	double cost_sum=0;
	for (int i = 0; i < test_d_lenght; i++) {
		double* actual_output = neurons_output(test_d[i].x, the_net);
		double vector_length=0;
		for (int j = 0; j < 10;j++){
			vector_length += pow(test_d[i].y[j] - actual_output[j],2);
		}
		vector_length = sqrt(vector_length);
		cost_sum += vector_length;
	}
	return cost_sum / (2 * test_d_lenght);
}

void gradient_descent(data* train_d, int train_d_length, int n_epohs, int mini_batch_size, double learning_rate, net* the_net, data* test_d, int test_d_length) {
	for (int i = 0; i < n_epohs; i++) {
		shuffle(train_d, train_d_length);///for each epoch shuffle whole training set
		for (int j = 0; j<train_d_length; j+=mini_batch_size) {///iterate applying "update_mb" for each batch(train_d_length div mini_batch_size should be =0(just a convention))		
			update_mb(&train_d[j], mini_batch_size, learning_rate, the_net);
		}

		for (int j = 0; j < 15; j++) {
			printf("b1:%f ", the_net->biases[1][j]);
		}printf("\n\n");

		for (int j = 0; j < 10; j++) {
			printf("b2:%f ", the_net->biases[2][j]);
		}printf("\n");

		/*for (int j = 0; j < 10; j++) {
			for (int k = 0; k < 10; k++) {
				printf("w[%d][%d]:%f ", j, k, the_net->weights[2][j][k]);
			}printf("\n\n");
		}*/

		double* output = neurons_output(test_d[i].x, the_net);
		for (int i = 0; i < 10; i++) {
			printf("out:%f\n", output[i]);
		}

		printf("Epoch %d: %d/%d\n\n\n", i, check(test_d, test_d_length, the_net), test_d_length);			
	}
}

