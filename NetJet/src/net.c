#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "../head/get_data.h"

typedef struct net {
	int n_layers;
	int* sizes;
	double** biases;
	double*** weights;
}net;

net* init_net(int* sizes,int const n_layers){
	net* the_net = malloc(sizeof(net));
	the_net->sizes = sizes;
	the_net->n_layers = n_layers;
	the_net->biases = (double**)malloc(sizeof(double*)*(n_layers-1));
	the_net->weights = (double***)malloc(sizeof(double**)*(n_layers - 1));
	for (int i = 1; i < n_layers;i++){//fill biases/weights for each neural layer except first
		the_net->biases[i] = (double*)malloc(sizeof(double)*sizes[i]);
		the_net->weights[i] = (double**)malloc(sizeof(double*)*sizes[i]);
		for (int j = 0; j < sizes[i];j++){//fill biases/weights for each neuron
			
			srand(((i*3467897)^(j*78951))%1328098134);//miserable attemps to write a real random
			the_net->biases[i][j] = -1 + (rand() % 20000)*0.0001;//randoming function in range from -1 till 1

			printf("b[%d][%d]:%f", i, j, the_net->biases[i][j]);
			the_net->weights[i][j] = (double*)malloc(sizeof(double)*sizes[i - 1]); printf("{");
			for (int k = 0; k < sizes[i - 1]; k++) {//fill weights for each neuron (in the count of neurons in previous layer)

				srand((((i * 3497) ^ (j * 751)) ^ (k * 4428)) % 7215242);
				the_net->weights[i][j][k] = -1 + (rand() % 20000)*0.0001;

				printf("w[%d][%d][%d]:%f ", i, j, k, the_net->weights[i][j][k]);
			} printf("} ");
		} printf("\n");
	}
	return the_net;
}

static double sigmoid_func(double const weighed_inp) {
	return 1 / (1 + exp(-1 * weighed_inp));
}

static double sigmoid_deriv(double const weighed_inp) {
    return sigmoid_func(weighed_inp)*(1-sigmoid_func(weighed_inp));
}

static double cost_deriv(double const actual_output, double const expecting_output) {
    return actual_output-expecting_output;
}

/*INPUT is the vector of data, and pointer to the net structure
 *OUTPUT is the pointer to result vector
 *(lenth of the vector should be the same as the count of neurons in the input layer of the net(just a convention)),*/
double* neurons_output(uint8_t* input,net* const the_net){
	double* input_prev = (double*)malloc(sizeof(double)*the_net->sizes[0]);
	double* input_next = (double*)malloc(sizeof(double)*the_net->sizes[1]);
	for(int i=0;i<the_net->sizes[0];i++){//converting input uint8_t data to double
		input_prev[i] = input[i];
	}
	for(int i = 1; i<the_net->n_layers;i++){
		for (int j = 0; j < the_net->sizes[i];j++){//iterate for each neuron
			double weighed_sum = 0;
			for(int k = 0; k < the_net->sizes[i-1]; k++){//summing the input neuron's values multiplying by weights
				weighed_sum += input_prev[k] * the_net->weights[i][j][k];
			}
			input_next[j] = sigmoid_func(weighed_sum+the_net->biases[i][j]);
			printf("%f ", input_next[j]);
		}
		printf("\n");
		free(input_prev);
		input_prev = input_next;
		free(input_next);
		input_next = (double*)malloc(sizeof(double)*the_net->sizes[i+1]);
	}
	printf("\n\n");
	return input_prev;	
}

static void shuffle(train_d* old_td, int size){
	int j;
	for(int i = 0;i<size;i++){
		j = rand() % size;
		char tmp_y = old_td[i].y;
		old_td[i].y = old_td[j].y;
		old_td[j].y = tmp_y;
		char tmp_x = (char) old_td[i].x;
		*old_td[i].x = (char) old_td[j].x;
		*old_td[j].x = tmp_x;
	}
}

static net* init_zero_net(net* net_template){
    net* zero_net = malloc(sizeof(net));//initializing copy of the obtaind net and fill all biases\weights by zeros
    zero_net->biases = (double**)malloc(sizeof(double*)*(net_template->n_layers - 1));
    zero_net->weights = (double***)malloc(sizeof(double**)*(net_template->n_layers - 1));
    for (int i = 1; i < net_template->n_layers; i++){
        zero_net->biases[i] = (double*)malloc(sizeof(double)*net_template->sizes[i]);
        zero_net->weights[i] = (double**)malloc(sizeof(double*)*net_template->sizes[i]);
        for (int j = 0; j < net_template->sizes[i]; j++){
            zero_net->biases[i][j] = 0;
            zero_net->weights[i][j] = (double*)malloc(sizeof(double)*net_template->sizes[i - 1]); printf("{");
            for (int k = 0; k < net_template->sizes[i - 1]; k++){
                zero_net->weights[i][j][k] = 0;
            }
        }
    }
    return zero_net;
}

static net* backpropagation(train_d x_y_tuple,net* the_net){//TODO test backpropagation method
    net* delta_gradient_net = init_zero_net(the_net);
    double** activation_value = (double**)malloc(sizeof(double*)*the_net->n_layers);//array with all activations vectors
    activation_value[0] = (double *) x_y_tuple.x;
    double** weighted_input = (double**)malloc(sizeof(double*)*(the_net->n_layers-1));//array with all weighted inputs vrctors

    //moving forvard
    for(int i = 1; i<the_net->n_layers;i++){
        activation_value[i] = (double*)malloc(sizeof(double)*the_net->sizes[i]);
        weighted_input[i] = (double*)malloc(sizeof(double)*the_net->sizes[i]);//first for weighted_input has index 1(not 0)
        for (int j = 0; j < the_net->sizes[i];j++){
            double weighed_sum = 0;
            for(int k = 0; k < the_net->sizes[i-1]; k++){//summing the input neuron's values multiplying by weights
                weighed_sum += activation_value[i-1][k] * the_net->weights[i][j][k];
            }
            activation_value[i][j] = weighed_sum+the_net->biases[i][j];
            weighted_input[i][j] = sigmoid_func(activation_value[i][j]);
        }
    }

    //computing output error
    int last = the_net->n_layers;//initializing the number of the last layer
    double* forward_layer_error = (double*)malloc(sizeof(double)*the_net->sizes[last]);//initializing output error
    for(int j = 0; j<the_net->sizes[last];j++){//iterate for the last layer
        forward_layer_error[j] = cost_deriv(activation_value[last][j],x_y_tuple.y)*sigmoid_deriv(weighted_input[last][j]);
        delta_gradient_net->biases[last][j] = forward_layer_error[j];
        for(int k = 0;k<the_net->sizes[last-1];k++){
            delta_gradient_net->weights[last][j][k] = activation_value[last-1][k]*forward_layer_error[j];
        }
    }

    //moving backward
    for(int i = last-1; i>1; i--){
        for(int j = 0; j<the_net->sizes[i];j++){
            //finding layer_error
            double* layer_error = (double*)malloc(sizeof(double)*the_net->sizes[i]);
            double weighted_error_sum = 0;
            for(int k = 0;k<the_net->sizes[i+1];k++){
                weighted_error_sum += forward_layer_error[k]*the_net->weights[i+1][k][j];//k - index for next layer neurons, j - index for this layer neurons
            }
            layer_error[j] = weighted_error_sum*sigmoid_deriv(weighted_input[i][j]);
            //finding delta weights\biases
            delta_gradient_net->biases[i][j] = layer_error[j];
            for(int k = 0;k<the_net->sizes[i-1];k++){
                delta_gradient_net->weights[i][j][k] = activation_value[i-1][k]*layer_error[j];
            }
            free(forward_layer_error);
            forward_layer_error = layer_error;
        }
    }
    return delta_gradient_net;
}

static void update_mb(train_d* mb,int mb_size, double lr, net* the_net){ //TODO test the update_mb function
	net* gradients_net = init_zero_net(the_net);
	for(int l = 0; l<mb_size;l++){
		net* delta_gradient_net = backpropagation(*(mb+l*sizeof(train_d)),the_net);//applying backpropagation method
		for (int i = 1; i < the_net->n_layers; i++){ //summing delta_gradient_net with gradient net
			for (int j = 0; j < the_net->sizes[i]; j++){
				gradients_net->biases[i][j]+=delta_gradient_net->biases[i][j];
				for (int k = 0; k < the_net->sizes[i - 1]; k++){
					gradients_net->weights[i][j][k] += delta_gradient_net->weights[i][j][k];
				}
			}
		}
	}
	for (int i = 1; i < the_net->n_layers; i++){ //updating weights\biases
		for (int j = 0; j < the_net->sizes[i]; j++){
			the_net->biases[i][j] = the_net->biases[i][j]-(lr/mb_size)*gradients_net->biases[i][j];
			for (int k = 0; k < the_net->sizes[i - 1]; k++){
				gradients_net->weights[i][j][k] = gradients_net->weights[i][j][k]-(lr/mb_size)*gradients_net->weights[i][j][k];
			}
		}
	}
}

void gradient_descent(train_d* td,int td_length, int n_epohs, int mini_batch_size, double learning_rate,net* the_net){
	for(int i = 0;i < n_epohs;i++){
		shuffle(td,td_length);//for each epoch shuffle whole training set
		for(int j = 0;j<td_length / mini_batch_size;j++){//iterate aplying "update_mb" for each batch(td_length div mini_batch_size should be =0(just a convention))
			update_mb((td+j*mini_batch_size),mini_batch_size,learning_rate,the_net);
		}
		printf("Epoch %d is done", i);
	}
}

