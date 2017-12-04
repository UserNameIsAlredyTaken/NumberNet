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

/*INPUT is the vector of data, and pointer to the net structure
 *OUTPUT is the pointer to result vector
 *(lenth of the vector should be the same as the count of neurons in the input layer of the net(just a convention)),*/
double* neurons_output(uint8_t* const input,net* const the_net){
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

void shuffle(train_d* old_td, int size){
	int j;
	for(int i = 0;i<size;i++){
		j = rand() % size;// TODO: test the function (especially this palace)
		char tmp_y = old_td[i].y;
		old_td[i].y = old_td[j].y;
		old_td[j].y = tmp_y;
		char tmp_x = old_td[i].x;
		*old_td[i].x = old_td[j].x;
		*old_td[j].x = tmp_x;
	}
}

void update_mb(train_d* mb,int mb_size, double lr, net* the_net){
	net* gradients_net = malloc(sizeof(net));//initializing copy of the obtaind by method net and fill all biases\weights by zeros
	gradients_net->biases = (double**)malloc(sizeof(double*)*(the_net->n_layers - 1));
	gradients_net->weights = (double***)malloc(sizeof(double**)*(the_net->n_layers - 1));
	for (int i = 1; i < the_net->n_layers; i++) {
		gradients_net->biases[i] = (double*)malloc(sizeof(double)*the_net->sizes[i]);
		gradients_net->weights[i] = (double**)malloc(sizeof(double*)*the_net->sizes[i]);
		for (int j = 0; j < the_net->sizes[i]; j++) {
			
			gradients_net->biases[i][j] = 0;

			printf("b[%d][%d]:%f", i, j, gradients_net->biases[i][j]);
			gradients_net->weights[i][j] = (double*)malloc(sizeof(double)*the_net->sizes[i - 1]); printf("{");
			for (int k = 0; k < the_net->sizes[i - 1]; k++) {
				
				gradients_net->weights[i][j][k] = 0;

				printf("w[%d][%d][%d]:%f ", i, j, k, gradients_net->weights[i][j][k]);
			} printf("} ");
		} printf("\n");
	}


}

void gradient_descent(train_d* td,int td_length, int n_epohs, int mini_batch_size, double learning_rate,net* the_net){
	for(int i = 0;i < n_epohs;i++){
		shuffle(td,td_length);//for each epoch shuffle whole training set
		for(int j = 0;j<td_length / mini_batch_size;j++){//iterate aplying "update_mb" for each batch
			update_mb((td+j*mini_batch_size),mini_batch_size,learning_rate,the_net);
		}
		printf("Epoch %d is done", i);
	}
}

