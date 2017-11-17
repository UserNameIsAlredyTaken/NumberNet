#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

typedef struct net {
	int n_layers;
	int* sizes;
	double** biases;
	double*** weights;
}net;

net* init_net(int* sizes,int n_layers){
	net* the_net = malloc(sizeof(net));
	the_net->sizes = sizes;
	the_net->n_layers = n_layers;
	the_net->biases = (double**)malloc(sizeof(double*)*(n_layers-1));
	for (int i = 1; i < n_layers;i++){//заполняем биасы для каждого нейронного слоя кроме первого
		the_net->biases[i] = (double*)malloc(sizeof(double)*sizes[i]);
		for (int j = 0; j < sizes[i];j++){//заполняем биасы для каждого нейрона			
			srand(((i*3467897)^(j*78951))%1328098134);//жалкие попытки написать реальный рандом
			the_net->biases[i][j] = -1 + (rand() % 20000)*0.0001;//рандомящая функция с диапозоном от -1 до 1
			printf("b[%d][%d]:%f ",i,j,the_net->biases[i][j]);
		}
		printf("\n");
	}
	printf("\n");
	printf("\n");
	the_net->weights = (double***)malloc(sizeof(double**)*n_layers-1);
	for (int i = 1; i < n_layers; i++){//заполняем биасы для каждого нейронного слоя кроме первого
		the_net->weights[i] = (double**)malloc(sizeof(double*)*sizes[i]);
		for (int j = 0; j < sizes[i]; j++){//заполняем веса для каждого нейрона
			the_net->weights[i][j] = (double*)malloc(sizeof(double)*sizes[i-1]);
			printf("[");
			for(int k = 0; k < sizes[i-1]; k++){//В каждый нейрон входит столько весов, сколько в предыдущем слое нейронов
				srand((((i * 3497) ^ (j * 751))^(k*4428)) % 7215242);
				the_net->weights[i][j][k] = -1 + (rand() % 20000)*0.0001;//рандомящая функция с диапозоном от -1 до 1
				printf("w[%d][%d][%d]:%f ", i, j, k, the_net->weights[i][j][k]);
			}
			printf("]");
		}
		printf("\n");
	}
	return the_net;
}

static double sigmoid_func(double const weighed_inp) {
	return 1 / (1 + exp(-1 * weighed_inp));
}

/*на входе входные данные
 *(длинна массива input обязательно должна соответствовать количеству входных нейронов),
 * на выходе результат*/
double* neurons_output(uint8_t* input,net* const the_net){
	double* input_prev = (double*)malloc(sizeof(double)*the_net->sizes[0]);
	double* input_next = (double*)malloc(sizeof(double)*the_net->sizes[1]);
	for(int i=0;i<the_net->sizes[0];i++){//преобразовываем входные данные типа uint8_t в double
		input_prev[i] = input[i];
	}
	for(int i = 1; i<the_net->n_layers;i++){
		for (int j = 0; j < the_net->sizes[i];j++){//проходимся по всем нейронам
			double weighed_sum = 0;
			for(int k = 0; k < the_net->sizes[i-1]; k++){//суммируем значение входных данных для нейрона, помноженных на веса
				weighed_sum += input_prev[k] * the_net->weights[i][j][k];
			}
			input_next[j] = sigmoid_func(weighed_sum+the_net->biases[i][j]);
			printf("%f ", input_next[j]);
		}
		printf("\n");
		input_prev = input_next;
		input_next = (double*)malloc(sizeof(double)*the_net->sizes[i+1]);
	}
	printf("\n");
	printf("\n");
	return input_prev;	
}

