#include <stdlib.h>
#include <time.h>
#include <stdio.h>

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
			//printf("b[%d][%d]:%f ",i,j,the_net->biases[i][j]);
		}
		printf("\n");
	}
	the_net->weights = (double***)malloc(sizeof(double**)*n_layers-1);
	for (int i = 1; i < n_layers; i++){//заполняем биасы для каждого нейронного слоя кроме первого
		the_net->weights[i] = (double**)malloc(sizeof(double*)*sizes[i]);
		for (int j = 0; j < sizes[i]; j++){//заполняем веса для каждого нейрона
			the_net->weights[i][j] = (double*)malloc(sizeof(double)*sizes[i-1]);
			for(int k = 1; k<sizes[i-1]; k++){//В каждый нейрон входит столько весов, сколько в предыдущем слое нейронов
				srand((((i * 3497) ^ (j * 751))^(k*4428)) % 7215242);
				the_net->weights[i][j][k] = -1 + (rand() % 20000)*0.0001;//рандомящая функция с диапозоном от -1 до 1
				printf("w[%d][%d][%d]:%f ", i, j, k, the_net->weights[i][j][k]);
			}
		}
	}
	return the_net;
}