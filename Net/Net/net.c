#include <stdlib.h>

typedef struct net {
	int n_layers;
	int* sizes;
	int** biases;
	int*** weights;
}net;

net* init_net(int* sizes,int n_layers){
	net* the_net = malloc(sizeof(net));
	the_net->sizes = sizes;
	the_net->n_layers = n_layers;
	for (int i = 1; i < n_layers;i++){//заполняем биасы для каждого нейронного слоя кроме первого
		for (int j = 0; j < sizes[i];j++){//заполняем биасы для каждого нейрона
			//здесь могла бы быть ваша рандомящая функция с диапозоном от -1 до 1
		}
	}
	for (int i = 1; i < n_layers; i++) {//заполняем веса для каждого нейронного слоя кроме первого
		for (int j = 0; j < sizes[i]; j++){//заполняем веса для каждого нейрона
			for(int k = 0; k<sizes[i-1]; k++){//для каждого нейрона соответствующий вес с нейроном из предыдущего слоя
				//здесь могла бы быть ваша рандомящая функция с диапозоном от -1 до 1
			}
		}
	}
	return the_net;
}