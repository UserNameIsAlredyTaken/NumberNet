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
	for (int i = 1; i < n_layers;i++){//��������� ����� ��� ������� ���������� ���� ����� �������
		for (int j = 0; j < sizes[i];j++){//��������� ����� ��� ������� �������
			//����� ����� �� ���� ���� ���������� ������� � ���������� �� -1 �� 1
		}
	}
	for (int i = 1; i < n_layers; i++) {//��������� ���� ��� ������� ���������� ���� ����� �������
		for (int j = 0; j < sizes[i]; j++){//��������� ���� ��� ������� �������
			for(int k = 0; k<sizes[i-1]; k++){//��� ������� ������� ��������������� ��� � �������� �� ����������� ����
				//����� ����� �� ���� ���� ���������� ������� � ���������� �� -1 �� 1
			}
		}
	}
	return the_net;
}