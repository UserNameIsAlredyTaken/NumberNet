#include "../head/get_data.h"
#include "../head/net.h"
#include <stdint.h>
#include <stdio.h>


int main(int argc, char* argv[]){
	train_d* td = get_train_data("C:\\t10k-labels.idx1-ubyte", "C:\\t10k-images.idx3-ubyte");	
	//int sizes[] = { 784,30,10 };
	int sizes[] = { 2,2,1 };
	net* the_net = init_net(sizes,sizeof(sizes)/sizeof(int));
	uint8_t test_input[] = {138,90};
	double* output = neurons_output(test_input/*td->x*/,the_net);
	for(int i = 0;i<the_net->sizes[the_net->n_layers-1];i++){
		printf("out[%d]:%f ",i,output[i]);
	}
	return 0;	
}