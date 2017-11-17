#include "get_data.h"
#include "net.h"
#include <stdint.h>
#include <stdio.h>


int main(int argc, char* argv[]){
	train_d* td = get_train_data("C:\\t10k-labels.idx1-ubyte", "C:\\t10k-images.idx3-ubyte");	
	int sizes[] = { 784,30,10 };
	net* the_net = init_net(sizes,sizeof(sizes)/sizeof(int));
	return 0;	
}
