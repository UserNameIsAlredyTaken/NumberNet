#include "get_data.h"
#include <stdint.h>


int main(int argc, char* argv[]){
	train_d* td = get_train_data("C:\\t10k-labels.idx1-ubyte", "C:\\t10k-images.idx3-ubyte");	

	return 0;	
}
