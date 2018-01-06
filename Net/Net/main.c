#include "get_data.h"
#include "net.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define TRAIN_DATA_LENGTH 60000
#define TEST_DATA_LENGTH 10000

int main(int argc, char* argv[]) {
	data* train_data = get_data("C:\\train-labels.idx1-ubyte", "C:\\train-images.idx3-ubyte", TRAIN_DATA_LENGTH);
	data* test_data = get_data("C:\\t10k-labels.idx1-ubyte", "C:\\t10k-images.idx3-ubyte", TEST_DATA_LENGTH);
	int sizes[] = { 784,15,10 };	
	net* the_net = init_net(sizes, sizeof(sizes) / sizeof(int));		
	gradient_descent(train_data, TRAIN_DATA_LENGTH, 30, 10, 3.0, the_net, test_data, TEST_DATA_LENGTH);
	return 0;
}
