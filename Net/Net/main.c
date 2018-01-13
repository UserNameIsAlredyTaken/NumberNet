#include "get_data.h"
#include "net.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define TRAIN_DATA_LENGTH 60000
#define TEST_DATA_LENGTH 10000

int main(int argc, char* argv[]) {
	data* train_data = get_data("C:\\Users\\danil\\Desktop\\NumberNet\\Net\\Net\\train-labels.idx1-ubyte", "C:\\Users\\danil\\Desktop\\NumberNet\\Net\\Net\\train-images.idx3-ubyte", TRAIN_DATA_LENGTH);
	data* test_data = get_data("C:\\Users\\danil\\Desktop\\NumberNet\\Net\\Net\\t10k-labels.idx1-ubyte", "C:\\Users\\danil\\Desktop\\NumberNet\\Net\\Net\\t10k-images.idx3-ubyte", TEST_DATA_LENGTH);
	int sizes[] = { 784,10,10 };	
	net* the_net = init_net(sizes, sizeof(sizes) / sizeof(int));	

	gradient_descent(train_data, TRAIN_DATA_LENGTH, 30, 10, 0.2, the_net, test_data, TEST_DATA_LENGTH);
	return 0;
}
