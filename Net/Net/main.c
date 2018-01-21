#include "get_data.h"
#include "net.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define TRAIN_DATA_LENGTH 60000
#define TEST_DATA_LENGTH 10000

int main(int argc, char* argv[]) {
	struct data* train_data = get_data("..\\Net\\train-labels.idx1-ubyte", "..\\Net\\train-images.idx3-ubyte", TRAIN_DATA_LENGTH);
	struct data* test_data = get_data("..\\Net\\t10k-labels.idx1-ubyte", "..\\Net\\t10k-images.idx3-ubyte", TEST_DATA_LENGTH);
	int sizes[] = { 784,30,10 };	
	struct net* the_net = init_net(sizes, sizeof(sizes) / sizeof(int));	

	struct net final_net = gradient_descent(train_data, TRAIN_DATA_LENGTH, 30, 10, 0.3, the_net, test_data, TEST_DATA_LENGTH);
	save_net(".", &final_net);
	return 0;
}
 