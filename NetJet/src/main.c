#include "../head/get_data.h"
#include "../head/net.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define TRAIN_DATA_LENGTH 60000
#define TEST_DATA_LENGTH 10000

int main(int argc, char* argv[]) {
    data* train_data = get_data("C:\\train-labels.idx1-ubyte", "C:\\train-images.idx3-ubyte", TRAIN_DATA_LENGTH);
    data* test_data = get_data("C:\\t10k-labels.idx1-ubyte", "C:\\t10k-images.idx3-ubyte", TEST_DATA_LENGTH);


    for (int i = 0; i < 10; i++) {
        double rand_v;
        srand(time(NULL));
        rand_v = (double)rand()/RAND_MAX*2.0-1.0;
        printf("%f\n", rand_v);
    }


    int sizes[] = { 3,2,2 };
    net* the_net = init_net(sizes, sizeof(sizes) / sizeof(int));
    gradient_descent(train_data, TRAIN_DATA_LENGTH, 30, 10, 3.0, the_net, test_data, TEST_DATA_LENGTH);
    return 0;
}
