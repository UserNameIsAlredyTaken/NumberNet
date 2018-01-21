#pragma once
#include <stdint.h>
#define NUMBER_OF_PIXELS 784

struct data {
	uint8_t x[NUMBER_OF_PIXELS];//array with lenghth = NUMBER_OF_PIXELS
	uint8_t y[10];//array with lenghth = 10
};


struct data* get_data(char *lables_file_name, char *images_file_name, int const number_of_items);
void save_net(char *file_name, struct net* final_net);