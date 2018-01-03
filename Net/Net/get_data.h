#pragma once
#include <stdint.h>
#define NUMBER_OF_PIXELS 784

typedef struct data {
	uint8_t x[NUMBER_OF_PIXELS];//array with lenghth = NUMBER_OF_PIXELS
	uint8_t y[10];//array with lenghth = 10
}data;


data* get_data(char *lables_file_name, char *images_file_name, int number_of_items);