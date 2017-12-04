#pragma once
#include <stdint.h>
#define NUMBER_OF_ITEMS 10000
#define NUMBER_OF_PIXELS 784

typedef struct train_d {
	char x[NUMBER_OF_PIXELS];
	char y;
}train_d;

train_d* get_train_data(char* lables_file_name, char* images_file_name);