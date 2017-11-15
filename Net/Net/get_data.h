#pragma once
#include <stdint.h>

typedef struct input_image {
	uint32_t n_of_rows;
	uint32_t n_of_cols;
	int* pixls;
}input_image;

typedef struct training_data_tuple {
	input_image x;
	int8_t y;
}train_d;

train_d* get_train_data(char* lables_file_name, char* images_file_name);