#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NUMBER_OF_ITEMS 10000
#define NUMBER_OF_PIXELS 784

typedef struct train_d{
	char x[NUMBER_OF_PIXELS];
	char y;
}train_d;

train_d* get_train_data(char* lables_file_name, char* images_file_name){	
	FILE *lables_file = fopen(lables_file_name, "rb");
	FILE *images_file = fopen(images_file_name, "rb");
	train_d* x_y_array = malloc(NUMBER_OF_ITEMS * sizeof(train_d));
	for(int i = 0; i < NUMBER_OF_ITEMS; i++){
		fread(&(x_y_array[i].y), sizeof(char), 1, lables_file);		
		fread(x_y_array[i].x, sizeof(char[NUMBER_OF_PIXELS]), 1, images_file);
	}
	fclose(lables_file);
	fclose(images_file);
	return x_y_array;
}
