#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NUMBER_OF_PIXELS 784


#pragma pack(push,1)
typedef struct data {
	uint8_t x[NUMBER_OF_PIXELS];///array with lenghth = NUMBER_OF_PIXELS
	uint8_t y[10];///array with lenghth = 10
}data;
#pragma pack(pop)

static void from_number_to_vector(uint8_t const number, data* td) {///the value of namber ranges from 0-9
	for (int j = 0; j<10; j++) {
		if (j == number) {
			td->y[j] = 1;
		}
		else {
			td->y[j] = 0;
		}
	}
}

data* get_data(char *lables_file_name, char *images_file_name, int const number_of_items) {	
	FILE *lables_file = fopen(lables_file_name, "rb");
	FILE *images_file = fopen(images_file_name, "rb");
	data* x_y_array = malloc(number_of_items * sizeof(data));
	uint8_t number_y;
	for (int i = 0; i < number_of_items; i++) {
		fread(&(number_y), sizeof(uint8_t), 1, lables_file);
		from_number_to_vector(number_y, &x_y_array[i]);
		fread(x_y_array[i].x, sizeof(uint8_t[NUMBER_OF_PIXELS]), 1, images_file);
	}
	fclose(lables_file);
	fclose(images_file);
	return x_y_array;
}

void save_net(char *file_name, struct net* final_net) {
	FILE *file = fopen(file_name, "wb");
	fwrite(final_net, sizeof(struct net*), 1, file);
	fclose(file);
}
