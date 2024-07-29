#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "image2vector.h"

typedef struct {
    int batch_size;
    int input_size;
    float **vectors;
    int *labels;
    char *path;
} DataLoader;

DataLoader* create_data_loader(int batch_size, int input_size, char *path);
void destroy_data_loader(DataLoader *data_loader);

int load_data(DataLoader *data_loader, int num_classes);
float* create_targets(int num_classes, int index);

#endif // DATA_LOADER_H
