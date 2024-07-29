#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "image2vector.h"

#define PATH "/Users/profjuvi/Datasets/MNIST/"

typedef struct {
    float **vectors;
    int *labels;
    int size;
} DataLoader;

DataLoader* create_data_loader(int batch_size);
void destroy_data_loader(DataLoader *data_loader);
void load(DataLoader *data_loader, int num_classes);

#endif // DATA_LOADER_H
