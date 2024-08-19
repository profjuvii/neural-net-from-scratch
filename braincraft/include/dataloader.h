#ifndef DATALOADER_H
#define DATALOADER_H

typedef struct {
    float *features;
    int label;
} Sample;

typedef struct {
    Sample *batch;
    char* dataset_path;
    int batch_size;
    int input_size;
} DataLoader;

DataLoader* create_dataloader(char* dataset_path, int batch_size, int input_size);
void destroy_dataloader(DataLoader *dataloader);

int get_next_batch(DataLoader *dataloader, int num_classes);
float* create_targets(int num_classes, int label);

#endif // DATALOADER_H
