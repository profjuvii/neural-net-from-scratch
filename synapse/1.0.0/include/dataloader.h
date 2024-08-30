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

void set_mean_std_params(const float mean, const float std);

DataLoader* create_dataloader(char *dataset_path, const int batch_size, const int input_size);
void destroy_dataloader(DataLoader *dataloader);
int get_next_batch(DataLoader *dataloader, const int num_classes);

#endif
