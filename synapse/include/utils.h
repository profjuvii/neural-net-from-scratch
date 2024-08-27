#ifndef UTILS_H
#define UTILS_H

void print_vector(float *vector, int size);
int index_of_max(float *vector, int size);
void normalize_vector(float *vector, int size, float mean, float std);
int auto_epoch(int epoch, int num_epochs);

#endif
