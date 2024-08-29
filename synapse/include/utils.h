#ifndef UTILS_H
#define UTILS_H

void print_vector(const float *vector, const int size);
int index_of_max(const float *vector, const int size);
void normalize_vector(float *vector, const int size, const float mean, const float std);
int auto_epoch(const int epoch, const int num_epochs);

#endif
