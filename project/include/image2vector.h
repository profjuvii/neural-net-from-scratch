#ifndef IMAGE2VECTOR_H
#define IMAGE2VECTOR_H

void image2vector(const char *filename, float *buffer, int *size);
void normalize_vector(float *vector, int size, float mean, float std);

#endif // IMAGE2VECTOR_H
