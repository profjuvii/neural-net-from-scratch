#include <stdio.h>
#include "utils.h"

void print_vector(float *vector, int size) {
    if (size <= 10) {
        for (int i = 0; i < size; ++i) {
            printf("%.6f ", vector[i]);
        }
    } else {
        for (int i = 0; i < 5; ++i) {
            printf("%.6f ", vector[i]);
        }
        printf("... ");
        for (int i = size - 5; i < size; ++i) {
            printf("%.6f ", vector[i]);
        }
    }
    printf("\n");
}

int index_of_max(float *vector, int size) {
    float max = *vector;
    int idx = 0;

    for (int i = 1; i < size; ++i) {
        if (*(vector + i) > max) {
            max = *(vector + i);
            idx = i;
        }
    }

    return idx;
}

void normalize_vector(float *vector, int size, float mean, float std) {
    for (int i = 0; i < size; ++i) {
        vector[i] = (vector[i] - mean) / std;
    }
}

int auto_epoch(int epoch, int num_epochs) {
    if (num_epochs <= 10) {
        return num_epochs > 0;
    }

    int divisor = 10;
    while (divisor * divisor < num_epochs) {
        divisor *= 10;
    }

    return (epoch % divisor == 0);
}
