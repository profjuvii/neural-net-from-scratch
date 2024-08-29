#include <stdio.h>
#include "utils.h"

void print_vector(const float *vector, const int size) {
    if (size == 0) {
        printf("Empty vector.\n");
        return;
    }

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

int index_of_max(const float *vector, const int size) {
    if (size == 0) return -1;

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

void normalize_vector(float *vector, const int size, const float mean, const float std) {
    if (size == 0 || std == 0.0f) return;

    for (int i = 0; i < size; ++i) {
        vector[i] = (vector[i] - mean) / std;
    }
}

int auto_epoch(const int epoch, const int num_epochs) {
    if (num_epochs <= 0) return 0;
    
    if (num_epochs <= 10) {
        return num_epochs > 0;
    }

    int divisor = 10;
    while (divisor * divisor < num_epochs) {
        divisor *= 10;
    }

    return (epoch % divisor == 0);
}
