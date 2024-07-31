#include <stdio.h>
#include "utils.h"

void print_vector(const char *text, const float *vector, int size, int width, int precision) {
    if (text && *text) {
        printf("%s", text);
    }
    for (int i = 0; i < size; ++i) {
        printf("%*.*f ", width > 0 ? width : 0, precision > 0 ? precision : 0, vector[i]);
    }
    printf("\n");
}

int find_max_index(const float *vector, int size) {
    int pos = 0;
    for (int i = 0; i < size; ++i) {
        if (vector[pos] < vector[i]) {
            pos = i;
        }
    }
    return pos;
}
