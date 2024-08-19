#include <math.h>
#include "loss_funcs.h"

float mse(float *predicts, float *targets, int size) {
    float sum_squared_errors = 0.0;

    for (int i = 0; i < size; ++i) {
        float error = predicts[i] - targets[i];
        sum_squared_errors += error * error;
    }

    return sum_squared_errors / size;
}

float cross_entropy(float *predicts, float *targets, int size) {
    float loss = 0.0;

    for (int i = 0; i < size; ++i) {
        float prediction = fmaxf(predicts[i], 1e-15);
        loss -= targets[i] * logf(prediction);
    }

    return loss;
}

double loss_func(LossFunction loss_func, float *predicts, float *targets, int size) {
    switch (loss_func) {
        case MSE:
            return mse(predicts, targets, size);
        case CROSS_ENTROPY:
            return cross_entropy(predicts, targets, size);
        default:
            return 0.0;
    }
}
