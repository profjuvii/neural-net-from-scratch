#include <math.h>
#include "losses.h"

float mse(float *predicts, float *targets, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float error = targets[i] - predicts[i];
        sum += error * error;
    }
    return 0.5f * sum / n;
}

float cross_entropy(float *predicts, float *targets, int n) {
    float loss = 0.0f;
    float epsilon = 1e-10f;

    for (int i = 0; i < n; ++i) {
        if (targets[i] > 0.0f) {
            loss -= targets[i] * logf(fmaxf(predicts[i], epsilon));
        }
    }

    return loss;
}

float loss_function(LossFunction loss_function, float *predicts, float *targets, int size) {
    switch (loss_function) {
        case MSE: return mse(predicts, targets, size);
        case CROSS_ENTROPY: return cross_entropy(predicts, targets, size);
    }
}