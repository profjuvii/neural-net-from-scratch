#ifndef LOSSES_H
#define LOSSES_H

typedef enum {
    MSE,
    CROSS_ENTROPY
} LossFunction;

float mse(const float *predicts, const float *targets, const int n);
float cross_entropy(const float *predicts, const float *targets, const int n);

#endif
