#ifndef LOSS_FUNCS_H
#define LOSS_FUNCS_H

typedef enum {
    MSE,
    CROSS_ENTROPY
} LossFunction;

float mse(float *predicts, float *targets, int size);
float cross_entropy(float *predicts, float *targets, int size);
double loss_func(LossFunction loss_func, float *predicts, float *targets, int size);

#endif // LOSS_FUNCS_H
