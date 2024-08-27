#ifndef LOSSES_H
#define LOSSES_H

typedef enum {
    MSE,
    CROSS_ENTROPY
} LossFunction;

float loss_function(LossFunction loss_function, float *predicts, float *targets, int size);

#endif
