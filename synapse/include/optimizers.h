#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

extern float momentum_param;

typedef enum {
    SGD,
    MOMENTUM
} Optimizer;

void apply_optimizer(
    Optimizer optimizer,
    float *weights,
    float *weight_grads,
    float *velocity,
    int size,
    float learning_rate
);

#endif
