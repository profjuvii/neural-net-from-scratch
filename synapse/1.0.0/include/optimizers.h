#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

typedef enum {
    SGD,
    Momentum,
    Adagrad,
    RMSprop,
    Adam
} Optimizer;

typedef struct {
    float *data;
} OptParams;

typedef struct {
    float *m;
    float *v;
} AdamParams;

void set_momentum_param(const float momentum);
void set_adagrad_param(const float epsilon);
void set_rmsprop_params(const float decay_rate, float epsilon);
void set_adam_params(const float beta1, float beta2);

void optimize(
    const Optimizer optimizer,
    const void *optimizer_params,
    float *weights,
    const float *gradients,
    const int size,
    const float learning_rate,
    const int t
);

#endif
