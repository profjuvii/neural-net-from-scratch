#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

typedef enum {
    SGD,
    MOMENTUM,
    ADAM
} Optimizer;

void sgd(float *weights, float *weight_grads, int size, float learning_size);
void momentum(float *weights, float *weight_grads, float *velocity, int size, float learning_size, float momentum);
void adam(float *weights, float *weight_grads, float *m, float *v, int size, float learning_size, float beta1, float beta2, float epsilon);

#endif // OPTIMIZERS_H
