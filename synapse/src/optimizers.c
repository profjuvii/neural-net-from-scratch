#include "optimizers.h"

float momentum_param = 0.9f;

void sgd(float *weights, float *weight_grads, int size, float learning_rate) {
    for (int i = 0; i < size; ++i) {
        weights[i] -= learning_rate * weight_grads[i]; 
    }
}

void momentum(float *weights, float *weight_grads, float *velocity, int size, float learning_rate) {
    for (int i = 0; i < size; ++i) {
        velocity[i] = momentum_param * velocity[i] + learning_rate * weight_grads[i];
        weights[i] -= velocity[i];
    }
}

void apply_optimizer(
    Optimizer optimizer,
    float *weights,
    float *weight_grads,
    float *velocity,
    int size,
    float learning_rate
) {
    switch (optimizer) {
        case SGD: return sgd(weights, weight_grads, size, learning_rate);
        case MOMENTUM: return momentum(weights, weight_grads, velocity, size, learning_rate);
        default: return;
    }
}
