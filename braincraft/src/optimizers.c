#include <math.h>
#include "optimizers.h"

void sgd(float *weights, float *weight_grads, int size, float learning_size) {
    for (int i = 0; i < size; ++i) {
        weights[i] -= learning_size * weight_grads[i];
    }
}

void momentum(float *weights, float *weight_grads, float *velocity, int size, float learning_size, float momentum) {
    for (int i = 0; i < size; ++i) {
        velocity[i] = momentum * velocity[i] - learning_size * weight_grads[i];
        weights[i] += velocity[i];
    }
}

void adam(float *weights, float *weight_grads, float *m, float *v, int size, float learning_size, float beta1, float beta2, float epsilon) {
    for (int i = 0; i < size; ++i) {
        m[i] = beta1 * m[i] + (1 - beta1) * weight_grads[i];
        v[i] = beta2 * v[i] + (1 - beta2) * weight_grads[i] * weight_grads[i];

        float m_hat = m[i] / (1 - beta1);
        float v_hat = v[i] / (1 - beta2);
        
        weights[i] -= learning_size * m_hat / (sqrt(v_hat) + epsilon);
    }
}
