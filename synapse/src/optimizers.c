#include <math.h>
#include "optimizers.h"

static float opt_momentum = 0.9f;
static float opt_epsilon = 1e-8f;
static float opt_decay_rate = 0.99f;
static float opt_beta1 = 0.9f;
static float opt_beta2 = 0.999f;

void set_momentum_param(const float momentum) {
    opt_momentum = momentum;
}

void set_adagrad_param(const float epsilon) {
    opt_epsilon = epsilon;
}

void set_rmsprop_params(const float decay_rate, const float epsilon) {
    opt_decay_rate = decay_rate;
    opt_epsilon = epsilon;
}

void set_adam_params(const float beta1, const float beta2) {
    opt_beta1 = beta1;
    opt_beta2 = beta2;
}

void sgd(float *weights, const float *gradients, const int size, const float learning_rate) {
    for (int i = 0; i < size; ++i) {
        weights[i] -= learning_rate * gradients[i];
    }
}

void momentum(
    float *weights,
    const float *gradients,
    float *velocity,
    const int size,
    const float learning_rate
) {
    for (int i = 0; i < size; ++i) {
        velocity[i] = opt_momentum * velocity[i] - learning_rate * gradients[i];
        weights[i] += velocity[i];
    }
}

void adagrad(
    float *weights,
    const float *gradients,
    float *cache,
    const int size,
    const float learning_rate
) {
    for (int i = 0; i < size; ++i) {
        cache[i] += gradients[i] * gradients[i];
        weights[i] -= learning_rate * gradients[i] / (sqrtf(cache[i]) + opt_epsilon);
    }
}

void rmsprop(
    float *weights,
    const float *gradients,
    float *cache,
    const int size,
    const float learning_rate
) {
    for (int i = 0; i < size; ++i) {
        cache[i] = opt_decay_rate * cache[i] + (1.0f - opt_decay_rate) * gradients[i] * gradients[i];
        weights[i] -= learning_rate * gradients[i] / (sqrtf(cache[i]) + opt_epsilon);
    }
}

void adam(
    float *weights,
    const float *gradients,
    float *m,
    float *v,
    const int size,
    const float learning_rate,
    const int t
) {
    float beta1_t = powf(opt_beta1, t);
    float beta2_t = powf(opt_beta2, t);

    for (int i = 0; i < size; ++i) {
        m[i] = opt_beta1 * m[i] + (1.0f - opt_beta1) * gradients[i];
        v[i] = opt_beta2 * v[i] + (1.0f - opt_beta2) * gradients[i] * gradients[i];

        float m_hat = m[i] / (1.0f - beta1_t);
        float v_hat = v[i] / (1.0f - beta2_t);

        weights[i] -= learning_rate * m_hat / (sqrtf(v_hat) + opt_epsilon);
    }
}

void optimize(
    const Optimizer optimizer,
    const void *optimizer_params,
    float *weights,
    const float *gradients,
    const int size,
    const float learning_rate,
    const int t
) {
    switch (optimizer) {
        case SGD: {
            sgd(weights, gradients, size, learning_rate);
            break;
        }
        case MOMENTUM:
        case ADAGRAD:
        case RMSPROP: {
            OptParams *params = (OptParams *)optimizer_params;
            switch (optimizer) {
                case MOMENTUM:
                    momentum(weights, gradients, params->data, size, learning_rate);
                    break;
                case ADAGRAD:
                    adagrad(weights, gradients, params->data, size, learning_rate);
                    break;
                case RMSPROP:
                    rmsprop(weights, gradients, params->data, size, learning_rate);
                    break;
                default:
                    break;
            }
            break;
        }
        case ADAM: {
            AdamParams *params = (AdamParams *)optimizer_params;
            adam(
                weights,
                gradients,
                params->m,
                params->v,
                size,
                learning_rate,
                t
            );
            break;
        }
    }
}
