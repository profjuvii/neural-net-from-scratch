#ifndef ACTIVATORS_H
#define ACTIVATORS_H

typedef enum {
    Linear,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    Softmax
} ActivationFunction;

void set_leaky_relu_param(const float alpha);

void softmax(const float *inputs, float *outputs, const int size);
float activation_function(const ActivationFunction activation_function, const float x);
float activation_function_derivative(const ActivationFunction activation_function, const float x);

#endif
