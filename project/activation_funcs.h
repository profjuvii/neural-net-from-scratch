#ifndef ACTIVATION_FUNCS_H
#define ACTIVATION_FUNCS_H

typedef enum {
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    SOFTMAX
} ActivationFunction;

float relu(float x);
float leaky_relu(float x, float alpha);
float sigmoid(float x);
float tanh_activation(float x);
void softmax(float *input, float *output, int size);

float relu_derivative(float x);
float leaky_relu_derivative(float x, float alpha);
float sigmoid_derivative(float x);
float tanh_derivative(float x);

#endif // ACTIVATION_FUNCS_H
