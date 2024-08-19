#ifndef ACTIVATION_FUNCS_H
#define ACTIVATION_FUNCS_H

typedef enum {
    LINEAR,
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

float relu_grad(float x);
float leaky_relu_grad(float x, float alpha);
float sigmoid_grad(float x);
float tanh_grad(float x);

float activation_func(ActivationFunction func, float x, float alpha);
float activation_func_grad(ActivationFunction func, float x, float alpha);

#endif // ACTIVATION_FUNCS_H
