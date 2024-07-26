#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "braincraft.h"

NeuralNetwork *create_network(int num_layers, Optimizer optimizer, LossFunction loss_func, float learning_rate, float momentum, float beta1, float beta2) {
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    nn->layers = (Layer *)malloc(num_layers * sizeof(Layer));
    nn->num_layers = num_layers;
    nn->optimizer = optimizer;

    if (optimizer == MOMENTUM) {
        for (int i = 0; i < num_layers; ++i) {
            Layer *layer = &nn->layers[i];
            layer->velocity = (float *)malloc(layer->input_size * sizeof(float));
            for (int j = 0; j < layer->input_size; ++j) {
                layer->velocity[j] = 0.0;
            }
        }
    } else if (optimizer == ADAM) {
        for (int i = 0; i < num_layers; ++i) {
            Layer *layer = &nn->layers[i];

            layer->m = (float *)malloc(layer->input_size * sizeof(float));
            layer->v = (float *)malloc(layer->input_size * sizeof(float));
            
            for (int j = 0; j < layer->input_size; ++j) {
                layer->m[j] = 0.0;
                layer->v[j] = 0.0;
            }
        }
    }

    nn->loss_func = loss_func;
    nn->learning_rate = learning_rate;
    nn->momentum = momentum;
    nn->beta1 = beta1;
    nn->beta2 = beta2;

    return nn;
}

void destroy_layer(Layer *layer) {
    if (layer == NULL) return;
    for (int i = 0; i < layer->output_size; ++i) {
        free(layer->neurons[i].weights);
    }
    free(layer->neurons);
    free(layer->outputs);
    free(layer->sums);
}

void destroy_network(NeuralNetwork *nn) {
    if (nn == NULL) return;
    for (int i = 0; i < nn->num_layers; ++i) {
        destroy_layer(&nn->layers[i]);
    }
    free(nn->layers);
    free(nn);
}

void init_layer(Layer *layer, int input_size, int output_size, ActivationFunction activation_func, float alpha) {
    if (layer == NULL) return;

    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation_func = activation_func;
    layer->alpha = alpha;

    layer->neurons = (Neuron *)malloc(output_size * sizeof(Neuron));
    float limit = sqrt(6.0 / (input_size + output_size));

    // Initialize weights and biases for each neuron using Xavier (Glorot) initialization
    for (int i = 0; i < output_size; ++i) {
        layer->neurons[i].weights = (float *)malloc(input_size * sizeof(float));
        for (int j = 0; j < input_size; ++j) {
            layer->neurons[i].weights[j] = (float)rand() / RAND_MAX * 2 * limit - limit;
        }
        layer->neurons[i].bias = (float)rand() / RAND_MAX * 2 * limit - limit;
    }

    layer->outputs = (float *)malloc(output_size * sizeof(float));
    layer->sums = (float *)malloc(output_size * sizeof(float));
}

float sum(Neuron *neuron, int input_size, float *inputs) {
    float res = 0.0;
    for (int i = 0; i < input_size; ++i) {
        res += neuron->weights[i] * inputs[i];
    }
    return res + neuron->bias;
}

float func(ActivationFunction func, float x, float alpha) {
    switch (func) {
        case LINEAR:
            return linear(x);
        case RELU:
            return relu(x);
        case LEAKY_RELU:
            return leaky_relu(x, alpha);
        case SIGMOID:
            return sigmoid(x);
        case TANH:
            return tanh_activation(x);
        case SOFTMAX:
            return x;
    }
    return x;
}

void forward_pass(NeuralNetwork *nn, float *inputs) {
    for (int i = 0; i < nn->num_layers; ++i) {
        Layer *layer = &nn->layers[i];

        for (int j = 0; j < layer->output_size; ++j) {
            float x = sum(&layer->neurons[j], layer->input_size, i == 0 ? inputs : nn->layers[i - 1].outputs);
            layer->sums[j] = x;
            if (layer->activation_func != SOFTMAX) {
                layer->outputs[j] = func(layer->activation_func, x, layer->alpha);
            }
        }

        if (layer->activation_func == SOFTMAX) {
            softmax(layer->sums, layer->outputs, layer->output_size);
        }
    }
    nn->predictions = nn->layers[nn->num_layers - 1].outputs;
}

float func_grad(ActivationFunction func, float x, float alpha) {
    switch (func) {
        case LINEAR:
            return linear_derivative();
        case RELU:
            return relu_derivative(x);
        case LEAKY_RELU:
            return leaky_relu_derivative(x, alpha);
        case SIGMOID:
            return sigmoid_derivative(x);
        case TANH:
            return tanh_derivative(x);
        case SOFTMAX:
            return x;
    }
    return x;
}

void mse_grad(float *outputs, float *targets, int size, float *grads) {
    for (int i = 0; i < size; ++i) {
        grads[i] = 2 * (outputs[i] - targets[i]) / size;
    }
}

void cross_entropy_softmax_grad(float *softmax_outputs, float *targets, int size, float *grads) {
    for (int i = 0; i < size; ++i) {
        grads[i] = softmax_outputs[i] - targets[i];
    }
}

void cross_entropy_grad(float *outputs, float *targets, int size, float *grads) {
    for (int i = 0; i < size; ++i) {
        if (outputs[i] > 0) {
            grads[i] = -targets[i] / outputs[i];
        } else {
            grads[i] = 0;
        }
    }
}

float *get_weight_grads(float *inputs, int size, float activation_grad) {
    float *weight_grads = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; ++i) {
        weight_grads[i] = activation_grad * inputs[i];
    }
    return weight_grads;
}

void backward_pass(NeuralNetwork *nn, float *targets) {
    Layer *output_layer = &nn->layers[nn->num_layers - 1];
    float *output_gradients = (float *)malloc(output_layer->output_size * sizeof(float));

    // Calculate gradients for the output layer based on the loss function
    if (nn->loss_func == MSE) {
        mse_grad(output_layer->outputs, targets, output_layer->output_size, output_gradients);
    } else if (nn->loss_func == CROSS_ENTROPY) {
        if (output_layer->activation_func == SOFTMAX) {
            // For softmax activation with cross entropy loss, the gradient is simplified
            cross_entropy_softmax_grad(output_layer->outputs, targets, output_layer->output_size, output_gradients);
        } else {
            // For cross entropy loss without softmax, use the general gradient
            cross_entropy_grad(output_layer->outputs, targets, output_layer->output_size, output_gradients);
        }
    }
    
    // Adjust gradients for non-softmax activations
    if (output_layer->activation_func != SOFTMAX) {
        for (int i = 0; i < output_layer->output_size; ++i) {
            float activation_grad = func_grad(output_layer->activation_func, output_layer->sums[i], output_layer->alpha);
            output_gradients[i] *= activation_grad;
        }
    }

    // Update weights and biases of the output layer
    for (int i = 0; i < output_layer->output_size; ++i) {
        float *weight_grads = get_weight_grads(output_layer->outputs, output_layer->input_size, output_gradients[i]);
        switch (nn->optimizer) {
            case SGD:
                sgd(output_layer->neurons[i].weights, weight_grads, output_layer->input_size, nn->learning_rate);
                break;
            case MOMENTUM:
                momentum(output_layer->neurons[i].weights, weight_grads, output_layer->velocity, output_layer->input_size, nn->learning_rate, nn->momentum);
                break;
            case ADAM:
                adam(output_layer->neurons[i].weights, weight_grads, output_layer->m, output_layer->v, output_layer->input_size, nn->learning_rate, nn->beta1, nn->beta2, 1e-8);
                break;
        }
        output_layer->neurons[i].bias -= nn->learning_rate * output_gradients[i];
        free(weight_grads);
    }

    // Backpropagation through hidden layers
    for (int l = nn->num_layers - 2; l >= 0; --l) {
        Layer *layer = &nn->layers[l];
        Layer *next_layer = &nn->layers[l + 1];
        float *hidden_gradients = (float *)calloc(layer->output_size, sizeof(float));

        // Compute gradients for each neuron in the current layer
        for (int i = 0; i < layer->output_size; ++i) {
            for (int j = 0; j < next_layer->output_size; ++j) {
                hidden_gradients[i] += next_layer->neurons[j].weights[i] * output_gradients[j];
            }
            hidden_gradients[i] *= func_grad(layer->activation_func, layer->sums[i], layer->alpha);
        }

        // Update weights and biases of the current layer
        for (int i = 0; i < layer->output_size; ++i) {
            float *weight_grads = get_weight_grads(layer->outputs, layer->input_size, hidden_gradients[i]);
            switch (nn->optimizer) {
                case SGD:
                    sgd(layer->neurons[i].weights, weight_grads, layer->input_size, nn->learning_rate);
                    break;
                case MOMENTUM:
                    momentum(layer->neurons[i].weights, weight_grads, layer->velocity, layer->input_size, nn->learning_rate, nn->momentum);
                    break;
                case ADAM:
                    adam(layer->neurons[i].weights, weight_grads, layer->m, layer->v, layer->input_size, nn->learning_rate, nn->beta1, nn->beta2, 1e-8);
                    break;
            }
            layer->neurons[i].bias -= nn->learning_rate * hidden_gradients[i];
            free(weight_grads);
        }

        free(output_gradients);
        output_gradients = hidden_gradients;
    }
    free(output_gradients);
}

float mse(float *predictions, float *targets, int size) {
    float sum_squared_errors = 0.0;
    for (int i = 0; i < size; ++i) {
        float error = predictions[i] - targets[i];
        sum_squared_errors += error * error;
    }
    return sum_squared_errors / size;
}
