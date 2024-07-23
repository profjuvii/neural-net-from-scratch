#include <stdio.h>
#include <stdlib.h>
#include "braincraft.h"


typedef struct {
    double *inputs;
    double *outputs;
    int size;
} SoftmaxParams;


NeuralNetwork *create_network(int num_layers) {
    NeuralNetwork *network = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    network->layers = (Layer *)malloc(num_layers * sizeof(Layer));
    network->num_layers = num_layers;
    return network;
}


void destroy_neurons(Neuron *neurons, int num_neurons) {
    if (neurons) {
        for (int i = 0; i < num_neurons; ++i) {
            free(neurons[i].weights);
        }
        free(neurons);
    }
}


void destroy_layers(Layer *layers, int num_layers) {
    if (layers) {
        for (int i = 0; i < num_layers; ++i) {
            destroy_neurons(layers[i].neurons, layers[i].num_neurons);
        }
        free(layers);
    }
}


void destroy_network(NeuralNetwork *network) {
    if (network) {
        destroy_layers(network->layers, network->num_layers);
        free(network);
    }
}


void init_layer(Layer *layer, int input_size, int num_neurons, ActivationFunc func, void *params) {
    layer->neurons = (Neuron *)malloc(num_neurons * sizeof(Neuron));
    layer->num_neurons = num_neurons;
    layer->input_size = input_size;
    layer->func = func;
    layer->params = params;
    
    for (int i = 0; i < num_neurons; ++i) {
        layer->neurons[i].weights = (double *)malloc(input_size * sizeof(double));
        layer->neurons[i].weight_gradients = (double *)calloc(input_size, sizeof(double));
        for (int j = 0; j < input_size; ++j) {
            layer->neurons[i].weights[j] = (double)rand() / RAND_MAX;
        }
        layer->neurons[i].bias = (double)rand() / RAND_MAX;
        layer->neurons[i].bias_gradient = 0.0;
    }
}


void print_neuron_info(const Neuron *neuron, int input_size, int index) {
    printf("%*sNeuron %02d:\n", 4, "", index + 1);
    printf("\tInput size: %d\n", input_size);
    printf("\tBias: %lf\n", neuron->bias);
    printf("\tBias gradient: %lf\n", neuron->bias_gradient);
    printf("\tOutput: %lf\n", neuron->output);
    printf("\tWeights: ");
    for (int i = 0; i < input_size; ++i) {
        printf("%lf ", neuron->weights[i]);
    }
    printf("\n\tWeight gradients: ");
    for (int i = 0; i < input_size; ++i) {
        printf("%lf ", neuron->weight_gradients[i]);
    }
    printf("\n\n");
}


void print_layer_info(const Layer *layer, int layer_index) {
    printf("---------\n");
    printf("Layer %02d:\n", layer_index + 1);
    printf("Number of neurons: %d\n", layer->num_neurons);
    for (int i = 0; i < layer->num_neurons; ++i) {
        print_neuron_info(&layer->neurons[i], layer->input_size, i);
    }
    printf("\n");
}


void print_network_info(const NeuralNetwork *network) {
    printf("Neural Network:\n");
    printf("Number of layers: %d\n", network->num_layers);
    for (int i = 0; i < network->num_layers; ++i) {
        print_layer_info(&network->layers[i], i);
    }
}


double relu(double x, void *params) {
    return x > 0 ? x : 0;
}


double relu_grad(double activation, void *params) {
    return activation > 0 ? 1.0 : 0.0;
}


double leaky_relu(double x, void* params) {
    double alpha = params ? *((double*)params) : 0.01;
    return x > 0 ? x : alpha * x;
}


double leaky_relu_grad(double activation, void* params) {
    double alpha = *((double*)params);
    return activation > 0 ? 1.0 : alpha;
}


double elu(double x, void *params) {
    double alpha = params ? *((double *)params) : 1.0;
    return x > 0 ? x : alpha * (exp(x) - 1);
}


double elu_grad(double activation, void* params) {
    double alpha = *((double*)params);
    return activation > 0 ? 1.0 : alpha * exp(activation);
}



double selu(double x, void *params) {
    double lambda = 1.0507;
    double alpha = 1.6733;
    return x > 0 ? lambda * x : lambda * alpha * (exp(x) - 1);
}


double selu_grad(double activation, void *params) {
    double lambda = 1.0507;
    double alpha = 1.6733;
    return activation > 0 ? lambda : lambda * alpha * exp(activation);
}


double hyperbolic_tangent(double x, void *params) {
    return tanh(x);
}


double tanh_grad(double activation, void* params) {
    return 1.0 - activation * activation;
}


double sigmoid(double x, void* params) {
    return 1.0 / (1.0 + exp(-x));
}


double sigmoid_grad(double actiation) {
    return actiation * (1.0 - actiation);
}


double softmax(double x, void *params) {
    SoftmaxParams *p = (SoftmaxParams *)params;

    double max = p->inputs[0];
    for (int i = 1; i < p->size; ++i) {
        if (p->inputs[i] > max) {
            max = p->inputs[i];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < p->size; ++i) {
        p->outputs[i] = exp(p->inputs[i] - max);
        sum += p->outputs[i];
    }

    for (int i = 0; i < p->size; ++i) {
        p->outputs[i] /= sum;
    }

    return 0.0;
}


void softmax_grad(double *grad_output, double *outputs, int size, double *grad_inputs) {
    for (int i = 0; i < size; ++i) {
        grad_inputs[i] = 0.0;
        for (int j = 0; j < size; ++j) {
            if (i == j) {
                grad_inputs[i] += outputs[i] * (1 - outputs[i]) * grad_output[j];
            } else {
                grad_inputs[i] -= outputs[i] * outputs[j] * grad_output[j];
            }
        }
    }
}


double swish(double x, void *params) {
    return x * sigmoid(x, NULL);
}


double swish_grad(double activation, void *params) {
    double sigmoid_val = sigmoid(activation, NULL);
    return sigmoid_val + activation * sigmoid_val * (1.0 - sigmoid_val);
}


double compute_weighted_sum(Neuron *neuron, double *inputs, int input_size) {
    double sum = 0.0;
    for (int i = 0; i < input_size; ++i) {
        sum += neuron->weights[i] * inputs[i];
    }
    return sum;
}


double *get_activations(Layer* layer) {
    double *activations = (double *)calloc(layer->num_neurons, sizeof(double));
    for (int i = 0; i < layer->num_neurons; ++i) {
        activations[i] = layer->neurons[i].output;
    }
    return activations;
}


void forward(NeuralNetwork *network, double *inputs) {
    double *activations = NULL;

    for (int i = 0; i < network->num_layers; ++i) {
        Layer *layer = &network->layers[i];
        activations = i == 0 ? inputs : get_activations(&network->layers[i - 1]);

        if (layer->func == softmax) {
            double *weighted_sum = (double *)calloc(layer->num_neurons, sizeof(double));
            double *outputs = (double *)calloc(layer->num_neurons, sizeof(double));

            for (int j = 0; j < layer->num_neurons; ++j) {
                weighted_sum[j] = compute_weighted_sum(&layer->neurons[j], activations, layer->input_size);
            }

            SoftmaxParams params;
            params.inputs = weighted_sum;
            params.outputs = outputs;
            params.size = layer->num_neurons;

            softmax(0.0, &params);

            for (int j = 0; j < layer->num_neurons; ++j) {
                layer->neurons[j].output = outputs[j];
            }

            free(weighted_sum);
            free(outputs);

        } else {
            for (int j = 0; j < layer->num_neurons; ++j) {
                double sum = compute_weighted_sum(&layer->neurons[j], activations, layer->input_size);
                layer->neurons[j].output = layer->func(sum, layer->params);
            }
        }

        if (i != 0) {
            free(activations);
        }
    }
}


void update_weights_sgd(Neuron *neuron, int input_size, double learning_rate) {
    for (int i = 0; i < input_size; ++i) {
        neuron->weights[i] += learning_rate * neuron->weight_gradients[i];
    }
    neuron->bias += learning_rate * neuron->bias_gradient;
}


void backpropagate(Layer* layer, double* inputs, double* targets, int output_size, double learning_rate) {
    for (int i = 0; i < layer->num_neurons; ++i) {
        Neuron* neuron = &layer->neurons[i];

        double grad = 0.0;
        if (layer->func == sigmoid) {
            grad = sigmoid_grad(neuron->output);
        } else if (layer->func == leaky_relu) {
            grad = leaky_relu_grad(neuron->output, layer->params);
        } else if (layer->func == elu) {
            grad = elu_grad(neuron->output, layer->params);
        } else if (layer->func == selu) {
            grad = selu_grad(neuron->output, layer->params);
        } else if (layer->func == hyperbolic_tangent) {
            grad = tanh_grad(neuron->output, NULL);
        } else if (layer->func == swish) {
            grad = swish_grad(neuron->output, NULL);
        } else if (layer->func == softmax) {
            continue;
        }

        double error = targets[i] - neuron->output;
        double delta = error * grad;

        for (int j = 0; j < layer->input_size; ++j) {
            neuron->weight_gradients[j] = delta * inputs[j];
        }
        neuron->bias_gradient = delta;

        update_weights_sgd(neuron, layer->input_size, learning_rate);
    }
}


void backward(NeuralNetwork *network, double* inputs, double* targets, int output_size, double learning_rate) {
    double* next_activations = NULL;
    double* prev_activations = NULL;

    for (int i = network->num_layers - 1; i > 0; --i) {
        prev_activations = get_activations(&network->layers[i - 1]);

        if (i == network->num_layers - 1) {
            if (network->layers[i].func == softmax) {
                double* softmax_outputs = (double*)malloc(output_size * sizeof(double));
                SoftmaxParams softmax_params = { inputs, softmax_outputs, output_size };
                softmax(0.0, &softmax_params);

                double* grad_outputs = (double*)malloc(output_size * sizeof(double));
                for (int j = 0; j < output_size; ++j) {
                    grad_outputs[j] = (softmax_outputs[j] - targets[j]);
                }

                double* grad_inputs = (double*)malloc(output_size * sizeof(double));
                softmax_grad(grad_outputs, softmax_outputs, output_size, grad_inputs);

                for (int j = 0; j < output_size; ++j) {
                    network->layers[i].neurons[j].bias_gradient = grad_inputs[j];
                    for (int k = 0; k < network->layers[i].input_size; ++k) {
                        network->layers[i].neurons[j].weight_gradients[k] = grad_inputs[j] * prev_activations[k];
                    }
                }

                free(softmax_outputs);
                free(grad_outputs);
                free(grad_inputs);
            } else {
                backpropagate(&network->layers[i], prev_activations, targets, output_size, learning_rate);
            }

        } else {
            next_activations = get_activations(&network->layers[i + 1]);
            backpropagate(&network->layers[i], prev_activations, next_activations, network->layers[i + 1].num_neurons, learning_rate);
            free(next_activations);
        }
        free(prev_activations);
    }

    next_activations = get_activations(&network->layers[1]);
    backpropagate(&network->layers[0], inputs, next_activations, network->layers[1].num_neurons, learning_rate);
    free(next_activations);
}


double mean_squared_error(double* targets, double* outputs, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        double error = targets[i] - outputs[i];
        sum += error * error;
    }
    return sum / size;
}
