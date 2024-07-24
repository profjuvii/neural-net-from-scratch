#include <stdio.h>
#include "braincraft.h"

void print_neural_network(NeuralNetwork *nn) {
    printf("Neural Network:\n");
    printf("Number of layers: %d\n", nn->num_layers);
    printf("Optimizer: ");
    switch (nn->optimizer) {
        case SGD: printf("SGD\n"); break;
        case MOMENTUM: printf("Momentum\n"); break;
        case ADAM: printf("Adam\n"); break;
    }
    printf("Learning rate: %f\n", nn->learning_rate);

    for (int i = 0; i < nn->num_layers; ++i) {
        Layer *layer = &nn->layers[i];
        printf("Layer %d:\n", i + 1);
        printf("  Input size: %d\n", layer->input_size);
        printf("  Output size: %d\n", layer->output_size);
        printf("  Activation function: ");
        switch (layer->activation_func) {
            case RELU: printf("ReLU\n"); break;
            case LEAKY_RELU: printf("Leaky ReLU\n"); break;
            case SIGMOID: printf("Sigmoid\n"); break;
            case TANH: printf("Tanh\n"); break;
            case SOFTMAX: printf("Softmax\n"); break;
        }
        printf("  Alpha: %f\n", layer->alpha);

        for (int j = 0; j < layer->output_size; ++j) {
            Neuron *neuron = &layer->neurons[j];
            printf("    Neuron %d:\n", j + 1);
            printf("      Weights: ");
            for (int k = 0; k < layer->input_size; ++k) {
                printf("%f ", neuron->weights[k]);
            }
            printf("\n");
            printf("      Bias: %f\n", neuron->bias);
            printf("      Output: %f\n", layer->outputs[j]);
        }
    }
}

int main() {
    int input_size = 10;
    int output_size = 2;

    float inputs[] = {1.0, 0.01, 0.6, 0.78, 0.01, 1.0, 0.5, 0.25, 0.6, 1.0};
    // float targets[] = {1.0, 0.0};

    int num_layers = 4;
    float learning_rate = 0.01;

    NeuralNetwork *nn = create_network(num_layers, SGD, learning_rate);

    init_layer(&nn->layers[0], input_size, 10, RELU, 0.0);
    init_layer(&nn->layers[1], 10, 8, RELU, 0.0);
    init_layer(&nn->layers[2], 8, 6, RELU, 0.0);
    init_layer(&nn->layers[3], 6, output_size, SOFTMAX, 0.0);

    forward_pass(nn, inputs);
    print_neural_network(nn);

    destroy_network(nn);

    return 0;
}
