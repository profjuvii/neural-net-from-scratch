#include <stdio.h>
#include "utils.h"
#include "braincraft.h"

void print_vector(const char *text, const float *vector, int size, int width, int precision) {
    if (text && *text) {
        printf("%s", text);
    }
    for (int i = 0; i < size; ++i) {
        printf("%*.*f ", width > 0 ? width : 0, precision > 0 ? precision : 0, vector[i]);
    }
    printf("\n");
}

int find_max_index(const float *vector, int size) {
    int pos = 0;
    for (int i = 0; i < size; ++i) {
        if (vector[pos] < vector[i]) {
            pos = i;
        }
    }
    return pos;
}

void print_neural_network(void *nn_ptr) {
    NeuralNetwork *nn = (NeuralNetwork *)nn_ptr;

    if (nn == NULL) {
        printf("Error: Neural network is NULL.\n");
        return;
    }

    printf("Neural Network:\n");
    printf("Number of Layers: %d\n\n", nn->num_layers);

    for (int l = 0; l < nn->num_layers; ++l) {
        Layer *layer = &nn->layers[l];
        printf("Layer %d:\n", l);
        printf("Input Size: %d\n", layer->input_size);
        printf("Output Size: %d\n", layer->output_size);
        printf("Activation Function: %d\n", layer->activation_func);
        printf("Alpha (for Leaky ReLU): %f\n", layer->alpha);
        printf("Number of Neurons: %d\n", layer->output_size);
        printf("Activations: ");
        for (int i = 0; i < layer->output_size; ++i) {
            printf("%f ", layer->activations[i]);
        }
        printf("\nSums: ");
        for (int i = 0; i < layer->output_size; ++i) {
            printf("%f ", layer->sums[i]);
        }
        printf("\n");

        for (int n = 0; n < layer->output_size; ++n) {
            Neuron *neuron = &layer->neurons[n];
            printf("  Neuron %d:\n", n);
            printf("    Bias: %f\n", neuron->bias);
            printf("    Bias Gradient: %f\n", neuron->bias_grad);

            printf("    Weights: ");
            for (int i = 0; i < layer->input_size; ++i) {
                printf("%f ", neuron->weights[i]);
            }
            printf("\n");

            printf("    Weight Gradients: ");
            for (int i = 0; i < layer->input_size; ++i) {
                printf("%f ", neuron->weight_grads[i]);
            }
            printf("\n");

            printf("    Velocity: ");
            for (int i = 0; i < layer->input_size; ++i) {
                printf("%f ", neuron->velocity[i]);
            }
            printf("\n");

            printf("    Moment (m): ");
            for (int i = 0; i < layer->input_size; ++i) {
                printf("%f ", neuron->m[i]);
            }
            printf("\n");

            printf("    Second Moment (v): ");
            for (int i = 0; i < layer->input_size; ++i) {
                printf("%f ", neuron->v[i]);
            }
            printf("\n");
        }

        printf("\n");
    }
}
