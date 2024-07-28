#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "braincraft.h"

void print_neural_network(NeuralNetwork *nn) {
    for (int i = 0; i < nn->num_layers; ++i) {
        Layer *layer = &nn->layers[i];
        printf("Layer %d:\n", i + 1);
        for (int j = 0; j < layer->output_size; ++j) {
            Neuron *neuron = &layer->neurons[j];
            printf("  Neuron %d:\n", j + 1);
            printf("    Weights: ");
            for (int k = 0; k < layer->input_size; ++k) {
                printf("%f ", neuron->weights[k]);
            }
            printf("\n");
            printf("    Bias: %f\n", neuron->bias);
            printf("    Output: %f\n\n", layer->outputs[j]);
        }
    }
}

float mse(float *predicts, float *targets, int size) {
    float sum_squared_errors = 0.0;

    for (int i = 0; i < size; ++i) {
        float error = predicts[i] - targets[i];
        sum_squared_errors += error * error;
    }

    return sum_squared_errors / size;
}

float cross_entropy_loss(float *predicts, float *targets, int size) {
    float loss = 0.0;

    for (int i = 0; i < size; ++i) {
        float prediction = fmaxf(predicts[i], 1e-15);
        loss -= targets[i] * logf(prediction);
    }

    return loss;
}

double loss_func(LossFunction loss_func, float *predicts, float *targets, int size) {
    switch (loss_func) {
        case MSE:
            return mse(predicts, targets, size);
        case CROSS_ENTROPY:
            return cross_entropy_loss(predicts, targets, size);
        default:
            return 0.0;
    }
}

int main() {
    srand((unsigned int)time(NULL));

    int num_layers = 3;
    int input_size = 10;
    int output_size = 2;

    float inputs[] = {
        0.946, -0.382, 0.758, -1.135, 0.278,
        -0.957, 0.504, -0.287, 0.639, -0.705
    };

    float targets[] = {
        0.333, 0.56
    };

    float learning_rate = 0.01;
    float momentum = 0.9;
    float beta1 = 0.9;
    float beta2 = 0.999;

    NeuralNetwork *nn = create_network(num_layers, ADAM, MSE, learning_rate, momentum, beta1, beta2);

    init_layer(&nn->layers[0], input_size, 5, SIGMOID, 0.0);
    init_layer(&nn->layers[1], 5, 3, SIGMOID, 0.0);
    init_layer(&nn->layers[2], 3, output_size, SIGMOID, 0.0);

    int num_epochs = 10000;

    for (int i = 0; i <= num_epochs; ++i) {
        forward_pass(nn, inputs);
        backward_pass(nn, inputs, targets);

        if (i % 10 == 0) {
            printf("Iteration: %d\nLoss: %f\n\n", i, loss_func(nn->loss_func, nn->predicts, targets, output_size));
        }
    }

    print_neural_network(nn);

    printf("Targets: ");
    for (int i = 0; i < output_size; ++i) {
        printf("%f ", targets[i]);
    }
    printf("\nPredicts: ");

    for (int i = 0; i < output_size; ++i) {
        printf("%f ", nn->predicts[i]);
    }
    printf("\n");

    destroy_network(nn);

    return 0;
}
