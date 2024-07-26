#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "braincraft.h"
#include "test_networks.h"

void generate_random_data(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
}

void print_data(float *data, int size) {
    for (int i = 0; i < size; i++) {
        printf("%f ", data[i]);
    }
    printf("\n\n");
}

void test_single_layer_perceptron() {
    printf("Testing Single Layer Perceptron\n");

    NeuralNetwork *nn = create_network(2, SGD, CROSS_ENTROPY, 0.01, 0.9, 0.9, 0.999);

    init_layer(&nn->layers[0], 2, 2, RELU, 0.01);
    init_layer(&nn->layers[1], 2, 2, SOFTMAX, 0.01);

    float inputs[2] = {0.5, 0.8};
    float targets[2] = {0.0, 1.0};

    forward_pass(nn, inputs);
    print_data(nn->predictions, 2);

    backward_pass(nn, targets);

    destroy_network(nn);
}

void test_one_hidden_layer_network() {
    printf("Testing One Hidden Layer Network\n");

    NeuralNetwork *nn = create_network(3, SGD, CROSS_ENTROPY, 0.01, 0.9, 0.9, 0.999);

    init_layer(&nn->layers[0], 2, 4, RELU, 0.01);
    init_layer(&nn->layers[1], 4, 3, RELU, 0.01);
    init_layer(&nn->layers[2], 3, 2, SOFTMAX, 0.01);

    float inputs[2] = {0.5, 0.8};
    float targets[2] = {0.0, 1.0};

    forward_pass(nn, inputs);
    print_data(nn->predictions, 2);

    backward_pass(nn, targets);

    destroy_network(nn);
}

void test_two_hidden_layers_network() {
    printf("Testing Two Hidden Layers Network\n");

    NeuralNetwork *nn = create_network(4, SGD, CROSS_ENTROPY, 0.01, 0.9, 0.9, 0.999);

    init_layer(&nn->layers[0], 2, 5, RELU, 0.01);
    init_layer(&nn->layers[1], 5, 5, RELU, 0.01);
    init_layer(&nn->layers[2], 5, 3, RELU, 0.01);
    init_layer(&nn->layers[3], 3, 2, SOFTMAX, 0.01);

    float inputs[2] = {0.5, 0.8};
    float targets[2] = {0.0, 1.0};

    forward_pass(nn, inputs);
    print_data(nn->predictions, 2);

    backward_pass(nn, targets);

    destroy_network(nn);
}

void test_regression_one_hidden_layer() {
    printf("Testing Regression with One Hidden Layer\n");

    NeuralNetwork *nn = create_network(2, SGD, MSE, 0.01, 0.9, 0.9, 0.999);

    init_layer(&nn->layers[0], 2, 4, RELU, 0.01);
    init_layer(&nn->layers[1], 4, 1, LINEAR, 0.01);

    float inputs[2] = {0.5, 0.8};
    float targets[1] = {1.0};

    forward_pass(nn, inputs);
    print_data(nn->predictions, 1);

    backward_pass(nn, targets);

    destroy_network(nn);
}

void test_binary_classification() {
    printf("Testing Binary Classification with Sigmoid Output\n");

    NeuralNetwork *nn = create_network(2, SGD, CROSS_ENTROPY, 0.01, 0.9, 0.9, 0.999);

    init_layer(&nn->layers[0], 2, 4, RELU, 0.01);
    init_layer(&nn->layers[1], 4, 1, SIGMOID, 0.01);

    float inputs[2] = {0.5, 0.8};
    float targets[1] = {1.0};

    forward_pass(nn, inputs);
    print_data(nn->predictions, 1);

    backward_pass(nn, targets);

    destroy_network(nn);
}

void test_classification_tanh() {
    printf("Testing Classification with Tanh Hidden Layers\n");

    NeuralNetwork *nn = create_network(3, SGD, CROSS_ENTROPY, 0.01, 0.9, 0.9, 0.999);

    init_layer(&nn->layers[0], 2, 5, TANH, 0.01);
    init_layer(&nn->layers[1], 5, 5, TANH, 0.01);
    init_layer(&nn->layers[2], 5, 2, SOFTMAX, 0.01);

    float inputs[2] = {0.5, 0.8};
    float targets[2] = {0.0, 1.0};

    forward_pass(nn, inputs);
    print_data(nn->predictions, 2);

    backward_pass(nn, targets);

    destroy_network(nn);
}

void test_classification_with_optimizers() {
    printf("Testing Classification with Different Optimizers\n");

    printf("Using SGD Optimizer:\n");
    NeuralNetwork *nn_sgd = create_network(2, SGD, CROSS_ENTROPY, 0.01, 0.9, 0.9, 0.999);
    init_layer(&nn_sgd->layers[0], 2, 4, RELU, 0.01);
    init_layer(&nn_sgd->layers[1], 4, 2, SOFTMAX, 0.01);
    float inputs[2] = {0.5, 0.8};
    float targets[2] = {0.0, 1.0};
    forward_pass(nn_sgd, inputs);
    print_data(nn_sgd->predictions, 2);
    backward_pass(nn_sgd, targets);
    destroy_network(nn_sgd);

    printf("Using Adam Optimizer:\n");
    NeuralNetwork *nn_adam = create_network(2, ADAM, CROSS_ENTROPY, 0.01, 0.9, 0.9, 0.999);
    init_layer(&nn_adam->layers[0], 2, 4, RELU, 0.01);
    init_layer(&nn_adam->layers[1], 4, 2, SOFTMAX, 0.01);
    forward_pass(nn_adam, inputs);
    print_data(nn_adam->predictions, 2);
    backward_pass(nn_adam, targets);
    destroy_network(nn_adam);
}

void train_network(NeuralNetwork *nn, float *inputs, float *targets, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        forward_pass(nn, inputs);
        backward_pass(nn, targets);
        if (epoch % 100 == 0) {
            float loss = mse(nn->predictions, targets, 1);
            printf("Epoch %d, Loss: %f\n", epoch, loss);
        }
    }
    printf("\n");
}

void test_training_with_different_optimizers() {
    printf("Testing Training with Different Optimizers\n");

    float inputs[2] = {0.5, 0.8};
    float targets[2] = {0.0, 1.0};
    int epochs = 1000;

    printf("Using SGD Optimizer:\n");
    NeuralNetwork *nn_sgd = create_network(2, SGD, CROSS_ENTROPY, 0.01, 0.9, 0.9, 0.999);
    init_layer(&nn_sgd->layers[0], 2, 4, RELU, 0.01);
    init_layer(&nn_sgd->layers[1], 4, 2, SOFTMAX, 0.01);
    train_network(nn_sgd, inputs, targets, epochs);
    destroy_network(nn_sgd);

    printf("Using Adam Optimizer:\n");
    NeuralNetwork *nn_adam = create_network(2, ADAM, CROSS_ENTROPY, 0.01, 0.9, 0.9, 0.999);
    init_layer(&nn_adam->layers[0], 2, 4, RELU, 0.01);
    init_layer(&nn_adam->layers[1], 4, 2, SOFTMAX, 0.01);
    train_network(nn_adam, inputs, targets, epochs);
    destroy_network(nn_adam);
}

void test_training_with_different_loss_functions() {
    printf("Testing Training with Different Loss Functions\n");

    float inputs[2] = {0.5, 0.8};
    float targets[2] = {0.0, 1.0};
    int epochs = 1000;

    printf("Using Cross Entropy Loss:\n");
    NeuralNetwork *nn_ce = create_network(2, SGD, CROSS_ENTROPY, 0.01, 0.9, 0.9, 0.999);
    init_layer(&nn_ce->layers[0], 2, 4, RELU, 0.01);
    init_layer(&nn_ce->layers[1], 4, 2, SOFTMAX, 0.01);
    train_network(nn_ce, inputs, targets, epochs);
    destroy_network(nn_ce);

    printf("Using MSE Loss:\n");
    NeuralNetwork *nn_mse = create_network(2, SGD, MSE, 0.01, 0.9, 0.9, 0.999);
    init_layer(&nn_mse->layers[0], 2, 4, RELU, 0.01);
    init_layer(&nn_mse->layers[1], 4, 1, LINEAR, 0.01);
    train_network(nn_mse, inputs, targets, epochs);
    destroy_network(nn_mse);
}

void test_training_with_different_activation_functions() {
    printf("Testing Training with Different Activation Functions\n");

    float inputs[2] = {0.5, 0.8};
    float targets[2] = {0.0, 1.0};
    int epochs = 1000;

    printf("Using ReLU Activation Function:\n");
    NeuralNetwork *nn_relu = create_network(2, SGD, CROSS_ENTROPY, 0.01, 0.9, 0.9, 0.999);
    init_layer(&nn_relu->layers[0], 2, 4, RELU, 0.01);
    init_layer(&nn_relu->layers[1], 4, 2, SOFTMAX, 0.01);
    train_network(nn_relu, inputs, targets, epochs);
    destroy_network(nn_relu);

    printf("Using Tanh Activation Function:\n");
    NeuralNetwork *nn_tanh = create_network(2, SGD, CROSS_ENTROPY, 0.01, 0.9, 0.9, 0.999);
    init_layer(&nn_tanh->layers[0], 2, 4, TANH, 0.01);
    init_layer(&nn_tanh->layers[1], 4, 2, SOFTMAX, 0.01);
    train_network(nn_tanh, inputs, targets, epochs);
    destroy_network(nn_tanh);
}

void test_training_with_different_parameters() {
    printf("Testing Training with Different Parameters\n");

    float inputs[2] = {0.5, 0.8};
    float targets[2] = {0.0, 1.0};
    int epochs = 1000;

    printf("Using Adam Optimizer with Different Parameters:\n");
    NeuralNetwork *nn_adam_params = create_network(2, ADAM, CROSS_ENTROPY, 0.001, 0.8, 0.9, 0.999);
    init_layer(&nn_adam_params->layers[0], 2, 4, RELU, 0.01);
    init_layer(&nn_adam_params->layers[1], 4, 2, SOFTMAX, 0.01);
    train_network(nn_adam_params, inputs, targets, epochs);
    destroy_network(nn_adam_params);

    printf("Using SGD Optimizer with Different Parameters:\n");
    NeuralNetwork *nn_sgd_params = create_network(2, SGD, CROSS_ENTROPY, 0.1, 0.5, 0.9, 0.999);
    init_layer(&nn_sgd_params->layers[0], 2, 4, RELU, 0.01);
    init_layer(&nn_sgd_params->layers[1], 4, 2, SOFTMAX, 0.01);
    train_network(nn_sgd_params, inputs, targets, epochs);
    destroy_network(nn_sgd_params);
}
