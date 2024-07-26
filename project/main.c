#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "braincraft.h"
#include "img_vec.h"

#define NUM_TRAINING_IMAGES 10
#define NUM_TESTING_IMAGES 10
#define INPUT_SIZE 784
#define NUM_CLASSES 10

void print_neural_network(const NeuralNetwork *nn) {
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
        const Layer *layer = &nn->layers[i];
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
            const Neuron *neuron = &layer->neurons[j];
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

void load_data(const char *directory, float inputs[][INPUT_SIZE], float targets[][NUM_CLASSES], int num_images) {
    for (int i = 0; i < num_images; ++i) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/%d.png", directory, i);

        FILE *file = fopen(filename, "rb");
        if (!file) {
            perror("Failed to open file");
            exit(EXIT_FAILURE);
        }

        size_t size;
        float *vector = png_to_vector(file, &size);
        fclose(file);

        if (size != INPUT_SIZE * sizeof(float)) {
            fprintf(stderr, "Unexpected vector size\n");
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < INPUT_SIZE; ++j) {
            inputs[i][j] = vector[j];
        }
        free(vector);

        for (int k = 0; k < NUM_CLASSES; ++k) {
            targets[i][k] = (k == i) ? 1.0f : 0.0f;
        }
    }
}

int main() {
    srand(time(NULL));

    float inputs_train[NUM_TRAINING_IMAGES][INPUT_SIZE];
    float targets_train[NUM_TRAINING_IMAGES][NUM_CLASSES];
    float inputs_test[NUM_TESTING_IMAGES][INPUT_SIZE];
    float targets_test[NUM_TESTING_IMAGES][NUM_CLASSES];

    load_data("data/train", inputs_train, targets_train, NUM_TRAINING_IMAGES);
    load_data("data/test", inputs_test, targets_test, NUM_TESTING_IMAGES);

    const int num_layers = 3;
    const float learning_rate = 0.01;
    const float momentum = 0.9;
    const float beta1 = 0.9, beta2 = 0.999;

    NeuralNetwork *nn = create_network(num_layers, SGD, CROSS_ENTROPY, learning_rate, momentum, beta1, beta2);

    init_layer(&nn->layers[0], INPUT_SIZE, 128, RELU, 0.0);
    init_layer(&nn->layers[1], 128, 64, RELU, 0.0);
    init_layer(&nn->layers[2], 64, NUM_CLASSES, SOFTMAX, 0.0);

    const int num_epochs = 1000;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (int i = 0; i < NUM_TRAINING_IMAGES; ++i) {
            forward_pass(nn, inputs_train[i]);
            backward_pass(nn, targets_train[i]);

            if (epoch % 100 == 0) {
                float loss = mse(nn->predictions, targets_train[i], NUM_CLASSES);
                printf("Epoch: %d, Image: %d, Loss: %f\n", epoch, i, loss);
            }
        }
        if (epoch % 100 == 0) {
            printf("\n");
        }
    }

    float **predicts = (float **)malloc(NUM_TESTING_IMAGES * sizeof(float *));
    if (predicts == NULL) {
        perror("Failed to allocate memory for predictions");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < NUM_TESTING_IMAGES; ++i) {
        predicts[i] = (float *)malloc(NUM_CLASSES * sizeof(float));
        if (predicts[i] == NULL) {
            perror("Failed to allocate memory for a prediction");
            exit(EXIT_FAILURE);
        }
        forward_pass(nn, inputs_test[i]);
        for (int j = 0; j < NUM_CLASSES; ++j) {
            predicts[i][j] = nn->predictions[j];
        }
    }

    for (int i = 0; i < NUM_TESTING_IMAGES; ++i) {
        printf("%d.png\n\tPredictions:\t", i);
        for (int j = 0; j < NUM_CLASSES; ++j) {
            printf("%f ", predicts[i][j]);
        }
        printf("\n\tTargets:\t");
        for (int j = 0; j < NUM_CLASSES; ++j) {
            printf("%f ", targets_test[i][j]);
        }
        printf("\n\n");
    }

    destroy_network(nn);
    for (int i = 0; i < NUM_TESTING_IMAGES; ++i) {
        free(predicts[i]);
    }
    free(predicts);

    return 0;
}
