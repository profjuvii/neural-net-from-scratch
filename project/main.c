#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "braincraft.h"
#include "img_vec.h"
#include "test_networks.h"

#define NUM_TRAINING_IMAGES 10
#define NUM_TESTING_IMAGES 10
#define INPUT_SIZE 784
#define NUM_CLASSES 10

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
    srand((unsigned int)time(NULL));

    // test_single_layer_perceptron();
    // test_one_hidden_layer_network();
    // test_two_hidden_layers_network();
    // test_regression_one_hidden_layer();
    // test_binary_classification();
    // test_classification_tanh();
    // test_classification_with_optimizers();
    // test_training_with_different_optimizers();
    // test_training_with_different_loss_functions();
    // test_training_with_different_activation_functions();
    // test_training_with_different_parameters();

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
