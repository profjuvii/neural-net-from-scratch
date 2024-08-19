#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "braincraft.h"
#include "dataloader.h"
#include "utils.h"

#define INPUT_SIZE 784
#define NUM_CLASSES 10

int main() {
    srand((unsigned int)time(NULL));

    int num_layers = 3;

    NeuralNetwork *nn = create_network(num_layers, ADAM, CROSS_ENTROPY, 0.0, 0.9, 0.999, NONE, 0.0);

    init_layer(&nn->layers[0], INPUT_SIZE, 128, RELU, 0.0);
    init_layer(&nn->layers[1], 128, 64, RELU, 0.0);
    init_layer(&nn->layers[2], 64, NUM_CLASSES, SOFTMAX, 0.0);

    int batch_size = 64;
    char *train_path = "/Users/profjuvi/Datasets/mnist_png/train";
    char *test_path = "/Users/profjuvi/Datasets/mnist_png/test";

    DataLoader *dataloader_train = create_dataloader(train_path, batch_size, INPUT_SIZE);
    DataLoader *dataloader_test = create_dataloader(test_path, batch_size, INPUT_SIZE);

    float learning_rate = 0.002;
    int num_epochs = 200;

    for (int i = 0; i <= num_epochs; ++i) {
        float total_loss = 0.0;
        get_next_batch(dataloader_train, NUM_CLASSES);

        for (int j = 0; j < batch_size; ++j) {
            float *features = dataloader_train->batch[j].features;
            float *targets = create_targets(NUM_CLASSES, dataloader_train->batch[j].label);

            forward_pass(nn, features);
            compute_gradients(nn, features, targets);

            double loss = loss_func(nn->loss_func, nn->predicts, targets, NUM_CLASSES);
            total_loss += loss;

            free(targets);
        }

        update_weights(nn, learning_rate);
        init_zero_gradients(nn);

        if (i % 10 == 0) {
            printf("Epoch: %d\n", i);
            printf("Total loss: %.9f\n\n", total_loss / batch_size);
        }
    }

    int correct_predicts = 0;
    get_next_batch(dataloader_test, NUM_CLASSES);

    for (int j = 0; j < batch_size; ++j) {
        float *features = dataloader_test->batch[j].features;
        int label = dataloader_test->batch[j].label;

        forward_pass(nn, features);

        int max_index = find_max_index(nn->predicts, NUM_CLASSES);
        correct_predicts += label == max_index;

        printf("Number %d: %s\n", label, label == max_index ? "True" : "False");
    }

    printf("Accuracy: %.1f%%\n", (float)correct_predicts / batch_size * 100);

    if (save_network(nn, "/Users/profjuvi/neural-net-from-scratch/models", "mnist_classifier_v2") == 0) {
        printf("The model was successfully saved.\n");
    }

    destroy_dataloader(dataloader_train);
    destroy_dataloader(dataloader_test);
    destroy_network(nn);

    return 0;
}
