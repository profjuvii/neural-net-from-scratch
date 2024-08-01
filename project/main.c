#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "braincraft.h"
#include "data_loader.h"
#include "utils.h"

int main() {
    srand((unsigned int)time(NULL));

    int input_size = 28 * 28;
    int num_classes = 10;

    int num_layers = 3;

    float learning_rate = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;

    NeuralNetwork *nn = create_network(num_layers, ADAM, CROSS_ENTROPY, 0.0, beta1, beta2, NONE, 0.0);

    init_layer(&nn->layers[0], input_size, 128, RELU, 0.0);
    init_layer(&nn->layers[1], 128, 64, RELU, 0.0);
    init_layer(&nn->layers[2], 64, num_classes, SOFTMAX, 0.0);

    int batch_size = 64;
    char *dataset_path = "/Users/profjuvi/Datasets/MNIST/";

    DataLoader *data_loader = create_data_loader(batch_size, input_size, dataset_path);

    int num_epochs = 100;

    // Training loop
    for (int i = 0; i <= num_epochs; ++i) {
        if (load_data(data_loader, num_classes) == -1) {
            fprintf(stderr, "Error: Failed to load data.\n");
            break;
        }

        float total_loss = 0.0;

        for (int j = 0; j < batch_size; ++j) {
            float *features = data_loader->vectors[j];
            float *targets = create_targets(num_classes, data_loader->labels[j]);

            forward_pass(nn, features);
            compute_gradients(nn, features, targets);

            double loss = loss_func(nn->loss_func, nn->predicts, targets, num_classes);
            total_loss += loss;

            if (i % 10 == 0) {
                printf("  Image: %d\n  Loss: %f\n", j + 1, loss);
                print_vector("    Predicts: ", nn->predicts, num_classes, 8, 2);
                print_vector("    Targets:  ", targets, num_classes, 8, 2);
                printf("\n");
            }

            free(targets);
        }

        update_weights(nn, learning_rate);
        init_zero_gradients(nn);

        if (i % 10 == 0) {
            printf("Iteration: %d\n", i);
            printf("Total loss: %f\n\n", total_loss / batch_size);
        }
    }

    load_data(data_loader, num_classes);
    int correct_predicts = 0;

    // Testing loop
    for (int j = 0; j < batch_size; ++j) {
        int label = data_loader->labels[j];
        float *features = data_loader->vectors[j];

        forward_pass(nn, features);

        int max_index = find_max_index(nn->predicts, num_classes);
        correct_predicts += label == max_index;

        printf("Number %d: %s\n", label, label == max_index ? "True" : "False");
    }

    printf("Accuracy: %.1f%%\n", (float)correct_predicts / batch_size * 100);

    if (save_network(nn, "/Users/profjuvi/neural-net-from-scratch/models/", "model_v1") == 0) {
        printf("The model was successfully saved.\n");
    }

    destroy_data_loader(data_loader);
    destroy_network(nn);

    return 0;
}
