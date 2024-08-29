#include <stdio.h>
#include <stdlib.h>
#include <synapse.h>
#include <time.h>

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));

    // Image preparation
    char *image_path;
    if (argc == 2) {
        image_path = argv[1];
    } else {
        fprintf(stderr, "Error: The image path must be specified.\n");
        return 1;
    }

    int input_size = 28 * 28;
    int num_classes = 10;

    float *features = (float *)malloc(input_size * sizeof(float));
    image2vector(image_path, features);
    normalize_vector(features, input_size, 0.5f, 0.5f);

    // Loading a pre-trained model
    char *model_path = "../models/mnist_classifier.bin";

    load_network(model_path);

    // Recognition
    forward(features);
    float *predicts = get_network_predictions();
    int number = index_of_max(predicts, num_classes);

    // Printing text to the console
    char *phrases[] = {
        "I believe this number is",
        "I think this number is",
        "It looks like this number is",
        "In my opinion, this number is",
        "I would say this number is",
        "My guess is that this number is",
        "I would estimate that this number is",
        "From what I see, this number appears to be",
        "I'd say this number is",
        "Based on what I see, this number seems to be"
    };

    int idx = rand() % 10;
    printf("%s %d.\n", phrases[idx], number);

    // Free memory
    destroy_network();

    return 0;
}
