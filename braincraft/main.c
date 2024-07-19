#include <stdio.h>
// #include <time.h>
#include "braincraft.h"


int main(){
    // srand(time(NULL));

    int num_layers = 3;
    int input_size = 4;

    Layer* network = (Layer*)calloc(num_layers, sizeof(Layer));

    init_layer(&network[0], 3, input_size);
    init_layer(&network[1], 2, network[0].num_neurons);
    init_layer(&network[2], 1, network[1].num_neurons);

    return 0;
}
