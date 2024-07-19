#include "braincraft.h"


void init_neuron(Neuron* neuron, int input_size){
    neuron->weights = (double*)calloc(input_size, sizeof(double));
    for(int i = 0; i < input_size; ++i){
        neuron->weights[i] = (double)rand() / RAND_MAX;
    }
    neuron->bias = (double)rand() / RAND_MAX;
}


void init_layer(Layer* layer, int num_neurons, int input_size){
    layer->neurons = (Neuron*)calloc(num_neurons, sizeof(Neuron));
    for(int i = 0; i < num_neurons; ++i){
        init_neuron(&layer->neurons[i], input_size);
    }
    layer->num_neurons = num_neurons;
}


double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}


double compute_neuron_output(Neuron* neuron, double* inputs, int input_size){
    double sum = 0.0;
    for(int i = 0; i < input_size; ++i){
        sum += neuron->weights[i] * inputs[i];
    }
    sum += neuron->bias;
    neuron->output = sigmoid(sum);
    return neuron->output;
}


void forward_pass(Layer* network, int num_layers, double* inputs, int input_size){
    for(int i = 0; i < num_layers; ++i){
        for(int j = 0; j < network[i].num_neurons; ++i){
            if(i == 0){
                compute_neuron_output(&network[i].neurons[j], inputs, input_size);
            } else{   
                compute_neuron_output(&network[i].neurons[j], inputs, network[i - 1].num_neurons);
            }
        }
    }
}


double mean_squared_error(double* targets, double* outputs, int size){
    double sum = 0.0;
    for(int i = 0; i < size; ++i){
        double error = targets[i] - outputs[i];
        sum += error * error;
    }
    return sum / size;
}


// FIXME: more layer
// void backpropagate(Neuron* network, double* inputs, double* targets, int input_size, int num_neurons, double learning_rate){
//     for(int i = 0; i < num_neurons; ++i){
//         double error = targets[i] - network[i].output;
//         for(int j = 0; j < input_size; ++j){
//             network[i].weights[j] += learning_rate * error * network[i].output * (1.0 - network[i].output) * inputs[j];
//         }
//         network[i].bias += learning_rate * error * network[i].output * (1.0 - network[i].output);
//     }
// }
