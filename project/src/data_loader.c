#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include "data_loader.h"

DataLoader* create_data_loader(int batch_size, int input_size, char *path) {
    DataLoader *data_loader = (DataLoader *)malloc(sizeof(DataLoader));

    data_loader->vectors = (float **)malloc(batch_size * sizeof(float *));
    for (int i = 0; i < batch_size; ++i) {
        data_loader->vectors[i] = (float *)malloc(input_size * sizeof(float));
    }

    data_loader->labels = (int *)malloc(batch_size * sizeof(int));

    data_loader->batch_size = batch_size;
    data_loader->input_size = input_size;
    data_loader->path = path;

    return data_loader;
}

void destroy_data_loader(DataLoader *data_loader) {
    if (data_loader == NULL) return;

    for (int i = 0; i < data_loader->batch_size; ++i) {
        if (data_loader->vectors[i] != NULL) {
            free(data_loader->vectors[i]);
        }
    }
    free(data_loader->vectors);
    free(data_loader->labels);
    free(data_loader);
}

int load_data(DataLoader *data_loader, int num_classes) {
    for (int i = 0; i < data_loader->batch_size; ++i) {
        int label = rand() % num_classes;

        char dir_path[256];
        snprintf(dir_path, sizeof(dir_path), "%s%d", data_loader->path, label);

        DIR *dir = opendir(dir_path);
        if (dir == NULL) {
            fprintf(stderr, "Error: Failed to open directory: %s\n", dir_path);
            exit(EXIT_FAILURE);
        }

        struct dirent *entry;
        char **file_names = NULL;
        int file_count = 0;

        while ((entry = readdir(dir)) != NULL) {
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            file_names = realloc(file_names, (file_count + 1) * sizeof(char *));
            file_names[file_count] = strdup(entry->d_name);
            file_count++;
        }
        closedir(dir);

        int file_index = rand() % file_count;
        char *file_name = file_names[file_index];

        char file_path[512];
        snprintf(file_path, sizeof(file_path), "%s/%s", dir_path, file_name);

        int size = 0;
        image2vector(file_path, data_loader->vectors[i], &size);
        data_loader->labels[i] = label;

        if (data_loader->input_size != size) return -1;

        free(file_names[file_index]);
        free(file_names);
    }

    return 0;
}

float* create_targets(int num_classes, int label) {
    float *targets = (float *)calloc(num_classes, sizeof(float));
    targets[label] = 1.0;
    return targets;
}