#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include "data_loader.h"

DataLoader* create_data_loader(int batch_size) {
    DataLoader *data_loader = (DataLoader *)malloc(sizeof(DataLoader));
    data_loader->vectors = (float **)malloc(batch_size * sizeof(float *));

    data_loader->labels = (int *)malloc(batch_size * sizeof(int));
    data_loader->size = batch_size;
    return data_loader;
}

void clear_data_loader(DataLoader *data_loader) {
    for (int i = 0; i < data_loader->size; ++i) {
        free(data_loader->vectors[i]);
    }
}

void destroy_data_loader(DataLoader *data_loader) {
    free(data_loader->vectors);
    free(data_loader->labels);
    free(data_loader);
}

void load(DataLoader *data_loader, int num_classes) {
    for (int i = 0; i < data_loader->size; ++i) {
        int label = rand() % num_classes;

        char dir_path[256];
        snprintf(dir_path, sizeof(dir_path), "%s%d", PATH, label);

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

        printf("%s\n", file_path);

        int size = 0;
        float* features = image2vector(file_path, &size);

        // copy_vector(data_loader->vectors[i], features, size);
        data_loader->labels[i] = label;

        free(features);
        free(file_names[file_index]);
        free(file_names);
    }
}
