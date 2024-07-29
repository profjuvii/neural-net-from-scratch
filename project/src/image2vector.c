#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include "image2vector.h"

unsigned char* read_png_file(const char *filename, int *width, int *height) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Failed to open file %s.\n", filename);
        return NULL;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png == NULL) {
        fprintf(stderr, "Error: Failed to create PNG read structure.\n");
        fclose(fp);
        return NULL;
    }

    png_infop info = png_create_info_struct(png);
    if (info == NULL) {
        fprintf(stderr, "Error: Failed to create PNG info structure.\n");
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        return NULL;
    }

    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "Error during PNG initialization.\n");
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return NULL;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    *width = png_get_image_width(png, info);
    *height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);

    png_read_update_info(png, info);

    int rowbytes = png_get_rowbytes(png, info);
    unsigned char *image_data = (unsigned char *)malloc((*height) * rowbytes);
    if (image_data == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for image.\n");
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return NULL;
    }

    png_bytep *row_pointers = (png_bytep *)malloc((*height) * sizeof(png_bytep));
    if (row_pointers == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for row pointers.\n");
        free(image_data);
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return NULL;
    }

    for (int y = 0; y < *height; ++y) {
        row_pointers[y] = image_data + y * rowbytes;
    }

    png_read_image(png, row_pointers);

    free(row_pointers);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);

    return image_data;
}

void convert_to_grayscale(unsigned char *image_data, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 4;
            unsigned char r = image_data[idx];
            unsigned char g = image_data[idx + 1];
            unsigned char b = image_data[idx + 2];
            unsigned char gray = (r + g + b) / 3;
            image_data[idx] = image_data[idx + 1] = image_data[idx + 2] = gray;
        }
    }
}

float* image2vector(const char *filename, int *size) {
    int width, height;
    unsigned char* image_data = read_png_file(filename, &width, &height);
    if (image_data == NULL) {
        return NULL;
    }

    convert_to_grayscale(image_data, width, height);

    int vector_size = width * height;
    *size = vector_size;

    printf("+\n");
    float *features = (float *)malloc(vector_size * sizeof(float));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            features[y * width + x] = (float)image_data[(y * width + x) * 4] / 255;
        }
    }
    printf("-\n");

    free(image_data);
    return features;
}
