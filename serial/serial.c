#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GRAY_LEVELS 256

void calculate_histogram(unsigned char *image, int size, int *histogram) {
    for (int i = 0; i < size; i++) {
        histogram[image[i]]++;
    }
}

int otsu_threshold(int *histogram, int total_pixels) {
    int sum = 0, sumB = 0, q1 = 0, q2 = 0;
    float max_var = 0.0, threshold = 0.0;

    for (int i = 0; i < GRAY_LEVELS; i++) {
        sum += i * histogram[i];
    }

    for (int i = 0; i < GRAY_LEVELS; i++) {
        q1 += histogram[i];
        if (q1 == 0) continue;

        q2 = total_pixels - q1;
        if (q2 == 0) break;

        sumB += i * histogram[i];
        float m1 = (float)sumB / q1;
        float m2 = (float)(sum - sumB) / q2;
        float var_between = (float)q1 * q2 * (m1 - m2) * (m1 - m2);

        if (var_between > max_var) {
            max_var = var_between;
            threshold = i;
        }
    }

    return (int)threshold;
}

void apply_threshold(unsigned char *image, int size, int threshold) {
    for (int i = 0; i < size; i++) {
        image[i] = (image[i] > threshold) ? 255 : 0;
    }
}

void read_rawimage(char *fname, unsigned long height, unsigned long width, unsigned char **image) {
    FILE *file = fopen(fname, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", fname);
        exit(EXIT_FAILURE);
    }
    for (unsigned long i = 0; i < height; i++) {
        fread(image[i], 1, width, file);
    }
    fclose(file);
}

void write_rawimage(char *fname, unsigned long height, unsigned long width, unsigned char **image) {
    FILE *file = fopen(fname, "w");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", fname);
        exit(EXIT_FAILURE);
    }
    for (unsigned long i = 0; i < height; i++) {
        fwrite(image[i], 1, width, file);
    }
    fclose(file);
}

int main(int argc, char *argv[]) {
    clock_t start_time, end_time;
    double total_time;

    start_time = clock(); // Έναρξη μέτρησης συνολικού χρόνου

    if (argc < 4) {
        printf("Usage: %s input_image height width output_image\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char *input_file = argv[1];
    unsigned long height = atoi(argv[2]);
    unsigned long width = atoi(argv[3]);
    char *output_file = argv[4];

    unsigned long total_pixels = height * width;
    unsigned char **image = (unsigned char **)malloc(height * sizeof(unsigned char *));
    for (unsigned long i = 0; i < height; i++) {
        image[i] = (unsigned char *)malloc(width * sizeof(unsigned char));
    }

    read_rawimage(input_file, height, width, image);

    unsigned char *flat_image = (unsigned char *)malloc(total_pixels * sizeof(unsigned char));
    for (unsigned long i = 0; i < height; i++) {
        memcpy(flat_image + i * width, image[i], width);
    }

    int *histogram = (int *)calloc(GRAY_LEVELS, sizeof(int));
    calculate_histogram(flat_image, total_pixels, histogram);

    int threshold = otsu_threshold(histogram, total_pixels);

    apply_threshold(flat_image, total_pixels, threshold);

    for (unsigned long i = 0; i < height; i++) {
        memcpy(image[i], flat_image + i * width, width);
    }

    write_rawimage(output_file, height, width, image);

    for (unsigned long i = 0; i < height; i++) {
        free(image[i]);
    }
    free(image);
    free(flat_image);
    free(histogram);

    end_time = clock(); // Τέλος μέτρησης συνολικού χρόνου
    total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Total Execution Time: %.6f seconds\n", total_time);

    return 0;
}

