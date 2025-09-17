#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GRAY_LEVELS 256

// Συνάρτηση για υπολογισμό ιστογράμματος
void calculate_histogram(unsigned char *image, int size, int *histogram) {
    for (int i = 0; i < size; i++) {
        histogram[image[i]]++;
    }
}

// Συνάρτηση για τον υπολογισμό του κατωφλίου μέσω Otsu
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

// Συνάρτηση για την εφαρμογή κατωφλίου
void apply_threshold(unsigned char *image, int size, int threshold) {
    for (int i = 0; i < size; i++) {
        image[i] = (image[i] > threshold) ? 255 : 0;
    }
}

// Συνάρτηση για ανάγνωση εικόνας
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

// Συνάρτηση για αποθήκευση εικόνας
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
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    unsigned char **image = NULL;
    unsigned long height, width, total_pixels;
    unsigned char *local_image = NULL;
    int *global_histogram = NULL, *local_histogram = NULL;

    if (argc < 4) {
        if (rank == 0)
            printf("Usage: %s input_image height width output_image\n", argv[0]);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    char *input_file = argv[1];
    height = atoi(argv[2]);
    width = atoi(argv[3]);
    char *output_file = argv[4];
    total_pixels = height * width;

    // Ανάγνωση εικόνας στον κύριο κόμβο
    if (rank == 0) {
        image = (unsigned char **)malloc(height * sizeof(unsigned char *));
        for (unsigned long i = 0; i < height; i++) {
            image[i] = (unsigned char *)malloc(width * sizeof(unsigned char));
        }
        read_rawimage(input_file, height, width, image);
    }

    int local_size = total_pixels / size;
    local_image = (unsigned char *)malloc(local_size * sizeof(unsigned char));

    // Διασπορά εικόνας στους κόμβους
    if (rank == 0) {
        unsigned char *flat_image = (unsigned char *)malloc(total_pixels * sizeof(unsigned char));
        for (unsigned long i = 0; i < height; i++) {
            memcpy(flat_image + i * width, image[i], width);
        }
        MPI_Scatter(flat_image, local_size, MPI_UNSIGNED_CHAR, local_image, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        free(flat_image);
    } else {
        MPI_Scatter(NULL, local_size, MPI_UNSIGNED_CHAR, local_image, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }

    // Υπολογισμός τοπικού ιστογράμματος
    local_histogram = (int *)calloc(GRAY_LEVELS, sizeof(int));
    calculate_histogram(local_image, local_size, local_histogram);

    // Συγχώνευση ιστογραμμάτων
    if (rank == 0) global_histogram = (int *)calloc(GRAY_LEVELS, sizeof(int));
    MPI_Reduce(local_histogram, global_histogram, GRAY_LEVELS, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int threshold = 0;
    if (rank == 0) {
        threshold = otsu_threshold(global_histogram, total_pixels);
        printf("Calculated Otsu threshold: %d\n", threshold);
    }

    // Μετάδοση του κατωφλίου στους κόμβους
    MPI_Bcast(&threshold, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Εφαρμογή κατωφλίου
    apply_threshold(local_image, local_size, threshold);

    // Συγχώνευση εικόνας
    if (rank == 0) {
        unsigned char *flat_image = (unsigned char *)malloc(total_pixels * sizeof(unsigned char));
        MPI_Gather(local_image, local_size, MPI_UNSIGNED_CHAR, flat_image, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        for (unsigned long i = 0; i < height; i++) {
            memcpy(image[i], flat_image + i * width, width);
        }
        free(flat_image);

        // Αποθήκευση εικόνας
        write_rawimage(output_file, height, width, image);
        for (unsigned long i = 0; i < height; i++) free(image[i]);
        free(image);
        free(global_histogram);
    } else {
        MPI_Gather(local_image, local_size, MPI_UNSIGNED_CHAR, NULL, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }

    free(local_image);
    free(local_histogram);

    MPI_Finalize();
    return 0;
}

