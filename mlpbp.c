///////////////////////////////////////////////////////////////////////////////
/* 
 * File: mlpbp.c
 * 
 * \@brief A (lazy, bad and unfinished) C implementation of a Multilayer
 * Perceptron with Backpropagation. Note that this is an unfinished code used 
 * for learning purposes only and a lot of errors may exist. Then, be careful 
 * using this ~crap~ code.
 * 
 * \@author Guilherme Oliveira Chagas (guilherme.o.chagas[a]gmail.com)
 * \@version 0.5
 * \@date This code was created on November 16, 2016, 09:30 AM
 * \@warning Apologizes about my poor english xD
 * \@copyright GNU Public License
 *
 * References:
 *
 */
///////////////////////////////////////////////////////////////////////////////

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

////////////////////////////////// Macros /////////////////////////////////////

#define MAX_NO_OF_ATTR 768 // max number of instance attributes
#define MAX_NO_OF_INST 60000
#define MAX_DOUBLE 100000000000.0

//////////////////////////////// Structures ///////////////////////////////////

// neuron struct
struct neuron
{
    int no_of_inputs; // bias + no. of neurons of previous layer
    double delta;
    double sigma;
    double *inputs; // inputs values
    double *out_weights;
    double *weights;
    double output;
};

// layer struct
struct layer
{
    int no_of_neurons;
    struct neuron *neurons;
};

// neural net struct
struct neural_net
{
    int no_of_neurons;
    int no_of_layers;
    struct layer *layers;
};

///////////////////////////// Global variables ////////////////////////////////

const static char *input_file = "../datasets/mnist/train-images.idx3-ubyte";
const static char *output_file = "./output_train.csv";
const static char *output_file2 = "./output_results.csv";
const static int no_of_lyrs = 2;
const static int layer_neurons[] = {250,784};
const static int max_epoch = 1000;
const static double perc_training = 0.80;
const static double eta = 0.05;

static int test_hits; // number of correct classifications
static int no_of_attr; // number of attributes
static int no_of_inst; // number of instances
static int no_of_class; // number of classes
static int test_set_size;
static int training_set_size;

static double test_inputs[MAX_NO_OF_INST][MAX_NO_OF_ATTR];
static double training_inputs[MAX_NO_OF_INST][MAX_NO_OF_ATTR];

static struct neural_net mlp_net; // multilayer perceptron neural network

///////////////////////////////// Prototypes //////////////////////////////////

// TODO

///////////////////////////////// Functions ///////////////////////////////////


///////////////////////////////////////////////////
/*
 * \@function rand_in_range()
 * \@brief TODO
 * \@param 
 */
///////////////////////////////////////////////////
double rand_in_range(const double min, const double max)
{
    return ((double)(rand() % 10000) / 10000.0) * (max - min) + min;
}


///////////////////////////////////////////////////
/*
 * \@function print_input_matrices()
 * \@brief TODO
 * \@param FILE: 
 * \@param double*: 
 */
///////////////////////////////////////////////////
void print_input_matrices()
{
    int i,j;

    printf("Training set size = %d \n", training_set_size);
    printf("Test set size = %d \n", test_set_size);

    printf("Training input: \n");
    for (i = 0; i < training_set_size; ++i)
    {
        printf("\t");
        for (j = 0; j < no_of_attr; ++j)
            printf(" %lf", training_inputs[i][j]);
        printf(" class =  %lf\n", training_inputs[i][j]);
    }

    printf("\nTest input: \n");
    for (i = 0; i < test_set_size; ++i)
    {
        printf("\t");
        for (j = 0; j < no_of_attr; ++j)
            printf(" %lf", test_inputs[i][j]);
        printf(" class =  %lf\n", test_inputs[i][j]);
    }


}


///////////////////////////////////////////////////
/*
 * \@function print_neural_net()
 * \@brief TODO
 * \@param FILE: 
 * \@param double*: 
 */
///////////////////////////////////////////////////
void print_neural_net()
{
    int i, j;
    struct neuron *nr;
    struct layer *lyr;

    for (i = 0; i <= mlp_net.no_of_layers; ++i) // input + others layers
    {
        lyr = &mlp_net.layers[i];

        for (j = 0; j < lyr->no_of_neurons; ++j) // 0 is bias
        {
            nr = &lyr->neurons[j];

            printf("N%d%d: %1.5f out\t", i, j, nr->output);
        }
        printf("\n");
    }
}


///////////////////////////////////////////////////
/*
 * \@function load_input_line()
 * \@brief TODO
 * \@param FILE: 
 * \@param double*: 
 */
///////////////////////////////////////////////////
void load_input_line(FILE *f, double *input_line)
{
    int i,j;
    int tmp;

    for (i = 0; i < no_of_attr; ++i)
    {
        if (fscanf(f, "%lf,", &input_line[i]) != 1)
        {
            fprintf(stderr,"ERROR - missing instance info! \n");
            exit(EXIT_FAILURE);
        }


    }

    if (fscanf(f, "%d\n", &tmp) != 1) // class
    {
        fprintf(stderr,"ERROR - missing instance info! \n");
        exit(EXIT_FAILURE);
    }

    // 3 classes 
    switch (tmp) // TODO: find a better solution for this
    {
    case 0:
        input_line[i] = 0.0;
        input_line[i+1] = 0.0;
        input_line[i+2] = 1.0;
        break;
    case 1:
        input_line[i] = 0.0;
        input_line[i+1] = 1.0;
        input_line[i+2] = 0.0;
        break;
    case 2:
        input_line[i] = 1.0;
        input_line[i+1] = 0.0;
        input_line[i+2] = 0.0;
        break;
    }
}


///////////////////////////////////////////////////
/*
 * \@function reverse_int()
 * \@brief TODO
 * \@param 
 */
///////////////////////////////////////////////////
int reverse_int(int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


///////////////////////////////////////////////////
/*
 * \@function load_input_idx_line()
 * \@brief TODO
 * \@param 
 */
///////////////////////////////////////////////////
void load_input_idx_line(FILE *f, const int i, const int r, const int c, 
    const int no_of_cols)
{
    unsigned char tmpc;
    int tmp;
    double normalized;

    if (fread((char*)&tmpc, sizeof(tmpc), 1, f) != 1)
    {
        fprintf(stderr, "ERROR - pixel value not found! \n");
        exit(EXIT_FAILURE);
    }

    tmp = (int)tmpc;

    normalized = (tmp - 0.0) / (255.0 - 0.0);

    training_inputs[i][(r * no_of_cols) + c] = normalized;
}


///////////////////////////////////////////////////
/*
 * \@function load_input()
 * \@brief TODO
 * \@param const char*:
 */
///////////////////////////////////////////////////
void load_input_idx(const char *fpath)
{
    int i,r,c;
    int magic_number;
        int no_of_images;
        int no_of_rows;
        int no_of_cols;

    FILE *f = NULL;

    if ((f = fopen(fpath, "rb")) == NULL)
    {
        fprintf(stderr,"ERROR - file \"%s\" was not found! \n", fpath);
        exit(EXIT_FAILURE);
    }

    if (fread(&magic_number, sizeof(magic_number), 1, f) != 1)
    {
        fprintf(stderr, "ERROR - magic number value! \n");
        exit(EXIT_FAILURE);
    }

    if (fread(&no_of_images, sizeof(no_of_images), 1, f) != 1)
    {
        fprintf(stderr, "ERROR - no_of_images value! \n");
        exit(EXIT_FAILURE);
    }
    
    if (fread(&no_of_rows, sizeof(no_of_rows), 1, f) != 1)
    {
        fprintf(stderr, "ERROR - no_of_rows value! \n");
        exit(EXIT_FAILURE);
    }

    if (fread(&no_of_cols, sizeof(no_of_cols), 1, f) != 1)
    {
        fprintf(stderr, "ERROR - no_of_cols value! \n");
        exit(EXIT_FAILURE);
    }

    magic_number = reverse_int(magic_number);
    no_of_images = reverse_int(no_of_images) - 59000;
    no_of_rows = reverse_int(no_of_rows);
    no_of_cols = reverse_int(no_of_cols);

    training_set_size = 0;

    for (i = 0; i < no_of_images; ++i)
    {
        for (r = 0; r < no_of_rows; ++r)
        {
            for (c = 0; c < no_of_cols; ++c)
            {
                load_input_idx_line(f, i, r, c, no_of_cols);
            }
        }

        ++training_set_size;
    }
}

///////////////////////////////////////////////////
/*
 * \@function load_input()
 * \@brief TODO
 * \@param const char*:
 */
///////////////////////////////////////////////////
void load_input(const char *fpath)
{
    int i;
    FILE *f = NULL;

    if ((f = fopen(fpath, "r")) == NULL)
    {
        fprintf(stderr,"ERROR - file \"%s\" was not found! \n", fpath);
        exit(EXIT_FAILURE);
    }

    if (fscanf(f, "%d %d %d\n", &no_of_inst,&no_of_attr,&no_of_class) != 3)
    {
        fprintf(stderr, "ERROR - missing instance information! \n");
        exit(EXIT_FAILURE);
    }

    test_set_size = 0;
    training_set_size = 0;

    for (i = 0; i < no_of_inst; ++i)
    {
        if (rand_in_range(0,1) <= perc_training)
        {
            load_input_line(f, training_inputs[training_set_size]);
            ++training_set_size;
        }
        else
        {
            load_input_line(f, test_inputs[test_set_size]);
            ++test_set_size;
        }
    }

    fclose(f);
}


///////////////////////////////////////////////////
/*
 * \@function print_header_output_file()
 * \@brief TODO
 * \@param
 */
///////////////////////////////////////////////////
void print_header_output_file()
{
    int i;
    FILE *of;

    of = fopen(output_file, "w");

    fprintf(of, "eta, perc_training, no_of_epochs, no_of_layers\n");
    fprintf(of, "%1.2lf, %1.2lf, %d, %d \n", eta, perc_training, 
        max_epoch, no_of_lyrs);

    for(i = 0; i < no_of_lyrs; ++i)
        fprintf(of, "layer-#%d, %d \n", i, layer_neurons[i]);

    fprintf(of, "\n");

    fclose(of);
}


///////////////////////////////////////////////////
/*
 * \@function print_training_data()
 * \@brief TODO
 * \@param
 */
///////////////////////////////////////////////////
void print_training_data(const int epoch, const double data)
{
    FILE *of;

    of = fopen(output_file, "a");

    fprintf(of, "%d, %1.5f\n", epoch, data);

    fclose(of);
}


///////////////////////////////////////////////////
/*
 * \@function print_test_results()
 * \@brief TODO
 * \@param
 */
///////////////////////////////////////////////////
void print_test_result()
{
    FILE *of;

    of = fopen(output_file, "a");

    fprintf(of, "Test results\n");
    fprintf(of, "hits, precision \n");

    fprintf(of, "%d, %1.5f\n", test_hits, 
        ((double)(test_hits * 100) / test_set_size));

    fclose(of);
}


///////////////////////////////////////////////////
/*
 * \@function create_layer()
 * \@brief TODO
 * \@param
 */
///////////////////////////////////////////////////
void create_layer(const int no_of_neurons, struct layer *lyr)
{
    lyr->neurons = malloc(no_of_neurons * sizeof(*lyr->neurons));
    lyr->no_of_neurons = no_of_neurons;
}


///////////////////////////////////////////////////
/*
 * \@function create_input_layer()
 * \@brief TODO
 * \@param
 */
///////////////////////////////////////////////////
void create_input_layer(const int no_of_neurons, struct layer *lyr)
{
    int i;
    struct neuron *nr;

    create_layer(no_of_neurons, lyr);

    // initial setup of input layer neurons
    for (i = 0; i < no_of_neurons; ++i) 
    {
        nr = &lyr->neurons[i];

        nr->inputs = malloc(sizeof(*nr->inputs));
        nr->no_of_inputs = 1;
    }
}


///////////////////////////////////////////////////
/*
 * \@function init_layer_neurons_setup()
 * \@brief TODO
 * \@param
 */
///////////////////////////////////////////////////
void init_layer_neurons_setup(const int layer_index, int no_of_in)
{
    int i,j;
    int temp;
    struct neuron *nr;
    struct layer *next_lyr;
    struct layer *prev_lyr;
    struct layer *lyr = &mlp_net.layers[layer_index];

    ++no_of_in; // no. of neurons of previous layer + bias

    for (i = 0; i < lyr->no_of_neurons; ++i)
    {
        nr = &lyr->neurons[i];

        nr->no_of_inputs = no_of_in;
        nr->inputs = malloc(no_of_in * sizeof(*nr->inputs));
        nr->weights = malloc(no_of_in * sizeof(*nr->weights));
        nr->out_weights = malloc(no_of_in * sizeof(*nr->out_weights));

        // bias
        nr->inputs[0] = 1;

        for (j = 0; j < no_of_in; ++j)
            nr->weights[j] = rand_in_range(0, 0.1);
    }
}


///////////////////////////////////////////////////
/*
 * \@function create_neural_net()
 * \@brief TODO
 * \@param
 */
///////////////////////////////////////////////////
void create_neural_net(const int no_of_layers, const int *layer_neurons)
{
    int i;
    int tmp;

    mlp_net.layers = malloc(no_of_layers * sizeof(*mlp_net.layers) + 1);
    mlp_net.no_of_layers = no_of_layers;

    for (i = 0; i <= no_of_layers; ++i) // input layer + layers
    {
        if (i == 0) // input layer
        {
            create_input_layer(no_of_attr, &mlp_net.layers[i]);
        }
        else
        {
            create_layer(layer_neurons[i-1], &mlp_net.layers[i]);
            
            tmp = mlp_net.layers[i-1].no_of_neurons;
            
            init_layer_neurons_setup(i, tmp);
        }
    }

    // TODO: return?
}


///////////////////////////////////////////////////
/*
 * \@function compute_neuron_sigma()
 * \@brief TODO
 * \@param
 */
///////////////////////////////////////////////////
void compute_neuron_sigma(const int layer_index, struct neuron *nr)
{
    int i;
    double sum = 0.0;
    struct layer *prev_lyr = &mlp_net.layers[layer_index-1]; 

    // printf("\t\t(%1.5f * %1.5f)", nr->weights[0], nr->inputs[0]);

    sum += nr->weights[0] * nr->inputs[0]; // bias

    for (i = 1; i < nr->no_of_inputs; ++i)
    {    
        nr->inputs[i] = prev_lyr->neurons[i-1].output;
        sum += nr->weights[i] * nr->inputs[i];

        // printf(" + (%1.5f * %1.5f)", nr->weights[i], nr->inputs[i]);
    }
    // printf(" = %1.5f\n", sum);

    nr->sigma = sum;
}


///////////////////////////////////////////////////
/*
 * \@function compute_neuron_sigma()
 * \@brief TODO
 * \@param
 */
///////////////////////////////////////////////////
void compute_activation_function(struct neuron *nr)
{
    switch (1) // TODO: implement other options
    {
    case 1:
        {
            nr->output = 1.0 / (1 + exp(-nr->sigma));
            break;
        }
    }
}


///////////////////////////////////////////////////
/*
 * \@function feed_forward()
 * \@brief TODO
 * \@param
 */
///////////////////////////////////////////////////
void feed_forward(double *instance)
{
    int i,j;
    struct neuron *nr;
    struct layer *lyr = &mlp_net.layers[0];

    // input layer
    for (i = 0; i < lyr->no_of_neurons; ++i) 
    {
        nr = &lyr->neurons[i];

        nr->inputs[0] = instance[i];
        nr->output = nr->inputs[0];
    }

    // hidden layers
    for (i = 1; i <= mlp_net.no_of_layers; ++i)
    {
        lyr = &mlp_net.layers[i];

        // printf("Hidden layer %d \n", i);

        for (j = 0; j < lyr->no_of_neurons; ++j)
        {
            // printf("\t Neuron %d \n", j);

            nr = &lyr->neurons[j];

            compute_neuron_sigma(i, nr);

            // printf("\t Sigma %1.5f \n", nr->sigma);

            // TODO: remove for last layer?
            compute_activation_function(nr);

            // printf("\t Output %1.5f \n", nr->output);
            // getchar();
        }
    }
}


///////////////////////////////////////////////////
/*
 * \@function back_propagation()
 * \@brief TODO
 * \@param
 */
///////////////////////////////////////////////////
double derivative(const double sigma)
{
    switch (1)
    {
    case 1:
        return exp(-sigma) / pow(1 + exp(-sigma), 2.0);
    }
}


///////////////////////////////////////////////////
/*
 * \@function update_weights()
 * \@brief TODO
 * \@param
 */
///////////////////////////////////////////////////
void update_weights(struct neuron *nr)
{
    int i;

    // printf("\t\t\t no_of_inputs = %d \n", nr->no_of_inputs);

    for (i = 0; i < nr->no_of_inputs; ++i)
    {
        // printf("\t\t\t\t i = %d \n", i);

        nr->weights[i] += (eta * nr->inputs[i] * nr->delta);
    }
}


///////////////////////////////////////////////////
/*
 * \@function compute_delta()
 * \@brief TODO
 * \@param
 */
///////////////////////////////////////////////////
void compute_delta(const int layer_index, const int neuron_index)
{
    int i;
    double tmp;
    double sum = 0.0;
    struct neuron *nr = &mlp_net.layers[layer_index].neurons[neuron_index];
    struct neuron *next_nr;
    struct layer *next_lyr;

    next_lyr = &mlp_net.layers[layer_index + 1];

    for (i = 0; i < next_lyr->no_of_neurons; ++i) 
    {
        next_nr = &next_lyr->neurons[i];
        tmp = next_nr->weights[neuron_index + 1]; // 0: bias
        sum += (tmp * next_nr->delta);
    }

    nr->delta = sum * derivative(nr->sigma);
}


///////////////////////////////////////////////////
/*
 * \@function back_propagation()
 * \@brief TODO
 * \@param
 */
///////////////////////////////////////////////////
double back_propagation(const int instance_index)
{
    int i,j;
    double t;
    double tmp = 0.0;
    double error = 0.0;
    struct layer *lyr;
    struct neuron *nr;

    lyr = &mlp_net.layers[mlp_net.no_of_layers]; // last layer

    // printf("\tAqui! 2.1 \n");

    for (i = 0; i < no_of_class; ++i) // lyr->no_of_neurons
    {
        nr = &lyr->neurons[i];

        // MUDEI AQUI
        // t = training_inputs[instance_index][no_of_attr + i];

        // MNIST
        t = training_inputs[instance_index][i];

        nr->delta = (t - nr->output) * derivative(nr->sigma);

        tmp = fabs(t - nr->output);

        // printf("\t\t nr.output = %1.5f \t target = %1.5f \n", nr->output, t);

        // getchar();

        error += pow(tmp, 2);

        update_weights(nr);
    }

    // printf("\tAqui! 2.2 \n");

    // hidden layers
    for (i = mlp_net.no_of_layers - 1; i != 0; --i)
    {
        lyr = &mlp_net.layers[i];

        for (j = 0; j < lyr->no_of_neurons; ++j)
        {
            // printf("\t\tAqui! 2.21 \n");

            compute_delta(i, j);

            // printf("\t\tAqui! 2.22 \n");

            update_weights(&lyr->neurons[j]);

            // printf("\t\tAqui! 2.23 \n");
        }
    }

    // printf("\tAqui! 2.3 \n");

    return error; // TODO: fix it!
}


///////////////////////////////////////////////////
/*
 * \@function train_neural_net()
 * \@brief TODO
 */
///////////////////////////////////////////////////
void train_neural_net()
{
    int i;
    int epoch = 0;
    double error = 0.0;
    double sum_err = 0.0;

    while (epoch < max_epoch) // TODO: or error min
    {
        sum_err = 0.0;
        error = 0.0;

        for (i = 0; i < training_set_size; ++i)
        {
            feed_forward(training_inputs[i]); // i: instance index

            error = back_propagation(i); // i: instance index

            sum_err += error;
        }

        sum_err /= (training_set_size * no_of_class);

        print_training_data(epoch, sum_err);

        printf("epoch = %d \n", epoch);

        ++epoch;
    }

    // TODO: return?
}


///////////////////////////////////////////////////
/*
 * \@function classify()
 * \@brief TODO
 */
///////////////////////////////////////////////////
double classify(const struct layer *last_lyr, const int instance_index)
{
    int i;
    int class_index;
    double tmp;
    double error;
    double sum_error = 0.0;
    double max_perc = 0;

    for (i = 0; i < no_of_class; ++i)
    {
        tmp = last_lyr->neurons[i].output;

        if (tmp > max_perc)
        {
            max_perc = tmp;
            class_index = no_of_attr + i;
        }


        error = fabs(last_lyr->neurons[i].output - 
            test_inputs[instance_index][no_of_attr + i]);

        sum_error += pow(error,2);
    }

    // MODIFIED FOR TESTS
    if (fabs(test_inputs[instance_index][class_index] - 1.0) < 0.000000001)
        ++test_hits;

    // if (fabs(test_inputs[instance_index][class_index] - 1.0) < 0.000000001)
    //     ++test_hits;

    return sum_error;
}


///////////////////////////////////////////////////
/*
 * \@function print_last_layer_output()
 * \@brief TODO
 */
///////////////////////////////////////////////////
void print_last_layer_output()
{
    int i;
    int tmp, tmp2;
    FILE *of;
    struct neuron *nr;
    struct layer *lyr;

    of = fopen(output_file2, "w");

    lyr = &mlp_net.layers[mlp_net.no_of_layers]; // last layer

    fprintf(of, "Output_layer, attributes, original_attributes\n");

    for (i = 0; i < no_of_class; ++i) // lyr->no_of_neurons
    {
        nr = &lyr->neurons[i];

        tmp = (int)(nr->output*255);
        tmp2 = (int)(training_inputs[0][i]*255);

        fprintf(of, "%d, %d \n", tmp, tmp2);
    }

    fclose(of);
}


///////////////////////////////////////////////////
/*
 * \@function test_neural_net()
 * \@brief TODO
 */
///////////////////////////////////////////////////
void test_neural_net()
{
    int i, j;
    int tmp;
    double error = 0.0;

    test_hits = 0;

    tmp = mlp_net.no_of_layers;

    // MUDEI AQUI
    feed_forward(training_inputs[0]);

    print_last_layer_output();

    // for (i = 0; i < test_set_size; ++i)
    // {
    //     feed_forward(test_inputs[i]);

    //     error += classify(&mlp_net.layers[tmp], i);
    // }

    // error /= (test_set_size * no_of_class);

    // print_test_result();
}


///////////////////////////////////////////////////
/*
 * \@function main()
 * \@brief main routine
 */
///////////////////////////////////////////////////
int main()
{
    srand((double)time(NULL));

    // TODO: comment!
    print_header_output_file();

    // TODO: comment!
    printf("Reading input file...\n");
    // load_input(input_file);
    
    // 4 MNIST tests
    load_input_idx(input_file); // MUDEI AQUI

    // 4 MNIST tests
    no_of_class = no_of_attr = 784; // MUDEI AQUI

    // // TODO: comment!
    printf("Creating neural net...\n");
    create_neural_net(no_of_lyrs, layer_neurons);

    // // TODO: comment!
    printf("Training neural net...\n");
    train_neural_net();

    // // print results autoenconder
    // print_last_layer_output(); // MUDEI AQUI

    // // TODO: comment!
    printf("Testing neural net...\n");
    test_neural_net();

    return EXIT_SUCCESS;
}