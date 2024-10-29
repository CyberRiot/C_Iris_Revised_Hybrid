#ifndef __NEURON_HPP
#define __NEURON_HPP

#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <sstream>

class neuron{
    public:
        double output;
        double delta; // term for gradient error, used for backprogagation
        std::vector<double>* weights; // pointers to the weights for connection to previous layers

        //constructor and de-constructor
        neuron(int previous_layer_size);
        ~neuron();

        void initialize_weights(int previous_layer_size);
        double activate(const std::vector<double>* inputs);
        void save_weights(std::ofstream* out);
        void load_weights(std::ifstream* in);
    private:
        double generate_random_number(double min, double max);
};

#endif