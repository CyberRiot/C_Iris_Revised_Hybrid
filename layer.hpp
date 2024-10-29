#ifndef __LAYER_HPP
#define __LAYER_HPP

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include "data_handler.hpp"
#include "neuron.hpp"
#include <chrono>
#include <iomanip>
#include <cstdint>

// Base Layer class for common functionalities
class layer {
public:
    virtual std::vector<double>* forward(std::vector<double>* input) = 0;
    virtual std::vector<double>* backward(std::vector<double>* gradients) = 0;
    virtual ~layer() = default;  // Virtual destructor for polymorphic cleanup
};

// Convolutional Layer class
class ConvLayer : public layer {
public:
    int filter_size;
    int num_filters;
    double learning_rate;

    std::vector<std::vector<neuron*>>* filters;  // 2D array of neurons
    std::vector<double>* input;  // Store the input for use in the backward pass

    ConvLayer(int filter_size, int num_filters, double learning_rate);
    ~ConvLayer();

    std::vector<double>* forward(std::vector<double>* input) override;
    std::vector<double>* backward(std::vector<double>* gradients) override;
    int get_pooled_output_size() const;

private:
    std::vector<std::vector<double>>* convolve(std::vector<std::vector<double>>* input, std::vector<neuron*>* filter);
    
    // New function for pooling
    std::vector<double>* average_pooling(std::vector<double>* input, int pooling_size);
};

// Recurrent Layer class (LSTM/GRU)
class RNNLayer : public layer {
public:
    int input_size;
    int hidden_size;
    double learning_rate;

    // The hidden state of the LSTM
    std::vector<double>* hidden_state;
    
    // The cell state (memory) of the LSTM
    std::vector<double>* cell_state; 

    // Neurons for each hidden state
    std::vector<neuron*>* hidden_neurons;  

    RNNLayer(int input_size, int hidden_size, double learning_rate);
    ~RNNLayer();

    std::vector<double>* forward(std::vector<double>* input) override;
    std::vector<double>* backward(std::vector<double>* gradients) override;

private:
    std::vector<double>* lstm_forward(std::vector<double>* input);
    std::vector<double>* lstm_backward(std::vector<double>* d_output, std::vector<double>* d_next_cell_state);
    double sigmoid(double x);
};


#endif