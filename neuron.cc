#include "../include/neuron.hpp"

//constructor and de-constructor
neuron::neuron(int previous_layer_size) {
    weights = new std::vector<double>(previous_layer_size);  // Initialize with correct size
    initialize_weights(previous_layer_size);
}

neuron::~neuron(){
    delete weights;
}

void neuron::initialize_weights(int previous_layer_size) {
    std::default_random_engine generator;
    double variance = 1.0 / sqrt(previous_layer_size);
    std::uniform_real_distribution<double> distribution(-variance, variance);  // Xavier initialization

    for (int i = 0; i < previous_layer_size; i++) {
        (*weights)[i] = distribution(generator);
    }
}

double neuron::activate(const std::vector<double>* inputs){
    double activation = weights->back();  // Bias term
    for(int i = 0; i < inputs->size(); i++){
        activation += weights->at(i) * inputs->at(i);  // Weighted sum
    }
    
    // Apply ReLU instead of Sigmoid
    output = std::max(0.0, activation);  // ReLU
    return output;
}

void neuron::save_weights(std::ofstream* out){
    for(double weight : *weights){
        *out << weight << " ";
    }
    *out << std::endl;
}

void neuron::load_weights(std::ifstream* in){
    weights->clear();
    std::string line;
    if(std::getline(*in, line)){
        std::istringstream iss(line);
        double weight;
        while(iss >> weight){
            weights->push_back(weight);
        }
    }
}

double neuron::generate_random_number(double min, double max){
    double random = (double)rand() / RAND_MAX;
    return min + random * (max - min);
}