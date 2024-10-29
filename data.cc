#include "../include/data.hpp"

// Constructor
data::data() : feature_vector(new std::vector<double>()), class_vector(new std::vector<int>()), label(0), enum_label(0), distance(0.0) {}

// De-Constructor
data::~data() {
    delete feature_vector;
    delete class_vector;
}

// Set and Adjust the feature vector
void data::set_feature_vector(std::vector<double> *vect) {
    delete feature_vector; // Free previous memory
    feature_vector = vect; // Assign new vector
}

void data::append_to_feature_vector(double val) {
    feature_vector->push_back(val); // Add value to the vector
}

// Set Vectors, Distance, and Labels
void data::set_label(int lab) {
    label = lab;
}

void data::set_distance(double val) {
    distance = val;
}

void data::set_class_vector(int counts) {
    delete class_vector; // Free previous memory
    class_vector = new std::vector<int>(counts, 0);
}

void data::set_enum_label(int lab) {
    enum_label = lab;
}

// Get Distance and Labels
double data::get_distance() const {
    return distance;
}

int data::get_label() const {
    return label;
}

int data::get_enum_label() const {
    return enum_label;
}

// Getters for vectors
std::vector<double>* data::get_feature_vector() const {
    return feature_vector;
}

std::vector<int>* data::get_class_vector() const {
    return class_vector;
}