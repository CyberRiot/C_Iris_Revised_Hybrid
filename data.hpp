#ifndef __DATA_HPP
#define __DATA_HPP

#include <vector>
#include <cstdint>
#include <iostream>

class data {
    std::vector<double> *feature_vector;
    std::vector<int> *class_vector;
    int label;
    int enum_label;
    double distance;

public:
    // Constructor and Destructor
    data();
    ~data();

    // Set and Adjust the feature vector
    void set_feature_vector(std::vector<double> *vect);
    void append_to_feature_vector(double val);

    // Set Vectors, Distance, and Labels
    void set_label(int lab);
    void set_distance(double val);
    void set_class_vector(int counts);
    void set_enum_label(int lab);

    // Get Distance and Labels
    double get_distance() const;
    int get_label() const;
    int get_enum_label() const;

    // Getters for vectors
    std::vector<double>* get_feature_vector() const;
    std::vector<int>* get_class_vector() const;
};

#endif