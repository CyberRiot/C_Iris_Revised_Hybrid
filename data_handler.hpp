#ifndef __DATA_HANDLER_HPP
#define __DATA_HANDLER_HPP

#include "data.hpp"
#include <vector>
#include <cstdint>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_set>
#include <algorithm>
#include <random>
#include "common.hpp"
#include <numeric>

class data_handler{
    std::vector<data *> *data_array; //This Array is set to take in the entire stream of binary png images into a list
    std::vector<data *> *training_data; //These three are being populated by the split_data() function and the percentage is split below
    std::vector<data *> *testing_data;
    std::vector<data *> *validation_data;
    common_data *cd;

    //labels vector to store the labels
    std::vector<int> *labels;

    //ints to control the class info
    int class_counts;
    int num_classes;
    int feature_vector_size;

    std::map<uint8_t, int> class_from_int;
    std::map<std::string, int> class_from_string;

    //TRAINING, TESTING, AND VALIDATION PERCENTAGE CONTROL
    const double TRAINING_DATA_SET_PERCENTAGE = 0.60;
    const double TESTING_DATA_SET_PERCENTAGE = 0.20;
    const double VALIDATION_DATA_SET_PERCENTAGE = 0.20;

    public:
        //Constructor and De-Constructor
        data_handler();
        ~data_handler();

        //Read the Polyphia dataset binary data
        void read_data_and_labels(std::string data_path, std::string labels_path);
        void split_data();
        void count_classes();

        int get_class_counts();

        std::vector<data *> *get_training_data();
        std::vector<data *> *get_testing_data();
        std::vector<data *> *get_validation_data();

        std::vector<int> *get_labels();
        void set_feature_vector_size(int vect_size);
};
#endif