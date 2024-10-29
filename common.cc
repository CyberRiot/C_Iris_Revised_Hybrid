#include "../include/common.hpp"

common_data::common_data(){
    common_training_data = nullptr;
    common_testing_data = nullptr;
    common_validation_data = nullptr;
}

common_data::~common_data(){
    delete common_training_data;
    delete common_testing_data;
    delete common_validation_data;
}

std::vector<data *> *common_data::get_common_training_data(){
    return common_training_data;
}

std::vector<data *> *common_data::get_common_testing_data(){
    return common_testing_data;
}

std::vector<data *> *common_data::get_common_validation_data(){
    return common_validation_data;
}

void common_data::set_common_training_data(std::vector<data *> *vect){
    common_training_data = vect;
}

void common_data::set_common_testing_data(std::vector<data *> *vect){
    common_testing_data = vect;
}

void common_data::set_common_validation_data(std::vector<data *> *vect){
    common_validation_data = vect;
}