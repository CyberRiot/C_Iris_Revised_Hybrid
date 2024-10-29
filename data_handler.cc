#include "../include/data_handler.hpp"

// Constructor
data_handler::data_handler() {
    data_array = new std::vector<data *>();
    training_data = new std::vector<data *>();
    testing_data = new std::vector<data *>();
    validation_data = new std::vector<data *>();
    labels = new std::vector<int>();  // Initialize labels vector
    cd = new common_data();

    feature_vector_size = 1920 * 1080;
}

// Destructor
data_handler::~data_handler() {
    delete data_array;
    delete training_data;
    delete testing_data;
    delete validation_data;
    delete labels;  // Free memory for labels
    delete cd;
}

void data_handler::read_data_and_labels(std::string data_path, std::string labels_path){
    std::cout << "Attempting to load data and labels in batches." << std::endl;

    // Set the image dimensions manually
    int width = 1920;
    int height = 1080;
    int channels = 1;
    
    // Calculate the feature vector size
    feature_vector_size = width * height * channels;
    std::cout << "Image Width: " << width << ", Height: " << height << ", Channels: " << channels << std::endl;
    std::cout << "Dynamically calculated feature vector size: " << feature_vector_size << std::endl;

    // Open the binary data file
    std::ifstream data_file(data_path, std::ios::binary);
    if (!data_file) {
        std::cerr << "Error opening data file: " << data_path << std::endl;
        exit(1);
    }

    // Open the labels CSV file
    std::ifstream labels_file(labels_path);
    if (!labels_file) {
        std::cerr << "Error opening labels file: " << labels_path << std::endl;
        exit(1);
    }

    std::string line;
    bool is_header = true;
    int label_count = 0;

    // Read the labels from the CSV, ignoring the class_name column
    while (std::getline(labels_file, line)) {
        if (is_header) {
            is_header = false;
            continue;
        }

        std::stringstream ss(line);
        std::string class_name;
        int label;

        std::getline(ss, class_name, ',');  // Ignore the class_name
        ss >> label;  // Read the actual label

        labels->push_back(label);
        label_count++;
    }

    std::cout << "Number of labels read from CSV: " << label_count << std::endl;

    const int BATCH_SIZE = 50;  // Define the batch size
    int frame_index = 0;
    
    while (data_file.peek() != EOF && frame_index < labels->size()) {
        // Create a temporary batch to store this batch of images
        std::vector<data *> temp_data_batch;

        // Process a batch of images
        for (int batch = 0; batch < BATCH_SIZE && frame_index < labels->size(); ++batch, ++frame_index) {
            std::vector<uint8_t> img_flat(feature_vector_size);

            // Read the image data
            data_file.read(reinterpret_cast<char*>(img_flat.data()), feature_vector_size);

            if (!data_file) {
                std::cerr << "Error reading binary data for sample " << frame_index << std::endl;
                break;
            }

            // Create a new data object, set its feature vector and label
            data *data_instance = new data();
            std::vector<double>* feature_vector = new std::vector<double>(img_flat.begin(), img_flat.end());
            data_instance->set_feature_vector(feature_vector);

            // Check for empty feature vectors
            if (feature_vector->empty()) {
                std::cerr << "Error: Empty feature vector for sample " << frame_index << std::endl;
            }

            // Set the label corresponding to the image
            data_instance->set_label((*labels)[frame_index]);
            temp_data_batch.push_back(data_instance);
        }

        // After the batch is processed, push it to the main data array
        data_array->insert(data_array->end(), temp_data_batch.begin(), temp_data_batch.end());

        std::cout << "Successfully processed batch of " << temp_data_batch.size() << " images." << std::endl;
    }

    // Close files
    std::cout << "Successfully loaded all data and labels in batches." << std::endl;
    data_file.close();
    labels_file.close();
}

void data_handler::split_data() {
    std::unordered_set<int> used_indexes;
    int training_size = data_array->size() * TRAINING_DATA_SET_PERCENTAGE;
    int testing_size = data_array->size() * TESTING_DATA_SET_PERCENTAGE;
    int validation_size = data_array->size() * VALIDATION_DATA_SET_PERCENTAGE;

    // Use a random device and shuffle
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data_array->begin(), data_array->end(), g);  // Shuffle the data

    // TRAINING DATA
    int count = 0;
    int index = 0;
    while (count < training_size) {
        training_data->push_back(data_array->at(index++));
        count++;
    }

    // TESTING DATA
    count = 0;
    while (count < testing_size) {
        testing_data->push_back(data_array->at(index++));
        count++;
    }

    // VALIDATION DATA
    count = 0;
    while (count < validation_size) {
        validation_data->push_back(data_array->at(index++));
        count++;
    }
    std::cout << "First Training Data Feature Vector Size: " << training_data->at(0)->get_feature_vector()->size() << std::endl;

    
    std::cout << "Training Data Size: " << training_data->size() 
              << "\nTesting Data Size: " << testing_data->size() 
              << "\nValidation Data Size: " << validation_data->size() << std::endl;
}

void data_handler::count_classes() {
    int count = 0;
    for (unsigned i = 0; i < data_array->size(); i++) {
        if (class_from_int.find(data_array->at(i)->get_label()) == class_from_int.end()) {
            class_from_int[data_array->at(i)->get_label()] = count;
            data_array->at(i)->set_enum_label(count);
            count++;
        } else {
            data_array->at(i)->set_enum_label(class_from_int[data_array->at(i)->get_label()]);
        }
    }
    class_counts = count;

    for (data *da : *data_array)
        da->set_class_vector(class_counts);

    printf("Successfully Extracted %d Unique Classes.\n", class_counts);
}

// Getter methods
int data_handler::get_class_counts() {
    return class_counts;
}

std::vector<data *> *data_handler::get_training_data() {
    return training_data;
}

std::vector<data *> *data_handler::get_testing_data() {
    return testing_data;
}

std::vector<data *> *data_handler::get_validation_data() {
    return validation_data;
}

std::vector<int> *data_handler::get_labels() {
    return labels;
}

//Setter Methods
void data_handler::set_feature_vector_size(int vect_size){
    feature_vector_size = vect_size;
}