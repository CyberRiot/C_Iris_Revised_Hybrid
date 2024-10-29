#include "network.hpp"

network::network(std::vector<int> *spec, int input_size, int num_classes, double learning_rate) {
    this->input_size = input_size;
    this->num_classes = num_classes;
    this->learning_rate = learning_rate;
    layers = new std::vector<layer *>();

    std::cout << "Starting layer initialization..." << std::endl;

    int current_input_size = input_size;  // Track the input size for each layer

    // Initialize the ConvLayer first
    int conv_filter_size = 3;  // Use filter size 3 for convolution
    int conv_num_filters = spec->at(0);  // First element in spec is the number of filters
    layers->push_back(new ConvLayer(conv_filter_size, conv_num_filters, learning_rate));
    
    std::cout << "ConvLayer initialized with filter size " << conv_filter_size << " and " << conv_num_filters << " filters." << std::endl;

    // Now, get the output size after convolution and pooling (or flattening if pooling is removed)
    ConvLayer* conv_layer = static_cast<ConvLayer*>(layers->back());

    // Run a dummy forward pass to get the size after convolution and pooling/flattening
    std::vector<double> dummy_input(input_size, 0.0);  // Create a dummy input with the original input size
    std::vector<double>* conv_output = conv_layer->forward(&dummy_input);
    int pooled_output_size = conv_output->size();  // Get the size after convolution and pooling/flattening

    std::cout << "Pooled output size from ConvLayer: " << pooled_output_size << std::endl;

    // Set up chunking for RNN input size if necessary
    int chunk_size = 8192;  // Adjust chunk size based on available resources
    int num_chunks = pooled_output_size / chunk_size;
    int remaining_elements = pooled_output_size % chunk_size;

    std::cout << "Chunking RNN input size into chunks of " << chunk_size << std::endl;
    std::cout << "Number of chunks: " << num_chunks << ", Remaining elements: " << remaining_elements << std::endl;

    // Use the pooled output size as the input size for the first RNNLayer
    current_input_size = chunk_size;

    // Initialize RNN layers in chunks
    for (int i = 1; i < spec->size(); i++) {
        int hidden_size = spec->at(i);
        std::cout << "Initializing RNNLayer with input size (chunked) " << current_input_size << " and hidden size " << hidden_size << "." << std::endl;

        layers->push_back(new RNNLayer(current_input_size, hidden_size, learning_rate));

        current_input_size = hidden_size;  // Update input size for the next RNNLayer
    }

    // Initialize the final output layer (RNNLayer) in chunks
    std::cout << "Initializing final RNNLayer with input size (chunked) " << current_input_size << " and output size " << num_classes << "." << std::endl;
    layers->push_back(new RNNLayer(current_input_size, num_classes, learning_rate));

    std::cout << "Network initialized with " << layers->size() << " layers." << std::endl;

    delete conv_output;  // Clean up the dummy input/output after determining sizes
}

network::~network() {
    for (layer *l : *layers) {
        delete l;
    }
    delete layers;
    close_debug_output();
}

std::vector<double>* network::fprop(data *d) {
    std::vector<double>* input = d->get_feature_vector();
    std::cout << "Forward Pass | Initial Feature Vector Size: " << input->size() << std::endl;

    for (int i = 0; i < layers->size(); ++i) {
        layer* current_layer = (*layers)[i];

        // Handling chunking for RNNLayer
        if (RNNLayer* rnn_layer = dynamic_cast<RNNLayer*>(current_layer)) {
            std::cout << "Processing RNNLayer with chunking." << std::endl;
            
            int chunk_size = 8192;  // Adjust the chunk size based on your system
            int num_chunks = input->size() / chunk_size;
            int remaining_elements = input->size() % chunk_size;

            std::cout << "Input size: " << input->size() << ", Chunk size: " << chunk_size << ", Number of chunks: " << num_chunks << std::endl;

            std::vector<double>* rnn_output = new std::vector<double>;

            // Process each chunk
            for (int j = 0; j < num_chunks; j++) {
                std::vector<double> chunk(input->begin() + j * chunk_size, input->begin() + (j + 1) * chunk_size);
                std::cout << "Processing chunk " << j + 1 << "/" << num_chunks << " with size: " << chunk.size() << std::endl;
                
                std::vector<double>* chunk_output = rnn_layer->forward(&chunk);
                rnn_output->insert(rnn_output->end(), chunk_output->begin(), chunk_output->end());
                delete chunk_output;
            }

            // Handle the remaining data if any
            if (remaining_elements > 0) {
                std::vector<double> chunk(input->end() - remaining_elements, input->end());
                std::cout << "Processing remaining chunk with size: " << chunk.size() << std::endl;
                
                std::vector<double>* chunk_output = rnn_layer->forward(&chunk);
                rnn_output->insert(rnn_output->end(), chunk_output->begin(), chunk_output->end());
                delete chunk_output;
            }

            input = rnn_output;
        } else {
            // For non-RNN layers
            input = current_layer->forward(input);
            std::cout << "Hit other condition" << std::endl;
        }

        if (input == nullptr || input->empty()) {
            std::cerr << "Error: Input feature vector became empty after forward pass in layer " << i << "!" << std::endl;
            exit(1);
        }

        std::cout << "Layer " << i << " forward pass completed with output size: " << input->size() << std::endl;
    }

    return input;
}

void network::bprop(data *d) {
    std::vector<double> *output = fprop(d);
    std::vector<double>* gradients = new std::vector<double>(output->size(), 0.0);

    std::vector<int> *class_vector = d->get_class_vector();
    for(size_t i = 0; i < output->size(); i++){
        double error = (*output)[i] - (*class_vector)[i];
        (*gradients)[i] = error * transfer_derivative((*output)[i]);
    }

    for(int i = layers->size() - 1; i >= 0; i--){
        layer *current_layer = (*layers)[i];
        std::cout << "Backpropagating through layer " << i << std::endl;  // Debugging
        gradients = current_layer->backward(gradients);
    }

    delete gradients;
}

void network::update_weights(data *d) {
    for(int i = 0; i < layers->size(); i++){
        layer* current_layer = (*layers)[i];

        //Check if current layer is a ConvLayer or RNNLayer
        if(ConvLayer* conv_layer = dynamic_cast<ConvLayer*>(current_layer)){
            //This is a Convolutional Layer, update weights for ConvLayer
            for(int f = 0; f < conv_layer->num_filters; f++){
                for(int j = 0; j < conv_layer->filter_size * conv_layer->filter_size; j++){
                    for(auto &n : (*conv_layer->filters)[f]){
                        //This is the rule for the simple weight update
                        n->weights->at(j) -= learning_rate * n->delta;
                    }
                }
            }
        }
        else if(RNNLayer* rnn_layer = dynamic_cast<RNNLayer*>(current_layer)){
            //Update weights for RNNLayer
            for(int h = 0; h < rnn_layer->hidden_size; h++){
                for(int w = 0; w < rnn_layer->input_size; w++){
                    (*rnn_layer->hidden_neurons)[h]->weights->at(w) -= learning_rate * (*rnn_layer->hidden_neurons)[h]->delta;
                }
            }
        }
    }
}

double network::transfer(double activation) {
    return 1.0 / (1.0 + std::exp(-activation));
}

double network::transfer_derivative(double output) {
    return output * (1 - output);
}

int network::predict(data *d) {
    std::vector<double>* output = fprop(d);
    return std::distance(output->begin(), std::max_element(output->begin(), output->end()));
}

// Training: Pass data through the network, forward pass only for now
void network::train(int epochs, double validation_threshold) {
    if (common_training_data == nullptr || common_training_data->empty()) {
        std::cerr << "Error: Training data is empty!" << std::endl;
        return;
    }

    // Open the debug log for writing
    std::ofstream debug_log("debug_output.txt", std::ios::app);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl;
        debug_log << "Epoch " << epoch + 1 << "/" << epochs << std::endl;

        double total_loss = 0.0;
        int sample_index = 0;

        for (data* d : *common_training_data) {
            std::vector<double>* input = d->get_feature_vector();

            if (input == nullptr || input->empty()) {
                std::cerr << "Error: Input feature vector is empty for sample " << sample_index << std::endl;
                debug_log << "Error: Input feature vector is empty for sample " << sample_index << std::endl;
                continue;
            }

            // Forward pass
            std::cout << "Forward pass for sample " << sample_index << std::endl;
            std::vector<double>* output = fprop(d);

            if (output == nullptr) {
                std::cerr << "Error: Forward pass produced nullptr for sample " << sample_index << std::endl;
                return;  // Exit if the output is nullptr
            } else if (output->empty()) {
                std::cerr << "Error: Forward pass produced empty output for sample " << sample_index << std::endl;
                return;  // Exit if the output is empty
            } else {
                std::cout << "Forward pass completed successfully for sample " << sample_index << std::endl;
                std::cout << "Output size: " << output->size() << std::endl;
            }
            
            // Check for valid output
            if (output == nullptr || output->empty()) {
                std::cerr << "Error: Forward pass produced empty output for sample " << sample_index << std::endl;
                debug_log << "Error: Forward pass produced empty output for sample " << sample_index << std::endl;
                continue;
            }

            // Calculate loss
            std::cout << "Calculating loss for sample " << sample_index << std::endl;
            double sample_loss = calculate_loss(output, d->get_class_vector());
            total_loss += sample_loss;

            // Backward pass
            std::cout << "Backward pass for sample " << sample_index << std::endl;
            bprop(d);

            // Update weights
            std::cout << "Updating weights for sample " << sample_index << std::endl;
            update_weights(d);

            sample_index++;
        }

        std::cout << "Epoch " << epoch + 1 << " completed. Total loss: " << total_loss << std::endl;
        debug_log << "Epoch " << epoch + 1 << " completed. Total loss: " << total_loss << std::endl;

        // Early stopping
        double validation_accuracy = validate();
        std::cout << "Validation Accuracy after epoch " << epoch + 1 << ": " << validation_accuracy * 100 << "%" << std::endl;
        debug_log << "Validation Accuracy after epoch " << epoch + 1 << ": " << validation_accuracy * 100 << "%" << std::endl;

        if (validation_accuracy >= validation_threshold) {
            std::cout << "Stopping early as validation accuracy reached " << validation_accuracy * 100 << "%" << std::endl;
            break;
        }
    }

    debug_log.close();
}

void network::set_debug_output(const std::string &filename) {
    debug_output.open(filename);
    if (!debug_output.is_open()) {
        std::cerr << "Error opening debug file: " << filename << std::endl;
    }
}

void network::close_debug_output() {
    if (debug_output.is_open()) {
        debug_output.close();
    }
}

void network::save_model(const std::string &filename) {
    std::ofstream outfile(filename);
    if(!outfile.is_open()){
        std::cerr << "Error: Could not open file" << filename << "for saving model." << std::endl;
        exit(1);
    }

    //Save each layer's weights
    for(layer* l : *layers){
        ConvLayer* conv_layer = dynamic_cast<ConvLayer*>(l);
        RNNLayer* rnn_layer = dynamic_cast<RNNLayer*>(l);
        if(conv_layer != nullptr){
            for(auto& filter : *conv_layer->filters){
                for(neuron* n : filter){
                    for(double weight : *n->weights){
                        outfile << weight << " ";
                    }
                    outfile << std::endl;
                }
            }
        }
        if(rnn_layer != nullptr){
            for(neuron* n : *rnn_layer->hidden_neurons){
                for(double weight : *n->weights){
                    outfile << weight << " ";
                }
                outfile << std::endl;
            }
        }
    }
    outfile.close();
    std::cout << "Model saved to " << filename << std::endl;
}

void network::load_model(const std::string &filename) {
    std::ifstream infile(filename);
    if(!infile.is_open()){
        std::cerr << "Error: could not open file " << filename << " for loading model." << std::endl;
        exit(1);
    }
    for(layer* l : *layers){
        ConvLayer* conv_layer = dynamic_cast<ConvLayer*>(l);
        RNNLayer* rnn_layer = dynamic_cast<RNNLayer*>(l);

        if(conv_layer != nullptr){
            for(auto& filter : *conv_layer->filters){
                for(neuron* n : filter){
                    for(double& weight : *n->weights){
                        infile >> weight;
                    }
                }
            }
        }
        if(rnn_layer != nullptr){
            for(neuron* n : *rnn_layer->hidden_neurons){
                for(double& weight : *n->weights){
                    infile >> weight;
                }
            }
        }
    }
    infile.close();
    std::cout << "Model loaded from " << filename << std::endl;
}

double network::validate() {
    if (common_validation_data == nullptr || common_validation_data->empty()) {
        std::cerr << "Error: Validation data is empty!" << std::endl;
        return 0.0;
    }

    int correct = 0;
    int total = 0;

    // Iterate over the validation data in common_validation_data
    for (data* d : *common_validation_data) {
        std::vector<double>* input = d->get_feature_vector();

        if (input == nullptr || input->empty()) {
            std::cerr << "Error: Input feature vector is empty!" << std::endl;
            continue;
        }

        // Forward pass through the network
        std::vector<double>* output = fprop(d);

        // Get the predicted class
        int predicted_class = std::distance(output->begin(), std::max_element(output->begin(), output->end()));

        if (d->get_class_vector()->at(predicted_class) == 1) {
            correct++;
        }
        total++;
    }

    return static_cast<double>(correct) / total;
}

double network::test() {
    if (common_testing_data == nullptr || common_testing_data->empty()) {
        std::cerr << "Error: Testing data is empty!" << std::endl;
        return 0.0;
    }

    int correct = 0;
    int total = 0;

    // Iterate over the testing data in common_testing_data
    for (data* d : *common_testing_data) {
        std::vector<double>* input = d->get_feature_vector();
        
        if (input == nullptr || input->empty()) {
            std::cerr << "Error: Input feature vector is empty!" << std::endl;
            continue;
        }

        // Forward pass through the network
        std::vector<double>* output = fprop(d);

        // Get the predicted class
        int predicted_class = std::distance(output->begin(), std::max_element(output->begin(), output->end()));

        if (d->get_class_vector()->at(predicted_class) == 1) {
            correct++;
        }
        total++;
    }

    return static_cast<double>(correct) / total;
}

void network::output_predictions(const std::string &filename, data_handler *dh) {
    // Output predictions to CSV or other file formats
}

double network::calculate_loss(std::vector<double>* output, std::vector<int>* class_vector) {
    double loss = 0.0;
    for (size_t i = 0; i < class_vector->size(); i++) {
        // Assuming binary cross-entropy loss for this example
        int target = (*class_vector)[i];  // Dereference to access the value
        double prediction = (*output)[i];
        
        // Binary cross-entropy loss calculation
        loss += -target * std::log(prediction) - (1 - target) * std::log(1 - prediction);
    }
    return loss;
}

// In Main
int main() {
    // Initialize the data handler
    data_handler *dh = new data_handler();
    dh->read_data_and_labels("F:\\Code\\VOOD\\data\\binary\\Polyphia.data", "F:\\Code\\VOOD\\data\\labels\\VOOD_labels.csv");
    dh->split_data();

    // Initialize the network with the inherited common_data's training set
    std::vector<int> *spec = new std::vector<int>{128, 64, 3};
    network *net = new network(spec, dh->get_training_data()->at(0)->get_feature_vector()->size(), 3, 0.01);

    // Set debug output file
    net->set_debug_output("debug_output.txt");

    // Assign the split data to the network's inherited common_data members
    net->set_common_training_data(dh->get_training_data());
    net->set_common_testing_data(dh->get_testing_data());
    net->set_common_validation_data(dh->get_validation_data());

    // Train the network
    std::cout << "Starting training..." << std::endl;
    net->train(10, 0.98);  // Train for 10 epochs

    // Save the trained model
    net->save_model("F:\\Code\\VOOD\\data\\saved_model\\trained_model.bin");
    std::cout << "Model saved successfully." << std::endl;

    // Test the model on test data
    double test_accuracy = net->test();
    std::cout << "Test Accuracy: " << test_accuracy << std::endl;

    // Cleanup
    delete dh;
    delete net;
    delete spec;

    return 0;
}
