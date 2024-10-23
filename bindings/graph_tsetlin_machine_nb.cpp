#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
namespace nb = nanobind;

// Update the external function declaration
extern "C" void launch_prepare_message_ta_state(unsigned int *d_global_ta_state,
                                                dim3 gridDim,
                                                dim3 blockDim,
                                                const int STATE_BITS,
                                                const int CLAUSES,
                                                const int MESSAGE_CHUNKS);



extern "C" void launch_update(curandState *state,
                              float s,
                              unsigned int *global_ta_state,
                              int number_of_nodes,
                              int graph_index,
                              int *clause_node,
                              int *number_of_include_actions,
                              int *X,
                              int *class_clause_update,
                              int LA_CHUNKS,
                              int CLAUSES,
                              int CLASSES,
                              int STATE_BITS,
                              int MAX_INCLUDED_LITERALS,
                              int BOOST_TRUE_POSITIVE_FEEDBACK,
                              int INT_SIZE,
                              dim3 gridDim,
                              dim3 blockDim);


class GraphTsetlinMachineState {
public:
    int number_of_outputs = 0;
    int number_of_clauses = 0;
    int number_of_literals = 0;
    int number_of_state_bits = 8;
    int boost_true_positive_feedback = 1;
    int T = 0;
    float q = 1.0f;
    int max_included_literals = 0;
    int negative_clauses = 1;
    int max_number_of_graph_nodes = 0;
    int message_size = 256;
    int message_bits = 2;

    // Computed variables
    int INT_SIZE = 32;
    int LA_CHUNKS = 0;
    int CLAUSE_CHUNKS = 0;
    int MESSAGE_LITERALS = 0;
    int MESSAGE_CHUNKS = 0;
    int NODE_CHUNKS = 0;
    unsigned int FILTER = 0;
    unsigned int MESSAGE_FILTER = 0;

    GraphTsetlinMachineState() {
        initializeCUDA();
    }

    ~GraphTsetlinMachineState() {
        // Clean up CUDA resources in the destructor
        cudaDeviceReset();
    }

    void initializeCUDA() {
        cudaError_t cudaStatus;

        // Get the number of CUDA devices
        int deviceCount = 0;
        cudaStatus = cudaGetDeviceCount(&deviceCount);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error(std::string("cudaGetDeviceCount failed! Error: ") + cudaGetErrorString(cudaStatus));
        }

        if (deviceCount == 0) {
            throw std::runtime_error("No CUDA-capable devices found!");
        }

        // Set to use the first available device
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error(std::string("cudaSetDevice failed! Error: ") + cudaGetErrorString(cudaStatus));
        }

        // Optional: Print some information about the selected device
        cudaDeviceProp deviceProp;
        cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error(std::string("cudaGetDeviceProperties failed! Error: ") + cudaGetErrorString(cudaStatus));
        }

        printf("Using CUDA Device %d: %s\n", 0, deviceProp.name);
    }

    void checkCUDAError(const char* message) {
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error(std::string(message) + " Error: " + cudaGetErrorString(cudaStatus));
        }
    }


    void update_state(const nb::dict& kwargs) {
        if (kwargs.contains("number_of_outputs")) number_of_outputs = nb::cast<int>(kwargs["number_of_outputs"]);
        if (kwargs.contains("number_of_clauses")) number_of_clauses = nb::cast<int>(kwargs["number_of_clauses"]);
        if (kwargs.contains("number_of_literals")) number_of_literals = nb::cast<int>(kwargs["number_of_literals"]);
        if (kwargs.contains("number_of_state_bits")) number_of_state_bits = nb::cast<int>(kwargs["number_of_state_bits"]);
        if (kwargs.contains("boost_true_positive_feedback")) boost_true_positive_feedback = nb::cast<int>(kwargs["boost_true_positive_feedback"]);
        if (kwargs.contains("T")) T = nb::cast<int>(kwargs["T"]);
        if (kwargs.contains("q")) q = nb::cast<float>(kwargs["q"]);
        if (kwargs.contains("max_included_literals")) max_included_literals = nb::cast<int>(kwargs["max_included_literals"]);
        if (kwargs.contains("negative_clauses")) negative_clauses = nb::cast<int>(kwargs["negative_clauses"]);
        if (kwargs.contains("max_number_of_graph_nodes")) max_number_of_graph_nodes = nb::cast<int>(kwargs["max_number_of_graph_nodes"]);
        if (kwargs.contains("message_size")) message_size = nb::cast<int>(kwargs["message_size"]);
        if (kwargs.contains("message_bits")) message_bits = nb::cast<int>(kwargs["message_bits"]);


        // Compute derived variables
        LA_CHUNKS = (number_of_literals - 1) / INT_SIZE + 1;
        CLAUSE_CHUNKS = (number_of_clauses - 1) / INT_SIZE + 1;
        MESSAGE_LITERALS = message_size * 2;
        MESSAGE_CHUNKS = (MESSAGE_LITERALS - 1) / INT_SIZE + 1;
        NODE_CHUNKS = (max_number_of_graph_nodes - 1) / INT_SIZE + 1;

        // Compute FILTER
        if (number_of_literals % 32 != 0) {
            FILTER = ~(0xffffffff << (number_of_literals % INT_SIZE));
        } else {
            FILTER = 0xffffffff;
        }

        // Compute MESSAGE_FILTER
        if (MESSAGE_LITERALS % 32 != 0) {
            MESSAGE_FILTER = ~(0xffffffff << (MESSAGE_LITERALS % INT_SIZE));
        } else {
            MESSAGE_FILTER = 0xffffffff;
        }

    }

     void prepare_message_ta_state(unsigned long long int cuda_array_ptr, nb::dict grid, nb::dict block) {
            // Extract grid and block dimensions
            dim3 gridDim(
                nb::cast<int>(grid["x"]),
                nb::cast<int>(grid["y"]),
                nb::cast<int>(grid["z"])
            );
            dim3 blockDim(
                nb::cast<int>(block["x"]),
                nb::cast<int>(block["y"]),
                nb::cast<int>(block["z"])
            );
            // Convert the pointer value to a CUDA device pointer
            unsigned int *d_global_ta_state = reinterpret_cast<unsigned int*>(cuda_array_ptr);

            // Launch the kernel using the CUDA function
            launch_prepare_message_ta_state(
                d_global_ta_state,
                gridDim,
                blockDim,
                number_of_state_bits,
                number_of_clauses,
                message_size
            );
        }


void update(nb::dict grid,
            nb::dict block,
            unsigned long long int curand_state_ptr,
            float s,
            unsigned long long int global_ta_state_ptr,
            int number_of_nodes,
            int graph_index,
            unsigned long long int clause_node_ptr,
            unsigned long long int number_of_include_actions_ptr,
            unsigned long long int X_ptr,
            unsigned long long int class_clause_update_ptr) {

    // Extract grid and block dimensions
    dim3 gridDim(
        nb::cast<int>(grid["x"]),
        nb::cast<int>(grid["y"]),
        nb::cast<int>(grid["z"])
    );
    dim3 blockDim(
        nb::cast<int>(block["x"]),
        nb::cast<int>(block["y"]),
        nb::cast<int>(block["z"])
    );

    // Convert the pointer values to CUDA device pointers
    curandState *d_curand_state = reinterpret_cast<curandState*>(curand_state_ptr);
    unsigned int *d_global_ta_state = reinterpret_cast<unsigned int*>(global_ta_state_ptr);
    int *d_clause_node = reinterpret_cast<int*>(clause_node_ptr);
    int *d_number_of_include_actions = reinterpret_cast<int*>(number_of_include_actions_ptr);
    int *d_X = reinterpret_cast<int*>(X_ptr);
    int *d_class_clause_update = reinterpret_cast<int*>(class_clause_update_ptr);

    // Launch the kernel using the CUDA function
    launch_update(d_curand_state, s, d_global_ta_state, number_of_nodes, graph_index,
                  d_clause_node, d_number_of_include_actions, d_X, d_class_clause_update,
                  this->LA_CHUNKS, this->number_of_clauses, this->number_of_outputs, this->number_of_state_bits,
                  this->max_included_literals, this->boost_true_positive_feedback, this->INT_SIZE,
                  gridDim, blockDim);

//    cudaDeviceSynchronize();
//    // Check for any CUDA errors
//    cudaError_t cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(cudaStatus));
//    }
}

};

NB_MODULE(graphtsetlinmachine, m) {
    nb::class_<GraphTsetlinMachineState>(m, "GraphTsetlinMachineState")
        .def(nb::init<>())
        .def("update", &GraphTsetlinMachineState::update)
        .def("update_state", &GraphTsetlinMachineState::update_state)
        .def("prepare_message_ta_state", &GraphTsetlinMachineState::prepare_message_ta_state)
        .def_rw("number_of_outputs", &GraphTsetlinMachineState::number_of_outputs)
        .def_rw("number_of_clauses", &GraphTsetlinMachineState::number_of_clauses)
        .def_rw("number_of_literals", &GraphTsetlinMachineState::number_of_literals)
        .def_rw("number_of_state_bits", &GraphTsetlinMachineState::number_of_state_bits)
        .def_rw("boost_true_positive_feedback", &GraphTsetlinMachineState::boost_true_positive_feedback)
        .def_rw("T", &GraphTsetlinMachineState::T)
        .def_rw("q", &GraphTsetlinMachineState::q)
        .def_rw("max_included_literals", &GraphTsetlinMachineState::max_included_literals)
        .def_rw("negative_clauses", &GraphTsetlinMachineState::negative_clauses)
        .def_rw("max_number_of_graph_nodes", &GraphTsetlinMachineState::max_number_of_graph_nodes)
        .def_rw("message_size", &GraphTsetlinMachineState::message_size)
        .def_rw("message_bits", &GraphTsetlinMachineState::message_bits);
}