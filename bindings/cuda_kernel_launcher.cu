#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdexcept>
#include <string>

// CUDA kernel declaration
extern "C" __global__ void prepare_message_ta_state(unsigned int *global_ta_state,
                                                    const int STATE_BITS,
                                                    const int CLAUSES,
                                                    const int MESSAGE_CHUNKS);

// CUDA kernel launcher
extern "C" void launch_prepare_message_ta_state(unsigned int *d_global_ta_state,
                                                dim3 gridDim,
                                                dim3 blockDim,
                                                const int STATE_BITS,
                                                const int CLAUSES,
                                                const int MESSAGE_CHUNKS) {
    prepare_message_ta_state<<<gridDim, blockDim>>>(d_global_ta_state, STATE_BITS, CLAUSES, MESSAGE_CHUNKS);
    cudaDeviceSynchronize();
}

// CUDA kernel implementation
extern "C"
{
    __global__ void prepare_message_ta_state(unsigned int *global_ta_state,
                                             const int STATE_BITS,
                                             const int CLAUSES,
                                             const int MESSAGE_CHUNKS)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (unsigned long long clause = index; clause < CLAUSES; clause += stride) {
            unsigned int *ta_state = &global_ta_state[clause*MESSAGE_CHUNKS*STATE_BITS];
            for (int message_ta_chunk = 0; message_ta_chunk < MESSAGE_CHUNKS; ++message_ta_chunk) {
                for (int b = 0; b < STATE_BITS-1; ++b) {
                    ta_state[message_ta_chunk*STATE_BITS + b] = ~0;
                }
                ta_state[message_ta_chunk*STATE_BITS + STATE_BITS - 1] = 0;
            }
        }
    }
}






///////////////////////////////////////////

// Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
__device__ inline void inc(unsigned int *ta_state, int chunk, unsigned int active, const int STATE_BITS)
{
    unsigned int carry, carry_next;
    int id = chunk * STATE_BITS;
    carry = active;
    for (int b = 0; b < STATE_BITS; ++b) {
        if (carry == 0)
            break;

        carry_next = ta_state[id + b] & carry; // Sets carry bits (overflow) passing on to next bit
        ta_state[id + b] = ta_state[id + b] ^ carry; // Performs increments with XOR
        carry = carry_next;
    }

    if (carry > 0) {
        for (int b = 0; b < STATE_BITS; ++b) {
            ta_state[id + b] |= carry;
        }
    }
}

// Decrement the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
__device__ inline void dec(unsigned int *ta_state, int chunk, unsigned int active, const int STATE_BITS)
{
    unsigned int carry, carry_next;
    int id = chunk * STATE_BITS;
    carry = active;
    for (int b = 0; b < STATE_BITS; ++b) {
        if (carry == 0)
            break;

        carry_next = (~ta_state[id + b]) & carry; // Sets carry bits (overflow) passing on to next bit
        ta_state[id + b] = ta_state[id + b] ^ carry; // Performs increments with XOR
        carry = carry_next;
    }

    if (carry > 0) {
        for (int b = 0; b < STATE_BITS; ++b) {
            ta_state[id + b] &= ~carry;
        }
    }
}

__device__ inline void update_clause(
    curandState *localState,
    float s,
    int target_sign,
    unsigned int *ta_state,
    int clause_output,
    int clause_node,
    int number_of_include_actions,
    int *X,
    const int LA_CHUNKS,
    const int STATE_BITS,
    const int MAX_INCLUDED_LITERALS,
    const int BOOST_TRUE_POSITIVE_FEEDBACK,
    const int INT_SIZE
)
{
    if (target_sign > 0) {
        // Type I Feedback
        for (int la_chunk = 0; la_chunk < LA_CHUNKS; ++la_chunk) {
            // Generate random bit values
            unsigned int la_feedback = 0;
            for (int b = 0; b < INT_SIZE; ++b) {
                if (curand_uniform(localState) <= 1.0/s) {
                    la_feedback |= (1 << b);
                }
            }

            if (clause_output && number_of_include_actions <= MAX_INCLUDED_LITERALS) {
                #if BOOST_TRUE_POSITIVE_FEEDBACK == 1
                    inc(ta_state, la_chunk, X[clause_node*LA_CHUNKS + la_chunk], STATE_BITS);
                #else
                    inc(ta_state, la_chunk, X[clause_node*LA_CHUNKS + la_chunk] & (~la_feedback), STATE_BITS);
                #endif

                dec(ta_state, la_chunk, (~X[clause_node*LA_CHUNKS + la_chunk]) & la_feedback, STATE_BITS);
            } else {
                dec(ta_state, la_chunk, la_feedback, STATE_BITS);
            }
        }
    } else if (target_sign < 0 && clause_output) {
        // Type II Feedback
        for (int la_chunk = 0; la_chunk < LA_CHUNKS; ++la_chunk) {
            inc(ta_state, la_chunk, (~X[clause_node*LA_CHUNKS + la_chunk]) & (~ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]), STATE_BITS);
        }
    }
}

__global__ void update_kernel(
    curandState *state,
    float s,
    unsigned int *global_ta_state,
    int number_of_nodes,
    int graph_index,
    int *clause_node,
    int *number_of_include_actions,
    int *X,
    int *class_clause_update,
    const int LA_CHUNKS,
    const int CLAUSES,
    const int CLASSES,
    const int STATE_BITS,
    const int MAX_INCLUDED_LITERALS,
    const int BOOST_TRUE_POSITIVE_FEEDBACK,
    const int INT_SIZE
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState localState = state[index];
    X = &X[graph_index * LA_CHUNKS];
    // Calculate clause output first
    for (unsigned long long clause = index; clause < CLAUSES; clause += stride) {
        unsigned int *ta_state = &global_ta_state[clause*LA_CHUNKS*STATE_BITS];
        for (unsigned long long class_id = 0; class_id < CLASSES; ++class_id) {
            update_clause(&localState, s, class_clause_update[class_id*CLAUSES + clause],
                          ta_state, clause_node[clause] != -1, clause_node[clause],
                          number_of_include_actions[clause], X, LA_CHUNKS, STATE_BITS,
                          MAX_INCLUDED_LITERALS, BOOST_TRUE_POSITIVE_FEEDBACK, INT_SIZE);
        }
    }
    state[index] = localState;
}

extern "C" void launch_update(
    curandState *state,
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
    dim3 blockDim
)
{
    update_kernel<<<gridDim, blockDim>>>(
        state, s, global_ta_state, number_of_nodes, graph_index,
        clause_node, number_of_include_actions, X, class_clause_update,
        LA_CHUNKS, CLAUSES, CLASSES, STATE_BITS, MAX_INCLUDED_LITERALS,
        BOOST_TRUE_POSITIVE_FEEDBACK, INT_SIZE
    );
}


