/**
 * @file gpu_kernels.h
 * @brief CUDA Kernel Declarations
 * 
 * CLASSIFICATION: ENTERPRISE PRODUCTION
 * SECURITY LEVEL: Governmental Grade (Level 5)
 * 
 * Declarations for all CUDA kernels used in GPU acceleration.
 * Implementation in gpu_kernels.cu
 * 
 * @author BTC GOLD Development Team
 * @version 6.0
 * @date January 2026
 */

#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

#include <cstdint>
#include <cuda_runtime.h>
#include "types.h"

namespace BTCGold {
namespace GPU {
namespace Kernels {

// ============================================================================
// ECDSA POINT MULTIPLICATION KERNELS
// ============================================================================

/**
 * @brief CUDA kernel: Point multiplication (private key → public key)
 * 
 * This is the critical path kernel. Optimized for maximum throughput.
 * Uses batched operations with cooperative groups for synchronization.
 * 
 * @param private_keys Array of 256-bit private keys (host)
 * @param public_keys_x Array of public key X coordinates (output)
 * @param public_keys_y Array of public key Y coordinates (output)
 * @param batch_size Number of keys to process
 * @param stream CUDA stream for async execution
 */
__global__ void kernel_ecdsa_point_multiply(
    const uint32_t* private_keys,
    uint32_t* public_keys_x,
    uint32_t* public_keys_y,
    size_t batch_size
);

/**
 * @brief Batched point multiplication with shared memory optimization
 * 
 * Version optimized for batch processing with reduced global memory access.
 * 
 * @param private_keys_batch Batch of private keys
 * @param public_x_batch Output X coordinates
 * @param public_y_batch Output Y coordinates
 * @param batch_size Keys in this batch
 * @param batch_offset Offset in larger dataset
 */
__global__ void kernel_ecdsa_batch_multiply(
    const uint32_t* private_keys_batch,
    uint32_t* public_x_batch,
    uint32_t* public_y_batch,
    size_t batch_size,
    size_t batch_offset
);

/**
 * @brief Compressed point generation kernel
 * 
 * Converts uncompressed public keys to compressed format (33 bytes).
 * 
 * @param public_keys_x Uncompressed X coordinates
 * @param public_keys_y Uncompressed Y coordinates
 * @param compressed_keys Output compressed keys (33 bytes each)
 * @param batch_size Number of keys
 */
__global__ void kernel_compress_pubkeys(
    const uint32_t* public_keys_x,
    const uint32_t* public_keys_y,
    uint8_t* compressed_keys,
    size_t batch_size
);

// ============================================================================
// HASH160 KERNELS (SHA256 + RIPEMD160)
// ============================================================================

/**
 * @brief CUDA kernel: SHA256 hash computation
 * 
 * Computes SHA256 hash of 33-byte compressed public keys.
 * Uses shared memory precomputed K constants.
 * 
 * @param input_data Compressed public keys (33 bytes each)
 * @param sha256_output SHA256 hashes (32 bytes each)
 * @param batch_size Number of hashes to compute
 */
__global__ void kernel_sha256(
    const uint8_t* input_data,
    uint8_t* sha256_output,
    size_t batch_size
);

/**
 * @brief CUDA kernel: RIPEMD160 hash computation
 * 
 * Computes RIPEMD160(SHA256(pubkey)) → 20-byte hash160.
 * This is the final Bitcoin address identifier.
 * 
 * @param sha256_input 32-byte SHA256 hashes
 * @param hash160_output 20-byte hash160 output
 * @param batch_size Number of hashes to compute
 */
__global__ void kernel_ripemd160(
    const uint8_t* sha256_input,
    uint8_t* hash160_output,
    size_t batch_size
);

/**
 * @brief Combined Hash160 kernel (SHA256 + RIPEMD160 pipeline)
 * 
 * Fused kernel that computes both hashes in one kernel launch.
 * Reduces memory round-trips.
 * 
 * @param compressed_pubkeys Compressed public keys (33 bytes)
 * @param hash160_output Final hash160 (20 bytes)
 * @param batch_size Number of keys
 */
__global__ void kernel_hash160_combined(
    const uint8_t* compressed_pubkeys,
    uint8_t* hash160_output,
    size_t batch_size
);

// ============================================================================
// DATABASE MATCHING KERNELS
// ============================================================================

/**
 * @brief CUDA kernel: Database matching with linear search
 * 
 * Searches hash160 values against target database.
 * Returns matching indices.
 * 
 * @param hash160_values Input hashes (20 bytes each)
 * @param database_targets Target hashes (20 bytes each)
 * @param database_size Number of targets
 * @param matches Output matches (index pairs)
 * @param match_count Number of matches found
 * @param batch_size Number of input hashes
 */
__global__ void kernel_database_match_linear(
    const uint8_t* hash160_values,
    const uint8_t* database_targets,
    size_t database_size,
    uint32_t* matches,
    uint32_t* match_count,
    size_t batch_size
);

/**
 * @brief CUDA kernel: Database matching with binary search
 * 
 * Requires sorted database. More efficient for large databases.
 * 
 * @param hash160_values Input hashes to search
 * @param sorted_database Sorted database of targets
 * @param database_size Database size
 * @param matches Output match results
 * @param batch_size Number of input hashes
 */
__global__ void kernel_database_match_binary(
    const uint8_t* hash160_values,
    const uint8_t* sorted_database,
    size_t database_size,
    uint32_t* matches,
    size_t batch_size
);

/**
 * @brief CUDA kernel: Texture memory based matching
 * 
 * Optimized for GPU with large L2 cache and texture cache.
 * 
 * @param hash160_values Input hashes
 * @param matches Match results
 * @param batch_size Number of hashes
 */
__global__ void kernel_database_match_texture(
    const uint8_t* hash160_values,
    uint32_t* matches,
    size_t batch_size
);

// ============================================================================
// SEARCH MODE KERNELS
// ============================================================================

/**
 * @brief Kernel: LINEAR mode key generation
 * 
 * Sequential key generation from start to start+count.
 * 
 * @param start_key Starting private key
 * @param key_increment Increment per thread
 * @param private_keys Output private keys
 * @param batch_size Keys to generate
 */
__global__ void kernel_mode_linear(
    const uint32_t* start_key,
    const uint32_t* key_increment,
    uint32_t* private_keys,
    size_t batch_size
);

/**
 * @brief Kernel: RANDOM mode (Xorshift128+ PRNG)
 * 
 * Generates random keys using thread-local PRNG state.
 * 
 * @param seed_state PRNG seed state per thread
 * @param private_keys Output random keys
 * @param batch_size Keys to generate
 */
__global__ void kernel_mode_random(
    uint64_t* seed_state,
    uint32_t* private_keys,
    size_t batch_size
);

/**
 * @brief Kernel: GEOMETRIC mode (3-phase intelligent search)
 * 
 * Implements geometric progression:
 * Phase 1: Linear range
 * Phase 2: Geometric expansion
 * Phase 3: High-order exploration
 * 
 * @param phase Current phase (0-2)
 * @param base_key Starting key
 * @param multiplier Phase multiplier
 * @param private_keys Output keys
 * @param batch_size Keys to generate
 */
__global__ void kernel_mode_geometric(
    int phase,
    const uint32_t* base_key,
    double multiplier,
    uint32_t* private_keys,
    size_t batch_size
);

/**
 * @brief Kernel: DOUBLING mode (power-of-2 progression)
 * 
 * Generates keys at 2^n positions.
 * 
 * @param exponents Exponent array for each key
 * @param base_key Starting key
 * @param private_keys Output keys
 * @param batch_size Keys to generate
 */
__global__ void kernel_mode_doubling(
    const uint32_t* exponents,
    const uint32_t* base_key,
    uint32_t* private_keys,
    size_t batch_size
);

// ============================================================================
// UTILITY KERNELS
// ============================================================================

/**
 * @brief Kernel: Batch deduplication
 * 
 * Removes duplicate keys from batch.
 * Uses atomics and shared memory.
 * 
 * @param input_keys Input key batch
 * @param output_keys Deduplicated output
 * @param output_count Output count
 * @param batch_size Input size
 */
__global__ void kernel_deduplicate_keys(
    const uint32_t* input_keys,
    uint32_t* output_keys,
    uint32_t* output_count,
    size_t batch_size
);

/**
 * @brief Kernel: Batch reduction (sum, count, etc)
 * 
 * Parallel reduction for statistics.
 * 
 * @param input_data Input array
 * @param output_result Reduction result
 * @param batch_size Array size
 */
__global__ void kernel_reduce_statistics(
    const uint64_t* input_data,
    uint64_t* output_result,
    size_t batch_size
);

/**
 * @brief Kernel: Memory pattern validation
 * 
 * Validates data integrity during transfer.
 * 
 * @param data Data to validate
 * @param pattern Expected pattern
 * @param size Data size
 * @param is_valid Output validation flag
 */
__global__ void kernel_validate_memory(
    const uint8_t* data,
    const uint8_t* pattern,
    size_t size,
    uint32_t* is_valid
);

// ============================================================================
// WRAPPER FUNCTIONS (for host code)
// ============================================================================

/**
 * @brief Wrapper to launch ECDSA kernel with error handling
 */
bool launch_ecdsa_kernel_wrapper(
    const uint32_t* d_privkeys,
    uint32_t* d_pubx,
    uint32_t* d_puby,
    size_t batch_size,
    cudaStream_t stream
);

/**
 * @brief Wrapper to launch Hash160 kernel
 */
bool launch_hash160_kernel_wrapper(
    const uint8_t* d_pubkeys,
    uint8_t* d_hashes,
    size_t batch_size,
    cudaStream_t stream
);

/**
 * @brief Wrapper to launch database matching kernel
 */
bool launch_database_kernel_wrapper(
    const uint8_t* d_hashes,
    const uint8_t* d_database,
    size_t db_size,
    uint32_t* d_matches,
    size_t batch_size,
    cudaStream_t stream
);

} // namespace Kernels
} // namespace GPU
} // namespace BTCGold

#endif // GPU_KERNELS_H
