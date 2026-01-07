#pragma once

#include "types.h"
#include <cstdint>
#include <array>
#include <string>
#include <cstring>

namespace btc_gold {

// ============================================================================
// BIGINT256 - 256-bit unsigned integer arithmetic
// Enterprise-grade implementation for Bitcoin key range operations
// ============================================================================

class BigInt256 {
public:
    // Internal representation: 4x 64-bit limbs (little-endian)
    std::array<uint64_t, 4> limbs;
    
    // ========================================================================
    // CONSTRUCTORS
    // ========================================================================
    
    BigInt256() : limbs{0, 0, 0, 0} {}
    
    explicit BigInt256(uint64_t val) : limbs{val, 0, 0, 0} {}
    
    explicit BigInt256(const PrivateKey& key) {
        // Convert 32-byte big-endian to 4x 64-bit little-endian limbs
        for (int i = 0; i < 4; i++) {
            limbs[i] = 0;
            for (int j = 0; j < 8; j++) {
                int byte_idx = 31 - (i * 8 + j); // Big-endian
                limbs[i] |= (static_cast<uint64_t>(key[byte_idx]) << (j * 8));
            }
        }
    }
    
    // ========================================================================
    // CONVERSION
    // ========================================================================
    
    // Convert to PrivateKey (32 bytes, big-endian)
    void to_privkey(PrivateKey& out) const {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                int byte_idx = 31 - (i * 8 + j); // Big-endian
                out[byte_idx] = static_cast<uint8_t>((limbs[i] >> (j * 8)) & 0xFF);
            }
        }
    }
    
    // Convert to hex string
    std::string to_hex() const {
        PrivateKey key;
        to_privkey(key);
        
        std::string result;
        result.reserve(64);
        const char* hex_chars = "0123456789abcdef";
        
        for (int i = 0; i < 32; i++) {
            result += hex_chars[key[i] >> 4];
            result += hex_chars[key[i] & 0xF];
        }
        
        return result;
    }
    
    // ========================================================================
    // COMPARISON
    // ========================================================================
    
    bool operator<(const BigInt256& other) const {
        // Compare from most significant to least
        for (int i = 3; i >= 0; i--) {
            if (limbs[i] < other.limbs[i]) return true;
            if (limbs[i] > other.limbs[i]) return false;
        }
        return false; // Equal
    }
    
    bool operator<=(const BigInt256& other) const {
        return *this < other || *this == other;
    }
    
    bool operator>(const BigInt256& other) const {
        return other < *this;
    }
    
    bool operator>=(const BigInt256& other) const {
        return other <= *this;
    }
    
    bool operator==(const BigInt256& other) const {
        return limbs[0] == other.limbs[0] &&
               limbs[1] == other.limbs[1] &&
               limbs[2] == other.limbs[2] &&
               limbs[3] == other.limbs[3];
    }
    
    bool operator!=(const BigInt256& other) const {
        return !(*this == other);
    }
    
    // Check if zero
    bool is_zero() const {
        return limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0;
    }
    
    // ========================================================================
    // ARITHMETIC - Addition
    // ========================================================================
    
    BigInt256& operator+=(uint64_t rhs) {
        uint64_t carry = rhs;
        
        for (int i = 0; i < 4 && carry > 0; i++) {
            uint64_t old = limbs[i];
            limbs[i] += carry;
            carry = (limbs[i] < old) ? 1 : 0; // Detect overflow
        }
        
        return *this;
    }
    
    BigInt256& operator+=(const BigInt256& rhs) {
        uint64_t carry = 0;
        
        for (int i = 0; i < 4; i++) {
            uint64_t old = limbs[i];
            limbs[i] += rhs.limbs[i] + carry;
            
            // Carry if: overflow from addition OR overflow from carry
            carry = (limbs[i] < old) || (carry && limbs[i] == old) ? 1 : 0;
        }
        
        return *this;
    }
    
    BigInt256 operator+(const BigInt256& rhs) const {
        BigInt256 result = *this;
        result += rhs;
        return result;
    }
    
    BigInt256 operator+(uint64_t rhs) const {
        BigInt256 result = *this;
        result += rhs;
        return result;
    }
    
    // Pre-increment
    BigInt256& operator++() {
        *this += 1;
        return *this;
    }
    
    // Post-increment
    BigInt256 operator++(int) {
        BigInt256 tmp = *this;
        ++(*this);
        return tmp;
    }
    
    // ========================================================================
    // ARITHMETIC - Subtraction
    // ========================================================================
    
    BigInt256& operator-=(uint64_t rhs) {
        uint64_t borrow = rhs;
        
        for (int i = 0; i < 4 && borrow > 0; i++) {
            uint64_t old = limbs[i];
            limbs[i] -= borrow;
            borrow = (limbs[i] > old) ? 1 : 0; // Detect underflow
        }
        
        return *this;
    }
    
    BigInt256& operator-=(const BigInt256& rhs) {
        uint64_t borrow = 0;
        
        for (int i = 0; i < 4; i++) {
            uint64_t old = limbs[i];
            limbs[i] -= rhs.limbs[i] + borrow;
            
            borrow = (limbs[i] > old) || (borrow && limbs[i] == old) ? 1 : 0;
        }
        
        return *this;
    }
    
    BigInt256 operator-(const BigInt256& rhs) const {
        BigInt256 result = *this;
        result -= rhs;
        return result;
    }
    
    // ========================================================================
    // ARITHMETIC - Multiplication (BigInt256 * uint64_t)
    // ========================================================================

    BigInt256 operator*(uint64_t rhs) const {
        BigInt256 result;
        uint64_t carry = 0;

        for (int i = 0; i < 4; i++) {
            __uint128_t prod = static_cast<__uint128_t>(limbs[i]) * rhs + carry;
            result.limbs[i] = static_cast<uint64_t>(prod);
            carry = static_cast<uint64_t>(prod >> 64);
        }

        // Note: Carry at the end is discarded (overflow for 256-bit result)
        // This is expected behavior for fixed-width arithmetic
        return result;
    }

    // ========================================================================
    // ARITHMETIC - Division (for range splitting)
    // ========================================================================
    
    BigInt256 operator/(uint64_t divisor) const {
        if (divisor == 0) {
            // Return zero on division by zero (safe fallback)
            return BigInt256();
        }
        
        BigInt256 result;
        uint64_t remainder = 0;
        
        // Long division from most significant to least
        for (int i = 3; i >= 0; i--) {
            // Combine remainder with current limb
            __uint128_t dividend = (static_cast<__uint128_t>(remainder) << 64) | limbs[i];
            result.limbs[i] = static_cast<uint64_t>(dividend / divisor);
            remainder = static_cast<uint64_t>(dividend % divisor);
        }
        
        return result;
    }
    
    // ========================================================================
    // UTILITY
    // ========================================================================
    
    // Get approximate value as double (for progress reporting)
    double to_double() const {
        double result = 0.0;
        double multiplier = 1.0;
        
        for (int i = 0; i < 4; i++) {
            result += static_cast<double>(limbs[i]) * multiplier;
            multiplier *= 18446744073709551616.0; // 2^64
        }
        
        return result;
    }
};

}  // namespace btc_gold
