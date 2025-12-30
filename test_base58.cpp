#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <openssl/sha.h>

static const std::string BASE58_ALPHABET = 
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

static std::vector<uint8_t> decode_base58(const std::string& encoded) {
    // Count leading zeros (represented as '1' in base58)
    size_t leading_zeros = 0;
    for (char c : encoded) {
        if (c == '1') leading_zeros++;
        else break;
    }
    
    // Allocate enough space for the result (maximum size)
    std::vector<uint8_t> result(leading_zeros + (encoded.length() * 733 / 1000) + 1, 0);
    
    // Process each base58 character
    for (char c : encoded) {
        size_t digit = BASE58_ALPHABET.find(c);
        if (digit == std::string::npos) {
            throw std::runtime_error("Invalid base58 character");
        }
        
        // Multiply the whole array by 58 and add the new digit
        int carry = static_cast<int>(digit);
        for (int i = result.size() - 1; i >= 0; --i) {
            carry += result[i] * 58;
            result[i] = carry & 0xFF;
            carry >>= 8;
        }
        
        if (carry > 0) {
            throw std::runtime_error("Base58 decode overflow");
        }
    }
    
    // Find the first non-zero byte
    auto first_non_zero = std::find_if(result.begin(), result.end(), 
                                       [](uint8_t b) { return b != 0; });
    
    // Skip leading zeros in the decoded result but keep the original leading zeros
    std::vector<uint8_t> final_result(leading_zeros, 0);
    final_result.insert(final_result.end(), first_non_zero, result.end());
    
    return final_result;
}

int main() {
    // Known valid Bitcoin addresses
    std::vector<std::string> test_addresses = {
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",  // Genesis block (Satoshi)
        "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",  // Known address
        "1dice8EMZqgUvrHKwcW1gimzhyT5sq64J",   // SatoshiDice
        "12c6DSiU4Rq3P4ZxziKxzrL5LmMBrzjrJX",  // Another known
    };
    
    int passed = 0;
    int failed = 0;
    
    for (const auto& address : test_addresses) {
        std::cout << "\nTesting: " << address << std::endl;
        
        try {
            auto decoded = decode_base58(address);
            
            std::cout << "  Decoded size: " << decoded.size() << " bytes" << std::endl;
            std::cout << "  Decoded hex: ";
            for (auto byte : decoded) {
                printf("%02x", byte);
            }
            std::cout << std::endl;
            
            if (decoded.size() != 25) {
                std::cout << "  ERROR: Expected 25 bytes!" << std::endl;
                failed++;
                continue;
            }
            
            // Verify checksum
            std::vector<uint8_t> payload(decoded.begin(), decoded.end() - 4);
            std::vector<uint8_t> checksum(decoded.end() - 4, decoded.end());
            
            unsigned char sha256_1[SHA256_DIGEST_LENGTH];
            unsigned char sha256_2[SHA256_DIGEST_LENGTH];
            
            SHA256(payload.data(), payload.size(), sha256_1);
            SHA256(sha256_1, SHA256_DIGEST_LENGTH, sha256_2);
            
            bool checksum_valid = (std::vector<uint8_t>(sha256_2, sha256_2 + 4) == checksum);
            std::cout << "  Checksum: " << (checksum_valid ? "\033[32mVALID\033[0m" : "\033[31mINVALID\033[0m") << std::endl;
            
            if (checksum_valid) {
                std::cout << "  Hash160: ";
                for (size_t i = 1; i < 21; i++) {
                    printf("%02x", decoded[i]);
                }
                std::cout << std::endl;
                passed++;
            } else {
                failed++;
            }
            
        } catch (const std::exception& e) {
            std::cout << "  ERROR: " << e.what() << std::endl;
            failed++;
        }
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    
    if (passed == test_addresses.size()) {
        std::cout << "\033[32m✓ ALL TESTS PASSED!\033[0m" << std::endl;
        return 0;
    } else {
        std::cout << "\033[31m✗ SOME TESTS FAILED\033[0m" << std::endl;
        return 1;
    }
}
