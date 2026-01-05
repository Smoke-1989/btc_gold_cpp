#pragma once

#include <cstdint>
#include <array>
#include <string>

namespace btc_gold {

// ============================================================================
// FUNDAMENTAL TYPES
// ============================================================================

using Hash160 = std::array<uint8_t, 20>;      // 160-bit hash
using PrivateKey = std::array<uint8_t, 32>;   // 256-bit private key
using PublicKey = std::array<uint8_t, 33>;    // 33-byte compressed public key

// ============================================================================
// KEY RESULT
// ============================================================================

struct KeyResult {
    PrivateKey privkey;
    Hash160 hash160;
    std::string address;
    std::string wif_compressed;
    std::string wif_uncompressed;
    bool found = false;
    int scan_mode = 0;  // 1=compressed, 2=uncompressed, 3=both
};

// ============================================================================
// CONFIGURATION v4.0 EXTERMINATOR
// ============================================================================

struct Config {
    // ========================================================================
    // OPERATION MODES (1-7)
    // ========================================================================
    enum Mode { 
        LINEAR = 1,              // Seqüêncial (+1, +1, +1...)
        RANDOM = 2,              // Aleatório puro
        GEOMETRIC = 3,            // 3 fases: Border, Ceiling, Hamming
        TERMINATOR = 4,           // Multiplicação regressiva
        DOUBLING = 5,             // v4.0: Multiplicação por 2 (logaritmo)
        HAMMING = 6,              // v4.0: Baixo peso de bits (2-3 bits)
        MODULAR_STRIDE = 7        // v4.0: Padrão customizado (k+m, k+2m...)
    };
    Mode mode = LINEAR;
    
    // ========================================================================
    // DATABASE INPUT TYPE
    // ========================================================================
    enum InputType { 
        ADDRESS = 1,      // Endereços Bitcoin (1A1z...)
        HASH160 = 2,      // HASH160 hex puro (40 chars)
        PUBKEY = 3        // Public Keys (02/03/04...)
    };
    InputType input_type = ADDRESS;
    
    // ========================================================================
    // SCAN MODE (Comprimido/Descomprimido)
    // ========================================================================
    enum ScanMode { 
        COMPRESSED = 1,        // 33 bytes (02/03 prefix)
        UNCOMPRESSED = 2,      // 65 bytes (04 prefix)
        BOTH = 3               // Ambos (mais lento)
    };
    ScanMode scan_mode = COMPRESSED;
    
    // ========================================================================
    // PARÂMETROS DE RANGE
    // ========================================================================
    uint64_t start_value = 1;              // Início do range (mode LINEAR/RANDOM)
    uint64_t end_value = 0xFFFFFFFFFFFFFF;  // Fim do range
    
    int range_min_bit = 1;    // Bit mínimo (modo GEOMETRIC/HAMMING/DOUBLING)
    int range_max_bit = 256;   // Bit máximo
    
    uint64_t multiplier = 2;   // Fator multiplicador (TERMINATOR/DOUBLING/MODULAR_STRIDE)
    uint64_t stride = 1;       // Passo entre threads (LINEAR)
    
    // ========================================================================
    // THREADING & PERFORMANCE
    // ========================================================================
    int num_threads = 0;       // 0 = auto-detect
    bool turbo_mode = true;    // v4.0: Ativa otimização agressiva no LINEAR
    bool batch_write = true;   // v4.0: Escreve hits em lote (menos mutex)
    
    // ========================================================================
    // COMPORTAMENTO
    // ========================================================================
    bool stop_on_find = false;  // Para ao encontrar primeira chave
    bool verbose = true;        // Logs detalhados
    bool use_gpu = false;       // v4.0: Usa GPU se disponível (futuro)
    
    // ========================================================================
    // ARQUIVOS
    // ========================================================================
    std::string database_file = "alvos.txt";
    std::string output_file = "found_gold.txt";
};

}  // namespace btc_gold
