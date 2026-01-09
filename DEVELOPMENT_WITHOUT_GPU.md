# 💻 Desenvolvimento sem GPU - Guia Completo

**Data**: Janeiro 2026  
**Versão**: BTC GOLD v5.1 + v6.0 GPU (Hybrid Architecture)  
**Classificação**: Enterprise Development  

---

## 📋 Situação

Você tem:
- ✅ Máquina de desenvolvimento (sem GPU)
- ✅ CUDA Toolkit instalado (ou não)
- ❌ Placa gráfica NVIDIA (ou driver não detectado)
- 🎯 Quer compilar e rodar o BTC GOLD na arquitetura atual

---

## 🎯 Solução: Compilação CPU-Only Híbrida

O projeto agora suporta **compilação dual-mode**:

| Modo | GPU Disponível? | Compilation | Runtime | Performance |
|------|-----------------|-------------|---------|-------------|
| **GPU v6** | ✅ SIM | `./build_gpu_v6.sh` | 500M keys/sec | 2.7x |
| **CPU Hybrid** | ❌ NÃO | `./build_cpu_only.sh` | 188M keys/sec | 1.0x (baseline) |
| **CPU-Only Force** | ✅ (ignorar) | `./build_cpu_only.sh` | 188M keys/sec | 1.0x |

---

## 🚀 Método 1: Quick Start (Recomendado)

### Passo 1: Clone o repositório

```bash
git clone https://github.com/Smoke-1989/btc_gold_cpp.git
cd btc_gold_cpp
git checkout feature/gpu-cuda-acceleration-v6
```

### Passo 2: Compile em modo CPU-Only

```bash
chmod +x build_cpu_only.sh
./build_cpu_only.sh
```

**O que acontece:**
- Cria pasta `build_cpu_only/`
- Compila **sem** GPU (sem dependências CUDA)
- Gera binário otimizado com `-O3 -march=native`
- Desativa todas as compilações `.cu` (CUDA)

### Passo 3: Execute

```bash
# Modo interativo (recomendado)
./build_cpu_only/bin/btc_gold

# Modo CLI
./build_cpu_only/bin/btc_gold --mode 0 --input targets.txt

# Ver ajuda
./build_cpu_only/bin/btc_gold --help
```

**Saída esperada:**
```
╔════════════════════════════════════╗
║ BTC GOLD C++ v5.1 PRODUCTION      ║
╚════════════════════════════════════╝

EXECUTION ENVIRONMENT
────────────────────────────────────
GPU Support:        NOT COMPILED
GPU Status:         DISABLED
Performance:        188M keys/sec (CPU)

To enable GPU acceleration:
  ./build_gpu_v6.sh -DENABLE_GPU=ON
  Or read: docs/GPU_ACCELERATION_V6.md
```

---

## 🔧 Método 2: Compilação Manual (Avançado)

Se você preferir controle total:

```bash
# 1. Crie diretório de build
mkdir build_cpu_only
cd build_cpu_only

# 2. Configure CMake sem GPU
cmake -DENABLE_GPU=OFF -CMAKE_BUILD_TYPE=Release ..

# 3. Compile
cmake --build . --config Release -j$(nproc)

# 4. Execute
./bin/btc_gold
```

### Opções CMake Úteis

```bash
# Compilação de debug (mais lento, mas com símbolos)
cmake -DENABLE_GPU=OFF -DCMAKE_BUILD_TYPE=Debug ..

# Apenas CPU, sanitizers ativados
cmake -DENABLE_GPU=OFF -DENABLE_SANITIZERS=ON ..

# CPU otimizado para processadores antigos
cmake -DENABLE_GPU=OFF -DCMAKE_CXX_FLAGS="-O3 -march=x86-64" ..
```

---

## 🔄 Integração Híbrida: Como Funciona

### Arquivo: `src/main_v5.cpp` (Novo)

O main foi reescrito com classe `HybridExecutor`:

```cpp
class HybridExecutor {
  - detect_gpu_availability()     // Procura GPU
  - has_gpu()                     // Retorna true/false
  - print_execution_mode()        // Mostra status
  - force_cpu()                   // Força CPU mesmo com GPU
};
```

### Fluxo de Detecção

```
┌─ Programa inicia
│
├─ HybridExecutor::detect_gpu_availability()
│   ├─ Se GPU support compilado (ENABLE_GPU=ON)
│   │   ├─ Tenta inicializar GPUEngine
│   │   ├─ Se sucesso: gpu_available = true
│   │   └─ Se falha: gpu_available = false
│   │
│   └─ Se GPU não compilado
│       └─ gpu_available = false (sempre)
│
├─ has_gpu() retorna status
│
└─ Executa com melhor engine disponível
    ├─ GPU (se available && !--cpu-only)
    └─ CPU (fallback ou força)
```

### Flags de Controle

```bash
# Força modo CPU mesmo com GPU disponível
./build_cpu_only/bin/btc_gold --cpu-only

# Benchmark GPU (se disponível)
./build_cpu_only/bin/btc_gold --benchmark-gpu

# Modo verboso (mostra status da GPU)
./build_cpu_only/bin/btc_gold --verbose
```

---

## ✅ Verificação: O Que Funciona

### Modos de Busca (Todos Funcionam)

```
✅ LINEAR       - Busca sequencial em range
✅ RANDOM       - Busca aleatória 256-bit
✅ GEOMETRIC    - Busca inteligente 3-fase
✅ TERMINATOR   - Progressão multiplicativa
✅ DOUBLING     - Potências de 2
✅ HAMMING      - Padrões low-weight
✅ MODULAR      - Progressão aritmética
✅ VANITY       - Matching de padrão
✅ ENTROPY      - Detecção weak entropy
✅ COLLISION    - Busca de endereços adjacentes
```

### Tipos de Input

```
✅ hash160      - Hash160 (20 bytes)
✅ hash256      - Hash256 (32 bytes)
✅ pubkey       - Chave pública descompactada
```

### Recursos de Sistema

```
✅ Multi-threading (--threads N)
✅ Stop on find (--stop-on-find)
✅ Verbose logging (--verbose)
✅ Interactive menu
✅ CLI automation
```

---

## 🎮 Exemplos de Uso

### Exemplo 1: Modo Interativo

```bash
$ ./build_cpu_only/bin/btc_gold

[Menu interativo aparece]
Escolha modo de busca: 0 (LINEAR)
Arquivo de entrada: targets.txt
Número de threads: 8
Start hex: 1
End hex: ffffffff...
```

### Exemplo 2: CLI - Busca LINEAR

```bash
./build_cpu_only/bin/btc_gold \
  --mode 0 \
  --input targets.txt \
  --threads 16 \
  --start-hex 1 \
  --end-hex ffffffff \
  --stop-on-find
```

### Exemplo 3: CLI - Busca GEOMETRIC

```bash
./build_cpu_only/bin/btc_gold \
  --mode 2 \
  --input targets.txt \
  --threads 12 \
  --min-range-bit 1 \
  --max-range-bit 256 \
  --verbose
```

### Exemplo 4: CLI - Busca RANDOM

```bash
./build_cpu_only/bin/btc_gold \
  --mode 1 \
  --input targets.txt \
  --threads 8 \
  --stop-on-find
```

---

## 📊 Performance Esperada

### CPU-Only (v5.1 Baseline)

```
Performance:        188M keys/sec
Por thread:         ~23-24M keys/sec (8 threads)
Memória RAM:        ~500 MB (modest)
Consume CPU:        100% (8 cores)
```

### Comparação com GPU

```
CPU (v5.1):         188M keys/sec (1.0x baseline)
GPU Single:         500M keys/sec (2.7x speedup)
GPU Dual:           1B keys/sec (5.3x speedup)
GPU 8x A100:        4B+ keys/sec (21x+ speedup)
```

---

## 🔀 Transição: CPU → GPU

Quando você tiver GPU disponível:

### Opção 1: Recompilar com GPU

```bash
# Limpe build anterior
rm -rf build_cpu_only

# Compile com GPU
./build_gpu_v6.sh

# Execute
./build_gpu_v6/bin/btc_gold_cuda --benchmark-gpu
```

### Opção 2: Manter Ambos os Binários

```bash
# Mantenha CPU build
./build_cpu_only/bin/btc_gold       # CPU (sempre)

# Adicione GPU build
./build_gpu_v6.sh
./build_gpu_v6/bin/btc_gold_cuda    # GPU (quando disponível)
```

---

## 🐛 Troubleshooting

### Problema: "CMake not found"

```bash
# Solução: Instale CMake
sudo apt-get install cmake  # Ubuntu/Debian
brew install cmake          # macOS
```

### Problema: "C++ compiler not found"

```bash
# Solução: Instale build tools
sudo apt-get install build-essential  # Ubuntu/Debian
xcode-select --install                 # macOS
```

### Problema: Build falha com erro obscuro

```bash
# Limpe build e tente novamente
rm -rf build_cpu_only
./build_cpu_only.sh --clean
```

### Problema: Execução muito lenta

```bash
# Aumente threads
./build_cpu_only/bin/btc_gold --threads 16

# Ou use modo mais eficiente
./build_cpu_only/bin/btc_gold --mode 2  # GEOMETRIC é mais eficiente
```

---

## 📚 Documentação Relacionada

- **[GPU_ACCELERATION_V6.md](docs/GPU_ACCELERATION_V6.md)** - Guia GPU completo
- **[README_GPU_v6.md](README_GPU_v6.md)** - Quick start GPU
- **[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)** - Build system
- **[GPU_IMPLEMENTATION_STATUS.md](GPU_IMPLEMENTATION_STATUS.md)** - Status técnico

---

## 💡 Dicas Profissionais

### Dica 1: Otimização de CPU

```bash
# Use mode GEOMETRIC para search inteligente
# Usa ~3x menos keys testadas vs LINEAR
./build_cpu_only/bin/btc_gold --mode 2
```

### Dica 2: Multi-threading

```bash
# Use threads = num_cores - 1 (deixe 1 free para sistema)
CPU_CORES=$(nproc)
THREADS=$((CPU_CORES - 1))
./build_cpu_only/bin/btc_gold --threads $THREADS
```

### Dica 3: Benchmark seu Sistema

```bash
# Crie small test file
echo "1234567890abcdef1234567890abcdef12345678" > tiny_test.txt

# Rode rápido test
./build_cpu_only/bin/btc_gold \
  --mode 1 \
  --input tiny_test.txt \
  --threads 8
```

### Dica 4: Logs para Análise

```bash
# Capture output
./build_cpu_only/bin/btc_gold \
  --mode 0 \
  --input targets.txt \
  --verbose 2>&1 | tee search.log
```

---

## 🎓 Arquitetura Técnica

### Compilação Condicional

```cpp
// src/main_v5.cpp
#ifdef ENABLE_GPU
    #include "gpu_engine.h"
    #define HAS_GPU_SUPPORT 1
#else
    #define HAS_GPU_SUPPORT 0
#endif
```

### CMake Integration

```cmake
# CMakeLists_GPU.txt
if(ENABLE_GPU)
    enable_language(CUDA)
    add_subdirectory(src/cuda)
    target_link_libraries(btc_gold ${CUDA_LIBRARIES})
else()
    message(STATUS "GPU support disabled")
endif()
```

---

## ✅ Próximos Passos

1. **Agora (CPU-only)**
   - ✅ Compile com `./build_cpu_only.sh`
   - ✅ Teste todos os 10 modos
   - ✅ Familiarize-se com o código

2. **Quando tiver GPU**
   - 📈 Recompile com `./build_gpu_v6.sh`
   - 📈 Compare performance (2.7x mais rápido)
   - 📈 Deploy em produção

3. **Para Escalabilidade**
   - 🚀 Cloud GPU (AWS p3, Google Cloud)
   - 🚀 Multi-GPU em servidor
   - 🚀 Docker + Kubernetes

---

## 📞 Suporte

Para issues específicas:

1. Consulte `docs/GPU_ACCELERATION_V6.md` para GPU
2. Verifique `README_GPU_v6.md` para setup
3. Leia inline comments em `src/main_v5.cpp`
4. Abra issue no GitHub

---

**Status**: ✅ Pronto para Desenvolvimento em CPU  
**Última Atualização**: 9 de Janeiro de 2026  
**Versão**: v6.0 Hybrid Architecture
