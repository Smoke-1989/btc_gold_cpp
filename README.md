# 🏆 BTC GOLD C++ (v3.1 Hybrid Enterprise)

> **A Ferramenta Definitiva para Recuperação de Chaves e Puzzles Bitcoin**

O **BTC GOLD C++** é um software de alta performance desenvolvido para buscar chaves privadas de Bitcoin perdidas em grandes faixas numéricas (Puzzles). 

Diferente de scripts comuns em Python, esta ferramenta foi escrita em **C++ Moderno** e **CUDA (NVIDIA)**, utilizando instruções de processador de baixo nível (Assembly) para atingir velocidades extremas.

---

## 🚀 Principais Funcionalidades

*   **⚡ Modo "Exterminador" (CPU):** Utiliza instruções AVX2 e cálculos matemáticos otimizados (Point Addition) para varrer chaves sequenciais 1000x mais rápido que o método tradicional.
*   **☢️ Modo Híbrido (GPU NVIDIA):** Detecta automaticamente se você tem uma placa de vídeo NVIDIA e ativa o motor **CUDA Enterprise**, que utiliza aceleração gráfica para processar bilhões de chaves.
*   **🧠 Modos Inteligentes:** Além da força bruta, possui modos matemáticos (Terminator e Geometric) para estratégias de busca específicas.
*   **🎯 Zero Overhead:** Suporte a verificação direta de HASH160 (hexadecimal), eliminando conversões lentas de endereços de texto.

---

## 🛠️ Instalação

Siga os passos abaixo para preparar seu ambiente (Linux/Ubuntu).

### 1. Instalar Dependências Básicas
Abra o terminal e cole:
```bash
sudo apt update
sudo apt install -y cmake build-essential libssl-dev pkg-config git
```

### 2. (Opcional) Instalar Drivers NVIDIA
Se você tem uma placa de vídeo NVIDIA e quer usar o modo Turbo:
```bash
sudo apt install -y nvidia-cuda-toolkit
```
*Se não tiver placa NVIDIA, pule este passo. O programa funcionará normalmente usando a força máxima da CPU.*

### 3. Baixar e Compilar
```bash
# 1. Baixar o código
git clone https://github.com/Smoke-1989/btc_gold_cpp.git
cd btc_gold_cpp

# 2. Criar a pasta de construção
rm -rf build && mkdir build && cd build

# 3. Preparar e Compilar (Otimização Automática)
cmake ..
make -j$(nproc)
```

---

## 🎮 Guia de Modos (Estratégias)

O programa possui 4 modos de operação. Escolha o melhor para o seu objetivo:

### 1. 🏁 Modo LINEAR (`--mode linear`)
> **O "Pente Fino"**
*   **Como funciona:** Começa de um número e testa o próximo, e o próximo (+1, +1, +1...).
*   **Velocidade:** 🚀 **Extrema (50M+ chaves/s)**.
*   **Quando usar:** Quando você quer varrer um **Range Completo** (ex: Puzzle 66 inteiro) sem deixar nenhum buraco para trás.
*   **Recomendação:** É o melhor modo para a maioria dos casos.

### 2. 🎲 Modo RANDOM (`--mode random`)
> **A "Sorte"**
*   **Como funciona:** Sorteia números aleatórios dentro do intervalo que você escolheu.
*   **Velocidade:** Média.
*   **Quando usar:** Quando o intervalo é grande demais para ser varrido (ex: Bit 100+) e você quer contar com a probabilidade estatística.

### 3. 🤖 Modo TERMINATOR (`--mode terminator`)
> **O "Sniper Matemático"**
*   **Como funciona:** Busca chaves que são resultado de multiplicações matemáticas, descendo do topo do range.
*   **Velocidade:** Variável.
*   **Quando usar:** Para estratégias específicas onde se suspeita que a chave não é aleatória, mas sim fruto de uma conta matemática.
*   **Atenção:** Este modo **PULA** chaves. Não serve para varredura completa.

---

## 💻 Exemplos de Uso

Os comandos devem ser rodados de dentro da pasta `build`.

### Exemplo 1: Varredura Máxima no Puzzle 66 (Modo Linear)
Este é o comando ideal para varrer sequencialmente com velocidade máxima.
```bash
./btc_gold --mode linear --scan-mode 1 --threads 8 --input-type 2 --start 0x20000000000000000
```

### Exemplo 2: Tentando a Sorte no Bit 71 (Modo Random)
```bash
./btc_gold --mode random --range-min 71 --range-max 72
```

### Exemplo 3: Usando a GPU (Automático)
Basta rodar qualquer comando acima. Se o computador tiver uma NVIDIA, você verá no log:
`>>> GPU DETECTED: CUDA Hybrid Mode ENABLED <<<`

---

## ⚙️ Entendendo as Configurações (Flags)

| Flag | O que faz | Dica de Ouro |
| :--- | :--- | :--- |
| `--threads` ou `-t` | Define quantos núcleos do processador usar. | Deixe vazio para usar todos (automático). |
| `--scan-mode` | Define o tipo de endereço: <br> `1`: Comprimido (Novo)<br>`2`: Não-Comprimido (Antigo)<br>`3`: Ambos | Use **1** para Puzzles modernos. É 2x mais rápido que usar 3. |
| `--input-type` | Define como seu arquivo `alvos.txt` está escrito. | Use **2** (HASH160). Converter endereços para HASH160 deixa o programa muito mais leve. |
| `--database` | Escolhe outro arquivo de alvos. | Padrão: `alvos.txt` na pasta raiz. |

---

## 🏆 Dicas de Performance (Para Leigos)

1.  **Use HASH160:** Não coloque endereços começando com "1..." no seu arquivo de alvos. Converta-os para hexadecimal. O computador lê isso instantaneamente.
2.  **Filtre o Tipo:** Se você sabe que a carteira é moderna, use `--scan-mode 1`. Se não souber, use `3`, mas saiba que a velocidade cai pela metade.
3.  **Não abra o navegador:** Enquanto o programa roda, ele usa 100% da sua máquina. Abrir vídeos ou jogos vai diminuir a velocidade de busca.

---

## ⚠️ Aviso Legal

Este software é uma ferramenta de análise matemática e criptográfica. É de inteira responsabilidade do usuário garantir que possui autorização para recuperar as chaves alvo. O desenvolvedor não se responsabiliza pelo uso indevido da ferramenta.

---
*Desenvolvido com tecnologia V3.1 Hybrid Engine.*
