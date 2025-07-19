# Detector de Números por Gestos de Mão com IA

Este projeto utiliza visão computacional e aprendizado de máquina para reconhecer gestos de mão que representam números de 0 a 5, usando a webcam do computador. O sistema foi desenvolvido em Python, utilizando OpenCV, MediaPipe e scikit-learn.

## Sumário
- [Descrição do Projeto](#descrição-do-projeto)
- [Funcionamento do Código](#funcionamento-do-código)
- [Treinamento do Modelo](#treinamento-do-modelo)
- [Como Executar](#como-executar)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Referências](#referências)

---

## Descrição do Projeto
O objetivo é detectar gestos de mão capturados pela webcam e classificá-los como números de 0 a 5. O projeto utiliza:
- **MediaPipe** para detecção dos pontos de referência (landmarks) da mão.
- **OpenCV** para captura e exibição das imagens da webcam.
- **scikit-learn** para treinar e utilizar o modelo de classificação dos gestos.

## Funcionamento do Código

### 1. Captura e Processamento da Imagem
O arquivo `main.py` faz a captura da imagem da webcam, processa com MediaPipe para detectar a mão e extrai os landmarks.

### 2. Extração de Features
A função `extrair_landmarks` (em `utils/feature_extractor.py`) transforma os landmarks em um vetor de características (features) que representa a posição dos pontos da mão.

### 3. Classificação do Gesto
O vetor de características é passado para o modelo treinado (carregado em `classify.py`), que retorna o número detectado e a confiança da predição.

### 4. Exibição do Resultado
O resultado é exibido na tela, sobreposto à imagem da webcam, e também impresso no console.

## Treinamento do Modelo

O treinamento do modelo é realizado em `model/treinar_modelo.py`:
1. **Coleta de Dados:**
   - Os dados dos gestos são coletados usando o script `utils/coletar_dados.py`, que salva os landmarks em arquivos CSV na pasta `data/`.
   - Cada arquivo representa um gesto (ex: `gesto_0.csv` para o número 0).
2. **Preparação dos Dados:**
   - Os arquivos CSV são lidos e combinados em um único dataset.
   - Os dados são rotulados conforme o gesto correspondente.
3. **Treinamento:**
   - Um modelo de classificação (Nesse projeto foi utilizado o SVM.) é treinado usando scikit-learn.
   - O modelo treinado é salvo em `model/classificador_gestos.pkl` usando joblib.

## Estrutura do Projeto
```
├── main.py                  # Código principal para detecção em tempo real
├── classify.py              # Carrega o modelo e faz a classificação
├── requirements.txt         # Dependências do projeto
├── README.md                # Documentação
├── data/                    # Dados coletados dos gestos
│   ├── gesto_0.csv
│   └── ...
├── model/
│   ├── classificador_gestos.pkl  # Modelo treinado
│   └── treinar_modelo.py         # Script de treinamento
└── utils/
    ├── coletar_dados.py          # Script para coleta de dados
    └── feature_extractor.py      # Função para extrair features dos landmarks
```

## Como Executar

### 1. Instale as dependências
Abra o terminal na pasta do projeto e execute:
```powershell
pip install -r requirements.txt
```

### 2. Treine o modelo (opcional, se quiser criar um novo modelo)
```powershell
python model/treinar_modelo.py
```

### 3. Execute o detector de gestos
```powershell
python main.py
```

A webcam será ativada e os gestos serão detectados em tempo real. Para sair, pressione `q`.

## Explicação Detalhada dos Arquivos

### main.py
- Inicializa MediaPipe e OpenCV.
- Captura frames da webcam.
- Detecta landmarks da mão.
- Extrai features e classifica o gesto.
- Exibe o resultado na tela.

### classify.py
- Carrega o modelo treinado (`classificador_gestos.pkl`).
- Função `classificar_gesto` recebe os landmarks e retorna o número do gesto e a confiança.

### utils/feature_extractor.py
- Função para transformar os landmarks em vetor de características.

### utils/coletar_dados.py
- Script para coletar dados dos gestos e salvar em CSV.

### model/treinar_modelo.py
- Lê os dados dos gestos.
- Treina o modelo de classificação.
- Salva o modelo treinado.

## Referências
- [Repositório original no GitHub](https://github.com/bitguardian/hand-gesture-ai.git)
- [Documentação MediaPipe](https://google.github.io/mediapipe/solutions/hands.html)
- [Documentação OpenCV](https://docs.opencv.org/)
- [Documentação scikit-learn](https://scikit-learn.org/)

---

Projeto desenvolvido para fins educacionais e demonstração de IA aplicada à visão computacional.
