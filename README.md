# Detecção Automatizada de Embarcações de Garimpo Ilegal usando Imagens SAR e Redes Neurais Profundas

Este repositório contém o código-fonte, a estrutura de dados e os recursos de desenvolvimento do **Trabalho de Conclusão de Curso (TCC)** em Ciência da Computação, apresentado ao Instituto de Ciências Exatas e Naturais da **Universidade Federal do Pará (UFPA)**.

## Objetivo

O objetivo principal deste projeto é a implementação e validação de um pipeline computacional baseado em **Deep Learning** para a monitorização e contagem automatizada de balsas e dragas de garimpo ilegal na calha do **Rio Madeira** (corredor Rondônia–Amazonas), utilizando imagens de Radar de Abertura Sintética (**SAR**) do satélite **Sentinel-1**.

---

## Estrutura do Repositório

```text
├── data/
│   ├── gabarito_2025_06.geojson      # Marcações de referência (Junho)
│   ├── gabarito_2025_07.geojson      # Marcações de referência (Julho)
│   └── gabarito_2025_08.geojson      # Marcações de referência (Validação - Agosto)
│
├── notebooks/
│   └── deteccao_garimpo_sar.ipynb  # Notebook principal com o pipeline completo
│
└── README.md
```

> **Nota:** Devido a limitações de armazenamento do GitHub, as imagens SAR originais (`.tif`) e os pesos treinados do modelo (`.pth`) não estão incluídos diretamente neste repositório. Esses recursos são geridos externamente através do Google Drive.

---

## Recursos Externos (Download)

Devido ao tamanho dos arquivos, o dataset completo (imagens SAR, gabaritos de referência e modelo treinado) é disponibilizado através de uma pasta compartilhada no Google Drive.

### Download dos Dados e Modelo

Clique no link abaixo para acessar a pasta compartilhada:

**[BAIXAR DADOS E MODELO TREINADO – GOOGLE DRIVE]([INSIRA_AQUI_O_LINK_DO_DRIVE](https://drive.google.com/drive/folders/1CrR6bZwz-VOAD_6aB_eJR7c_lw8040SB?usp=drive_link))**

O conteúdo disponibilizado inclui:

- Imagens SAR originais utilizadas no estudo (`.tif`);
- Arquivos de referência (`.geojson`);
- Modelo treinado (`unet_resnet34.pth`);
- Estrutura de diretórios compatível com o notebook.
---

## Metodologia

O sistema utiliza uma arquitetura de segmentação semântica **U-Net** com encoder **ResNet34**, inicializada com pesos pré-treinados da **ImageNet**.

### 1. Recorte de Imagens

As imagens SAR originais (polarização **VV**, modo **IW**) são divididas em blocos de **256×256 pixels** com sobreposição espacial (*stride*) de **128 pixels**.

### 2. Amostragem Negativa

Foram incluídas regiões propositalmente livres de atividade garimpeira (por exemplo, trechos do rio próximos a Humaitá–AM), permitindo que a rede aprendesse a distinguir embarcações de elementos naturais que frequentemente geram falsos positivos, como bancos de areia expostos durante a estiagem.

### 3. Função de Perda Híbrida

Para lidar com o forte desbalanceamento entre classes, foi utilizada uma combinação ponderada de:

```python
Loss = 0.5 × BCE + 0.5 × Dice Loss
```

### 4. Pós-processamento

* Test Time Augmentation (TTA);
* Reconstrução da cena completa a partir dos blocos preditos;
* Suavização gaussiana para reduzir artefatos nas bordas;
* Contagem final das embarcações em nível de cena inteira.

---

## Resultados Obtidos

Os experimentos demonstraram que a arquitetura **ResNet34** apresentou maior capacidade de generalização frente ao ruído granular (*speckle*) característico das imagens SAR quando comparada à **ResNet50**, que apresentou tendência mais acentuada ao sobreajuste (*overfitting*).

### Métricas de Validação (Cena Inteira – Agosto/2025)

| Métrica                  | Valor     |
| ------------------------ | --------- |
| **F1-Score**             | **0.841** |
| Precisão (*Precision*)   | 0.763     |
| Sensibilidade (*Recall*) | 0.935     |
| IoU de Pixel             | 0.4857    |
| Limiar de Confiança      | 0.40      |

### Destaque

O modelo foi capaz de localizar **29 das 31 embarcações reais**, atingindo uma taxa de captura de aproximadamente **93,5%**.

---

## Como Reproduzir o Projeto?

O notebook foi desenvolvido para execução no **Google Colab** integrado ao **Google Drive**.

### Configuração Inicial

1. Acesse o link do Google Drive disponibilizado em "Recursos Externos (Downloads)".

2. Faça uma das seguintes opções:

   - Baixe a pasta **Dados_TCC_Garimpo** para sua conta;
   - Ou adicione a pasta compartilhada diretamente ao seu Google Drive.

3. Certifique-se de que a pasta esteja localizada na raiz do seu Google Drive:

```text
/content/drive/MyDrive/Dados_TCC_Garimpo
```

A estrutura esperada é:

```text
Dados_TCC_Garimpo/
├── imagens_sar_originais/
├── gabarito_2025_06.geojson
├── gabarito_2025_07.geojson
├── gabarito_2025_08.geojson
└── unet_resnet34.pth
```

> **Importante:** Os caminhos utilizados pelo notebook assumem que a pasta `Dados_TCC_Garimpo` está localizada diretamente na raiz do Google Drive. Caso ela seja movida para outro diretório, será necessário ajustar manualmente os caminhos definidos no código.

---

### Opção A — Inferência com Pesos Treinados

Recomendado para quem deseja apenas executar as detecções.

1. Baixe o arquivo:

```text
unet_resnet34.pth
```

2. Coloque o arquivo na pasta:

```text
Dados_TCC_Garimpo/
```

3. Abra o notebook:

```text
notebooks/deteccao_garimpo_sar.ipynb
```

4. Execute:

   * Célula 1
   * Célula 2
   * Pule a Célula 3 (Treinamento)
   * Execute as Células 4 e 5

O modelo carregará os pesos previamente treinados e gerará os mapas e relatórios de detecção.

---

### Opção B — Treinamento Completo

Para reproduzir integralmente os experimentos:

1. Abra o notebook no Google Colab;
2. Ative uma GPU (T4 ou superior);
3. Execute todas as células em sequência.

O treinamento é realizado por até **80 épocas**, utilizando:

* Otimizador **AdamW**;
* Early Stopping com **20 épocas de paciência**;
* Semente global:

```python
SEED = 42
```

> Pequenas diferenças na terceira ou quarta casa decimal podem ocorrer devido a variações de hardware entre GPUs disponibilizadas pelo Google Colab.

---

## Tecnologias Utilizadas

### Deep Learning

* PyTorch
* Segmentation Models PyTorch (SMP)

### Processamento Geoespacial

* Rasterio
* GeoPandas
* Shapely

### Data Augmentation

* Albumentations

### Análise e Visualização

* NumPy
* SciPy
* Matplotlib

Todas as dependências são instaladas automaticamente na primeira célula do notebook.

---

## Autor

**Fellipe Machado Castro**

Graduando em Ciência da Computação

**Universidade Federal do Pará (UFPA)**

[fellipemachado77@gmail.com](mailto:fellipemachado77@gmail.com)

---

## Licença

Este projeto foi desenvolvido exclusivamente para fins acadêmicos como parte do Trabalho de Conclusão de Curso (TCC) em Ciência da Computação da Universidade Federal do Pará (UFPA).
