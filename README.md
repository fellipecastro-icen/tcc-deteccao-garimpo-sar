# Detecção Automatizada de Embarcações de Garimpo Ilegal usando Imagens SAR e Redes Neurais Profundas

Este repositório contém o código-fonte, a estrutura de dados e os recursos de desenvolvimento do **Trabalho de Conclusão de Curso (TCC)** em Ciência da Computação, apresentado ao Instituto de Ciências Exatas e Naturais da **Universidade Federal do Pará (UFPA)**.

## Objetivo

O objetivo principal deste projeto é a implementação e validação de um pipeline computacional baseado em **Deep Learning** para a monitorização e contagem automatizada de embarcações de garimpo ilegal na calha do **Rio Madeira** (corredor Rondônia–Amazonas), utilizando imagens de Radar de Abertura Sintética (**SAR**) do satélite **Sentinel-1**.

---

## Estrutura do Repositório

```text
├── data/
│   ├── gabarito_2025_06.geojson          # Marcações de referência (Junho)
│   ├── gabarito_2025_07.geojson          # Marcações de referência (Julho)
│   └── gabarito_2025_08.geojson          # Marcações de referência (Validação - Agosto)
│
├── notebooks/
│   ├── 01_inferencia_garimpo_sar.ipynb   # Audita modelo pré-treinado e gera métricas por objeto
│   └── 02_treinamento_garimpo_sar.ipynb  # Pipeline completo de engenharia de dados e treinamento
│
└── README.md
```

> **Nota:** Devido a limitações de armazenamento do GitHub, as imagens SAR originais (`.tif`) e os pesos treinados do modelo (`.pth`) não estão incluídos diretamente neste repositório. Esses recursos são geridos externamente através do Google Drive.

---

## Recursos Externos (Download)

Devido ao tamanho dos arquivos, o dataset completo (imagens SAR, gabaritos de referência e modelos treinados) é disponibilizado através de uma pasta compartilhada no Google Drive.

### Download dos Dados e Modelos

Clique no link abaixo para acessar a pasta compartilhada:

**[BAIXAR DADOS E MODELO TREINADO – GOOGLE DRIVE](https://drive.google.com/drive/folders/1CrR6bZwz-VOAD_6aB_eJR7c_lw8040SB?usp=drive_link)**

O conteúdo disponibilizado inclui:

* Imagens SAR originais utilizadas no estudo (`.tif`);
* Arquivos de referência (`.geojson`);
* Modelos treinados (`unet_resnet34.pth` e `unet_resnet50.pth`).

---

## Metodologia

O sistema utiliza uma arquitetura de segmentação semântica **U-Net** com encoder **ResNet34**, inicializada com pesos pré-treinados da ImageNet.

### 1. Recorte de Imagens

As imagens SAR originais (polarização VV, modo IW) são divididas em blocos de **256 × 256 pixels** com sobreposição espacial (*stride*) de **128 pixels**.

### 2. Amostragem Negativa

Foram incluídas regiões propositalmente livres de atividade garimpeira (por exemplo, trechos do rio próximos a Humaitá–AM), permitindo que a rede aprendesse a distinguir embarcações de elementos naturais que frequentemente geram falsos positivos, como bancos de areia expostos durante a estação seca.

### 3. Função de Perda Híbrida

Para lidar com o forte desbalanceamento entre classes, foi utilizada uma combinação ponderada de:

```text
Loss = 0.5 × BCE + 0.5 × Dice Loss
```

### 4. Pós-processamento

* Test Time Augmentation (TTA);
* Reconstrução da cena completa a partir dos blocos preditos;
* Suavização gaussiana para reduzir artefatos nas bordas;
* Validação rigorosa em nível de objeto (contagem real de embarcações).

---

## Resultados Obtidos

Os experimentos demonstraram que a arquitetura **ResNet34** apresentou maior capacidade de generalização frente ao ruído granular (*speckle*) característico das imagens SAR quando comparada à **ResNet50**, que apresentou tendência mais acentuada ao sobreajuste (*overfitting*).

### Métricas Finais da Cena Completa (Validação – Agosto/2025)

Abaixo estão os resultados oficiais alcançados pela ResNet34 após a fusão dos recortes e filtragem morfológica do rio inteiro.

| Métrica                | Valor  |
| ---------------------- | ------ |
| F1-Score               | 0.841  |
| Precisão (Precision)   | 0.763  |
| Sensibilidade (Recall) | 0.935  |
| Falsos Positivos (FP)  | 9      |
| Limiar de Confiança    | 0.40   |

### Destaque

O modelo foi capaz de localizar **29 das 31 embarcações reais**, atingindo uma taxa de captura de aproximadamente **93,5%**.

---

## Como Reproduzir o Projeto?

Os algoritmos foram desenvolvidos para execução nativa no **Google Colab**, utilizando o **Google Drive** como repositório de dados.

Siga os passos abaixo para garantir o funcionamento correto dos caminhos de arquivo.

### Passo 1: Configuração dos Dados no Google Drive

1. Acesse o link do Google Drive disponibilizado acima;
2. Adicione um atalho da pasta `Dados_TCC_Garimpo` ou faça o download e upload para o seu próprio Drive;
3. Certifique-se de que a pasta esteja localizada exatamente na raiz do seu Drive.

Caminho esperado pelo código:

```text
/content/drive/MyDrive/Dados_TCC_Garimpo
```

### Passo 2: Execução dos Notebooks

Faça o download dos arquivos `.ipynb` da pasta `notebooks/` deste repositório GitHub e faça o upload deles diretamente na interface do Google Colab.

#### Opção A — Inferência Rápida (Recomendado)

Para quem deseja auditar o modelo pré-treinado, visualizando as detecções e calculando a matriz de confusão em nível de objeto.

**Notebook:**

```text
01_inferencia_garimpo_sar.ipynb
```

**Instruções:**

* Abra o notebook no Colab;
* Conecte sua conta do Google Drive;
* Clique em **Run All (Executar Tudo)**.

O modelo fará a varredura e salvará as imagens processadas na pasta de resultados.

#### Opção B — Treinamento Completo

Para reproduzir integralmente os experimentos do zero, desde a geração do dataset (extração de chips) até o treinamento da rede neural.

**Notebook:**

```text
02_treinamento_garimpo_sar.ipynb
```

**Instruções:**

* Abra o notebook no Colab;
* Ative aceleração por hardware (**GPU T4 ou superior**);
* Execute todas as células.

> **Nota:** Este script salvará os pesos gerados como unet_[ENCODER_NAME]_novo_treino.pth, garantindo que os pesos oficiais do estudo não sejam sobrescritos acidentalmente. O treinamento ocorre por até 80 épocas, com paciência de 20 épocas para Early Stopping.

---

## Tecnologias Utilizadas

### Deep Learning

* PyTorch
* Segmentation Models PyTorch (SMP)

### Processamento Geoespacial

* Rasterio
* GeoPandas
* Shapely

### Data Augmentation e Visualização

* Albumentations
* NumPy
* SciPy
* Matplotlib

> Todas as dependências são instaladas automaticamente na primeira célula de cada notebook.

---

## Autor

**Fellipe Machado Castro**
Graduando em Ciência da Computação
Universidade Federal do Pará (UFPA)

[fellipemachado77@gmail.com](mailto:fellipemachado77@gmail.com)

---

## Licença

Este projeto foi desenvolvido exclusivamente para fins acadêmicos como parte do Trabalho de Conclusão de Curso (TCC) em Ciência da Computação da Universidade Federal do Pará (UFPA).
