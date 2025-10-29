# aircraft-detection-classification-denoising
End-to-end deep learning pipeline for aircraft image analysis — including detection (YOLO11), classification (EfficientNetV2S &amp; DenseNet121), and denoising (U-Net), using satellite and aerial datasets.

# 🛰️ AeroVision Pipeline

**Detecção, Classificação e Denoising de Imagens de Aeronaves Militares com Redes Neurais Convolucionais**

---

## 📘 Descrição do Projeto

Este projeto implementa um **pipeline completo de visão computacional** para análise de imagens de aeronaves obtidas por satélite.  
O sistema integra três módulos principais — **detecção, classificação e denoising** — baseados em **redes neurais convolucionais (CNNs)**, permitindo a identificação, categorização e restauração de imagens de aeronaves militares em diferentes condições de captura.

O pipeline foi desenvolvido no contexto da **Residência em Tecnologias Aeroespaciais (IA)** do **Instituto Hardware BR**, em parceria com a **Universidade Estadual de Campinas (UNICAMP)**.

---

## 🚀 Estrutura do Pipeline

| Etapa | Arquitetura | Objetivo |
|-------|--------------|----------|
| **1. Detecção** | YOLO11 (Ultralytics) | Localizar aeronaves em imagens de satélite com alta precisão. |
| **2. Classificação** | EfficientNetV2S & DenseNet121 | Classificar aeronaves em oito classes militares distintas. |
| **3. Denoising** | U-Net | Remover ruído gaussiano e restaurar detalhes estruturais. |

---

## 🧠 Arquiteturas Utilizadas

- **YOLO11** – Detecção de objetos em tempo real com alta eficiência e precisão.  
- **EfficientNetV2S** – Rede convolucional com *compound scaling*, equilibrando profundidade, largura e resolução.  
- **DenseNet121** – Conexões densas para melhor propagação de gradientes e reutilização de características.  
- **U-Net** – Estrutura simétrica encoder–decoder com *skip connections* para reconstrução de imagens degradadas.

---

## 🗂️ Estrutura do Repositório

```
AeroVision-Pipeline/
│
├── detection/
│   ├── train_yolo11.ipynb
│   ├── dataset.yaml
│   ├── results/
│   │   ├── training_map.png
│   │   ├── training_pr.png
│   │   └── detection_samples.png
│
├── classification/
│   ├── train_efficientnet.ipynb
│   ├── train_densenet.ipynb
│   ├── results/
│   │   ├── loss_comparison.png
│   │   ├── accuracy_comparison.png
│   │   └── confusion_matrix.png
│
├── denoising/
│   ├── unet_denoising.ipynb
│   ├── results/
│   │   ├── training_metrics.png
│   │   └── denoising_samples.png
│
├── data/
│   ├── aircraft-detection-with-yolov8/
│   ├── airplanes-satellite-imagery/
│   └── aircraft-classification/
│
├── figures/
│   ├── architecture_unet.png
│   ├── efficientnet_vs_params.png
│   └── datasets_examples.png
│
├── README.md
└── requirements.txt
```

---

## 📊 Resultados Principais

| Tarefa | Arquitetura | Métricas-Chave |
|--------|--------------|----------------|
| **Detecção** | YOLO11 | mAP@0.5 = 0.89 · mAP@0.5:0.95 = 0.73 |
| **Classificação** | EfficientNetV2S / DenseNet121 | Acurácia de validação ≈ 97% (ambas) |
| **Denoising** | U-Net | PSNR = 31.4 dB · SSIM = 0.93 |

*(valores ilustrativos — substitua pelos reais do seu `results.csv`)*

---

## 🧩 Tecnologias Utilizadas

- **Python 3.10+**
- **TensorFlow / Keras**
- **PyTorch (Ultralytics YOLO)**
- **Google Colab (GPU gratuita)**
- **OpenCV / NumPy / Matplotlib**
- **scikit-learn / seaborn**

---

## ⚙️ Configuração de Ambiente

1. Clone o repositório:
   ```bash
   git clone https://github.com/gabrielbaltazar/AeroVision-Pipeline.git
   cd AeroVision-Pipeline
   ```

2. Crie e ative o ambiente virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux
   .venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛰️ Datasets

Foram utilizados **dois datasets públicos disponíveis no Kaggle**:

1. [Aircraft Detection with YOLOv8](https://www.kaggle.com/datasets/tokarooo/aircraft-detection-with-yolov8)  
2. [Airplanes Satellite Imagery](https://www.kaggle.com/datasets/zidane10aa/airplanes-satellite-imagery)

---

## 🧾 Citação

Se este projeto for útil para o seu trabalho acadêmico, cite como:

```bibtex
@article{baltazar2025aerovision,
  title={Aplicação de Redes Neurais Convolucionais para Detecção, Classificação e Denoising de Imagens de Aeronaves Geradas por Satélite},
  author={Baltazar, Gabriel Alves},
  year={2025},
  institution={Instituto Hardware BR / UNICAMP}
}
```

---

## 📫 Contato

**Gabriel Alves Baltazar**  
Departamento de Telecomunicações – UNICAMP  
📧 g234628@dac.unicamp.br  
🔗 [LinkedIn](https://www.linkedin.com/in/gabrielbaltazar) • [GitHub](https://github.com/gabrielbaltazar)

---

## 📈 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
