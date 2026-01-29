# EASESPOT - Anomaly Detection System

![EASESPOT Logo](novologo.png)

Sistema de detecção de anomalias em imagens têxteis usando Machine Learning.

## 🎯 Funcionalidades

- **Detecção de Anomalias**: Identifica defeitos em tecidos automaticamente
- **4 Modelos Disponíveis**: PatchCore, PaDiM, SPADE e Autoencoder
- **Classificação de 41 Defeitos**: Mapeamento automático para tipos de defeitos reais
- **Ajuste de Threshold**: Slider para calibrar sensibilidade do detector
- **Cálculo de Custo e Prioridade**: Estimativa de impacto por defeito
- **Interface Moderna**: Design premium com branding EASESPOT

## 🚀 Como Usar

### Executar Localmente

```bash
# Clonar repositório
git clone https://github.com/SEU_USUARIO/easespot-anomaly-detection.git
cd easespot-anomaly-detection

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar app
streamlit run app.py
```

### Deploy no Streamlit Cloud

1. Faça push do repositório para o GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte seu repositório GitHub
4. Selecione o branch `main` e arquivo `app.py`
5. Clique em "Deploy"

## 📁 Estrutura do Projeto

```
├── app.py                  # Aplicação Streamlit principal
├── requirements.txt        # Dependências Python
├── novologo.png           # Logo EASESPOT
├── fundo.png              # Imagem de fundo
├── padim_model.pkl        # Modelo PaDiM treinado
├── patchcore_model.pkl    # Modelo PatchCore treinado
├── spade_model.pkl        # Modelo SPADE treinado
├── autoencoder_model.pkl  # Modelo Autoencoder treinado (Git LFS)
└── README.md              # Este arquivo
```

## 🤖 Modelos Disponíveis

| Modelo | Descrição |
|--------|-----------|
| **PatchCore** | Memory bank com KNN - Alta precisão para detecção de patches anômalos |
| **PaDiM** | Modelagem de distribuição gaussiana por patches |
| **SPADE** | Pirâmide semântica multi-escala |
| **Autoencoder** | Rede convolucional que detecta anomalias por erro de reconstrução |

## 📊 Classificação de Defeitos

O sistema classifica automaticamente anomalias em **41 tipos de defeitos**:

- Defeitos dimensionais
- Falhas no tecido
- Furos e rasgos
- Pontos quebrados
- Sem resistência
- Sujo/manchas
- Tecido escapando
- Tonalidade entre partes/peças

Cada defeito inclui:
- **Prioridade**: 🔴 Crítica, 🟠 Alta, 🟡 Média, 🟢 Baixa
- **Custo Estimado**: Baseado em prioridade, frequência e severidade
- **Ação Requerida**: Indicação de urgência

## ⚙️ Ajuste de Threshold

Use o slider na sidebar para ajustar a sensibilidade:
- **Multiplicador > 1.0**: Menos detecções (mais conservador, reduz falsos positivos)
- **Multiplicador < 1.0**: Mais detecções (mais sensível)

## 🔧 Tecnologias

- **Python 3.9+**
- **Streamlit** - Interface web
- **PyTorch** - Deep Learning
- **ResNet18** - Feature extraction
- **scikit-learn** - Machine Learning

## 📄 Licença

© 2026 EASESPOT - Todos os direitos reservados.
