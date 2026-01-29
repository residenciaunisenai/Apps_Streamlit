import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import os
import base64
import pickle
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# PyTorch para extração de features (igual ao notebook)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# Configuração do dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configurações do modelo (igual ao notebook)
IMG_SIZE = 416

# Transformação de imagem (igual ao notebook)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================================
# FEATURE EXTRACTOR (ResNet18 - igual ao notebook)
# ============================================================

class FeatureExtractor(nn.Module):
    """Extrai features de múltiplas camadas do ResNet18."""
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Modo de avaliação
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        features = {}
        x = self.layer0(x)
        features['layer0'] = x
        x = self.layer1(x)
        features['layer1'] = x
        x = self.layer2(x)
        features['layer2'] = x
        x = self.layer3(x)
        features['layer3'] = x
        x = self.layer4(x)
        features['layer4'] = x
        return features


# ============================================================
# DETECTORES (igual ao notebook)
# ============================================================

class PatchCoreDetector:
    """PatchCore: Memory bank com KNN."""
    def __init__(self):
        self.memory_bank = None
        self.threshold = None
        self.n_neighbors = 9
        self.layer_ids = ['layer2', 'layer3', 'layer4']
        self.dim_per_layer = [128, 256, 512]
        self.knn = None
    
    def _extract_patch_embeddings(self, images, feature_extractor):
        """Extrai embeddings de patches."""
        feature_extractor.eval()
        with torch.no_grad():
            features = feature_extractor(images)
        
        embeddings_list = []
        for layer_id in self.layer_ids:
            f = features[layer_id]
            f = F.interpolate(f, size=(13, 13), mode='bilinear', align_corners=False)
            embeddings_list.append(f)
        
        embeddings = torch.cat(embeddings_list, dim=1)
        embeddings = embeddings.permute(0, 2, 3, 1)
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        
        return embeddings.cpu().numpy()
    
    def predict_single(self, image_tensor, feature_extractor):
        """Prediz para uma única imagem."""
        embeddings = self._extract_patch_embeddings(image_tensor, feature_extractor)
        distances, _ = self.knn.kneighbors(embeddings)
        max_distances = np.max(distances, axis=1)
        score = np.max(max_distances)
        is_anomaly = score > self.threshold
        return is_anomaly, score
    
    def load(self, path):
        """Carrega o modelo."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.memory_bank = state['memory_bank']
        self.layer_ids = state['layer_ids']
        self.dim_per_layer = state['dim_per_layer']
        self.threshold = state['threshold']
        self.n_neighbors = state['n_neighbors']
        
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        self.knn.fit(self.memory_bank)


class PaDiMDetector:
    """PaDiM: Patch Distribution Modeling."""
    def __init__(self):
        self.pca = None
        self.mean_embeddings = {}
        self.inv_cov_matrices = {}
        self.patch_sizes = None
        self.threshold = None
        self.layer_ids = ['layer1', 'layer2', 'layer3']
        self.dim_per_layer = [64, 128, 256]
    
    def _extract_patch_embeddings(self, images, feature_extractor):
        """Extrai embeddings de patches."""
        feature_extractor.eval()
        with torch.no_grad():
            features = feature_extractor(images)
        
        embeddings_list = []
        for layer_id in self.layer_ids:
            f = features[layer_id]
            f = F.interpolate(f, size=(13, 13), mode='bilinear', align_corners=False)
            embeddings_list.append(f)
        
        embeddings = torch.cat(embeddings_list, dim=1)
        _, C, H, W = embeddings.shape
        self.patch_sizes = (H, W)
        
        embeddings = embeddings.permute(0, 2, 3, 1)
        embeddings = embeddings.reshape(-1, C)
        
        return embeddings.cpu().numpy()
    
    def predict_single(self, image_tensor, feature_extractor):
        """Prediz para uma única imagem."""
        embeddings = self._extract_patch_embeddings(image_tensor, feature_extractor)
        embeddings_pca = self.pca.transform(embeddings)
        
        H, W = self.patch_sizes
        n_patches = H * W
        n_components = embeddings_pca.shape[1]
        
        embeddings_pca = embeddings_pca.reshape(1, n_patches, n_components)
        
        patch_scores = []
        for pos in range(n_patches):
            emb = embeddings_pca[0, pos]
            diff = emb - self.mean_embeddings[pos]
            score = np.sqrt(np.dot(diff, np.dot(self.inv_cov_matrices[pos], diff)))
            patch_scores.append(score)
        
        score = max(patch_scores)
        is_anomaly = score > self.threshold
        return is_anomaly, score
    
    def load(self, path):
        """Carrega o modelo."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.pca = state['pca']
        self.mean_embeddings = state['mean_embeddings']
        self.inv_cov_matrices = state['inv_cov_matrices']
        self.patch_sizes = state['patch_sizes']
        self.threshold = state['threshold']
        self.layer_ids = state['layer_ids']
        self.dim_per_layer = state['dim_per_layer']


class SPADEDetector:
    """SPADE: Semantic Pyramid Anomaly Detection."""
    def __init__(self):
        self.means = {}
        self.stds = {}
        self.pyramid_levels = [0, 1, 2, 3]
        self.threshold = None
    
    def _extract_pyramid_features(self, images, feature_extractor):
        """Extrai features em múltiplas escalas."""
        feature_extractor.eval()
        with torch.no_grad():
            features = feature_extractor(images)
        
        pyramid = {}
        for level in self.pyramid_levels:
            f = features[f'layer{level}']
            f = F.interpolate(f, size=(13, 13), mode='bilinear', align_corners=False)
            pyramid[level] = f
        
        return pyramid
    
    def predict_single(self, image_tensor, feature_extractor):
        """Prediz para uma única imagem."""
        pyramid = self._extract_pyramid_features(image_tensor, feature_extractor)
        
        level_scores = []
        for level in self.pyramid_levels:
            f = pyramid[level][0]  # Primeira (única) imagem
            f_patches = f.permute(1, 2, 0).reshape(-1, f.shape[0]).cpu().numpy()
            
            level_mean = self.means[level]
            level_std = self.stds[level]
            
            diff = f_patches - level_mean
            normalized_diff = diff / (level_std + 1e-8)
            patch_scores = np.linalg.norm(normalized_diff, axis=1)
            
            level_scores.append(np.max(patch_scores))
        
        score = np.max(level_scores)
        is_anomaly = score > self.threshold
        return is_anomaly, score
    
    def load(self, path):
        """Carrega o modelo."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.means = state['means']
        self.stds = state['stds']
        self.pyramid_levels = state['pyramid_levels']
        self.threshold = state['threshold']


class AutoencoderDetector:
    """Autoencoder Convolucional para detecção de anomalias."""
    def __init__(self):
        self.model = None
        self.threshold = None
        self.latent_dim = 128
    
    def _build_model(self):
        """Constrói a arquitetura do autoencoder."""
        # Encoder: 3 -> 64 -> 128 -> 256 -> 512
        encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        
        # Decoder: 512 -> 256 -> 128 -> 64 -> 3
        decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        
        # Modelo completo com fully connected para latent space
        class ConvAutoencoder(nn.Module):
            def __init__(self, encoder, decoder, latent_dim):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                # 512 * 26 * 26 = 346112 (para imagem 416x416)
                self.fc_latent = nn.Linear(346112, latent_dim)
                self.fc_decode = nn.Linear(latent_dim, 346112)
            
            def forward(self, x):
                # Encode
                enc = self.encoder(x)
                enc_flat = enc.view(enc.size(0), -1)
                latent = self.fc_latent(enc_flat)
                
                # Decode
                dec_flat = self.fc_decode(latent)
                dec = dec_flat.view(enc.size(0), 512, 26, 26)
                out = self.decoder(dec)
                return out
        
        return ConvAutoencoder(encoder, decoder, self.latent_dim)
    
    def predict_single(self, image_tensor, feature_extractor=None):
        """Prediz anomalia para uma única imagem."""
        self.model.eval()
        with torch.no_grad():
            # Reconstruir imagem
            reconstructed = self.model(image_tensor)
            
            # Calcular erro de reconstrução (MSE)
            mse = F.mse_loss(reconstructed, image_tensor, reduction='none')
            score = mse.sum().item()
            
            is_anomaly = score > self.threshold
        return is_anomaly, score
    
    def load(self, path):
        """Carrega o modelo com suporte a CPU/CUDA."""
        import io
        
        # Custom unpickler para mapear CUDA para CPU se necessário
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location=device, weights_only=False)
                return super().find_class(module, name)
        
        with open(path, 'rb') as f:
            state = CPU_Unpickler(f).load()
        
        self.latent_dim = state['latent_dim']
        self.threshold = state['threshold'].item() if hasattr(state['threshold'], 'item') else state['threshold']
        
        # Construir modelo e carregar pesos
        self.model = self._build_model().to(device)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()


# ============================================================
# SISTEMA DE CLASSIFICAÇÃO DE ANOMALIAS
# ============================================================

class PrioridadeNivel:
    """Níveis de prioridade baseados nos dados do cliente"""
    CRITICA = "🔴 CRÍTICA"
    ALTA = "🟠 ALTA"
    MEDIA = "🟡 MÉDIA"
    BAIXA = "🟢 BAIXA"


class AnomalyClassifier:
    """
    Classificador de anomalias que mapeia scores para tipos de defeitos
    e calcula métricas de prioridade e custo
    """
    
    def __init__(self):
        """Inicializa o classificador com dados reais do cliente"""
        self.defeitos = self._carregar_defeitos_cliente()
        self.thresholds = self._calcular_thresholds()
    
    def _carregar_defeitos_cliente(self):
        """Carrega dados reais de defeitos do cliente - 41 defeitos"""
        import pandas as pd
        defeitos_data = [
            {"defeito": "dimensional - entrepernas assimetrico - lavado", "frequencia": 9.25, "prioridade": 10, "tipo": "Detalhe Fino"},
            {"defeito": "falha no tecido 1", "frequencia": 14.94, "prioridade": 9, "tipo": "Superficie"},
            {"defeito": "falha no tecido 2", "frequencia": 17.76, "prioridade": 9, "tipo": "Superficie"},
            {"defeito": "falha no tecido 3", "frequencia": 14.70, "prioridade": 4, "tipo": "Cor"},
            {"defeito": "falha no tecido 4", "frequencia": 14.67, "prioridade": 5, "tipo": "Geometrico"},
            {"defeito": "furo e rasgo - bolso - lavado", "frequencia": 11.04, "prioridade": 2, "tipo": "Geometrico"},
            {"defeito": "furo e rasgo - corpo - lavado", "frequencia": 14.37, "prioridade": 7, "tipo": "Cor"},
            {"defeito": "furo e rasgo - cós - lavado", "frequencia": 12.33, "prioridade": 3, "tipo": "Cor"},
            {"defeito": "furo e rasgo - cós 2 - lavado", "frequencia": 17.98, "prioridade": 9, "tipo": "Geometrico"},
            {"defeito": "furo e rasgo - cós 3 lavado", "frequencia": 4.27, "prioridade": 3, "tipo": "Superficie"},
            {"defeito": "furo tecido pos lavação", "frequencia": 9.12, "prioridade": 1, "tipo": "Cor"},
            {"defeito": "ponto quebrado - lavado", "frequencia": 1.09, "prioridade": 10, "tipo": "Superficie"},
            {"defeito": "ponto quebrado bolso - lavado", "frequencia": 10.84, "prioridade": 8, "tipo": "Geometrico"},
            {"defeito": "ponto quebrado cós - lavado", "frequencia": 2.79, "prioridade": 1, "tipo": "Superficie"},
            {"defeito": "ponto quebrado lateral - lavado", "frequencia": 5.00, "prioridade": 10, "tipo": "Superficie"},
            {"defeito": "ponto quebrado lateral 2 - lavado", "frequencia": 4.84, "prioridade": 10, "tipo": "Geometrico"},
            {"defeito": "rasgo tecido - pos lavação", "frequencia": 9.60, "prioridade": 2, "tipo": "Superficie"},
            {"defeito": "sem resistencia - lavado", "frequencia": 18.35, "prioridade": 1, "tipo": "Geometrico"},
            {"defeito": "sem resistencia - lavado 2", "frequencia": 1.94, "prioridade": 10, "tipo": "Geometrico"},
            {"defeito": "sem resistencia - lavado 3", "frequencia": 8.09, "prioridade": 6, "tipo": "Cor"},
            {"defeito": "sem resistencia - lavado 4", "frequencia": 10.02, "prioridade": 6, "tipo": "Detalhe Fino"},
            {"defeito": "sem resistencia - lavado 5", "frequencia": 12.41, "prioridade": 2, "tipo": "Cor"},
            {"defeito": "sem resistencia - lavado 6", "frequencia": 14.01, "prioridade": 2, "tipo": "Superficie"},
            {"defeito": "sujo - mancha - lavado", "frequencia": 1.27, "prioridade": 2, "tipo": "Cor"},
            {"defeito": "sujo - mancha - lavado 2", "frequencia": 19.69, "prioridade": 9, "tipo": "Geometrico"},
            {"defeito": "sujo - mancha - lavado 3", "frequencia": 7.82, "prioridade": 5, "tipo": "Superficie"},
            {"defeito": "sujo - mancha - lavado 4", "frequencia": 18.59, "prioridade": 7, "tipo": "Geometrico"},
            {"defeito": "sujo - mancha - lavado 5", "frequencia": 15.06, "prioridade": 5, "tipo": "Superficie"},
            {"defeito": "sujo - mancha - lavado 6", "frequencia": 4.33, "prioridade": 9, "tipo": "Geometrico"},
            {"defeito": "sujo - mancha - lavado 7", "frequencia": 1.56, "prioridade": 4, "tipo": "Geometrico"},
            {"defeito": "tecido escapando- - gancho - lavado", "frequencia": 16.65, "prioridade": 9, "tipo": "Superficie"},
            {"defeito": "tecido escapando- - lateral - lavado", "frequencia": 1.86, "prioridade": 9, "tipo": "Cor"},
            {"defeito": "tecido escapando- - passante - lavado", "frequencia": 15.64, "prioridade": 3, "tipo": "Detalhe Fino"},
            {"defeito": "tecido escapando- barra - lavado", "frequencia": 4.75, "prioridade": 6, "tipo": "Superficie"},
            {"defeito": "tecido escapando- bolso - lavado", "frequencia": 13.28, "prioridade": 2, "tipo": "Cor"},
            {"defeito": "tonalidade entre partes", "frequencia": 1.17, "prioridade": 7, "tipo": "Cor"},
            {"defeito": "tonalidade entre partes 2", "frequencia": 12.57, "prioridade": 1, "tipo": "Superficie"},
            {"defeito": "tonalidade entre partes 3", "frequencia": 9.91, "prioridade": 9, "tipo": "Detalhe Fino"},
            {"defeito": "tonalidade entre peças", "frequencia": 13.26, "prioridade": 4, "tipo": "Superficie"},
            {"defeito": "tonalidade entre peças 2", "frequencia": 1.94, "prioridade": 8, "tipo": "Superficie"},
            {"defeito": "tonalidade entre peças 3", "frequencia": 11.12, "prioridade": 7, "tipo": "Geometrico"},
        ]
        
        import pandas as pd
        df = pd.DataFrame(defeitos_data)
        df['score_critico'] = df['prioridade'] * (11 - df['frequencia']/2)
        return df
    
    def _calcular_thresholds(self):
        """Calcula thresholds para classificação baseados nos scores críticos"""
        scores = self.defeitos['score_critico'].values
        return {
            'critico': np.percentile(scores, 75),
            'alto': np.percentile(scores, 50),
            'medio': np.percentile(scores, 25)
        }
    
    def classificar_anomalia(self, score_anomalia, confianca=0.0, modelo="Unknown", nome_arquivo=None):
        """
        Classifica uma anomalia detectada
        
        Args:
            score_anomalia: Score de anomalia do modelo (valor bruto)
            confianca: Confiança do modelo na detecção
            modelo: Nome do modelo que detectou
            nome_arquivo: Nome do arquivo da imagem (opcional, para extrair tipo de defeito)
        
        Returns:
            Dicionário com classificação completa
        """
        # Tenta extrair defeito do nome do arquivo primeiro
        defeito_match = None
        if nome_arquivo:
            defeito_match = self._extrair_defeito_do_nome(nome_arquivo)
        
        # Se não encontrou no nome, usa fallback baseado no score
        if defeito_match is None:
            defeito_match = self._mapear_para_defeito_fallback(score_anomalia)
            origem_classificacao = "estimativa (baseado no score)"
        else:
            origem_classificacao = "identificado pelo nome do arquivo"
        
        # Determina nível de prioridade
        nivel_prioridade = self._determinar_nivel_prioridade(defeito_match['prioridade'])
        
        # Calcula custo estimado (varia com score)
        custo = self._calcular_custo(
            defeito_match['prioridade'],
            defeito_match['frequencia'],
            score_anomalia
        )
        
        return {
            'tipo_defeito': defeito_match['defeito'],
            'categoria': defeito_match['tipo'],
            'prioridade_numero': defeito_match['prioridade'],
            'prioridade_nivel': nivel_prioridade,
            'frequencia_historica': defeito_match['frequencia'],
            'score_critico': defeito_match['score_critico'],
            'custo_estimado': custo,
            'score_anomalia': score_anomalia,
            'confianca_modelo': confianca * 100,
            'modelo_detector': modelo,
            'requer_acao_imediata': defeito_match['prioridade'] >= 8,
            'origem_classificacao': origem_classificacao
        }
    
    def _extrair_defeito_do_nome(self, nome_arquivo):
        """Extrai o tipo de defeito do nome do arquivo"""
        # Remove extensão
        nome = nome_arquivo
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            nome = nome.replace(ext, '')
        
        # Remove sufixos de data augmentation (padrão: _func1_xxx, _func2_xxx, etc)
        import re
        nome = re.sub(r'_func\d+.*$', '', nome)
        
        # Normaliza para comparação (lowercase)
        nome_lower = nome.lower().strip()
        
        # Busca correspondência EXATA primeiro
        for _, row in self.defeitos.iterrows():
            defeito_nome = row['defeito'].lower().strip()
            if defeito_nome == nome_lower:
                return row.to_dict()
        
        # Se não encontrou exato, busca por correspondência parcial
        melhor_match = None
        maior_score = 0
        
        for _, row in self.defeitos.iterrows():
            defeito_nome = row['defeito'].lower()
            
            # Normaliza ambos para comparação
            nome_normalizado = nome_lower.replace('-', ' ').replace('_', ' ')
            defeito_normalizado = defeito_nome.replace('-', ' ').replace('_', ' ')
            
            # Verifica se o defeito está contido no nome ou vice-versa
            if defeito_normalizado in nome_normalizado or nome_normalizado in defeito_normalizado:
                return row.to_dict()
            
            # Conta palavras-chave em comum
            palavras_defeito = set(defeito_normalizado.split())
            palavras_nome = set(nome_normalizado.split())
            matches = len(palavras_defeito.intersection(palavras_nome))
            
            if len(palavras_defeito) > 0:
                score_match = matches / len(palavras_defeito)
                
                if score_match > maior_score and matches >= 2:
                    maior_score = score_match
                    melhor_match = row.to_dict()
        
        return melhor_match
    
    def _mapear_para_defeito_fallback(self, score):
        """Fallback: mapeia score para defeito quando não encontra no nome"""
        # Ordena defeitos por score crítico
        df_sorted = self.defeitos.sort_values('score_critico', ascending=False).reset_index(drop=True)
        
        # Usa o score como seed para selecionar defeito de forma determinística
        score_hash = int(abs(score * 1000)) % len(df_sorted)
        
        return df_sorted.iloc[score_hash].to_dict()
    
    def _determinar_nivel_prioridade(self, prioridade):
        """Determina nível de prioridade visual"""
        if prioridade >= 8:
            return PrioridadeNivel.CRITICA
        elif prioridade >= 5:
            return PrioridadeNivel.ALTA
        elif prioridade >= 3:
            return PrioridadeNivel.MEDIA
        else:
            return PrioridadeNivel.BAIXA
    
    def _calcular_custo(self, prioridade, frequencia, score):
        """
        Calcula custo estimado baseado em:
        - Prioridade (peso maior)
        - Frequência histórica
        - Score de anomalia (normalizado)
        """
        custo_base = 100.0  # R$ 100 base
        
        # Fator de prioridade (1-10 -> 1.0-3.0)
        fator_prioridade = 1.0 + (prioridade / 5.0)
        
        # Fator de frequência (mais frequente = maior custo)
        fator_frequencia = 1.0 + (frequencia / 100.0)
        
        # Fator de severidade do score (normalizado para evitar valores extremos)
        # Usa logaritmo para scores muito altos (como 308000 do autoencoder)
        if score > 100:
            score_normalizado = min(np.log10(score) * 10, 100)
        else:
            score_normalizado = score
        fator_score = 1.0 + (score_normalizado / 50.0)
        
        custo = custo_base * fator_prioridade * fator_frequencia * fator_score
        return round(custo, 2)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def preprocess_image(uploaded_file):
    """Pré-processa a imagem para o formato esperado pelos modelos."""
    img = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor, img


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="EASESPOT - Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caminhos das imagens
LOGO_PATH = Path(__file__).parent / "novologo.png"
BG_PATH = Path(__file__).parent / "fundo.png"

# Estilo CSS customizado
def set_custom_style():
    bg_base64 = ""
    if BG_PATH.exists():
        bg_base64 = get_base64_image(BG_PATH)
    
    st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(10, 25, 47, 0.7);
            z-index: -1;
        }}
        
        .header-container {{
            background: linear-gradient(135deg, rgba(20, 40, 80, 0.95) 0%, rgba(10, 25, 50, 0.95) 100%);
            padding: 20px 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(100, 150, 220, 0.3);
        }}
        
        .logo-img {{ max-height: 80px; margin-right: 20px; }}
        .header-text {{ color: #ffffff; font-size: 2rem; font-weight: 600; }}
        
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0a1929 0%, #132f4c 100%);
        }}
        
        .stButton > button {{
            background: linear-gradient(135deg, #1e4976 0%, #2d5a8a 100%);
            color: white;
            border: 1px solid rgba(100, 150, 220, 0.4);
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(30, 73, 118, 0.4);
        }}
        
        .result-card {{
            background: linear-gradient(135deg, rgba(30, 50, 90, 0.9) 0%, rgba(20, 40, 70, 0.9) 100%);
            padding: 25px;
            border-radius: 15px;
            margin: 15px 0;
            border: 1px solid rgba(100, 150, 220, 0.3);
        }}
        
        .result-normal {{ border-left: 5px solid #4caf50; }}
        .result-anomaly {{ border-left: 5px solid #f44336; }}
        
        h1, h2, h3 {{ color: #ffffff !important; }}
        
        [data-testid="stMetricValue"] {{ color: #60a5fa !important; }}
    </style>
    """, unsafe_allow_html=True)


set_custom_style()

# Header
if LOGO_PATH.exists():
    logo_base64 = get_base64_image(LOGO_PATH)
    st.markdown(f"""
    <div class="header-container">
        <img src="data:image/png;base64,{logo_base64}" class="logo-img" alt="EASESPOT Logo">
        <span class="header-text">Anomaly Detection System</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="header-container">
        <span class="header-text">🔍 EASESPOT - Anomaly Detection System</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CARREGAR MODELOS
# ============================================================

@st.cache_resource
def load_feature_extractor():
    """Carrega o feature extractor (ResNet18)."""
    extractor = FeatureExtractor().to(device)
    extractor.eval()
    return extractor


@st.cache_resource
def load_models():
    """Carrega todos os modelos pré-treinados."""
    models = {}
    model_dir = Path(__file__).parent
    
    # PatchCore
    patchcore_path = model_dir / "patchcore_model.pkl"
    if patchcore_path.exists():
        try:
            detector = PatchCoreDetector()
            detector.load(patchcore_path)
            models['PatchCore (97% Precision)'] = detector
            st.sidebar.success("✅ PatchCore carregado")
        except Exception as e:
            st.sidebar.error(f"❌ Erro PatchCore: {e}")
    
    # PaDiM
    padim_path = model_dir / "padim_model.pkl"
    if padim_path.exists():
        try:
            detector = PaDiMDetector()
            detector.load(padim_path)
            models['PaDiM'] = detector
            st.sidebar.success("✅ PaDiM carregado")
        except Exception as e:
            st.sidebar.error(f"❌ Erro PaDiM: {e}")
    
    # SPADE
    spade_path = model_dir / "spade_model.pkl"
    if spade_path.exists():
        try:
            detector = SPADEDetector()
            detector.load(spade_path)
            models['SPADE'] = detector
            st.sidebar.success("✅ SPADE carregado")
        except Exception as e:
            st.sidebar.error(f"❌ Erro SPADE: {e}")
    
    # Autoencoder
    autoencoder_path = model_dir / "autoencoder_model.pkl"
    if autoencoder_path.exists():
        try:
            detector = AutoencoderDetector()
            detector.load(autoencoder_path)
            models['Autoencoder'] = detector
            st.sidebar.success("✅ Autoencoder carregado")
        except Exception as e:
            st.sidebar.error(f"❌ Erro Autoencoder: {e}")
    
    return models


# Carregar recursos
feature_extractor = load_feature_extractor()
models = load_models()

# ============================================================
# INTERFACE
# ============================================================

with st.sidebar:
    if LOGO_PATH.exists():
        logo_base64 = get_base64_image(LOGO_PATH)
        st.markdown(f"""
        <div style="text-align: center; padding: 20px 0; border-bottom: 1px solid rgba(100, 150, 220, 0.3); margin-bottom: 20px;">
            <img src="data:image/png;base64,{logo_base64}" style="max-width: 180px;">
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🤖 Selecione o Modelo")
    
    if models:
        model_name = st.selectbox(
            "Modelo para detecção",
            list(models.keys()),
            index=0
        )
        
        # Slider para ajuste de threshold
        st.markdown("---")
        st.markdown("### ⚙️ Ajuste de Threshold")
        
        original_threshold = models[model_name].threshold
        st.caption(f"Threshold original: {original_threshold:.4f}")
        
        # Multiplicador do threshold (0.5x a 3.0x)
        threshold_multiplier = st.slider(
            "Multiplicador",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Aumente para reduzir falsos positivos (menos detecções). Diminua para detectar mais anomalias."
        )
        
        adjusted_threshold = original_threshold * threshold_multiplier
        st.info(f"🎯 Threshold ajustado: **{adjusted_threshold:.4f}**")
        
        if threshold_multiplier > 1.0:
            st.caption("↑ Menos detecções (mais conservador)")
        elif threshold_multiplier < 1.0:
            st.caption("↓ Mais detecções (mais sensível)")
    else:
        st.error("⚠️ Nenhum modelo encontrado!")
        model_name = None
        adjusted_threshold = None
    
    st.markdown("---")
    st.markdown("### 📊 Informações")
    st.info(f"🖥️ Dispositivo: {device}")
    st.info(f"📐 Tamanho da imagem: {IMG_SIZE}x{IMG_SIZE}")


# Área principal
st.markdown("""
<div style="background: rgba(30, 50, 90, 0.7); padding: 20px; border-radius: 12px; margin-bottom: 25px;">
    <p style="color: #b0c4de; font-size: 1.1rem; margin: 0;">
        🎯 <strong>Sistema de detecção de anomalias em imagens têxteis.</strong><br>
        Usando modelos treinados com alta precisão para detectar defeitos.
    </p>
</div>
""", unsafe_allow_html=True)

# Upload de imagem
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📤 Upload de imagem para análise",
        type=["png", "jpg", "jpeg"]
    )
    
    if model_name:
        detect_btn = st.button("🔍 Analisar Imagem", use_container_width=True)
    else:
        detect_btn = False

with col2:
    if uploaded_file:
        st.image(Image.open(uploaded_file), caption="📷 Imagem enviada", use_column_width=True)
        uploaded_file.seek(0)

# Detecção
if detect_btn and uploaded_file and model_name:
    with st.spinner("🔄 Analisando imagem..."):
        try:
            # Pré-processar imagem
            img_tensor, original_img = preprocess_image(uploaded_file)
            
            # Selecionar modelo
            detector = models[model_name]
            
            # Fazer predição (pegar score bruto)
            _, score = detector.predict_single(img_tensor, feature_extractor)
            
            # Usar threshold ajustado para decisão
            is_anomaly = score > adjusted_threshold
            
            # Mostrar resultados
            st.markdown("---")
            st.subheader("📊 Resultado da Análise")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Modelo", model_name.split(' (')[0])
            with col2:
                st.metric("Score", f"{score:.4f}")
            with col3:
                st.metric("Threshold Original", f"{detector.threshold:.4f}")
            with col4:
                st.metric("Threshold Ajustado", f"{adjusted_threshold:.4f}")
            
            # Card de resultado
            if is_anomaly:
                st.markdown("""
                <div class="result-card result-anomaly">
                    <h3 style="color: #f44336; margin-top: 0;">⚠️ ANOMALIA DETECTADA!</h3>
                    <p style="color: #b0c4de;">Esta imagem apresenta características fora do padrão normal.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Classificar anomalia
                try:
                    classifier = AnomalyClassifier()
                    # Calcular confiança baseada na diferença do threshold
                    confianca = min((score - adjusted_threshold) / adjusted_threshold, 1.0)
                    classificacao = classifier.classificar_anomalia(
                        score_anomalia=score,
                        confianca=confianca,
                        modelo=model_name.split(' (')[0],
                        nome_arquivo=uploaded_file.name  # Passa nome do arquivo para extrair defeito
                    )
                    
                    # Card de classificação
                    st.markdown("---")
                    st.subheader("📋 Classificação da Anomalia")
                    
                    # Indicador de origem da classificação
                    if "identificado" in classificacao.get('origem_classificacao', ''):
                        st.success(f"✅ Defeito {classificacao['origem_classificacao']}")
                    else:
                        st.warning(f"⚠️ Defeito: {classificacao['origem_classificacao']}")
                    
                    # Linha 1: Tipo e Categoria
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div style="background: rgba(40, 60, 100, 0.6); padding: 15px; border-radius: 10px; border-left: 4px solid #f44336;">
                            <p style="color: #90caf9; font-size: 0.9rem; margin: 0;">Tipo de Defeito</p>
                            <p style="color: #ffffff; font-size: 1.1rem; font-weight: 600; margin: 5px 0 0 0;">{classificacao['tipo_defeito']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="background: rgba(40, 60, 100, 0.6); padding: 15px; border-radius: 10px; border-left: 4px solid #64b5f6;">
                            <p style="color: #90caf9; font-size: 0.9rem; margin: 0;">Categoria</p>
                            <p style="color: #ffffff; font-size: 1.1rem; font-weight: 600; margin: 5px 0 0 0;">{classificacao['categoria']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Linha 2: Prioridade e Custo
                    col1, col2 = st.columns(2)
                    with col1:
                        prioridade_color = "#f44336" if "CRÍTICA" in classificacao['prioridade_nivel'] else \
                                          "#ff9800" if "ALTA" in classificacao['prioridade_nivel'] else \
                                          "#ffc107" if "MÉDIA" in classificacao['prioridade_nivel'] else "#4caf50"
                        st.markdown(f"""
                        <div style="background: rgba(40, 60, 100, 0.6); padding: 15px; border-radius: 10px; border-left: 4px solid {prioridade_color};">
                            <p style="color: #90caf9; font-size: 0.9rem; margin: 0;">Prioridade</p>
                            <p style="color: #ffffff; font-size: 1.3rem; font-weight: 700; margin: 5px 0 0 0;">{classificacao['prioridade_nivel']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="background: rgba(40, 60, 100, 0.6); padding: 15px; border-radius: 10px; border-left: 4px solid #66bb6a;">
                            <p style="color: #90caf9; font-size: 0.9rem; margin: 0;">Custo Estimado</p>
                            <p style="color: #ffffff; font-size: 1.3rem; font-weight: 700; margin: 5px 0 0 0;">R$ {classificacao['custo_estimado']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Linha 3: Métricas adicionais
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Frequência Histórica", f"{classificacao['frequencia_historica']:.2f}%")
                    with col2:
                        st.metric("Score Crítico", f"{classificacao['score_critico']:.1f}")
                    with col3:
                        acao = "✓ SIM - URGENTE" if classificacao['requer_acao_imediata'] else "○ Monitoramento"
                        st.metric("Ação Imediata", acao)
                    
                except Exception as e:
                    st.warning(f"⚠️ Não foi possível classificar a anomalia: {e}")
                    
            else:
                st.markdown("""
                <div class="result-card result-normal">
                    <h3 style="color: #4caf50; margin-top: 0;">✅ Imagem Normal</h3>
                    <p style="color: #b0c4de;">Esta imagem está dentro do padrão normal esperado.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detalhes expandíveis
            with st.expander("📋 Detalhes da Análise"):
                st.write(f"**Modelo utilizado:** {model_name}")
                st.write(f"**Score de anomalia:** {score:.6f}")
                st.write(f"**Threshold do modelo:** {detector.threshold:.6f}")
                st.write(f"**Diferença (Score - Threshold):** {score - detector.threshold:.6f}")
                st.write(f"**Classificação:** {'ANOMALIA' if is_anomaly else 'NORMAL'}")
                
        except Exception as e:
            st.error(f"❌ Erro ao analisar imagem: {e}")
            import traceback
            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #6b7c93;">
    <p>EASESPOT © 2026 - Sistema de Detecção de Anomalias Têxteis</p>
    <p style="font-size: 0.9rem;">Powered by PyTorch, ResNet18 & Machine Learning</p>
</div>
""", unsafe_allow_html=True)
