
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Caminhos das pastas do projeto
pasta_projeto = Path(__file__).resolve().parent
caminho_imagem = pasta_projeto / "imagens" / "agricultura.jpg"
pasta_resultados = pasta_projeto / "resultados"

# Garante que a pasta de resultados exista
pasta_resultados.mkdir(exist_ok=True)

# Lê a imagem, inclusive em caminhos com acentos
dados_imagem = np.fromfile(str(caminho_imagem), dtype=np.uint8)
imagem = cv2.imdecode(dados_imagem, cv2.IMREAD_COLOR)

if imagem is None:
    raise FileNotFoundError("Imagem agricultura.jpg não encontrada!")

# OpenCV lê imagens em BGR; convertemos para RGB
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# Converte os pixels para float para permitir divisões
imagem_float = imagem_rgb.astype(np.float32)

# Separa os três canais de cor
R = imagem_float[:, :, 0]
G = imagem_float[:, :, 1]
B = imagem_float[:, :, 2]

# Calcula o Green Leaf Index (GLI) para cada pixel
numerador = 2 * G - R - B
denominador = 2 * G + R + B

gli = np.divide(
    numerador,
    denominador,
    out=np.zeros_like(numerador),
    where=denominador != 0
)

# Exibe a imagem original e o mapa GLI
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(imagem_rgb)
plt.title("Imagem original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gli, cmap="RdYlGn", vmin=-1, vmax=1)
plt.title("Indice de vegetacao GLI")
plt.colorbar(label="Valor GLI")
plt.axis("off")

plt.tight_layout()

# Salva o resultado na pasta resultados
plt.savefig(pasta_resultados / "mapa_gli.png", dpi=200)

plt.close()

print("Mapa GLI gerado com sucesso!")


# SEGMENTAÇÃO DA VEGETAÇÃO

# Converte o GLI de [-1, 1] para [0, 255]
# O OpenCV utiliza imagens de 8 bits no método de Otsu
gli_8bits = ((gli + 1) * 127.5).clip(0, 255).astype(np.uint8)

# Calcula automaticamente o limiar pelo método de Otsu
limiar, mascara = cv2.threshold(
    gli_8bits,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Converte o limiar para a escala original do GLI
limiar_gli = (limiar / 127.5) - 1

# Evita classificar valores negativos de GLI como vegetação
# A máscara final exige GLI positivo e acima do limiar de Otsu
# Máscara gerada pelo método de Otsu
mascara_otsu = np.where(
    (mascara == 255) & (gli > 0),
    255,
    0
).astype(np.uint8)

# Máscara utilizando um limiar fixo de GLI
mascara_fixa = np.where(
    gli > 0.10,
    255,
    0
).astype(np.uint8)

# Compara os percentuais de vegetação
percentual_otsu = np.count_nonzero(mascara_otsu) / mascara_otsu.size * 100
percentual_fixo = np.count_nonzero(mascara_fixa) / mascara_fixa.size * 100

print(f"Cobertura com Otsu: {percentual_otsu:.2f}%")
print(f"Cobertura com GLI > 0.10: {percentual_fixo:.2f}%")

# Utiliza temporariamente a máscara de limiar fixo
mascara = mascara_fixa

# Calcula a porcentagem de pixels classificados como vegetação
pixels_vegetacao = np.count_nonzero(mascara)
pixels_totais = mascara.size

percentual = (pixels_vegetacao / pixels_totais) * 100

print(f"Limiar de Otsu (GLI): {limiar_gli:.4f}")
print(f"Cobertura vegetal estimada: {percentual:.2f}%")

# Mantém apenas os pixels classificados como vegetação
vegetacao_destacada = cv2.bitwise_and(
    imagem_rgb,
    imagem_rgb,
    mask=mascara
)


# ETAPA 4 - COMPARAÇÃO DOS MÉTODOS

# Cria uma figura com a imagem original e as duas máscaras
plt.figure(figsize=(15, 5))

# Imagem original
plt.subplot(1, 3, 1)
plt.imshow(imagem_rgb)
plt.title("Imagem original")
plt.axis("off")

# Máscara gerada pelo método de Otsu
plt.subplot(1, 3, 2)
plt.imshow(mascara_otsu, cmap="gray", vmin=0, vmax=255)
plt.title(f"GLI + Otsu: {percentual_otsu:.2f}%")
plt.axis("off")

# Máscara gerada pelo limiar fixo
plt.subplot(1, 3, 3)
plt.imshow(mascara_fixa, cmap="gray", vmin=0, vmax=255)
plt.title(f"GLI > 0.10: {percentual_fixo:.2f}%")
plt.axis("off")

plt.tight_layout()

# Salva a comparação
plt.savefig(
    pasta_resultados / "comparacao_metodos.png",
    dpi=200
)

# Salva cada máscara separadamente
cv2.imwrite(
    str(pasta_resultados / "mascara_otsu.png"),
    mascara_otsu
)

cv2.imwrite(
    str(pasta_resultados / "mascara_limiar_fixo.png"),
    mascara_fixa
)

plt.close()

print("Comparação salva com sucesso!")



# LIMPEZA DAS MÁSCARAS E CONTORNOS

# Cria um elemento estruturante de 7x7 pixels
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (7, 7)
)

# Função para limpar uma máscara binária
def limpar_mascara(mascara):
    # Preenche pequenos buracos
    fechada = cv2.morphologyEx(
        mascara, cv2.MORPH_CLOSE, kernel
    )

    # Remove pequenos pontos isolados
    limpa = cv2.morphologyEx(
        fechada, cv2.MORPH_OPEN, kernel
    )

    return limpa


# Aplica a limpeza aos dois métodos
mascara_otsu_limpa = limpar_mascara(mascara_otsu)
mascara_fixa_limpa = limpar_mascara(mascara_fixa)


# Função para desenhar os contornos sobre a imagem
def desenhar_contornos(imagem_rgb, mascara):
    # Identifica os contornos externos das regiões brancas
    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Ignora contornos muito pequenos (ruídos)
    contornos_validos = [
        contorno for contorno in contornos
        if cv2.contourArea(contorno) > 500
    ]

    # Cria uma cópia da imagem original
    resultado = imagem_rgb.copy()

    # Desenha os contornos em vermelho
    cv2.drawContours(
        resultado,
        contornos_validos,
        -1,
        (255, 0, 0),
        3
    )

    return resultado, len(contornos_validos)


# Gera as imagens com os contornos
contornos_otsu, quantidade_otsu = desenhar_contornos(
    imagem_rgb, mascara_otsu_limpa
)

contornos_fixo, quantidade_fixo = desenhar_contornos(
    imagem_rgb, mascara_fixa_limpa
)

# Calcula a cobertura após a limpeza
cobertura_otsu_limpa = (
    np.count_nonzero(mascara_otsu_limpa)
    / mascara_otsu_limpa.size * 100
)

cobertura_fixa_limpa = (
    np.count_nonzero(mascara_fixa_limpa)
    / mascara_fixa_limpa.size * 100
)

print(f"Otsu após limpeza: {cobertura_otsu_limpa:.2f}%")
print(f"Limiar fixo após limpeza: {cobertura_fixa_limpa:.2f}%")
print(f"Contornos Otsu: {quantidade_otsu}")
print(f"Contornos limiar fixo: {quantidade_fixo}")

# Compara visualmente os contornos
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(imagem_rgb)
plt.title("Imagem original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(contornos_otsu)
plt.title("Contornos - Otsu")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(contornos_fixo)
plt.title("Contornos - Limiar fixo")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    pasta_resultados / "comparacao_contornos.png",
    dpi=200
)

plt.close()

# Salva também as máscaras limpas
cv2.imwrite(
    str(pasta_resultados / "mascara_otsu_limpa.png"),
    mascara_otsu_limpa
)

cv2.imwrite(
    str(pasta_resultados / "mascara_fixa_limpa.png"),
    mascara_fixa_limpa
)

print("Contornos gerados com sucesso!")
