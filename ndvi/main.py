
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path


# Caminhos dos arquivos
pasta_projeto = Path(__file__).resolve().parent
pasta_imagens = pasta_projeto / "imagens"
pasta_resultados = pasta_projeto / "resultados"

pasta_resultados.mkdir(exist_ok=True)

# Lê as duas bandas do Sentinel-2
with rasterio.open(pasta_imagens / "B04.tiff") as src:
    vermelho = src.read(1).astype(np.float32)
    perfil_vermelho = src.profile
    mascara_vermelho = src.read_masks(1) > 0

    # Mostra as informações geográficas da imagem
    print("Resolução do arquivo:", src.res)
    print("Sistema de coordenadas:", src.crs)

with rasterio.open(pasta_imagens / "B08.tiff") as src:
    infravermelho = src.read(1).astype(np.float32)
    perfil_infravermelho = src.profile
    mascara_infravermelho = src.read_masks(1) > 0

# Verifica se as duas bandas estão alinhadas
if (
    perfil_vermelho["width"] != perfil_infravermelho["width"]
    or perfil_vermelho["height"] != perfil_infravermelho["height"]
    or perfil_vermelho["transform"] != perfil_infravermelho["transform"]
    or perfil_vermelho["crs"] != perfil_infravermelho["crs"]
):
    raise ValueError("As bandas B04 e B08 não estão alinhadas!")

# Calcula o NDVI
numerador = infravermelho - vermelho
denominador = infravermelho + vermelho

pixels_validos = (
    mascara_vermelho
    & mascara_infravermelho
    & (denominador != 0)
)

ndvi = np.full(vermelho.shape, np.nan, dtype=np.float32)

np.divide(
    numerador,
    denominador,
    out=ndvi,
    where=pixels_validos
)

# Mostra algumas informações
print("Dimensões da imagem:", ndvi.shape)
print("Resolução espacial:", perfil_vermelho["transform"])
print(f"NDVI mínimo: {np.nanmin(ndvi):.4f}")
print(f"NDVI máximo: {np.nanmax(ndvi):.4f}")

# Cria a visualização
plt.figure(figsize=(10, 7))

imagem = plt.imshow(
    ndvi,
    cmap="RdYlGn",
    vmin=-1,
    vmax=1
)

plt.colorbar(imagem, label="NDVI")
plt.title("NDVI - Sentinel-2 (B08 e B04)")
plt.axis("off")

plt.tight_layout()

# Salva o mapa na pasta resultados
plt.savefig(
    pasta_resultados / "mapa_ndvi.png",
    dpi=200
)

plt.close()

print("Mapa NDVI gerado com sucesso!")


# CLASSIFICAÇÃO DA VEGETAÇÃO PELO NDVI

# Define um limiar para identificar possível vegetação
limiar_ndvi = 0.3

# Cria a máscara binária (255 = vegetação, 0 = não vegetação)
mascara_ndvi = np.where(
    pixels_validos & (ndvi > limiar_ndvi),
    255,
    0
).astype(np.uint8)

# Calcula a porcentagem somente sobre os pixels válidos
total_validos = np.count_nonzero(pixels_validos)
total_vegetacao = np.count_nonzero(mascara_ndvi)

percentual_vegetacao = (total_vegetacao / total_validos) * 100

print(f"Cobertura com NDVI > {limiar_ndvi}: {percentual_vegetacao:.2f}%")

# Compara o mapa NDVI com a máscara
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
plt.title("Mapa NDVI")
plt.colorbar(label="NDVI")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mascara_ndvi, cmap="gray", vmin=0, vmax=255)
plt.title(f"Vegetacao detectada: {percentual_vegetacao:.2f}%")
plt.axis("off")

plt.tight_layout()

# Salva o resultado
plt.savefig(
    pasta_resultados / "comparacao_ndvi.png",
    dpi=200
)

plt.close()

print("Comparacao NDVI gerada com sucesso!")

