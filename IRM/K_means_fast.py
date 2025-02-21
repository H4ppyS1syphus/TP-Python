from sklearn.cluster import KMeans
import pyvista as pv
import numpy as np
from skimage import io
from scipy.ndimage import gaussian_filter, convolve
import scipy.ndimage as ndi
from skimage.segmentation import flood
import os

# Constants
color_map = 'turbo'
voxel_volume_mm3 = 0.9765635 * 0.9765635 * 1  # Volume of a voxel in mm^3

# Fonction pour charger une image 3D
def load_image_3D(path):
    img = io.imread(path)
    return img.astype(np.float32)

# Fonction pour appliquer un filtre gaussien
def gaussian_filter_volume(img_float, sigma=1):
    img_filtered = gaussian_filter(img_float, sigma=sigma)
    return img_filtered

# Fonction pour trouver le point d'intensité maximale
def find_max_intensity_point(volume):
    kernel = np.ones((3, 3, 3))
    local_sums = convolve(volume, kernel, mode='constant', cval=0)
    max_position = np.unravel_index(np.argmax(local_sums), local_sums.shape)
    return max_position

# Fonction pour normaliser l'intensité du volume
def normalize_volume_intensity(img_float):
    max_intensity = np.max(img_float)
    if max_intensity > 0:
        normalized_volume = img_float / max_intensity
    else:
        normalized_volume = img_float
    return normalized_volume

# Fonction pour appliquer K-moyens
def kmeans_clustering_on_volume(volume, n_clusters=3):
    # Aplatir le volume 3D en 2D (chaque voxel devient une ligne)
    original_shape = volume.shape
    volume_flat = volume.flatten().reshape(-1, 1)
    
    # Appliquer l'algorithme K-moyens
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(volume_flat)
    
    # Récupérer les étiquettes et les centroids
    labels = kmeans.labels_
    
    # Reconvertir les étiquettes en forme 3D
    volume_segmented = labels.reshape(original_shape)
    
    return volume_segmented, kmeans.cluster_centers_

# Fonction pour obtenir le cluster contenant le point d'intensité maximale
def get_cluster_containing_max_point(volume_segmented, max_intensity_point):
    # Identifier le cluster auquel appartient le point d'intensité maximale
    cluster_id = volume_segmented[max_intensity_point]
    
    # Créer un masque pour extraire ce cluster
    cluster_mask = volume_segmented == cluster_id
    
    # Extraire le volume du cluster
    cluster_volume = cluster_mask.astype(np.float32)  # Assurer que c'est un volume binaire
    
    return cluster_volume

def visualize_volumes_with_subplots(volume1, volume2, volume3, title="Volumes 3D"):
    plotter = pv.Plotter(shape=(1, 3))  # 1 ligne, 3 colonnes
    
    # Afficher le premier volume dans la première sous-figure
    plotter.subplot(0, 0)
    plotter.add_volume(volume1, cmap=color_map, opacity="sigmoid_6", name="Cluster Volume")
    plotter.add_text("Cluster Volume", position='upper_left')

    # Afficher le deuxième volume dans la deuxième sous-figure
    plotter.subplot(0, 1)
    plotter.add_volume(volume2, cmap=color_map, opacity="sigmoid_6", name="Segmented Volume")
    plotter.add_text("Segmented Volume", position='upper_left')

    # Afficher le troisième volume dans la troisième sous-figure
    plotter.subplot(0, 2)
    plotter.add_volume(volume3, cmap=color_map, opacity="sigmoid_6", name="Final Volume")
    plotter.add_text("Final Volume", position='upper_left')
    
    # Afficher toutes les sous-figures en même temps avec un seul show()
    plotter.show()

# Function to apply region growing
def region_growing(volume_filtered, seed_point, tolerance=10):
    segmented = flood(volume_filtered, seed_point, tolerance=tolerance)
    return segmented

# Fonction pour remplacer les voxels non nuls d'un volume par ceux d'un autre volume
def replace_non_zero_voxels(volume1, volume2):
    # Si les volumes sont de type pyvista.DataSet, convertissons-les en tableaux numpy
    if isinstance(volume1, pv.DataSet):
        volume1_data = volume1.point_data.active_scalars.reshape(volume1.dimensions, order="F")
    else:
        volume1_data = volume1

    if isinstance(volume2, pv.DataSet):
        volume2_data = volume2.point_data.active_scalars.reshape(volume2.dimensions, order="F")
    else:
        volume2_data = volume2

    # Assurez-vous que les dimensions sont compatibles
    assert volume1_data.shape == volume2_data.shape, "Les deux volumes doivent avoir les mêmes dimensions."
    
    # Remplacer les valeurs non nulles
    volume_resultant = np.where(volume1_data != 0, volume2_data, volume1_data)
    
    # Retourner le résultat sous forme de pyvista ou numpy
    if isinstance(volume1, pv.DataSet):
        return pv.wrap(volume_resultant)
    else:
        return volume_resultant

def combine_volumes(volume_1, volume_2):
    # Si volume_1 ou volume_2 sont des objets pyvista, les convertir en tableaux numpy
    if isinstance(volume_1, pv.DataSet):
        volume_1 = volume_1.point_data.active_scalars.reshape(volume_1.dimensions, order="F")
    if isinstance(volume_2, pv.DataSet):
        volume_2 = volume_2.point_data.active_scalars.reshape(volume_2.dimensions, order="F")
    
    # Créer un volume vide avec la même forme que les volumes d'entrée
    combined_volume = np.zeros_like(volume_1)
    
    # Appliquer la condition : si un des volumes a une valeur de 0, le voxel du résultat sera 0
    combined_volume[(volume_1 != 0) & (volume_2 != 0)] = volume_1[(volume_1 != 0) & (volume_2 != 0)]
    
    return combined_volume

def calculate_volume(img_float):
    if isinstance(img_float, pv.DataSet):
        img_float = img_float.point_data.active_scalars.reshape(img_float.dimensions, order="F")
    liquid_voxels = np.sum(img_float > 0)
    volume_mm3 = liquid_voxels * voxel_volume_mm3
    return volume_mm3

def visualize_all_clusters(volume_segmented, n_clusters, color_map='turbo'):
    plotter = pv.Plotter(shape=(1, n_clusters))  # 1 ligne, n colonnes selon le nombre de clusters

    for cluster_id in range(n_clusters):
        cluster_mask = (volume_segmented == cluster_id).astype(np.float32)
        cluster_volume_pv = pv.wrap(cluster_mask)

        plotter.subplot(0, cluster_id)
        plotter.add_volume(cluster_volume_pv, cmap=color_map, opacity="sigmoid_6", name=f"Cluster {cluster_id}")
        plotter.add_text(f"Cluster {cluster_id}", position='upper_left')

    plotter.show()
    
def save_volume(volume, file_path):
    if isinstance(volume, pv.DataSet):
        volume = volume.point_data.active_scalars.reshape(volume.dimensions, order="F")
    np.save(file_path, volume)
    print(f"Volume sauvegardé sous {file_path}")

path = 'c003.tif'
sigma_value = 1.5
tolerance_value = 0.40
# Workflow principal
img_float_1 = load_image_3D(path)
img_float_1 = normalize_volume_intensity(img_float_1)
volume_gaussian_1 = gaussian_filter_volume(img_float_1, sigma=sigma_value)

# Appliquer K-moyens sur le volume filtré
n_clusters = 6  # Nombre de clusters souhaités
volume_segmented, centroids = kmeans_clustering_on_volume(volume_gaussian_1, n_clusters)

# Trouver le point d'intensité maximale
max_intensity_point = find_max_intensity_point(volume_gaussian_1)

# Récupérer le cluster contenant ce point
cluster_volume = get_cluster_containing_max_point(volume_segmented, max_intensity_point)

# Appliquer la segmentation par croissance régionale
segmented_volume = region_growing(volume_gaussian_1, max_intensity_point, tolerance=tolerance_value)

# Convertir en objet pyvista pour les visualisations
segmented_volume_pv = pv.wrap(segmented_volume.astype(np.float32))

# Remplacer les voxels non nuls du volume segmenté par ceux du volume gaussien filtré
volume_resultant = replace_non_zero_voxels(segmented_volume_pv, volume_gaussian_1)

# Combiner les volumes résultants
final_volume = combine_volumes(volume_resultant, cluster_volume)

# Appeler la fonction pour visualiser tous les clusters
visualize_all_clusters(volume_segmented, n_clusters)
# Afficher le volume final
visualize_volumes_with_subplots(cluster_volume, volume_resultant, final_volume)


# Save les résultats
file_name = os.path.splitext(os.path.basename(path))[0]

save_volume(volume_segmented, file_name + '_volume_segmented.npy')
save_volume(cluster_volume, file_name + '_cluster_volume.npy')
save_volume(volume_resultant, file_name + '_resultant_volume.npy')
save_volume(final_volume, file_name + '_final_volume.npy')

print(calculate_volume(cluster_volume))
print(calculate_volume(volume_resultant))
print(calculate_volume(final_volume))









