import numpy as np


def centroid_distance(image_emb, audio_emb):
    image_centroid = image_emb.mean(axis=0)
    audio_centroid = audio_emb.mean(axis=0)
    return float(np.linalg.norm(image_centroid - audio_centroid))


METRICS = {
    'centroid_distance': centroid_distance,
}
