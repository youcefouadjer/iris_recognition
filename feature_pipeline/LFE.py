import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from scipy.special import factorial
from numpy.linalg import pinv
import mahotas


def extract_blocks(image, block_size=(32, 32), stride=32):
    """
    Divide image into non-overlapping blocks.
    Returns list of blocks.
    """
    h, w = image.shape
    bh, bw = block_size
    blocks = []

    for y in range(0, h - bh + 1, stride):
        for x in range(0, w - bw + 1, stride):
            block = image[y:y+bh, x:x+bw]
            blocks.append(block)

    return blocks


# ============================================================
# Log-Gabor feature
# ============================================================
def log_gabor_filter(block, sigma=0.55):
    """
    Apply simplified Log-Gabor filter (1 scale).
    Returns mean energy of filtered block.
    """
    block = block.astype(np.float32)
    rows, cols = block.shape

    # FFT
    fft = np.fft.fft2(block)
    fft_shift = np.fft.fftshift(fft)

    # Frequency grid
    y, x = np.mgrid[-rows//2:rows//2, -cols//2:cols//2]
    radius = np.sqrt(x**2 + y**2)
    radius[radius == 0] = 1

    # Log-Gabor
    log_gabor = np.exp(-(np.log(radius / (rows/4))**2) / (2 * (np.log(sigma))**2))

    # Apply filter
    filtered = fft_shift * log_gabor
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))

    # Feature: mean + std
    #return np.array([result.mean(), result.std(), result.max()])
    return np.array([result.mean(), result.std()])


# ============================================================
# Zernike Moments
# ============================================================
# def zernike_moments(block, order=5):
#     """
#     Compute magnitude of Zernike moments up to given order.
#     """
#     block = cv2.resize(block, (64, 64))
#     block = block.astype(np.float32) / 255.0

#     rows, cols = block.shape
#     y, x = np.indices((rows, cols))
#     x = 2*(x-cols/2)/cols
#     y = 2*(y-rows/2)/rows
#     r = np.sqrt(x**2 + y**2)
#     theta = np.arctan2(y, x)

#     mask = r <= 1
#     features = []

#     for n in range(order+1):
#         for m in range(-n, n+1, 2):
#             if (n - abs(m)) % 2 == 0:
#                 radial = r**n
#                 moment = np.sum(block[mask] * radial[mask] * 
#                                 np.exp(-1j*m*theta[mask]))
#                 features.append(np.abs(moment))

#     return np.array(features)


def zernike_moments(block, degree=5, size=64):
    """
    Compute rotation-invariant Zernike moments
    using Mahotas implementation.

    Parameters:
    - degree : maximum Zernike degree (paper uses 5)
    - size   : resizing dimension (block → size x size)

    Returns:
    - 1D numpy array of Zernike magnitudes
    """

    # Resize block to square
    block = cv2.resize(block, (size, size))

    # Normalize to uint8 (required by mahotas)
    block = cv2.normalize(block, None, 0, 255,
                          cv2.NORM_MINMAX).astype(np.uint8)

    # Radius = half size
    radius = size // 2

    # Compute Zernike moments
    features = mahotas.features.zernike_moments(
        block,
        radius=radius,
        degree=degree
    )

    return np.array(features)

def zernike_moments_v2(block, degree=5, size=64):
    """
    Compute rotation-invariant Zernike moments using Mahotas.
    Research-correct version with circular masking.
    """

    # Resize to square
    block = cv2.resize(block, (size, size))

    # Normalize to uint8
    block = cv2.normalize(block, None, 0, 255,
                          cv2.NORM_MINMAX).astype(np.uint8)

    radius = size // 2

    # ---- Circular mask (important) ----
    y, x = np.ogrid[:size, :size]
    center = size // 2
    mask = (x - center)**2 + (y - center)**2 <= radius**2

    block_masked = np.zeros_like(block)
    block_masked[mask] = block[mask]

    # Compute Zernike moments
    features = mahotas.features.zernike_moments(
        block_masked,
        radius=radius,
        degree=degree
    )

    return np.array(features)


# ============================================================
# LBP feature
# ============================================================
def lbp_features(block, P=8, R=1):
    """
    Compute histogram of LBP.
    """
    lbp = local_binary_pattern(block, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, P+3),
                           range=(0, P+2))
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-6)
    return hist

# ============================================================
# Extract features for one image modality
# ============================================================
def extract_modality_features(image):
    """
    Extract block-based LGB, ZM, LBP features for one modality.
    Returns concatenated vector.
    """
    blocks = extract_blocks(image)

    G, Z, L = [], [], []

    for b in blocks:
        G.append(log_gabor_filter(b))
        Z.append(zernike_moments_v2(b))
        L.append(lbp_features(b))

    G = np.concatenate(G)
    Z = np.concatenate(Z)
    L = np.concatenate(L)

    return G, Z, L


# ============================================================
# Feature concatenation (Mi)
# ============================================================
def fuse_modalities(face, left_iris, right_iris):
    """
    Build unified feature vector for one subject.
    """
    Gif, Zif, Lif = extract_modality_features(face)
    Gil, Zil, Lil = extract_modality_features(left_iris)
    Gir, Zir, Lir = extract_modality_features(right_iris)

    # Gi, Zi, Li
    Gi = np.concatenate([Gif, Gil, Gir])
    Zi = np.concatenate([Zif, Zil, Zir])
    Li = np.concatenate([Lif, Lil, Lir])

    # Mi = [Gi, Zi, Li]
    Mi = np.concatenate([Gi, Zi, Li])

    return Mi