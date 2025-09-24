import numpy as np
from scipy.signal import convolve2d
import cv2

import os

def psnr(imageA, imageB):
    """
    PSNR: The higher the PSNR value, the closer the reconstructed image is to the original image, and thus, the better the quality.
          It is expressed in decibels (dB), and higher values indicate a higher quality.
    """
    mse = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    mse /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
    if mse == 0:
        return float('inf')  # avoid division by zero
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def ssim(imageA, imageB, C1=6.5025, C2=58.5225, window_size=11, sigma=1.5):
    """
    Calculate the SSIM (Structural Similarity Index) between two images.
    SSIM: The SSIM value ranges from -1 to 1, where 1 indicates the highest similarity between the two images.
          A value of 1 means that the two images are identical, and a value of -1 means that the two images are completely different.
    """
    window = np.outer(np.exp(-np.arange(-(window_size // 2), window_size // 2 + 1) ** 2 / (2 * sigma ** 2)),
                      np.exp(-np.arange(-(window_size // 2), window_size // 2 + 1) ** 2 / (2 * sigma ** 2)))
    window /= np.sum(window)

    mu1 = convolve2d(imageA, window, mode='valid')
    mu2 = convolve2d(imageB, window, mode='valid')
    sigma1_sq = convolve2d(imageA ** 2, window, mode='valid') - mu1 ** 2
    sigma2_sq = convolve2d(imageB ** 2, window, mode='valid') - mu2 ** 2
    sigma12 = convolve2d(imageA * imageB, window, mode='valid') - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)

def multi_channel_ssim(imageA, imageB):
    """
    Compute SSIM over the three channels separately and then average.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    ssim_sum = 0
    for i in range(imageA.shape[2]):
        channelA = imageA[:, :, i].astype(np.float64)
        channelB = imageB[:, :, i].astype(np.float64)
        ssim_sum += ssim(channelA, channelB, C1, C2)
    return ssim_sum / imageA.shape[2]

authentic_path = 'DeepImageBlending/background/rajeev_new'
synthetic_path = 'DeepImageBlending/results/rajeev_new'
all_file_names = os.listdir(synthetic_path)

all_psnr = []
all_ssim = []

for id_file_name in all_file_names:
    authentic_id_path = os.path.join(authentic_path, id_file_name.split('.')[0] + '.jpg')
    synthetic_id_path = os.path.join(synthetic_path, id_file_name)
    authentic_image = cv2.imread(authentic_id_path)
    synthetic_image = cv2.imread(synthetic_id_path)
    # synthetic_image = cv2.resize(synthetic_image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    # cv2.imwrite(os.path.join('DeepImageBlending/results/synthetic_rajeev_id_512', synthetic_id_path.split('/')[-1]), synthetic_image)
    psnr_value = psnr(authentic_image, synthetic_image)
    ssim_value = multi_channel_ssim(authentic_image, synthetic_image)
    print(f"ID: {id_file_name}")
    print(f"PSNR: {psnr_value}")
    print(f"SSIM: {ssim_value}")
    all_psnr.append(psnr_value)
    all_ssim.append(ssim_value)

mean = np.mean(all_psnr)
std = np.std(all_psnr)

print("Mean:", mean)
print("Standard Deviation:", std)

mean = np.mean(all_ssim)
std = np.std(all_ssim)

print("Mean:", mean)
print("Standard Deviation:", std)
