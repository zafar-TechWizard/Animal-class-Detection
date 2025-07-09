import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern
from scipy.stats import skew, kurtosis





scaler = joblib.load("./model/scaler.pkl")
pca = joblib.load("./model/pca.pkl")
svm_model = joblib.load("./model/final_svm_model.pkl")
label_encoder = joblib.load("./model/label_encoder.pkl")
selector = joblib.load("./model/selector.pkl") 



def extract_features(image):
    features = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HOG
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    features.extend(hog_features)

    # LBP
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-7)
    features.extend(lbp_hist)

    # Color Moments
    for color_space in [image, hsv]:
        for channel in range(3):
            channel_data = color_space[:, :, channel].flatten()
            features.append(np.mean(channel_data))
            features.append(np.std(channel_data))
            features.append(skew(channel_data))
            features.append(kurtosis(channel_data))

    # Color Correlogram
    for channel in range(3):
        ch = image[:, :, channel]
        shifted_right = np.roll(ch, 1, axis=1)
        shifted_down = np.roll(ch, 1, axis=0)
        features.append(np.mean(ch[:, :-1] == shifted_right[:, :-1]))
        features.append(np.mean(ch[:-1, :] == shifted_down[:-1, :]))

    # Haralick
    gray_scaled = (gray / 16).astype(int)
    for d in [1, 2, 3]:
        glcm = np.zeros((16, 16))
        for i in range(gray_scaled.shape[0]):
            for j in range(gray_scaled.shape[1] - d):
                glcm[gray_scaled[i, j], gray_scaled[i, j + d]] += 1
        glcm_norm = glcm / (glcm.sum() + 1e-7)
        features.append(np.sum(np.square(np.arange(16) - np.arange(16).reshape(-1, 1)) * glcm_norm))
        features.append(np.sum(glcm_norm / (1 + np.square(np.arange(16) - np.arange(16).reshape(-1, 1)))))
        features.append(np.sum(np.square(glcm_norm)))

    return np.array(features)










def predict_animal_species(image_path):
    
    image = cv2.imread(image_path)
    image = cv2.resize(image, (200, 200))

    features = extract_features(image)
    features = np.nan_to_num(features)

    features_selected = selector.transform([features]) 

    features_scaled = scaler.transform(features_selected)

    features_pca = pca.transform(features_scaled)

    pred_encoded = svm_model.predict(features_pca)[0]
    return label_encoder.inverse_transform([pred_encoded])[0]


if __name__ == "__main__":
    image_path = "image.png"  
    species = predict_animal_species(image_path)
    print(f"The predicted species for the image is: {species}")