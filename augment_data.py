import pickle
import numpy as np
from sklearn.utils import shuffle

"""
SIMPLE DATA AUGMENTATION untuk Hand Landmarks
Tidak perlu Kaggle/GPU - jalan di CPU biasa!

Augmentasi yang aman untuk hand landmarks:
1. Horizontal flip (mirror)
2. Small rotation
3. Small scaling
4. Small translation (shift)
5. Add small noise
"""

print("=" * 70)
print("🔄 DATA AUGMENTATION - Simple & CPU-Friendly")
print("=" * 70)

# Load original data
print("\n📂 Loading original data...")
data_dict = pickle.load(open('./data.pickle', 'rb'))
original_data = np.array(data_dict['data'])
original_labels = np.array(data_dict['labels'])

print(f"✓ Original dataset: {len(original_data)} samples")

# ============================================
# AUGMENTATION FUNCTIONS
# ============================================

def horizontal_flip(landmarks):
    """Mirror flip (x coordinates)"""
    augmented = landmarks.copy()
    # Flip x coordinates (setiap 2 element, karena format: x, y, x, y, ...)
    augmented[::2] = 1.0 - augmented[::2]  # x_new = 1 - x_old (karena normalized 0-1)
    return augmented

def add_noise(landmarks, noise_level=0.02):
    """Add small random noise"""
    noise = np.random.normal(0, noise_level, landmarks.shape)
    augmented = landmarks + noise
    # Clip to valid range [0, 1]
    augmented = np.clip(augmented, 0, 1)
    return augmented

def scale(landmarks, scale_factor=1.1):
    """Scale landmarks slightly"""
    # Find center
    x_coords = landmarks[::2]
    y_coords = landmarks[1::2]
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    
    augmented = landmarks.copy()
    # Scale around center
    augmented[::2] = center_x + (x_coords - center_x) * scale_factor
    augmented[1::2] = center_y + (y_coords - center_y) * scale_factor
    
    # Clip to valid range
    augmented = np.clip(augmented, 0, 1)
    return augmented

def rotate_slight(landmarks, angle_deg=5):
    """Rotate landmarks slightly"""
    angle_rad = np.deg2rad(angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    # Find center
    x_coords = landmarks[::2]
    y_coords = landmarks[1::2]
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    
    augmented = landmarks.copy()
    
    # Rotate around center
    for i in range(0, len(landmarks), 2):
        x = landmarks[i] - center_x
        y = landmarks[i+1] - center_y
        
        new_x = x * cos_angle - y * sin_angle + center_x
        new_y = x * sin_angle + y * cos_angle + center_y
        
        augmented[i] = new_x
        augmented[i+1] = new_y
    
    # Clip to valid range
    augmented = np.clip(augmented, 0, 1)
    return augmented

# ============================================
# APPLY AUGMENTATION
# ============================================

print("\n🔄 Applying augmentations...")

augmented_data = []
augmented_labels = []

# Keep original data
augmented_data.extend(original_data)
augmented_labels.extend(original_labels)
print(f"   ✓ Original: {len(original_data)} samples")

# Augmentation strategies
augmentations = [
    ("Horizontal Flip", horizontal_flip),
    ("Noise (light)", lambda x: add_noise(x, 0.01)),
    ("Noise (medium)", lambda x: add_noise(x, 0.02)),
    ("Scale Up", lambda x: scale(x, 1.05)),
    ("Scale Down", lambda x: scale(x, 0.95)),
    ("Rotate +5°", lambda x: rotate_slight(x, 5)),
    ("Rotate -5°", lambda x: rotate_slight(x, -5)),
]

for aug_name, aug_func in augmentations:
    print(f"   ✓ Applying {aug_name}...", end=" ")
    count = 0
    for sample, label in zip(original_data, original_labels):
        try:
            augmented_sample = aug_func(sample)
            augmented_data.append(augmented_sample)
            augmented_labels.append(label)
            count += 1
        except Exception as e:
            print(f"\n   ⚠️  Error: {e}")
            continue
    print(f"{count} samples")

# Shuffle
augmented_data, augmented_labels = shuffle(
    np.array(augmented_data), 
    np.array(augmented_labels), 
    random_state=42
)

print(f"\n📊 Augmentation complete:")
print(f"   Original:  {len(original_data)} samples")
print(f"   Augmented: {len(augmented_data)} samples")
print(f"   Increase:  {len(augmented_data) / len(original_data):.1f}x")

# ============================================
# SAVE AUGMENTED DATA
# ============================================

print(f"\n💾 Saving augmented data...")

# Backup original
import shutil
if os.path.exists('data.pickle'):
    shutil.copy('data.pickle', 'data_original.pickle')
    print("   ✓ Original data backed up to 'data_original.pickle'")

# Save augmented
augmented_dict = {
    'data': augmented_data.tolist(),
    'labels': augmented_labels.tolist()
}

with open('data_augmented.pickle', 'wb') as f:
    pickle.dump(augmented_dict, f)
print("   ✓ Augmented data saved to 'data_augmented.pickle'")

print(f"\n{'='*70}")
print(f"✨ AUGMENTATION COMPLETED!")
print(f"{'='*70}")
print(f"\n📋 Next steps:")
print(f"   1. Copy augmented data:")
print(f"      cp data_augmented.pickle data.pickle")
print(f"   2. Train with augmented data:")
print(f"      python train_classifier.py")
print(f"\n   Or restore original:")
print(f"      cp data_original.pickle data.pickle")
print(f"{'='*70}")