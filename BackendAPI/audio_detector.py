import librosa
import numpy as np

def detect_fake(filename, model):
    sound_signal, sample_rate = librosa.load(filename, res_type="kaiser_fast")
    
    mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=13)
    
    # Asegurar que haya al menos 200 frames
    if mfcc_features.shape[1] < 200:
        # Padding con ceros si tiene menos de 200
        pad_width = 200 - mfcc_features.shape[1]
        mfcc_features = np.pad(mfcc_features, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Cortar si tiene más
        mfcc_features = mfcc_features[:, :200]
    
    # Reshape
    input_features = mfcc_features.reshape(1, 13, 200, 1)

    result_array = model.predict(input_features)
    result_classes = ["Deepfake", "Real"]
    result = np.argmax(result_array[0])
    
    distribution = f"Softmax distribution: FAKE={result_array[0][0]:.4f}, REAL={result_array[0][1]:.4f}"
    
    return result_classes[result], distribution