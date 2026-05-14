# ==================== 1. IMPORT LIBRARY ====================
import numpy as np          # untuk operasi matriks dan array numerik
import pandas as pd         # untuk membaca file CSV
import re                   # untuk mencocokkan pola teks (regex), digunakan ekstrak tahun
from sklearn.model_selection import train_test_split  # opsional, untuk membagi data (kita buat manual)
from sklearn.preprocessing import StandardScaler      # untuk normalisasi fitur

# ==================== 2. FUNGSI PREPROCESSING DATA ====================
def extract_year(title):
    """Mengambil tahun dari judul film yang formatnya 'Nama Film (tahun)'"""
    match = re.search(r'\((\d{4})\)', title)   # cari pola 4 digit dalam kurung
    return int(match.group(1)) if match else 1990  # jika ditemukan, konversi ke int, else 1990

def clean_title(title):
    """Menghapus bagian tahun dari judul, menyisakan nama film saja"""
    return re.sub(r'\s*\(\d{4}\)', '', title)   # ganti spasi + (tahun) dengan string kosong

def load_and_preprocess(filepath):
    # baca file CSV
    df = pd.read_csv(filepath)
    
    # buat kolom 'year' dari hasil ekstraksi tahun pada judul
    df['year'] = df['title'].apply(extract_year)
    
    # buat kolom judul bersih (tanpa tahun)
    df['title_clean'] = df['title'].apply(clean_title)
    
    # buat kolom 'title_length' = panjang karakter judul bersih
    df['title_length'] = df['title_clean'].apply(len)
    
    # target (y): apakah film bergenre Comedy? (1 jika ya, 0 jika tidak)
    df['is_comedy'] = df['genres'].str.contains('Comedy', na=False).astype(int)
    
    # pilih fitur: tahun dan panjang judul
    X = df[['year', 'title_length']].values   # bentuk (jumlah_data, 2)
    y = df['is_comedy'].values.reshape(-1, 1) # bentuk (jumlah_data, 1)
    return X, y

# ==================== 3. MEMUAT DATA ====================
X_raw, y_raw = load_and_preprocess('movies.csv')

# StandardScaler: mengubah fitur agar memiliki mean=0 dan std=1 (penting untuk ANN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)   # fit dan transform fitur
X = X_scaled.T      # transpos: bentuk (2, jumlah_data) -> (n_features, m)
y = y_raw.T         # transpos: bentuk (1, jumlah_data) -> (output_size, m)

print(f"Ukuran data: {X.shape}, {y.shape}")
print(f"Proporsi film Comedy dalam dataset: {np.mean(y)*100:.2f}%")

# ==================== 4. FUNGSI-FUNGSI ANN ====================
def inisialisasi_parameter(input_size, hidden_size, output_size):
    """
    Membuat parameter (bobot dan bias) secara acak.
    input_size : jumlah neuron input (2)
    hidden_size: jumlah neuron hidden layer (4)
    output_size: jumlah neuron output (1)
    """
    np.random.seed(42)   # agar hasil inisialisasi konsisten setiap dijalankan
    
    # W1: bobot dari input ke hidden, ukuran (hidden_size, input_size)
    # diinisialisasi dengan bilangan acak kecil (standar 0.01)
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    # b1: bias hidden layer, ukuran (hidden_size, 1)
    b1 = np.zeros((hidden_size, 1))
    
    # W2: bobot dari hidden ke output, ukuran (output_size, hidden_size)
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    # b2: bias output layer, ukuran (output_size, 1)
    b2 = np.zeros((output_size, 1))
    
    parameter = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameter

def sigmoid(Z):
    """Fungsi aktivasi sigmoid: mengubah Z menjadi probabilitas antara 0 dan 1"""
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    """Fungsi aktivasi ReLU: max(0, Z)"""
    return np.maximum(0, Z)

def relu_derivative(Z):
    """Turunan dari ReLU: 1 jika Z>0, 0 jika Z<=0"""
    return (Z > 0).astype(int)

def forward_propagation(X, parameter):
    """Proses perambatan maju (forward) untuk menghasilkan output"""
    W1 = parameter["W1"]
    b1 = parameter["b1"]
    W2 = parameter["W2"]
    b2 = parameter["b2"]
    
    # Layer 1 (hidden): Z1 = W1.X + b1, lalu A1 = ReLU(Z1)
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    
    # Layer 2 (output): Z2 = W2.A1 + b2, lalu A2 = sigmoid(Z2) -> probabilitas
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    # Simpan nilai antara (cache) untuk digunakan pada backward propagation
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

def komputasi_biaya(Y, A2):
    """
    Menghitung binary cross-entropy loss (cost)
    Y : label sebenarnya (0 atau 1)
    A2: output prediksi (probabilitas)
    """
    m = Y.shape[1]   # jumlah sampel
    # Tambahkan epsilon 1e-8 untuk menghindari log(0)
    biaya = -np.sum(Y * np.log(A2 + 1e-8) + (1 - Y) * np.log(1 - A2 + 1e-8)) / m
    return np.squeeze(biaya)   # hilangkan dimensi berlebih

def backward_propagation(X, Y, parameter, cache):
    """
    Menghitung gradien (turunan) dari setiap parameter terhadap loss
    X : input (2, m)
    Y : label (1, m)
    parameter: berisi W1,b1,W2,b2
    cache: berisi Z1,A1,Z2,A2 dari forward propagation
    """
    m = X.shape[1]
    W2 = parameter["W2"]
    
    # dZ2 = A2 - Y (turunan loss terhadap Z2)
    dZ2 = cache["A2"] - Y
    dW2 = np.dot(dZ2, cache["A1"].T) / m   # gradien terhadap W2
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m   # gradien terhadap b2
    
    # dZ1 = (W2^T . dZ2) * relu_derivative(Z1)
    dZ1 = np.dot(W2.T, dZ2) * relu_derivative(cache["Z1"])
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads

def parameter_terbaru(parameter, grads, learning_rate):
    """Update parameter dengan gradient descent: param = param - learning_rate * grad"""
    for key in parameter.keys():
        parameter[key] -= learning_rate * grads["d" + key]
    return parameter

def train_neural_network(X, Y, input_size, hidden_size, output_size, epochs=1000, learning_rate=0.01):
    """Melatih ANN selama epoch tertentu"""
    parameter = inisialisasi_parameter(input_size, hidden_size, output_size)
    
    for i in range(epochs):
        # Forward propagation
        A2, cache = forward_propagation(X, parameter)
        # Hitung loss
        biaya = komputasi_biaya(Y, A2)
        # Backward propagation (hitung gradien)
        grads = backward_propagation(X, Y, parameter, cache)
        # Update parameter
        parameter = parameter_terbaru(parameter, grads, learning_rate)
        
        if i % 100 == 0:   # cetak loss setiap 100 epoch
            print(f"Epoch {i}: Cost = {biaya:.6f}")
    return parameter

def predict(X, parameter):
    """Melakukan prediksi dengan model yang sudah dilatih"""
    A2, _ = forward_propagation(X, parameter)
    # Threshold 0.5: jika prob > 0.5 -> kelas 1 (Comedy), else 0
    return (A2 > 0.5).astype(int)

# ==================== 5. LATIH DAN EVALUASI ====================
# Karena kita tidak ingin random split setiap run, kita buat split manual konsisten
m = X.shape[1]               # jumlah total sampel
indices = np.random.permutation(m)   # acak urutan indeks
split = int(0.8 * m)         # 80% untuk training
train_idx, test_idx = indices[:split], indices[split:]

X_train = X[:, train_idx]    # data training (fitur)
y_train = y[:, train_idx]    # label training
X_test  = X[:, test_idx]     # data testing
y_test  = y[:, test_idx]     # label testing

print(f"Jumlah data training: {X_train.shape[1]}, testing: {X_test.shape[1]}")

# Latih ANN
input_size  = X.shape[0]     # = 2 (tahun dan panjang judul)
hidden_size = 4              # bisa dicoba-coba
output_size = 1              # binary classification

trained_params = train_neural_network(
    X_train, y_train,
    input_size, hidden_size, output_size,
    epochs=2000,    # jumlah iterasi
    learning_rate=0.1
)

# Prediksi pada data test
y_pred = predict(X_test, trained_params)
akurasi = np.mean(y_pred == y_test) * 100
print(f"\nAkurasi pada data test: {akurasi:.2f}%")

# Tampilkan beberapa contoh prediksi
print("\nContoh hasil prediksi untuk 10 film pertama (test set):")
for i in range(min(10, X_test.shape[1])):
    pred = "Comedy" if y_pred[0, i] == 1 else "Non-Comedy"
    actual = "Comedy" if y_test[0, i] == 1 else "Non-Comedy"
    print(f"Film ke-{i+1} -> Prediksi: {pred} | Aktual: {actual}")