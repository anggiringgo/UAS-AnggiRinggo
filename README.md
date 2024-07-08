## UAS - Anggi Ringgo Kuncoro Jati - 311910743

### 1.	Code: Baca gambar dan ubah menjadi gambar RGB.
```
import numpy as np
import matplotlib.pyplot as plt
import cv2

%matplotlib inline

# Read in the image
image = cv2.imread('data/foto.jpg')

# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)

```
![image](https://github.com/anggiringgo/Lab2Web/assets/81921974/187a715a-34e5-4e4b-a9d5-c66d23a22d45)


### 2. Sekarang kita harus menyiapkan data untuk K. Gambar adalah bentuk 3 dimensi tetapi untuk menerapkan K clustering pada gambar tersebut kita perlu membentuknya kembali menjadi array 2 dimensi.
#### Code:
```
# Membentuk ulang gambar menjadi susunan piksel 2D dan 3 nilai warna (RGB)
pixel_vals = image.reshape((-1,3))

# Ubah ke float type
pixel_vals = np.float32(pixel_vals)
```

### 3. Sekarang kita akan mengimplementasikan algoritma K untuk mensegmentasi suatu gambar.
#### Code: Mengambil k = 3, artinya algoritma akan mengidentifikasi 3 cluster pada gambar.
```
#baris kode di bawah ini menentukan kriteria agar algoritme berhenti berjalan, 
#yang akan terjadi adalah 100 iterasi dijalankan atau epsilon (yang merupakan akurasi yang dibutuhkan)
#menjadi 85%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

# lalu lakukan K clustering dengan jumlah cluster yang ditetapkan sebagai 3
# juga pusat acak pada awalnya dipilih untuk pengelompokan k
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# mengonversi data menjadi nilai 8-bit
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# membentuk ulang data menjadi dimensi gambar asli
segmented_image = segmented_data.reshape((image.shape))

plt.imshow(segmented_image)
```
## 4. Output
![image](https://github.com/anggiringgo/Lab2Web/assets/81921974/7e3ce7c8-60b9-432b-930c-1f51cb6b94ec)

## 5. Sekarang jika kita mengubah nilai k menjadi 6, kita mendapatkan Output berikut:
![image](https://github.com/anggiringgo/Lab2Web/assets/81921974/384cc8a0-606e-458e-a138-156af23e8087)

