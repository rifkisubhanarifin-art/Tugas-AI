# Regresi Logistik

Regresi logistik berbeda dari regresi linear biasa. Kalau regresi linear menghasilkan angka (misalnya omzet), regresi logistik menghasilkan kategori, biasanya berupa 0 atau 1.

Contohnya:

- 1 = produk laris
- 0 = produk tidak laris

Jadi, model ini digunakan untuk klasifikasi, bukan prediksi angka.

Cara kerjanya adalah dengan menghitung probabilitas (kemungkinan). Misalnya:

- Produk A → peluang laris 80%
- Produk B → peluang laris 30%

Lalu biasanya dibuat batas (threshold), misalnya:

- Jika > 50% → dianggap laris
- Jika < 50% → tidak laris

Contoh di dropshipping:
Input:

- harga produk
- rating
- jumlah review

Output:

- apakah produk layak dijual atau tidak

Regresi logistik sangat berguna untuk membantu pengambilan keputusan, terutama saat kita harus memilih dari banyak pilihan.
