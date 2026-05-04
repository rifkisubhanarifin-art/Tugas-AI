import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Membaca data dari file CSV
data = pd.read_csv('Mall_Customers.csv')
# Menampilkan judul aplikasi
st.title('Analisis Data dengan Streamlit')
# Menampilkan data dalam bentuk tabel
st.subheader('Data Pelanggan Mall')
st.dataframe(data)
# Menampilkan statistik deskriptif
st.subheader('Statistik Deskriptif')
st.write(data.describe())
# Menampilkan distribusi usia pelanggan
st.subheader('Distribusi Usia Pelanggan')
plt.figure(figsize=(10, 6))
plt.hist(data['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribusi Usia Pelanggan')
plt.xlabel('Usia')
plt.ylabel('Frekuensi')
st.pyplot(plt)
# Menampilkan distribusi pendapatan tahunan pelanggan
st.subheader('Distribusi Pendapatan Tahunan Pelanggan')
plt.figure(figsize=(10, 6))
plt.hist(data['Annual Income (k$)'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribusi Pendapatan Tahunan Pelanggan')
plt.xlabel('Pendapatan Tahunan (k$)')
plt.ylabel('Frekuensi')
st.pyplot(plt)
# Menampilkan distribusi skor pengeluaran pelanggan
st.subheader('Distribusi Skor Pengeluaran Pelanggan')
plt.figure(figsize=(10, 6))
plt.hist(data['Spending Score (1-100)'], bins=20, color='salmon', edgecolor='black')
plt.title('Distribusi Skor Pengeluaran Pelanggan')
plt.xlabel('Skor Pengeluaran (1-100)')
plt.ylabel('Frekuensi')
st.pyplot(plt)
# Menampilkan hubungan antara usia dan skor pengeluaran
st.subheader('Hubungan antara Usia dan Skor Pengeluaran')
plt.figure(figsize=(10, 6))
plt.scatter(data['Age'], data['Spending Score (1-100)'], color='orange', edgecolor='black')
plt.title('Hubungan antara Usia dan Skor Pengeluaran')
plt.xlabel('Usia')
plt.ylabel('Skor Pengeluaran (1-100)')
st.pyplot(plt)

# Menampilkan hubungan antara pendapatan tahunan dan skor pengeluaran
st.subheader('Hubungan antara Pendapatan Tahunan dan Skor Pengeluaran')
plt.figure(figsize=(10, 6))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], color='purple', edgecolor='black')
plt.title('Hubungan antara Pendapatan Tahunan dan Skor Pengeluaran')
plt.xlabel('Pendapatan Tahunan (k$)')
plt.ylabel('Skor Pengeluaran (1-100)')
st.pyplot(plt)

#Kesimpulan
st.subheader('Kesimpulan')
text = '''Dari analisis data pelanggan mall, dapat disimpulkan bahwa terdapat variasi dalam usia, pendapatan tahunan, dan skor pengeluaran pelanggan. Distribusi usia menunjukkan bahwa sebagian besar pelanggan berada dalam rentang usia 20-40 tahun. Distribusi pendapatan tahunan menunjukkan bahwa sebagian besar pelanggan memiliki pendapatan antara 40-70 k$. Sedangkan distribusi skor pengeluaran menunjukkan bahwa sebagian besar pelanggan memiliki skor pengeluaran antara 40-60. Hubungan antara usia dan skor pengeluaran menunjukkan bahwa tidak ada pola yang jelas, sementara hubungan antara pendapatan tahunan dan skor pengeluaran menunjukkan bahwa pelanggan dengan pendapatan lebih tinggi cenderung memiliki skor pengeluaran yang lebih tinggi.'''

st.write(f'<div style="text-align: justify;">{text}</div>', unsafe_allow_html=True)