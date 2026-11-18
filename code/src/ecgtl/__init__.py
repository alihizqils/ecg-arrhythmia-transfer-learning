"""Pipeline eksperimen studi kinerja transfer learning untuk deteksi aritmia EKG.

Paket ini mengimplementasikan rancangan eksperimen penelitian:
konversi sinyal EKG 1D menjadi citra 2D grayscale, penanganan ketidakseimbangan
kelas lewat pembobotan pada fungsi kerugian, perbandingan empat skenario
transfer learning pada tiga arsitektur lightweight, serta validasi
interpretabilitas memakai SHAP.
"""

__version__ = "0.1.0"

CLASSES = ("N", "S", "V", "AF", "VT")
