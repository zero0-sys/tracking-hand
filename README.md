# Gesture-to-Text & Text-to-Speech (Realtime)

**Hand Gesture Recognition menggunakan MediaPipe + LSTM**

## Deskripsi Proyek

Proyek ini merupakan sistem **pengenalan gesture tangan secara realtime** menggunakan kamera (IP Webcam / webcam), yang mampu:

* mengenali **gesture tangan dinamis**
* mengonversi gesture menjadi **teks**
* menyusun **kalimat otomatis**
* membacakan hasilnya menggunakan **Text-to-Speech (TTS)** secara **offline**

Sistem ini dirancang agar:

* stabil (tidak berubah saat gesture ragu)
* tidak crash walaupun frame drop
* mudah ditambah gesture baru

---

## Fitur Utama

* Realtime hand tracking (MediaPipe)
* Deep Learning (LSTM)
* Multi-gesture recognition
* Live text (tidak menumpuk)
* Kalimat otomatis (`aku + saja + baik`)
* Text-to-Speech (offline)
* Auto-clear teks setelah tangan hilang
* Gesture khusus (`CLEAR`, `STOP`)
* Class `NONE` untuk gesture kosong (anti salah deteksi)

---

## Arsitektur Sistem

```
Camera (IP Webcam)
        ↓
MediaPipe Hands (21 landmark)
        ↓
Feature Extraction (126 fitur)
        ↓
Sequence Buffer (30 frame)
        ↓
LSTM Model
        ↓
Gesture Label
        ↓
Text Display + TTS
```

---

## 📁 Struktur Folder

```
detect-hand-for-text/
│
├── data/
│   ├── aku/
│   ├── saja/
│   ├── baik/
│   ├── CLEAR/
│   ├── STOP/      (opsional)
│   └── NONE/      (WAJIB)
│
├── gesture/
│   ├── collect_data.py
│   ├── train_model.py
│   └── realtime_run.py
│
├── ar310/         # virtual environment (Python 3.10)
├── gesture_lstm.h5
└── README.md
```

---

## Dataset

* Setiap gesture direkam sebagai **30 frame**
* Setiap frame memiliki **126 fitur** (21 landmark × xyz × 2 tangan)
* Format data: `.npy`

### Aturan Dataset (WAJIB)

| Aturan        | Keterangan         |
| ------------- | ------------------ |
| Jumlah sample | 50–100 per gesture |
| Gesture mirip | ❌ Tidak disarankan |
| Class NONE    | ✅ Paling banyak    |
| Folder kosong | ❌ Dilarang         |

---

## Instalasi

### Python

Gunakan **Python 3.10**

```bash
python --version
```

### Virtual Environment

```bash
python -m venv ar310
source ar310/bin/activate
```

### Install Dependency

```bash
pip install opencv-python mediapipe numpy tensorflow pyttsx3
```

Linux (audio):

```bash
sudo apt install espeak-ng libespeak1
```

---

## Setup Kamera

Gunakan **IP Webcam (Android)**
Contoh URL:

```python
IP_CAM = "http://192.168.1.4:8080/video"
```

Pastikan:

* HP & laptop satu WiFi
* Resolusi disarankan: **640×480**

---

## ✋ Mengumpulkan Data Gesture

```bash
./ar310/bin/python gesture/collect_data.py
```

Masukkan label:

```
Masukkan label gesture: aku
```

Lakukan gesture **pelan & konsisten** sampai:

```
Frames: 30/30
Saved: aku 0
```

---

## Training Model

```bash
./ar310/bin/python gesture/train_model.py
```

Output:

```
gesture_lstm.h5
```

**WAJIB train ulang jika menambah / menghapus gesture**

---

##  Menjalankan Realtime Gesture + TTS

```bash
./ar310/bin/python gesture/realtime_run.py
```

### Cara Pakai

* Tampilkan gesture → teks muncul
* Ganti gesture → teks berganti
* Gesture `CLEAR` → hapus teks
* Gesture `STOP` → baca kalimat
* Tangan turun 3 detik → auto-clear

---

## Sistem Anti Salah Deteksi

Sistem menggunakan:

* Confidence threshold
* Top-2 confidence margin
* Stable frame counter
* Class `NONE` untuk gesture kosong

Hasil:

* Tidak random STOP
* Tidak berubah saat gesture ragu
* Tidak crash

---

## Text-to-Speech

* Offline (pyttsx3)
* Tidak perlu internet
* Anti pengulangan suara

---

## Pengembangan Lanjutan

* Android Native App
* Grammar otomatis
* Custom voice / voice cloning
* Pose-based classifier (lebih ringan)
* API / Web App

---

## Catatan Penting

* Gesture **harus berbeda jelas**
* Dataset **lebih penting dari model**
* Jangan campur gesture transisi
* NONE class sangat krusial

---

## Author

Dikembangkan sebagai sistem **Gesture-to-Text & TTS realtime**
menggunakan **MediaPipe & Deep Learning**

---

## Lisensi

Proyek ini bersifat **bebas digunakan untuk pembelajaran & pengembangan**.

