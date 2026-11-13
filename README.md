# GoEmotions-Ekman

Proyek klasifikasi emosi multi-label berbasis teks dengan pendekatan fine-tuning model bahasa `DistilBERT` menggunakan dataset GoEmotions yang dipetakan ke 7 label emosi Ekman:

- anger
- disgust
- fear
- joy
- sadness
- surprise
- neutral

Project ini mencakup pipeline pelatihan, evaluasi, analisis kesalahan, serta aplikasi antarmuka web untuk melakukan prediksi emosi dari teks dalam bahasa Inggris.

## Fitur utama

- Klasifikasi multi-label emosi pada teks
- Fine-tuning model `distilbert-base-uncased`
- Validasi dataset dan pengecekan label per split
- Eksperimen konfigurasi model (E1 hingga E6)
- Evaluasi metrik seperti micro-F1, macro-F1, dan Hamming loss
- Inference untuk prediksi single text
- Antarmuka GUI berbasis Streamlit

## Struktur proyek

```text
GoEmotions-Ekman/
├── README.md
├── main.py
├── requirements.txt
├── GUI/
│   └── app.py
├── data/
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   └── final/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── notebooks/
│   └── eda.ipynb
├── outputs/
│   ├── final_config.json
│   ├── checkpoints/
│   │   └── E6/
│   └── metrics/
│       ├── E2_summary.json
│       ├── E3_summary.json
│       ├── E4_summary.json
│       ├── E5_summary.json
│       ├── E6_summary.json
│       ├── final_test_metrics.json
│       └── ...
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── error_analysis.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── model.py
│   ├── preprocessing.py
│   └── train.py
├── Rancangan_Tahap_1_MLFT.ipynb
└── .gitignore
```

## Teknologi yang dipakai

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Datasets
- Pandas
- NumPy
- scikit-learn
- Streamlit

## Persiapan lingkungan

1. Clone repository:

```bash
git clone https://github.com/rahelitapasaribu/GoEmotions-Ekman.git
cd GoEmotions-Ekman
```

2. Buat virtual environment (opsional tapi disarankan):

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependency:

```bash
pip install -r requirements.txt
```

## Dataset

Dataset yang dipakai berada di folder `data/final/` dengan tiga split:

- `train.csv`
- `validation.csv`
- `test.csv`

Setiap baris berisi:

- `text`: kalimat input
- label emosi: `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, `neutral`

Validasi dilakukan untuk memastikan:

- teks tidak null atau kosong
- label numeric 0/1
- setiap baris minimal memiliki satu label aktif
- tidak ada data duplikat antar split
- setiap label memiliki cukup contoh positif

## Menjalankan training

Project menyediakan eksperimen konfigurasi yang terdefinisi pada `src/config.py`.

Cara menjalankan training:

```bash
python main.py --experiment E1
```

Eksperimen yang tersedia:

- `E1` : baseline
- `E2` : learning rate naik
- `E3` : learning rate lebih tinggi
- `E4` : batch size lebih besar
- `E5` : epoch lebih lama
- `E6` : max_length lebih panjang

Contoh lain:

```bash
python main.py --experiment E6
```

Setiap eksperimen akan:

- memuat train/validation dataset
- membangun tokenizer dan dataloader
- melatih model DistilBERT
- menyimpan checkpoint di `outputs/checkpoints/<experiment_name>/`
- menyimpan history metrik ke `outputs/metrics/<experiment_name>_history.csv`
- menyimpan ringkasan eksperimen ke `outputs/metrics/<experiment_name>_summary.json`

## Prediksi menggunakan model

Model final yang digunakan oleh aplikasi inference disimpan pada konfigurasi `outputs/final_config.json`.

Untuk menjalankan prediksi lewat Python:

```python
from src.inference import EmotionPredictor

predictor = EmotionPredictor()
result = predictor.predict("I am so happy to finally finish this project!")

print(result["predicted_labels"])
print(result["probabilities"])
```

## Menjalankan GUI

Aplikasi web dibuat dengan Streamlit. Jalankan: 

```bash
streamlit run GUI/app.py
```

Aplikasi ini memungkinkan pengguna memasukkan teks dan menampilkan:

- emosi yang terdeteksi
- skor probabilitas tiap label
- threshold yang digunakan
- visualisasi hasil klasifikasi

## Hasil model

Model terbaik yang dipilih pada proyek ini adalah eksperimen `E6` dengan threshold `0.4` berdasarkan metrik validation macro-F1.

Ringkasan hasil evaluasi final:

- validation macro F1: sekitar `0.7098`
- validation micro F1: sekitar `0.7433`
- test micro F1: sekitar `0.7259`
- test macro F1: sekitar `0.6712`
- hamming loss: sekitar `0.1584`

Detail hasil dapat dilihat di:

- `outputs/final_config.json`
- `outputs/metrics/final_test_metrics.json`
- `outputs/metrics/*_summary.json`

## Pipeline utama

Proyek ini mengikuti alur kerja berikut:

1. Load dataset train, validation, dan test
2. Validasi struktur dataset dan label
3. Tokenisasi teks dengan tokenizer DistilBERT
4. Fine-tuning model multi-label classification
5. Evaluasi hasil pada validation/test set
6. Menyimpan checkpoint terbaik
7. Menggunakan model final untuk inference
8. Menyajikan prediksi di UI Streamlit

## Catatan penting

- Model dirancang untuk klasifikasi multi-label, sehingga satu kalimat dapat memiliki lebih dari satu emosi aktif.
- Label default threshold yang dipakai adalah `0.5`, namun konfigurasi final model menentukan threshold `0.4` untuk hasil terbaik.
- Proyek ini fokus pada teks berbahasa Inggris.

## Kontributor

Proyek ini dikembangkan sebagai tugas/eksperimen Machine Learning for Text dalam konteks klasifikasi emosi.

## Referensi

- GoEmotions dataset
- DistilBERT
- Ekman emotion taxonomy
