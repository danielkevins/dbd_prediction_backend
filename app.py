from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import sqlite3
import joblib  # Mengganti pickle dengan joblib
import numpy as np
import traceback

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. LOAD PIPELINE SVR DAN DAFTAR FITUR
# ==========================================
try:
    # Memuat file pipeline yang berisi scaler dan model sekaligus
    model_package = joblib.load('pipeline_svr_dbd.joblib')
    
    pipeline_svr = model_package['pipeline']
    fitur_wajib = model_package['fitur_urutan'] 
    
    print("Pipeline SVR dan Daftar Fitur berhasil dimuat!")
    print(f"Fitur yang digunakan: {fitur_wajib}")
except Exception as e:
    print(f"Gagal memuat pipeline: {e}")

def clean_numeric(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.')
    try:
        return float(val)
    except:
        return 0.0

# ==========================================
# 2. ENDPOINT DASHBOARD STATS 
# ==========================================
@app.route('/api/dashboard-stats', methods=['GET'])
def get_dashboard_stats():
    tahun_input = request.args.get('tahun', default='2024')
    try:
        df = pd.read_csv('data_dbd_mergeNew.csv')
        
        df['jumlah_penduduk'] = df['jumlah_penduduk'].apply(clean_numeric)
        df['kasus'] = df['kasus'].apply(clean_numeric).astype(int)
        df['kasus_meninggal'] = df['kasus_meninggal'].apply(clean_numeric).astype(int)
        df['periode_dt'] = pd.to_datetime(df['periode'], format='%m/%Y')
        df['tahun'] = df['periode_dt'].dt.year
        df['kecamatan'] = df['kecamatan'].str.upper().str.strip()

        df_year = df[df['tahun'] == int(tahun_input)].copy()

        total_kasus = int(df_year['kasus'].sum())
        total_meninggal = int(df_year['kasus_meninggal'].sum())
        
        kec_summary = df_year.groupby('kecamatan').agg({
            'kasus': 'sum',
            'kasus_meninggal': 'sum',
            'jumlah_penduduk': 'mean'
        })
        
        kec_summary['ir_tahunan'] = (kec_summary['kasus'] / kec_summary['jumlah_penduduk']) * 100000
        
        map_data = {}
        for index, row in kec_summary.iterrows():
            map_data[index] = {
                "ir": round(float(row['ir_tahunan']), 2),
                "kasus": int(row['kasus']),
                "meninggal": int(row['kasus_meninggal'])
            }

        monthly = df_year.groupby(df_year['periode_dt'].dt.month)['kasus'].sum()
        bulan_nama = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']
        trend_data = [{"label": bulan_nama[m-1], "value": int(v)} for m, v in monthly.items()]

        rendah = int(len(kec_summary[kec_summary['ir_tahunan'] < 20]))
        sedang = int(len(kec_summary[(kec_summary['ir_tahunan'] >= 20) & (kec_summary['ir_tahunan'] <= 55)]))
        tinggi = int(len(kec_summary[kec_summary['ir_tahunan'] > 55]))

        top_kec = kec_summary['kasus'].sort_values(ascending=False).head(10).to_dict()

        return jsonify({
            "cards": {
                "total_kasus": total_kasus,
                "avg_ir": round((total_kasus / kec_summary['jumlah_penduduk'].sum()) * 100000, 2) if not kec_summary.empty else 0,
                "total_meninggal": total_meninggal,
            },
            "map_data": map_data,
            "monthly_trend": trend_data,
            "ir_distribution": {
                "Rendah (<20)": rendah,
                "Sedang (20-55)": sedang,
                "Tinggi (>55)": tinggi
            },
            "top_kecamatan": top_kec
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ==========================================
# 3. ENDPOINT PREDIKSI DINAMIS SVR PIPELINE
# ==========================================
@app.route('/api/predict-full', methods=['GET'])
def get_predict_full():
    try:
        # --- A. Ambil Data Aktual Terbaru dari Database ---
        conn = sqlite3.connect('database_dbd_cuaca.db')
        # PENTING: Tambahkan max_temp dan avg_temp pada query SELECT
        query = "SELECT periode, kasus, rainfall, avg_humidity, max_temp, avg_temp FROM data_kasus_cuaca ORDER BY periode DESC LIMIT 5" 
        df_exog = pd.read_sql_query(query, conn)
        conn.close()

        if df_exog.empty:
            return jsonify({"error": "Database kosong!"}), 400
            
        df_exog = df_exog.iloc[::-1].reset_index(drop=True)

        hist_labels = [pd.to_datetime(d).strftime('%b %y') for d in df_exog['periode']]
        hist_values = [int(v) for v in df_exog['kasus']]

        # --- B. Buat 1 Baris Masa Depan ---
        last_date = pd.to_datetime(df_exog['periode'].iloc[-1])
        future_date = last_date + pd.DateOffset(months=1)
        
        # Sertakan semua kolom cuaca dengan nilai NaN untuk baris masa depan
        dummy_df = pd.DataFrame({
            'periode': [future_date],
            'kasus': [np.nan],
            'rainfall': [np.nan],
            'avg_humidity': [np.nan],
            'max_temp': [np.nan],
            'avg_temp': [np.nan]
        })
        
        df_combined = pd.concat([df_exog, dummy_df], ignore_index=True)

        # --- C. Feature Engineering (Eksklusif untuk 5 Fitur Pilihan) ---
        # 1. kasus_lag_1
        df_combined['kasus_lag_1'] = df_combined['kasus'].shift(1)
        
        # 2. max_temp_lag_1
        df_combined['max_temp_lag_1'] = df_combined['max_temp'].shift(1)
        
        # 3. avg_temp_roll_mean_3
        df_combined['avg_temp_roll_mean_3'] = df_combined['avg_temp'].shift(1).rolling(window=3).mean()
        
        # 4. avg_humidity_lag_1
        df_combined['avg_humidity_lag_1'] = df_combined['avg_humidity'].shift(1)
        
        # 5. rainfall_roll_mean_3
        df_combined['rainfall_roll_mean_3'] = df_combined['rainfall'].shift(1).rolling(window=3).mean()

        # --- D. Eksekusi Prediksi ---
        # Ambil hanya baris terakhir (baris target prediksi)
        df_prediksi = df_combined.tail(1).copy()
        pred_labels = [d.strftime('%b %y') for d in df_prediksi['periode']]
        
        # Filter dataframe HANYA mengambil kolom yang ada di fitur_wajib
        # Ini mencegah error urutan kolom dan kolom yang tidak dikenali
        df_model_input = df_prediksi[fitur_wajib]

        if df_model_input.isnull().values.any():
             return jsonify({"error": "Data histori di database tidak cukup (Butuh minimal 4 bulan data beruntun untuk menghitung rolling mean)."}), 400

        # Prediksi menggunakan Pipeline (Scaler otomatis berjalan di dalam fungsi ini)
        predictions = pipeline_svr.predict(df_model_input)
        
        # Format hasil
        pred_values = [max(0, round(float(v))) for v in predictions]

        # --- E. Kirim Response ke Frontend ---
        return jsonify({
            "historical": { 
                "labels": hist_labels, 
                "values": hist_values 
            },
            "forecast": { 
                "labels": pred_labels, 
                "values": pred_values 
            }
        })

    except KeyError as e:
        traceback.print_exc()
        return jsonify({"error": f"Kolom tidak ditemukan di database: {str(e)}"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

#if __name__ == '__main__':
#    app.run(debug=True, port=5000)

app = app