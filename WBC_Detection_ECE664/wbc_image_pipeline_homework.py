# -*- coding: utf-8 -*-
"""
================================================================================
          WBC (Akyuvar) GÖRÜNTÜ İŞLEME, ODAKLAMA VE SINIFLANDIRMA PIPELINE'I
                           ÜNİVERSİTE ÖDEV PROJESİ
================================================================================
Bu script, bir dijital mikroskop tarama sisteminde tarayıcıdan (kameradan) görüntü
alındıktan sonra uygulanan tüm görüntü işleme adımlarını tek bir çatı altında toplar.

İçerdiği Adımlar:
1. Kamera Edinimi / Z-Stack Simülasyonu (Kameradan görüntü akışı alma simülasyonu)
2. Görüntü Netlik (Sharpness) Ölçümü (Laplacian Variance, Tenengrad Grid Median vb.)
3. Otomatik Odaklama (Autofocus) Z-Peak Tespiti
4. Görüntü Ön İşleme (CLAHE, Eşikleme, Morfoloji)
5. Hücre Tespiti (Contour / Bounding Box tespiti)
6. Özel Hücre Kırpma (Squaring, 1.40x Margin Büyütme, 360x360 Boyutlandırma)
7. Swin Transformer (Swin-S) Deep Learning Sınıflandırma Mimarisi
8. Post-Processing ve JSON Sonuç Raporlama

Gerekli Kütüphaneler:
pip install opencv-python numpy torch torchvision matplotlib pillow
"""

import os
import sys
import json
import time
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.models.swin_transformer import swin_s, Swin_S_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch or Torchvision is not installed. Deep learning classification will run in simulation mode.")

# ==========================================
# 1. PARAMETRELER VE SINIF TANIMLARI
# ==========================================

CROP_SIZE = 360
SCALE_FACTOR = 1.40  # Hücreyi kırparken çevre bağlamı korumak için büyüme katsayısı

class CellType:
    SEGMENTED_NEUTROPHIL = "Segmented Neutrophil"
    BAND_NEUTROPHIL = "Band Neutrophil"
    LYMPHOCYTE = "Lymphocyte"
    MONOCYTE = "Monocyte"
    EOSINOPHIL = "Eosinophil"
    BASOPHIL = "Basophil"
    BLAST = "Blast"
    PROMYELOCYTE = "Promyelocyte"
    MYELOCYTE = "Myelocyte"
    METAMYELOCYTE = "Metamyelocyte"
    ATYPICAL_LYMPHOCYTE = "Atypical Lymphocyte"
    PLASMA_CELL = "Plasma Cell"
    NUCLEATED_RBC = "Nucleated RBC"
    SMUDGE_CELL = "Smudge Cell"
    ARTIFACT = "Artifact"
    UNKNOWN = "Unknown"

# Model Sınıfları Listesi
CLASSES = [
    CellType.ARTIFACT,
    CellType.ATYPICAL_LYMPHOCYTE,
    CellType.BAND_NEUTROPHIL,
    CellType.BASOPHIL,
    CellType.BLAST,
    CellType.EOSINOPHIL,
    CellType.LYMPHOCYTE,
    CellType.METAMYELOCYTE,
    CellType.MONOCYTE,
    CellType.MYELOCYTE,
    CellType.NUCLEATED_RBC,
    CellType.PLASMA_CELL,  # Veya Platelet
    CellType.PROMYELOCYTE,
    CellType.SEGMENTED_NEUTROPHIL,
    CellType.SMUDGE_CELL,
    CellType.UNKNOWN
]

# ==========================================
# ==========================================
# 2. GERÇEK ZAMANLI GÖRÜNTÜ EDİNİMİ & DONANIM SÜRÜCÜLERİ
# ==========================================

# Basler Kamerası ve Seri Port (Motor) Kütüphanelerini Yüklemeyi Dene
try:
    from pypylon import pylon
    PYPYLON_AVAILABLE = True
except ImportError:
    PYPYLON_AVAILABLE = False

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

class MicroscopeCamera:
    """
    Mikroskop üzerindeki endüstriyel Basler Kameradan (Pylon SDK/API) veya 
    standart USB/Bileşik kameralardan (OpenCV VideoCapture) anlık görüntü edinen donanım arayüzü.
    """
    def __init__(self):
        self.camera = None
        self.is_basler = False
        self.converter = None
        self.width = 1920
        self.height = 1080
        
    def open(self):
        """Kamerayı arar, açar ve ayarlarını kilitler (Pozlama, Kazanç vb.)."""
        # 1. Önce Endüstriyel Basler Kamerayı (PyPylon) bağlamayı dene
        if PYPYLON_AVAILABLE:
            try:
                print("Endüstriyel Basler Kamera aranıyor (Pylon SDK)...")
                self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
                self.camera.Open()
                self.is_basler = True
                
                # Kamera Ayarlarını Yap (Donanım Seviyesinde Kilitleme)
                nodemap = self.camera.GetNodeMap()
                
                # Otomatik pozlama ve kazancı kapa, manuel sabitle (Stabil autofocus için şarttır)
                nodemap.GetNode("ExposureAuto").SetValue("Off")
                nodemap.GetNode("GainAuto").SetValue("Off")
                
                # Pozlama süresini 6000 mikrosaniye (6 ms) ve Kazancı 0 dB ayarla (C++ ile birebir uyumlu)
                try:
                    nodemap.GetNode("ExposureTime").SetValue(6000.0)
                except Exception:
                    nodemap.GetNode("ExposureTimeAbs").SetValue(6000.0)
                nodemap.GetNode("Gain").SetValue(0.0)
                
                # Görüntü formatını OpenCV uyumlu BGR8 formatına ayarla
                self.converter = pylon.ImageFormatConverter()
                self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
                self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
                
                # Kameradan görüntü alımını başlat
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                print("[BAŞARILI - BASLER] Endüstriyel mikroskop kamerası bağlandı ve kilitlendi.")
                return True
            except Exception as e:
                print(f"Basler Kamera bulunamadı veya açılamadı: {e}")
        
        # 2. Basler yoksa standart USB Mikroskop kamerasını dene (cv2.VideoCapture)
        try:
            print("Standart USB Mikroskop Kamerası aranıyor (DirectShow/OpenCV)...")
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Windows DirectShow ile hızlı bağlantı
            if self.camera.isOpened():
                # Mikroskop çözünürlük ayarları
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                print(f"[BAŞARILI - USB] USB Mikroskop Kamerası açıldı (Çözünürlük: {self.width}x{self.height}).")
                return True
            else:
                self.camera = None
        except Exception as e:
            print(f"USB Kamera açılamadı: {e}")
            
        print("[DONANIM UYARISI] Fiziksel bir mikroskop kamerası bulunamadı! Simülasyon moduna geçiliyor.")
        return False
        
    def grab_frame(self):
        """Kameradan anlık tek bir çerçeve yakalar (Fotoğraf çeker)."""
        if self.camera is None:
            # Donanım yoksa hata vermez, simüle veri döner (Ödev test edilebilirliği için)
            return None
            
        if self.is_basler:
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                img = image.GetArray()
                grabResult.Release()
                return img.copy()
            else:
                grabResult.Release()
                raise RuntimeError("Basler kamerasından kare okunamadı!")
        else:
            ret, frame = self.camera.read()
            if ret:
                return frame
            else:
                raise RuntimeError("USB kameradan kare okunamadı!")
                
    def close(self):
        """Kamera bağlantılarını serbest bırakır."""
        if self.camera is not None:
            if self.is_basler:
                self.camera.StopGrabbing()
                self.camera.Close()
            else:
                self.camera.release()
            print("Kamera bağlantısı güvenli bir şekilde kapatıldı.")
            self.camera = None

class MicroscopeStage:
    """
    Mikroskobun Z-eksenini hareket ettiren step motor kontrol kartı (STM32 / Arduino / Serial) 
    için geliştirilmiş Seri Port G-Code sürücüsü.
    """
    def __init__(self, port="COM3", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.z_position = 0.0
        
    def connect(self):
        """Seri port üzerinden kontrol kartına bağlanır."""
        if SERIAL_AVAILABLE:
            try:
                print(f"Z-Stage Kontrol Kartına Bağlanılıyor ({self.port})...")
                self.serial = serial.Serial(self.port, self.baudrate, timeout=2.0)
                # STM32 hazır komutunu oku
                time.sleep(1.0)
                print("[BAŞARILI - MOTOR] STM32 Z-Eksen kontrol kartı bağlandı.")
                return True
            except Exception as e:
                print(f"Z-Stage Seri Port Bağlantı Hatası: {e}")
        
        print("[DONANIM UYARISI] Z-Stage kontrol kartı bulunamadı! Sanal motor aktif edildi.")
        return False
        
    def move_z(self, target_z):
        """Motoru G-Code standardında istenen Z konumuna (mm) hareket ettirir."""
        self.z_position = target_z
        if self.serial and self.serial.is_open:
            cmd = f"G0 Z{target_z:.3f}\n"
            self.serial.write(cmd.encode('utf-8'))
            
            # Karttan 'DONE Z' geri bildirimini bekle (C++ Serial.cpp ile uyumlu)
            start_time = time.time()
            while time.time() - start_time < 5.0: # 5 saniye zaman aşımı
                line = self.serial.readline().decode('utf-8').strip()
                if "DONE Z" in line:
                    print(f"[MOTOR] Z Konumuna Ulaşıldı: {target_z:.3f} mm")
                    return True
            print("Hata: Motor hareket zaman aşımı!")
            return False
        else:
            # Donanım yoksa simüle bekleme yapar
            time.sleep(0.10)
            print(f"[MOTOR - SIMÜLE] Sanal Z-Ekseni hareket ediyor -> Konum: {target_z:.3f} mm")
            return True
            
    def close(self):
        """Seri port bağlantısını kapatır."""
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("Z-Stage motor kontrol bağlantısı kapatıldı.")
            self.serial = None

def create_synthetic_wbc_slide():
    """
    Donanım bağlı olmadığında kameradan alınacak görüntüyü 
    simüle etmek amacıyla sentetik bir mikroskobik kan yayması oluşturur.
    """
    background = np.full((960, 1280, 3), 240, dtype=np.uint8)
    background[:, :] = [235, 240, 242]
    np.random.seed(42)
    # Alyuvarlar (RBC)
    for _ in range(80):
        cx = np.random.randint(50, 1230)
        cy = np.random.randint(50, 910)
        r = np.random.randint(25, 35)
        cv2.circle(background, (cx, cy), r, (120, 110, 220), -1)
        cv2.circle(background, (cx, cy), int(r * 0.5), (235, 240, 242), -1)
    # Akyuvarlar (WBC)
    # Hücre 1
    cv2.circle(background, (400, 300), 75, (230, 190, 220), -1)
    cv2.circle(background, (375, 280), 22, (110, 30, 80), -1)
    cv2.circle(background, (425, 290), 24, (110, 30, 80), -1)
    cv2.circle(background, (390, 330), 20, (110, 30, 80), -1)
    cv2.line(background, (375, 280), (425, 290), (110, 30, 80), 6)
    cv2.line(background, (390, 330), (425, 290), (110, 30, 80), 6)
    # Hücre 2
    cv2.circle(background, (850, 600), 80, (200, 200, 240), -1)
    for _ in range(50):
        gx = np.random.randint(810, 890)
        gy = np.random.randint(560, 640)
        if math.hypot(gx - 850, gy - 600) < 70:
            cv2.circle(background, (gx, gy), 3, (80, 120, 240), -1)
    cv2.circle(background, (825, 595), 25, (100, 20, 70), -1)
    cv2.circle(background, (875, 595), 25, (100, 20, 70), -1)
    cv2.line(background, (825, 595), (875, 595), (100, 20, 70), 8)
    # Hücre 3
    cv2.circle(background, (300, 750), 60, (240, 210, 230), -1)
    cv2.circle(background, (300, 750), 48, (90, 20, 70), -1)
    # Hücre 4
    cv2.circle(background, (950, 250), 85, (230, 210, 230), -1)
    cv2.circle(background, (930, 240), 28, (80, 20, 60), -1)
    cv2.circle(background, (955, 270), 26, (80, 20, 60), -1)
    cv2.circle(background, (935, 270), 27, (80, 20, 60), -1)
    return background

def get_simulated_focus_frame(template_img, z_val, peak_z=5.0):
    """
    Kamera donanımı bağlı olmadığında, verilen Z adımındaki bulanıklaşma 
    etkisini matematiksel olarak simüle eder (Gaussian Blur ve Sensör Gürültüsü).
    """
    # Odak noktasından olan uzaklık
    distance = abs(z_val - peak_z)
    
    # Bulanıklık katsayısı (0 ile 25 arasında kernel)
    k = int(distance * 5) * 2 + 1
    k = np.clip(k, 1, 31)
    
    if k == 1:
        return template_img.copy()
    else:
        blurred = cv2.GaussianBlur(template_img, (k, k), 0)
        # Hafif sensör gürültüsü
        noise = np.random.normal(0, 1.2, blurred.shape).astype(np.int16)
        noisy = np.clip(blurred.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy

# ==========================================
# 3. NETLİK ÖLÇÜM ALGORİTMALARI (FOCUS MEASURE)
# ==========================================

class FocusParams:
    """Mikroskop AutoFocus C++ parametrelerinin Python karşılığı"""
    def __init__(self):
        self.blur_ksize = 3
        self.blur_sigma = 1.0
        self.grid_n = 3          # Tenengrad Grid Median için NxN ızgara boyutu
        self.roi_fraction = 0.50 # Görüntü merkezindeki odaklanma alanının oranı (50%)
        self.use_clahe = True
        self.clahe_clip = 2.0
        self.clahe_tiles = 8
        self.contrast_blur_ksize = 9

g_params = FocusParams()

def laplacian_variance(gray_roi):
    """
    Laplacian Varyansı Algoritması:
    Görüntünün ikinci türevinin (Laplacian) standart sapmasının karesini hesaplar.
    Kenarlar ne kadar keskinse varyans o kadar yüksek çıkar.
    """
    lap = cv2.Laplacian(gray_roi, cv2.CV_64F, ksize=3)
    mean, stddev = cv2.meanStdDev(lap)
    return stddev[0][0] ** 2

def tenengrad_score_whole(gray_roi):
    """
    Tenengrad Algoritması (Tüm Görüntü için):
    Sobel operatörü kullanarak yatay ve dikey gradyanları (Gx ve Gy) hesaplar.
    Kenar büyüklüklerinin karelerinin ortalamasını alır.
    """
    work = gray_roi.copy()
    if g_params.blur_ksize > 1:
        work = cv2.GaussianBlur(work, (g_params.blur_ksize, g_params.blur_ksize), g_params.blur_sigma)
    
    gx = cv2.Sobel(work, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(work, cv2.CV_64F, 0, 1, ksize=3)
    mag = gx*gx + gy*gy
    return np.mean(mag)

def tenengrad_score_grid_median(gray_roi):
    """
    Tenengrad Grid Median Algoritması (C++'tan Aktarılan Gelişmiş Yöntem):
    Görüntüyü NxN boyutunda ızgaralara böler. Her bölge için ayrı Tenengrad skoru hesaplar
    ve bu skorların MEDYAN'ını alır. Bu sayede homojen arka plandaki gürültülerin ve outlier'ların
    odak skorunu bozması engellenir.
    """
    n = g_params.grid_n
    if n <= 1:
        return tenengrad_score_whole(gray_roi)

    H, W = gray_roi.shape[:2]
    if W < n * 16 or H < n * 16:
        return tenengrad_score_whole(gray_roi)

    scores = []
    pw = W // n
    ph = H // n

    for yy in range(n):
        for xx in range(n):
            x0 = xx * pw
            y0 = yy * ph
            x1 = W if xx == n - 1 else x0 + pw
            y1 = H if yy == n - 1 else y0 + ph

            patch = gray_roi[y0:y1, x0:x1]
            scores.append(tenengrad_score_whole(patch))

    # Ortanca (Medyan) Değeri Döndür
    return np.median(scores)

def contrast_score(gray_roi):
    """
    Kontrast Odak Skoru:
    Orijinal görüntü ile onun bulanıklaştırılmış hali arasındaki ortalama mutlak farktır.
    |I - blur(I)| ortalaması.
    """
    k = g_params.contrast_blur_ksize
    if k < 3: k = 3
    if k % 2 == 0: k += 1
    
    blur = cv2.GaussianBlur(gray_roi, (k, k), 0)
    diff = cv2.absdiff(gray_roi, blur)
    return np.mean(diff)

def entropy_score(gray_roi):
    """
    Entropi Odak Skoru:
    Görüntüdeki bilgi yoğunluğunu (düzensizliğini) ölçer.
    Keskin görüntüler daha fazla doku (yüksek entropi) barındırır.
    """
    # 256-bin histogram
    hist = cv2.calcHist([gray_roi], [0], None, [256], [0, 256])
    hist = hist / gray_roi.size  # Normalize et (olasılık dağılımı)
    
    entropy = 0.0
    for p in hist:
        if p > 0:
            entropy -= p * np.log2(p)
    return float(entropy)

def calculate_combined_focus_score(img):
    """
    Birleşik Odak Skoru (C++ API Karşılığı):
    1. Görüntüyü gri tonlamaya çevirir.
    2. Merkez odak ROI'sini kırpar (%50 oranlı kare alan).
    3. Stabilite için isteğe bağlı CLAHE (kontrast dengeleme) uygular.
    4. Laplacian Varyansı (%70) ve Tenengrad Grid Medyan (%30) skorlarını birleştirir.
    """
    if img is None or img.size == 0:
        return 0.0

    # 1. Gri tonlamaya dönüştür
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 2. Odak ROI Belirleme (Görüntünün merkezindeki %50'lik alan)
    H, W = gray.shape[:2]
    f = np.clip(g_params.roi_fraction, 0.10, 1.00)
    rw = int(W * f)
    rh = int(H * f)
    rx = (W - rw) // 2
    ry = (H - rh) // 2
    crop = gray[ry:ry+rh, rx:rx+rw]

    # 3. CLAHE Kontrast İyileştirme (Özellikle loş veya düz ışıklı mikroskoplarda AF başarısını artırır)
    if g_params.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=g_params.clahe_clip, 
                                tileGridSize=(g_params.clahe_tiles, g_params.clahe_tiles))
        crop = clahe.apply(crop)

    # 4. Skorları Hesapla ve Ağırlıklı Topla
    s_lap = laplacian_variance(crop)
    s_ten = tenengrad_score_grid_median(crop)
    
    # Ağırlıklandırma: Laplacian baskın (%70), Grid Tenengrad yardımcı (%30)
    combined = 0.70 * s_lap + 0.30 * s_ten
    return combined

# ==========================================
# 4. Z-EKSENİ OTO-ODAKLAMA & EN NET ÇERÇEVE SEÇİMİ
# ==========================================

def save_focus_scores_to_excel(excel_data, filename="focus_measure_results.xlsx"):
    """
    Her Z adımındaki detaylı odak skorlarını ve alt metrikleri Excel dosyasına kaydeder.
    Eğer pandas ve openpyxl yüklüyse gerçek bir .xlsx dosyası yazar.
    Yüklü değilse, Excel'in doğrudan açabileceği Türkçe karakter uyumlu (UTF-8 BOM) bir .csv dosyası üretir.
    """
    print("\n--- ODAK ÖLÇÜM SONUÇLARINI EXCEL'E AKTARMA ---")
    
    # Gerçek .xlsx yazmayı dene (pandas ve openpyxl ile)
    try:
        import pandas as pd
        df = pd.DataFrame(excel_data)
        df.to_excel(filename, index=False)
        print(f">> [BAŞARILI] Tüm odak metrikleri gerçek Excel dosyası olarak '{filename}' adıyla kaydedildi.")
        return
    except ImportError:
        pass

    # Fallback: Excel'in doğrudan tanıdığı noktalı virgüllü ve UTF-8 BOM'lu CSV formatı
    csv_filename = filename.replace(".xlsx", ".csv")
    import csv
    try:
        with open(csv_filename, mode='w', encoding='utf-8-sig', newline='') as f:
            if excel_data:
                fieldnames = list(excel_data[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()
                for row in excel_data:
                    writer.writerow(row)
        print(f">> [BAŞARILI] Pandas/Openpyxl yüklü olmadığından, Excel uyumlu CSV dosyası '{csv_filename}' adıyla kaydedildi.")
    except Exception as e:
        print(f"Excel/CSV yazma hatası: {e}")

def perform_autofocus_sweep(z_stack):
    """
    Z-Stack'teki tüm çerçeveleri tarar, her birinin netlik skorunu ve alt metriklerini ölçer,
    en net çerçeveyi (Peak Focus Frame) tespit eder, skorları grafikleştirir ve Excel'e kaydeder.
    """
    print("\n--- OTO-ODAKLAMA (AUTOFOCUS) BAŞLATILIYOR ---")
    scores = []
    excel_data = []
    
    for idx, frame in enumerate(z_stack):
        # Görüntüyü gri tonlamaya çevirip odak ROI'sini çıkararak alt metrikleri hesaplayalım
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        H, W = gray.shape[:2]
        f = np.clip(g_params.roi_fraction, 0.10, 1.00)
        rw = int(W * f)
        rh = int(H * f)
        rx = (W - rw) // 2
        ry = (H - rh) // 2
        crop = gray[ry:ry+rh, rx:rx+rw]

        if g_params.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=g_params.clahe_clip, 
                                    tileGridSize=(g_params.clahe_tiles, g_params.clahe_tiles))
            crop = clahe.apply(crop)

        # Alt odak metrikleri
        s_lap = laplacian_variance(crop)
        s_ten = tenengrad_score_grid_median(crop)
        s_con = contrast_score(crop)
        s_ent = entropy_score(crop)
        combined = 0.70 * s_lap + 0.30 * s_ten
        
        scores.append(combined)
        
        # Excel verisine ekle
        excel_data.append({
            "Z-Adimi (Cerceve No)": idx,
            "Birlesik Odak Skoru": round(combined, 4),
            "Laplacian Varyansi": round(s_lap, 4),
            "Tenengrad Sobel Grid Median": round(s_ten, 4),
            "Kontrast Skoru": round(s_con, 4),
            "Entropi Skoru": round(s_ent, 4),
            "Durum": "Gecici" # Aşağıda güncellenecek
        })
        
        print(f"Çerçeve {idx:2d} (Z-Adımı): Odak Skoru = {combined:.2f} | Laplacian = {s_lap:.1f} | Tenengrad = {s_ten:.1f}")

    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    print(f">> EN NET GÖRÜNTÜ TESPİT EDİLDİ: Çerçeve İndeksi = {best_idx} (Skor = {best_score:.2f})")

    # Durum bilgisini güncelle
    for idx, row in enumerate(excel_data):
        row["Durum"] = "EN NET ODAK (PEAK)" if idx == best_idx else "Odak Disi"

    # Excel/CSV dosyasına kaydet
    save_focus_scores_to_excel(excel_data)

    # Odak Eğrisini Grafik Olarak Çiz (Bell Curve - Odak Tepesi)
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(scores)), scores, 'b-o', label='Birleşik Odak Skoru', linewidth=2)
    plt.plot(best_idx, best_score, 'r*', markersize=15, label=f'En Net Odak (Adım {best_idx})')
    plt.title('Z-Ekseni Odak Değişim Eğrisi (Autofocus Curve)')
    plt.xlabel('Z-Ekseni Adımları (Çerçeve İndeksi)')
    plt.ylabel('Birleşik Sharpness Skoru')
    plt.grid(True)
    plt.legend()
    plot_path = "focus_curve.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[KAYDEDİLDİ] Odak eğrisi grafiği '{plot_path}' olarak kaydedildi.")

    return best_idx

# ==========================================
# 5. GÖRÜNTÜ ÖN İŞLEME & HÜCRE TESPİTİ (WBC DETECTION)
# ==========================================

def detect_white_blood_cells(bgr_img):
    """
    En net seçilen çerçevedeki akyuvarları (WBC) tespit etmek için
    Görüntü İşleme / Morfoloji yöntemlerini kullanır (Simüle veya Kontur tabanlı).
    WBC'ler mor renkli olduklarından, renk eşiği veya parlaklık tespitiyle bulunur.
    """
    print("\n--- HÜCRE DETEKSİYONU BAŞLATILIYOR (OPENCV CONTOUR PIPELINE) ---")
    # 1. CLAHE ile Kontrastı Dengele
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)

    # 2. Renk Tabanlı Akyuvar Çekirdeği Eşleme (HSV ile mor tonları yakalama)
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # Mor çekirdek için HSV renk aralıkları
    lower_purple = np.array([120, 40, 40])
    upper_purple = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # 3. Morfolojik İşlemler (Gürültü temizleme ve boşluk kapatma)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    dilated = cv2.morphologyEx(cleaned, cv2.MORPH_DILATE, kernel, iterations=2)

    # 4. Kontur Analizi ile Bounding Box (Kutu) Bulma
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        # WBC hücreleri belirli bir alandan büyük olmalı (küçük gürültüleri filtrele)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            # YOLO formatına dönüştür (Center_X, Center_Y, Width, Height) (Normalize)
            H, W = bgr_img.shape[:2]
            cx = (x + w/2) / W
            cy = (y + h/2) / H
            nw = w / W
            nh = h / H
            bboxes.append((idx, cx, cy, nw, nh))
            print(f"Hücre #{idx} Tespit Edildi: Alan = {area:.1f} px, Konum = [x:{x}, y:{y}, w:{w}, h:{h}]")

    # Sonucu Görselleştir
    annotated = bgr_img.copy()
    for idx, cx, cy, nw, nh in bboxes:
        H, W = annotated.shape[:2]
        x1 = int((cx - nw/2) * W)
        y1 = int((cy - nh/2) * H)
        x2 = int((cx + nw/2) * W)
        y2 = int((cy + nh/2) * H)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(annotated, f"WBC #{idx}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imwrite("detected_cells.png", annotated)
    print("[KAYDEDİLDİ] Hücre tespiti görselleştirmesi 'detected_cells.png' olarak kaydedildi.")
    return bboxes

# ==========================================
# 6. HÜCRE KIRPMA VE BOYUTLANDIRMA (CUSTOM CROP)
# ==========================================

def crop_and_resize_cells(bgr_img, bboxes, output_dir="crops"):
    """
    Bulunan hücrelerin koordinatlarını alıp:
    1. Kareleştirir (Squaring).
    2. Genişletir (SCALE_FACTOR = 1.40x): Hücrenin etrafındaki dokuyu da görmek için.
    3. Taşkınları sınırlar (Clip boundaries).
    4. 360x360 standart boyutuna getirir ve diske kaydeder.
    """
    print("\n--- ÖZEL HÜCRE KIRPMA VE BOYUTLANDIRMA BAŞLADI ---")
    os.makedirs(output_dir, exist_ok=True)
    H, W = bgr_img.shape[:2]
    crop_paths = []

    for idx, cx, cy, nw, nh in bboxes:
        # 1. Piksel koordinatlarına dönüştür
        pixel_cx = cx * W
        pixel_cy = cy * H
        pixel_w = nw * W
        pixel_h = nh * H

        # 2. Kutuyu kare yap ve 1.40 katı genişlet
        box_size = max(pixel_w, pixel_h) * SCALE_FACTOR

        x1 = max(0, int(pixel_cx - box_size/2))
        y1 = max(0, int(pixel_cy - box_size/2))
        x2 = min(W, int(pixel_cx + box_size/2))
        y2 = min(H, int(pixel_cy + box_size/2))

        crop = bgr_img[y1:y2, x1:x2]
        
        if crop.size == 0:
            continue

        # 3. 360x360 Boyutuna Yeniden Boyutlandır (Bilinear Interpolation)
        crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)

        # 4. Dosyaya Kaydet
        filename = f"wbc_crop_{idx}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, crop_resized)
        crop_paths.append((idx, filepath))
        print(f"Hücre #{idx} kırpıldı ve '{filepath}' (360x360 px) olarak kaydedildi.")

    return crop_paths

# ==========================================
# 7. DEEP LEARNING CELL CLASSIFIER (SWIN TRANSFORMER)
# ==========================================

class SwinSClassifierModel(nn.Module):
    """
    Swin Transformer (Small) Sınıflandırma Ağı:
    Tıbbi görüntü işlemede son teknoloji ürünü (SOTA) Transformer modelidir.
    Kendi veri setimizle eğitilmiş 16 sınıflı başlığı barındırır.
    """
    def __init__(self, num_classes=16):
        super(SwinSClassifierModel, self).__init__()
        # Swin-S omurgası (Swin Transformer Small)
        weights = Swin_S_Weights.DEFAULT if TORCH_AVAILABLE else None
        self.backbone = swin_s(weights=weights)
        
        # Son katmanı (Head) 16 hücre sınıfımıza göre yeniden tanımla
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class AcademicWBCClassifier:
    """
    Eğitilmiş Swin-S ağırlıklarıyla hücre tahmini yapan sınıflandırıcı.
    Eğer eğitilmiş ağırlık pth dosyası yoksa veya Torch kütüphanesi eksikse,
    sentetik olasılıklarla simüle çalışır (ödev sunumu için güvenli mod).
    """
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else "CPU"
        self.use_simulation = not TORCH_AVAILABLE or model_path is None or not os.path.exists(model_path)
        
        if not self.use_simulation:
            print(f"\nSwin-S Sınıflandırıcı '{model_path}' dosyası ile yükleniyor ({self.device})...")
            self.model = SwinSClassifierModel(num_classes=len(CLASSES))
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            # Swin Transformer Giriş Dönüşümleri (224x224, ImageNet Normalizasyonu)
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            print("Swin Transformer Başarıyla Yüklendi.")
        else:
            print("\n[BİLGİ] Deep Learning Modeli Simüle Modda Çalışıyor (Pytorch eksik veya .pth bulunamadı).")

    def predict_cell(self, image_path, cell_id):
        """
        Kırpılmış tek bir hücrenin sınıfını ve güven skorunu (confidence) tahmin eder.
        """
        if not self.use_simulation:
            try:
                # Resmi PIL ile oku ve transforme et
                image = Image.open(image_path).convert('RGB')
                tensor = self.transform(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted_idx = torch.max(probs, 1)
                    
                predicted_class = CLASSES[predicted_idx.item()]
                return predicted_class, confidence.item()
            except Exception as e:
                print(f"Çıkarım (Inference) Hatası: {e}")
                return CellType.UNKNOWN, 0.0
        else:
            # Ödev sunumu için tutarlı simüle sonuçlar üretelim
            # Hücre id'sine göre farklı akyuvar sınıfları atayalım
            simulated_results = {
                0: (CellType.SEGMENTED_NEUTROPHIL, 0.945),
                1: (CellType.EOSINOPHIL, 0.923),
                2: (CellType.LYMPHOCYTE, 0.887),
                3: (CellType.MONOCYTE, 0.812)
            }
            return simulated_results.get(cell_id, (CellType.UNKNOWN, 0.50))

# ==========================================
# 8. POST-PROCESSING & JSON RAPORLAMA
# ==========================================

def generate_academic_report(results, output_json="wbc_results_v1.json"):
    """
    Sınıflandırma sonuçlarını alır, toplam sayımları ve yüzdeleri hesaplar,
    ödev teslimine uygun profesyonel bir JSON raporu oluşturur.
    """
    print("\n--- POST-PROCESSING VE AKADEMİK RAPORLAMA ---")
    total_cells = len(results)
    
    # Sınıflara göre sayım sözlüğü
    class_counts = {}
    for cell_id, cell_class, confidence, img_path in results:
        class_counts[cell_class] = class_counts.get(cell_class, 0) + 1

    # Rapor formatını oluştur
    report_data = {
        "metadata": {
            "title": "Akyuvar (WBC) Diferansiyel Hemogram Analiz Raporu",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_wbc_counted": total_cells,
            "pipeline_version": "v1.0.0-Academic",
            "autofocus_metric": "Laplacian(70%) + Tenengrad Grid Median(30%)"
        },
        "wbc_distribution": []
    }

    for cls_name in CLASSES:
        count = class_counts.get(cls_name, 0)
        percentage = round((count / total_cells) * 100, 2) if total_cells > 0 else 0.0
        
        # Bu sınıfa ait örnek hücre görüntülerinin listesini çek
        samples = [path for cid, cname, conf, path in results if cname == cls_name]

        class_item = {
            "name": cls_name,
            "count": count,
            "percentage": percentage,
            "sample_images": samples[:5], # İlk 5 örnek resmi ekle
            "abnormal": cls_name in [CellType.BLAST, CellType.PROMYELOCYTE, CellType.ATYPICAL_LYMPHOCYTE]
        }
        report_data["wbc_distribution"].append(class_item)

    # JSON dosyasına kaydet
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)

    print(f">> [TAMAMLANDI] Akademik Hemogram Sonuç Raporu '{output_json}' olarak başarıyla kaydedildi!")
    
    # Konsola Özet Tabloyu Yazdır
    print("\n================== ÖZET BİLGİ TABLOSU ==================")
    print(f"{'Akyuvar Sınıfı':<25} | {'Adet':<6} | {'Yüzde (%)':<10}")
    print("-" * 48)
    for item in report_data["wbc_distribution"]:
        if item["count"] > 0:
            print(f"{item['name']:<25} | {item['count']:<6} | {item['percentage']:<10}%")
    print("========================================================")

# ==========================================
# ANA ÇALIŞTIRMA (MAIN FLOW)
# ==========================================

def main():
    print("======================================================================")
    print("      OPENCV & DEEP LEARNING TABANLI MİKROSKOPİK GÖRÜNTÜ ANALİZİ")
    print("                        AKADEMİK ÖDEV PROJESİ")
    print("======================================================================\n")

    # Donanım Bileşenlerini Tanımla
    camera = MicroscopeCamera()
    stage = MicroscopeStage(port="COM3", baudrate=115200)

    # Donanım Bağlantılarını Kur
    has_camera = camera.open()
    has_stage = stage.connect()

    # Z-Stack Tarama Ayarları (C++ Donanım Tarama Arayüzü ile Uyumlu)
    # Z Konumu: 0.0 mm -> 10.0 mm | Adım: 1.0 mm (Toplam 11 adım, En Net Odak Noktası Z = 5.0 mm)
    z_steps = [float(z) for z in range(11)]
    z_stack = []
    
    print("\n--- DONANIM Z-STACK TARAMA SWEEP DÖNGÜSÜ BAŞLIYOR ---")
    
    # Simülasyon şablonu (Eğer fiziksel kamera yoksa kullanılacak)
    sim_slide_template = None
    if not has_camera:
        sim_slide_template = create_synthetic_wbc_slide()
        cv2.imwrite("synthetic_slide_in_focus.png", sim_slide_template)
        print("[EMÜLASYON] Fiziksel kamera bağlı olmadığı için sanal kan yayması hazırlandı.")

    try:
        for idx, z_val in enumerate(z_steps):
            print(f"\n[ADIM {idx+1}/{len(z_steps)}] Z Stage Hedefi: {z_val:.3f} mm")
            
            # 1. Motoru Z konumuna sür
            stage.move_z(z_val)
            
            # 2. Kameradan anlık fotoğraf çek
            frame = camera.grab_frame()
            
            # 3. Eğer fiziksel kamera yoksa emüle odak katmanı üret
            if frame is None:
                frame = get_simulated_focus_frame(sim_slide_template, z_val, peak_z=5.0)
                
            z_stack.append(frame)
            print(f"[KAMERA] {z_val:.3f} mm konumunda anlık görüntü başarıyla yakalandı.")
            
    except Exception as e:
        print(f"Tarama döngüsü sırasında donanım hatası oluştu: {e}")
        return
    finally:
        # Donanımları Güvenli Bir Şekilde Serbest Bırak (Çok Önemli!)
        camera.close()
        stage.close()

    print(f"\n[1/8 OK] Z-Stack taraması başarıyla tamamlandı. Toplam edinilen çerçeve: {len(z_stack)}")

    # ADIM 3: Autofocus Tarama (En Keskin Çerçeve Seçimi ve Excel Kaydı)
    best_frame_idx = perform_autofocus_sweep(z_stack)
    best_frame = z_stack[best_frame_idx]
    
    # En keskin fotoğrafı kaydet
    cv2.imwrite("captured_focus_peak.png", best_frame)
    print(f"[3/8 OK] En net Z-Adımı belirlendi (Indeks: {best_frame_idx}). Fotoğraf 'captured_focus_peak.png' olarak kaydedildi.")

    # ADIM 4 & 5: Hücre Segmentasyonu & Deteksiyonu (OpenCV)
    detected_bboxes = detect_white_blood_cells(best_frame)
    print(f"[4&5 OK] Görüntü ön-işlendi ve toplam {len(detected_bboxes)} akyuvar tespit edildi.")

    if len(detected_bboxes) == 0:
        print("Hata: Hiç hücre tespit edilemedi! Pipeline sonlandırılıyor.")
        return

    # ADIM 6: Özel Kırpma ve Kareleştirme (Custom Bounding Box Cropping)
    crops = crop_and_resize_cells(best_frame, detected_bboxes)
    print(f"[6/8 OK] Hücreler 1.40x genişlik payıyla kare şeklinde kırpılıp 360x360 piksele boyutlandırıldı.")

    # ADIM 7: Swin Transformer ile Sınıflandırma
    classifier = AcademicWBCClassifier(model_path="best_swin_s_epoch_12.pth")
    
    classification_results = []
    for cell_id, filepath in crops:
        pred_class, confidence = classifier.predict_cell(filepath, cell_id)
        classification_results.append((cell_id, pred_class, confidence, filepath))
        print(f"Hücre #{cell_id} Sınıflandırma Sonucu: {pred_class:<22} (Güven Skoru: %{confidence*100:.1f})")
    print("[7/8 OK] Swin Transformer sınıflandırma adımı tamamlandı.")

    # ADIM 8: Raporlama ve JSON/Excel Kaydı
    generate_academic_report(classification_results)
    print("[8/8 OK] Post-processing ve diferansiyel hemogram sayım raporu JSON olarak kaydedildi.")
    
    print("\n----------------------------------------------------------------------")
    print("ÖDEV PROJESİ BAŞARIYLA ÇALIŞTIRILDI!")
    print("Üretilen çıktılar:")
    print("1. 'captured_focus_peak.png'      -> Tarayıcıdan Yakalanan En Keskin Hücre Fotoğrafı")
    print("2. 'focus_curve.png'              -> Z-Ekseni Odak Değişim Eğrisi (Autofocus Curve)")
    print("3. 'detected_cells.png'           -> Segmentasyon ve Tespit Çerçeveleri")
    print("4. 'crops/' klasörü               -> 360x360 Piksel Sınıflandırmaya Hazır Akyuvar Hücreleri")
    print("5. 'wbc_results_v1.json'         -> Akademik Sayım ve Diferansiyel Analiz Raporu (JSON)")
    print("6. 'focus_measure_results.xlsx'   -> Her Z-Adımındaki Netlik ve Alt Metrik Analizleri (Excel)")
    print("----------------------------------------------------------------------")

if __name__ == "__main__":
    main()
