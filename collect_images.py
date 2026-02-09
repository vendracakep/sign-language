import os
import cv2
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 5
dataset_size = 300  # ← NAIKKAN DARI 100 ke 300!

cap = cv2.VideoCapture(0)

# Set resolusi yang sama dengan inference
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("=" * 70)
print("📸 DATA COLLECTION - IMPROVED VERSION")
print("=" * 70)
print(f"Target: {number_of_classes} kelas × {dataset_size} gambar = {number_of_classes * dataset_size} total")
print()
print("⚠️  TIPS UNTUK AKURASI TINGGI:")
print("   1. Variasi POSISI tangan (kiri, tengah, kanan)")
print("   2. Variasi JARAK kamera (dekat, sedang, jauh)")
print("   3. Variasi ANGLE (lurus, miring, diputar sedikit)")
print("   4. Variasi LIGHTING (terang, redup)")
print("   5. Variasi BACKGROUND (beda-beda)")
print("   6. JEDA antar capture untuk variasi pose")
print("=" * 70)
print()

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print(f"\n{'='*70}")
    print(f"📂 KELAS {j}: Mengumpulkan {dataset_size} gambar")
    print(f"{'='*70}")
    
    # INSTRUKSI SPESIFIK PER TAHAP
    stages = [
        {"name": "Posisi TENGAH", "count": int(dataset_size * 0.3), "delay": 100},
        {"name": "Posisi KIRI", "count": int(dataset_size * 0.2), "delay": 100},
        {"name": "Posisi KANAN", "count": int(dataset_size * 0.2), "delay": 100},
        {"name": "Jarak DEKAT", "count": int(dataset_size * 0.15), "delay": 100},
        {"name": "Jarak JAUH", "count": int(dataset_size * 0.15), "delay": 100},
    ]
    
    total_collected = 0
    
    for stage_idx, stage in enumerate(stages):
        print(f"\n🎯 Tahap {stage_idx + 1}/{len(stages)}: {stage['name']}")
        print(f"   Target: {stage['count']} gambar")
        print(f"   💡 Tip: Gerakkan tangan perlahan, jangan diam!")
        
        done = False
        while True:
            ret, frame = cap.read()
            
            # Flip untuk mirror effect (lebih natural)
            frame = cv2.flip(frame, 1)
            
            # Overlay instruksi
            overlay_text = [
                f"KELAS {j} - {stage['name']}",
                f"Progress: {total_collected}/{dataset_size}",
                f"Tahap ini: 0/{stage['count']}",
                "",
                "Tekan 'M' untuk MULAI"
            ]
            
            y_offset = 30
            for line in overlay_text:
                cv2.putText(frame, line, (30, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                y_offset += 40
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(25)
            if key == ord('m') or key == ord('M'):
                break
            elif key == 27:  # ESC
                print("\n⚠️  Collection dibatalkan!")
                cap.release()
                cv2.destroyAllWindows()
                exit()
        
        # COLLECT dengan delay yang lebih lama
        stage_counter = 0
        last_capture_time = time.time()
        
        while stage_counter < stage['count']:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            
            current_time = time.time()
            time_since_last = int((current_time - last_capture_time) * 1000)  # ms
            
            # Progress overlay
            progress_pct = int((stage_counter / stage['count']) * 100)
            
            # Draw progress bar
            bar_width = 400
            bar_height = 30
            bar_x = 50
            bar_y = frame.shape[0] - 100
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            # Fill
            fill_width = int(bar_width * (stage_counter / stage['count']))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                         (0, 255, 0), -1)
            # Border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (255, 255, 255), 2)
            
            # Text
            cv2.putText(frame, f"KELAS {j} - {stage['name']}", (30, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{stage_counter}/{stage['count']} ({progress_pct}%)", (30, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Total: {total_collected}/{dataset_size}", (30, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
            
            # Countdown timer untuk next capture
            time_until_next = max(0, stage['delay'] - time_since_last)
            if time_until_next > 0:
                cv2.putText(frame, f"Next: {int(time_until_next)}ms", 
                           (frame.shape[1] - 250, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                # CAPTURE!
                cv2.putText(frame, "CAPTURING!", (frame.shape[1] - 250, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Data Collection', frame)
            cv2.waitKey(1)
            
            # Save frame dengan delay
            if time_since_last >= stage['delay']:
                filename = f"{total_collected:04d}.jpg"  # 0000.jpg, 0001.jpg, etc.
                cv2.imwrite(os.path.join(DATA_DIR, str(j), filename), frame)
                stage_counter += 1
                total_collected += 1
                last_capture_time = current_time
                
                # Print progress setiap 10 gambar
                if stage_counter % 10 == 0:
                    print(f"   ✓ {stage_counter}/{stage['count']} collected...")
        
        print(f"   ✅ Tahap {stage['name']} selesai!")
    
    print(f"\n🎉 KELAS {j} SELESAI! Total: {total_collected} gambar")
    print("   Tekan ENTER untuk lanjut ke kelas berikutnya...")
    input()

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 70)
print("✨ DATA COLLECTION SELESAI!")
print("=" * 70)
print(f"Total gambar: {number_of_classes * dataset_size}")
print()
print("📋 Next steps:")
print("   1. python create_dataset.py  # Extract features")
print("   2. python train_classifier.py  # Train model")
print("   3. python inference.py  # Test!")
print("=" * 70)