<div align="right" dir="rtl">

[⬅️ العودة إلى الصفحة الرئيسية (`README.md`)](../README.md)

# النماذج المدربة ودليل التشغيل (Models, CLI Usage & Roadmap)

## 1. جرد النماذج والأوزان المحفوظة (`Trained Checkpoints`)

| الوزن (`Checkpoint`) | المرحلة | المعمارية والدقة | الحجم | الأداء المرجعي | الدور التشغيلي |
| :--- | :---: | :--- | :---: | :--- | :--- |
| **`exp02_ep35_champion_1024.pt`** ⭐ | المرحلة 2 | `YOLO11s` (`1024px`, `nc=1`) | `19.2 MB` | **`52.82%` Std / `54.96%` TTA** (`AIResQ`) | **النموذج البطل للبحث والإنقاذ عابر النطاقات** |
| **`yolo11s_pro_800px_best.pt`** | المرحلة 1 | `YOLO11s` (`800px`, `nc=4`, `CLAHE`) | `18.30 MB` | **`89.63%` All-4 / `93.38%` Person** (`HIT-UAV`) | الأفضل للرصد التكتيكي متعدد الفئات + `BoT-SORT GMC` |
| **`yolo11s_640px_best.pt`** | المرحلة 1 | `YOLO11s` (`640px`, `nc=4`, `CLAHE`) | `18.28 MB` | `81.73%` Std / `84.73%` TTA (`HIT-UAV`) | استدلال فائق السرعة (`8.4 ms/img`) |
| **`best_thermal_unet.pt`** | المرحلة 1 | `Thermal U-Net` (`256px`, `nc=3`) | `7.39 MB` | `98.89%` Pixel Acc / `63.65%` mIoU | عزل البصمات الحرارية على مستوى البكسل |
| **`exp01_coco_v2_best.pt`** | المرحلة 2 | `YOLO11s` (`1024px`, `nc=1`) | `19.2 MB` | `99.45%` داخلي / `46.23%` (`AIResQ`) | خط الأساس المرجعي (`Baseline`) |

*(تُنشر ملفات `.pt` عبر تبويب **Releases** في المستودع وتوضع داخل مجلد `weights/` محلياً.)*

---

## 2. التثبيت والتشغيل عبر سطر الأوامر (`CLI`)

</div>

```bash
# 1. استنساخ المستودع وتثبيت المكتبات
git clone https://github.com/Ahmed-Al-Masqari/AeroTherm-SAR.git
cd AeroTherm-SAR
pip install -r requirements.txt

# 2. تنزيل مجموعات البيانات الخام من Zenodo مع التحقق من MD5
python src/download_datasets.py --out raw_datasets

# 3. الاستدلال الفوري بالنموذج البطل (Exp-02-ep35) مع TTA
python src/infer.py --weights weights/exp02_ep35_champion_1024.pt --source thermal_frame.jpg --imgsz 1024 --tta

# 4. تشغيل التتبع التكتيكي BoT-SORT + Sparse Optical Flow GMC على فيديو حراري
python src/track_botsort.py   --weights weights/yolo11s_pro_800px_best.pt   --source uav_thermal_stream.mp4   --tracker configs/sar_botsort_gmc.yaml   --output outputs/sar_tracked_hud.mp4
```

<div align="right" dir="rtl">

---

## 3. خارطة الطريق القادمة (`Next Milestones`)

- **إطلاق `Exp-03`:** تدريب `YOLO11s @ 1024px` على `Unified_SAR_Person_v2` (`5,676` صورة تدريب متوازنة ثلاثياً) لمعالجة فجوة الأهداف القريبة (`>=1200 px²`) على `AIResQ`.
- **اختبار استئصال التطبيع الحراري (`Exp-04`):** اختبار أثر `CLAHE` / التطبيع التكيفي على صور التباين الضعيف (`Low-Contrast`).
- **التقييم الميداني على فيديوهات الدرون الحقيقية:** اختبار `BoT-SORT GMC` على `DJI_0003.mov` و`DJI_0024.mov` و`DJI_0026.mov`.
- **النشر الطرفي (`Edge Deployment`):** تصدير النموذج البطل إلى `ONNX` و`TensorRT (FP16)` لحواسيب الدرون مثل `NVIDIA Jetson Orin`.

</div>
