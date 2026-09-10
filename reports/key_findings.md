# یافته‌های کلیدی پروژه

## EDA (فاز ۱)
- کلاس هدف شدیداً نامتوازن: 96.61% سالم در برابر 3.39% خراب
- قوی‌ترین فیچرهای مرتبط با خرابی (بر اساس Boxplot و همبستگی): Torque (0.19)، Tool wear (0.11)
- همبستگی بالا بین Air temperature و Process temperature (0.88)، و بین Rotational speed و Torque (-0.88)
- ستون‌های TWF، HDF، PWF، OSF دارای Data Leakage قطعی بودند (هر رکورد خراب در این ستون‌ها، همیشه Machine failure=1 بود)
- RNF فاقد رابطه معنادار با خرابی بود (احتمالاً نویز/خرابی تصادفی)
- نرخ خرابی بر اساس Type: L=3.92%، M=2.77%، H=2.09%

## پیش‌پردازش (فاز ۲)
- حذف 7 ستون (شناسه + نشتی داده)
- One-Hot Encoding برای Type
- Train/Test Split: 80/20 با stratify
- StandardScaler (fit فقط روی Train)
- SMOTE فقط روی Train: از 271 به 7729 نمونه خراب (تعادل 50/50)

## مقایسه مدل‌ها (Cross-Validation روی Train)
| مدل | F1 (CV میانگین) |
|---|---|
| Logistic Regression | 0.8304 |
| Random Forest | 0.9830 |
| XGBoost | 0.9884 |

## ارزیابی نهایی روی Test (فاز ۴)
| مدل | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.1511 | 0.8088 | 0.2546 | 0.8958 |
| Random Forest | 0.5050 | 0.7500 | 0.6036 | 0.9743 |
| XGBoost | 0.6548 | 0.8088 | 0.7237 | 0.9698 |

**مدل منتخب نهایی: XGBoost** — بهترین تعادل بین Recall و Precision.

## Feature Importance
مهم‌ترین فیچرها (هر دو مدل RF و XGBoost موافقند): Torque، Rotational speed، Tool wear.
نکته: اهمیت بالای Rotational speed با وجود همبستگی خطی ضعیف (-0.04) نشان‌دهنده تعامل غیرخطی با Torque است.

## تصمیمات کلیدی و توجیه آنها
- SMOTE به‌جای وزن‌دهی کلاس: برای مقابله با نامتوازن بودن شدید
- معیار F1/Recall به‌جای Accuracy: چون Accuracy با کلاس نامتوازن گمراه‌کننده است
- Threshold پیش‌فرض 0.5 حفظ شد: بررسی Threshold های دیگر (0.3-0.6) تعادل بهتری ارائه نداد
- فاز RUL (اختیاری) اجرا نشد: به دلیل محدودیت زمانی و اولویت با فازهای اصلی پروپوزال