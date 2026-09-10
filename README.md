# پیش‌بینی خرابی تجهیزات خط تولید خودرو با یادگیری ماشین

پروژه کارشناسی مهندسی کامپیوتر | سید امیرمحمد علوی | استاد راهنما: دکتر فاطمه اسماعیلی خلیل سرایی

## معرفی پروژه

این پروژه یک سیستم نگهداری و تعمیرات پیش‌بینانه (Predictive Maintenance) بر پایه یادگیری ماشین طراحی می‌کند که با تحلیل داده‌های سنسوری تجهیزات خط تولید خودرو، وقوع خرابی را پیش از رخداد آن پیش‌بینی می‌کند.

## دیتاست

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) از UCI Machine Learning Repository — شامل ۱۰٬۰۰۰ رکورد شبیه‌سازی‌شده از سنسورهای صنعتی.

## ساختار پروژه

## روش کار

1. تحلیل اکتشافی داده (EDA) و شناسایی الگوهای کلیدی
2. پیش‌پردازش: حذف ستون‌های دارای نشتی داده (Data Leakage)، One-Hot Encoding، نرمال‌سازی، متعادل‌سازی با SMOTE
3. آموزش و مقایسه سه مدل: Logistic Regression، Random Forest، XGBoost
4. تنظیم فراپارامتر با RandomizedSearchCV
5. ارزیابی نهایی روی داده تست با معیارهای Precision، Recall، F1-Score، ROC-AUC

## نتایج نهایی (مدل منتخب: XGBoost)

| معیار (کلاس خرابی) | مقدار |
|---|---|
| Precision | 0.6548 |
| Recall | 0.8088 |
| F1-Score | 0.7237 |
| Accuracy کلی | 0.9790 |
| ROC-AUC | 0.9698 |

مهم‌ترین فیچرهای پیش‌بینی‌کننده: Torque، Rotational speed، Tool wear.

## نحوه اجرا

```bash
cd src
python pipeline.py
```

## کتابخانه‌های مورد نیاز

pandas, numpy, scikit-learn, xgboost, imbalanced-learn, matplotlib, seaborn, joblib