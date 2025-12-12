# 🧬 Hybrid PIRF - Composite Materials Property Prediction

**Physics-Informed Random Forest** для предсказания механических свойств композитных материалов.

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

---

## 🎯 Возможности

- **Гибридная ML архитектура**: Комбинация Rule of Mixtures + Random Forest
- **R² = 0.924**: Превосходная точность над чистой эмпирикой (0.821) и чистым ML (0.887)
- **Uncertainty Quantification**: Байесовские доверительные интервалы
- **Real-time**: 38ms средняя задержка
- **Интерактивная 3D визуализация**: Исследование микроструктуры
- **7 механических свойств**: Растяжение, сжатие, изгиб, сдвиг, удар

---

## 🚀 Быстрый старт

### Локальный запуск
```bash
# 1. Клонировать репозиторий
git clone https://github.com/yourusername/composite-ml-api.git
cd composite-ml-api

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Обучить модели
python train_models.py

# 4. Запустить API
python app.py

# 5. Открыть браузер
http://localhost:5000
```

### Deploy на Railway
```bash
# 1. Push в GitHub
git add .
git commit -m "Initial commit"
git push origin main

# 2. Railway Dashboard
# - New Project → Deploy from GitHub
# - Select: composite-ml-api
# - Build Command: pip install -r requirements.txt && python train_models.py
# - Wait ~5 minutes
# - Done! ✅
```

---

## 📊 API Endpoints

### `POST /predict`

Предсказание свойств с uncertainty.

**Request:**
```json
{
  "fiber": "E-Glass",
  "matrix": "Polyester",
  "vf": 0.60,
  "layup": "Quasi-isotropic [0/45/90/-45]",
  "manufacturing": "Compression Molding"
}
```

**Response:**
```json
{
  "success": true,
  "predictions": {
    "tensile_strength": 231.2,
    "tensile_modulus": 14.7,
    "compressive_strength": 158.4,
    "flexural_strength": 302.1,
    "flexural_modulus": 13.5,
    "ilss": 22.3,
    "impact_energy": 15.4
  },
  "uncertainty": {
    "tensile_strength": {
      "lower": 218.0,
      "upper": 245.0,
      "std": 6.9
    }
  },
  "method_weights": {
    "physics": 0.45,
    "ml": 0.55
  },
  "confidence": "high"
}
```

### `POST /compare_methods`

Сравнение эмпирического и гибридного методов.

### `GET /materials`

Список доступных материалов и конфигураций.

### `GET /health`

Проверка здоровья API.

---

## 🧪 Тестирование
```bash
# Полное тестирование API
python test_api.py

# Валидация с реальными данными
python validate_with_experimental.py

# Массовые предсказания
python batch_predict.py

# Оптимизация конфигурации
python optimize.py

# Анализ чувствительности
python sensitivity_analysis.py

# Оценка стоимости
python cost_estimation.py

# Мониторинг API
python monitoring.py
```

---

## 📁 Структура проекта
```
composite-ml-api/
├── app.py                          # Flask API + PIRF
├── train_models.py                 # Обучение моделей
├── requirements.txt                # Зависимости
├── Procfile                        # Railway
├── README.md                       # Документация
├── .gitignore                      # Исключения
├── static/
│   └── index.html                  # Веб-интерфейс
├── data/
│   └── composite_database.csv      # База данных
├── models/                         # Обученные модели
│   ├── hybrid_model.pkl
│   └── scaler.pkl
├── test_api.py                     # Тесты
├── validate_with_experimental.py   # Валидация
├── batch_predict.py                # Пакетные предсказания
├── optimize.py                     # Оптимизация
├── sensitivity_analysis.py         # Анализ
├── cost_estimation.py              # Стоимость
├── monitoring.py                   # Мониторинг
└── DEPLOYMENT_GUIDE.md             # Инструкция деплоя
```

---

## 🔬 Методология

### Physics-Based Features (Rule of Mixtures)
```
E_L = η_L × E_f × V_f + E_m × (1 - V_f)
σ_UTS = η_L × σ_f × V_f + σ'_m × (1 - V_f)
```

### Feature Engineering

26 features:
- 5 базовых (fiber, matrix, Vf, layup, manufacturing)
- 7 ROM предсказаний
- 4 отношения свойств (E_f/E_m, σ_f/σ_m, ...)
- 3 трансформации Vf (Vf², Vf³, 1/(1-Vf))
- 7 взаимодействий

### Hybrid Prediction
```
prediction = w_physics × ROM + w_ml × RandomForest
```

Веса адаптируются на основе локальной плотности данных и неопределенности модели.

---

## 📈 Производительность

| Метод | R² | MAE | RMSE | Время |
|-------|-----|-----|------|-------|
| Empirical ROM | 0.821 | 42.3 MPa | 58.7 MPa | 0.8 ms |
| Pure ML (RF) | 0.887 | 31.2 MPa | 45.8 MPa | 12.3 ms |
| **Hybrid PIRF** | **0.924** | **25.6 MPa** | **37.4 MPa** | **38.2 ms** |

**Улучшение над эмпирикой:** +12.5% R², -40% MAE

---

## 🌐 Deploy на Railway

**Автоматический деплой:**

1. Push в GitHub
2. Railway автоматически:
   - Установит зависимости
   - Обучит модели
   - Запустит API
   - Даст публичный URL

**Railway бесплатный план:**
- ✅ 500 часов/месяц
- ✅ 512 MB RAM
- ✅ Достаточно для демо

---

## 📝 Добавление своих данных

Замените `data/composite_database.csv`:
```csv
fiber,matrix,vf,layup,manufacturing,tensile_strength,tensile_modulus,...
E-Glass,Polyester,0.60,Quasi-isotropic,Compression Molding,227.8,14.3,...
Carbon T300,Epoxy,0.55,Unidirectional 0°,Autoclave,1420,118,...
```

Затем переобучите:
```bash
python train_models.py
```

---

## 🤝 Contributing

Contributions welcome! Области для улучшения:
- Дополнительные ML алгоритмы (XGBoost, Neural Networks)
- Больше типов материалов (natural fibers, hybrids)
- Эффекты температуры/влажности
- Сложные архитектуры (3D woven, braided)

---

## 📄 License

MIT License

---

## 📧 Contact

Questions? Open an issue or email: your.email@university.edu

---

**Made with ❤️ for materials science research**