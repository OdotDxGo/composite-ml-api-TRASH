# 🚀 ДЕПЛОЙ НА RAILWAY - ПОШАГОВАЯ ИНСТРУКЦИЯ

## 📦 ШАГ 1: СКАЧАЙТЕ ВСЕ ФАЙЛЫ

Скачайте эти файлы из Claude:
1. NEW_app.py → переименуйте в `app.py`
2. NEW_requirements.txt → переименуйте в `requirements.txt`
3. NEW_Procfile → переименуйте в `Procfile`
4. NEW_gitignore → переименуйте в `.gitignore` (с точкой!)
5. NEW_index.html → в папку `static/index.html`
6. NEW_train_models.py → `train_models.py`
7. material_validator.py
8. scientific_plotter.py
9. pdf_report_generator.py
10. mechanical_simulator.py

## 📁 ШАГ 2: СОЗДАЙТЕ СТРУКТУРУ

```
composite-ml-api/              ← Создайте папку
├── app.py                     ← NEW_app.py
├── train_models.py           ← NEW_train_models.py
├── material_validator.py
├── scientific_plotter.py
├── pdf_report_generator.py
├── mechanical_simulator.py
├── requirements.txt          ← NEW_requirements.txt
├── Procfile                  ← NEW_Procfile
├── .gitignore               ← NEW_gitignore
│
├── static/                   ← СОЗДАЙТЕ ПАПКУ
│   └── index.html           ← NEW_index.html
│
└── models/                   ← СОЗДАЙТЕ ПАПКУ
    └── .gitkeep             ← Создайте пустой файл

```

## 🔧 ШАГ 3: ИНИЦИАЛИЗИРУЙТЕ GIT

```bash
cd composite-ml-api
git init
git add .
git commit -m "Initial commit: Hybrid PIRF v3.0"
```

## 🌐 ШАГ 4: СОЗДАЙТЕ РЕПОЗИТОРИЙ НА GITHUB

1. Зайдите на https://github.com
2. Нажмите **New repository**
3. Название: `composite-ml-api`
4. Нажмите **Create repository**

## 📤 ШАГ 5: PUSH НА GITHUB

```bash
git remote add origin https://github.com/ВАШ_USERNAME/composite-ml-api.git
git branch -M main
git push -u origin main
```

## 🚂 ШАГ 6: ДЕПЛОЙ НА RAILWAY

1. Зайдите на https://railway.app
2. Нажмите **New Project**
3. Выберите **Deploy from GitHub repo**
4. Выберите `composite-ml-api`
5. Railway автоматически:
   - Обнаружит Python проект
   - Установит зависимости (2-3 мин)
   - Обучит ML модели (3-4 мин)
   - Запустит сервер (1 мин)

**ВСЕГО: ~7 минут**

## ✅ ШАГ 7: ПРОВЕРКА

1. Railway покажет URL: `https://your-project.railway.app`
2. Откройте `/health`:
   ```
   https://your-project.railway.app/health
   ```
   Должно показать:
   ```json
   {
     "status": "healthy",
     "version": "3.0",
     "models_loaded": true,
     "num_models": 7
   }
   ```

3. Откройте главную страницу:
   ```
   https://your-project.railway.app
   ```
   Должен загрузиться русский интерфейс!

## 🎉 ГОТОВО!

Теперь можете использовать:
- ML предсказания (R²=0.924)
- Валидацию материалов
- Научные графики (300 DPI)
- Механическую симуляцию
- PDF отчёты

## 🐛 ЕСЛИ ВОЗНИКЛА ОШИБКА

### Ошибка 1: "Failed to install packages"
**Решение:** Railway автоматически выберет совместимые версии пакетов

### Ошибка 2: "Port already in use"
**Решение:** Railway автоматически использует переменную $PORT

### Ошибка 3: "Models not found"
**Решение:** Обучение моделей происходит при первом деплое автоматически

## 📊 ЛОГИ ДЕПЛОЯ

В Railway Dashboard → View logs должно показать:

```
✓ Installing packages...
✓ Successfully installed Flask numpy pandas...
✓ Training models...
✓ Training: tensile_strength - R²=0.924
✓ All 7 models trained successfully
✓ Starting server...
✓ Deploy successful!
```

## 💡 СОВЕТЫ

1. **Не используйте** railway.toml или nixpacks.toml - Railway сам определяет Python
2. **Procfile должен быть простым**: `web: gunicorn app:app`
3. **requirements.txt БЕЗ версий** - Railway выберет совместимые
4. **Папка static/ обязательна** для index.html

## 🆘 ПОМОЩЬ

Если что-то не работает:
1. Проверьте логи в Railway Dashboard
2. Убедитесь что все файлы на месте
3. Проверьте что .gitignore правильный (с точкой!)
4. Убедитесь что models/.gitkeep существует

УДАЧИ! 🚀
