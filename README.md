# WildHack Solo Track: Time Series Forecasting Challenge

Успешное решение задачи прогнозирования загруженности маршрутов Wildberries на горизонте 4 часов для соревнования **Wildhack**.

**Результат**: 🥉 **37-е место в приватном лидерборде, бронзовая медаль**

## 📋 Описание проекта

### Задача
Прогнозирование загруженности **маршрутов доставки Wildberries** на 8 шагов вперед (30-минутные интервалы, 4 часа горизонта). Модель должна предсказать значения показателя `target_1h` для каждого маршрута в каждый момент времени в будущем.

### Метрика
$$\text{WAPE} + |\text{RBias}| = \frac{\sum |y_{pred} - y_{true}|}{\sum y_{true}} + \left|\frac{\sum y_{pred}}{\sum y_{true}} - 1\right|$$

- **WAPE** (Weighted Absolute Percentage Error) — взвешенная абсолютная ошибка в процентах
- **RBias** (Relative Bias) — относительное смещение суммарного прогноза

---

## 🏗️ Архитектура решения

Решение представляет собой **каскадный ансамбль** трёх семейств моделей, каждая из которых оптимизирована для разной части горизонта прогноза:

### 1. LightGBM (Градиентный бустинг)
**Применение**: Шаги 1–2 горизонта (первый час)

- **Стратегия**: Прямой многошаговый прогноз (DIRECT) — отдельная модель для каждого из 8 шагов
- **Целевой показатель**: `target_1h` с лог-нормализацией относительно медианы маршрута
- **Параметры**:
  - `objective`: regression_l1 (MAE-оптимизация)
  - `learning_rate`: 0.04
  - `num_leaves`: 255 (глубокие деревья)
  - `GPU-ускорение`, `max_bin`: 63
  - Early stopping: 200 раундов без улучшения (до 5000 итераций)

- **Почему эффективен**: Информация о текущем состоянии маршрута максимально ценна на коротком горизонте

### 2. Temporal Fusion Transformer (TFT)
**Применение**: Шаги 2–5 горизонта (средний диапазон)

- **Библиотека**: NeuralForecast
- **Архитектура**:
  - Input size: 96 (48 часов ретроспективы)
  - Hidden size: 32
  - LSTM-слои: 2
  - Attention heads: 4
  - Dropout: 0.1
  - Precision: float16-mixed (экономия VRAM)

- **Входные данные**:
  - История: log1p-нормализованный таргет (96 шагов)
  - Будущие ковариаты: час, минута, день недели, флаг субботы, тайм-слот

- **Почему эффективен**: Трансформер эффективно использует долгосрочные паттерны и будущих ковариат

- **Особенность**: Обучается с чекпоинтированием каждые 50 шагов → при повторном запуске автоматически продолжается с последнего checkpoint

### 3. N-HiTS × 5 Seeds (Neural Hierarchical Interpolation)
**Применение**: Шаги 6–8 горизонта (длинный горизонт)

- **Архитектура**:
  - Input size: 672 (2 недели ретроспективы)
  - 3 блока иерархического пулинга
  - Pool kernel size: [16, 4, 1]
  - Frequency downsampling: [48, 8, 1]
  - MLP per block: [256, 256]
  - Dropout: 0.15

- **Ансамблирование**: 5 моделей с разными seeds (42–46) → средн арифметическое
- **Калибровка**: Множитель NHITS_CAL = 1.010 (небольшое смещение вверх)

- **Почему эффективен**: Не требует внешних ковариат, устойчив к длинным горизонтам

---

## 🔥 Feature Engineering (200+ признаков)

### Временные признаки
- `hour`, `minute`, `dayofweek`, `dayofmonth`, `month` (прямые)
- `is_weekend`, `is_saturday` (флаги)
- `time_slot` — индекс 30-минутного слота (0–47)
- Гармонические кодировки (sin/cos):
  - `hour_sin`, `hour_cos`
  - `dow_sin`, `dow_cos`
  - `slot_sin`, `slot_cos`

### Лаги целевого показателя
Шаги: 1, 2, 3, 4, 6, 8, 12, 16, 24, 48, 96, 336

### Скользящие статистики (окна 4, 8, 16, 48)
- `rmean_*` — скользящее среднее
- `rstd_*` — скользящее стандартное отклонение
- `rmax_*` — скользящий максимум
- `rmin_*` — скользящий минимум
- `ewm_*` — экспоненциальные скользящие средние (spans 4, 12, 24)

### Различия и тренды
- `diff_1`, `diff_4`, `diff_8`, `diff_16`, `diff_48` — разности таргета
- `target_trend` — линейный тренд за последние 8 точек

### Профили маршрутов (по исторической выборке)
- `route_mean`, `route_std`, `route_median` — базовая статистика
- `route_q10`, `route_q25`, `route_q75`, `route_q90` — квантили
- `route_slot_mean` — среднее по тайм-слотам
- `route_dow_mean` — среднее по дням недели
- **Собтни**:
  - `route_sat_slot_mean` — среднее по субботним слотам
  - `route_sat_recent_mean` — среднее по субботам последних 4 недель (более актуально)

### Кросс-маршрутные признаки
- `cross_mean`, `cross_std`, `cross_median` — агрегации по всем маршрутам на метку времени

### Производные признаки
- `dev_from_mean` — отклонение от среднего маршрута
- `rel_dev` — нормализованное отклонение
- `dev_from_slot` — отклонение от среднего по слоту
- `dev_from_cross` — отклонение от кросс-маршрутного среднего
- `target_norm` — нормализованный таргет

### Признаки статусов (для каждого status_*)
- Лаги: 1, 4, 8, 16, 24, 48
- Скользящие статистики (windows 4, 8, 16)
- Различия (diff_1, diff_4)

---

## 📊 Оптимизация: Каскадная стратегия

На **holdout-множестве** (25 октября 2025, 11:00–14:30) подбирается оптимальное распределение:

| Шаг | Горизонт | Рекомендуемый источник |
|-----|----------|----------------------|
| 1 | 30 мин | **100% LGBM** |
| 2 | 1 час | 50% LGBM + 30% TFT + 20% NHiTS |
| 3 | 1.5 часа | **60% TFT + 40% NHiTS** |
| 4 | 2 часа | **60% TFT + 40% NHiTS** |
| 5 | 2.5 часа | 40% TFT + 60% NHiTS |
| 6 | 3 часа | 20% TFT + 80% NHiTS |
| 7 | 3.5 часа | **100% NHiTS** |
| 8 | 4 часа | **100% NHiTS** |

**Мягкий каскад** (`cascade_soft`) показал лучший результат на публичном лидерборде благодаря плавному переходу между источниками.

---

## 🚀 Как запустить

### Требования

```bash
pip install numpy pandas torch pytorch-lightning lightgbm xgboost scikit-learn neuralforecast
```

### Основные зависимости

| Пакет | Версия | Назначение |
|-------|--------|-----------|
| `pytorch` | 2.0+ | Нейронные сети, GPU ускорение |
| `neuralforecast` | 0.5+ | TFT, N-HiTS |
| `lightgbm` | 4.0+ | Градиентный бустинг |
| `xgboost` | 2.0+ | Альтернативный бустинг |
| `pandas` | 1.5+ | Обработка данных |
| `numpy` | 1.24+ | Численные операции |
| `pytorch-lightning` | 2.0+ | Обучение с callback'ами |

### Запуск ноутбука

```bash
# Вариант 1: основной пайплайн с feature engineering и всеми моделями
jupyter notebook wb_final_full.ipynb

# Вариант 2: пайплайн с загруженными предсказаниями LGBM/XGB
jupyter notebook wb_final_with_preds.ipynb
```

### Структура данных

```
train_solo_track.parquet:
- route_id (ID маршрута)
- timestamp (время)
- target_1h (целевой показатель)
- status_* (статусные признаки)

test_solo_track.parquet:
- id (ID сампла)
- route_id, timestamp (как в train)
```

### Выходные файлы

Ноутбук генерирует несколько сабмитов:

```
submission_solo_nhits_ens5.csv         # Лучший NHiTS ансамбль
submission_solo_tft_cal.csv             # Калибрированный TFT
submission_solo_cascade_best.csv        # Автоматически найденный каскад
submission_solo_cascade_soft.csv        # 🏆 Мягкий каскад (финальный выбор)
submission_solo_lgbm1_tft25_nhits68.csv # Альтернативный каскад
submission_solo_lgbm12_nhits38.csv      # LGBM-первый каскад
```

---

## 🔧 Особенности реализации

### 1. Чекпоинтирование TFT
```python
# Модель автоматически продолжает обучение с последнего checkpoint
ckpt = find_ckpt(CKPT_DIR)
if ckpt:
    nf_tft = NeuralForecast.load(path=CKPT_DIR)
else:
    nf_tft = NeuralForecast(models=[make_tft(...)], freq='30T')
    nf_tft.fit(df=df_tft_fit, val_size=FP)
```

### 2. Log1p-нормализация
```python
df_base['y'] = np.log1p(df_base['y'] / df_base['unique_id'].map(route_medians))
```
Нормализация по медиане маршрута снижает вариативность и улучшает обучение моделей.

### 3. GPU-оптимизация
- LightGBM: GPU-ускорение на устройстве кода
- PyTorch-Lightning: `precision='16-mixed'` → 40% экономия VRAM
- Batch size: 8 для TFT, 16 для NHiTS

### 4. Monkey patch для PyTorch-Lightning
Обход конфликтов параметров при специфических версиях:
```python
_orig_init = pl.Trainer.__init__
def _patched_init(self, *args, **kwargs):
    forbidden = {'lstm_layers', 'num_heads', 'attn_dropout', ...}
    for k in forbidden:
        kwargs.pop(k, None)
    _orig_init(self, *args, **kwargs)
pl.Trainer.__init__ = _patched_init
```

---

## 📈 Результаты

### Публичный лидерборд
- **Место**: 37 из ~500+ участников
- **Медаль**: 🥉 Бронза
- **Сабмит**: `cascade_soft` (мягкий каскад)

---

## 📁 Структура репозитория

```
wildhack_solo-track/
├── wb_final_full.ipynb              # Полный пайплайн (feature eng + все модели)
├── wb_final_with_preds.ipynb        # С загруженными предсказаниями LGBM/XGB
├── data/
│   └── preds_v6/                    # Предвычисленные предсказания
│       ├── lgbm_val_*.npy           # LGBM validation (8 шагов)
│       ├── lgbm_tst_*.npy           # LGBM test (8 шагов)
│       ├── xgb_val_*.npy            # XGB validation
│       ├── xgb_tst_*.npy            # XGB test
│       └── y_val_*.npy              # True values validation
├── LICENSE
└── README.md
```

---

## 👤 Создание и тестирование

Проект разработан и протестирован на:
- **Kaggle Notebooks** (T4 GPU 16GB, 30 часов compute)
- **Python 3.10+**
- **CUDA 11.8+**

---

## 📝 Лицензия

Apache License