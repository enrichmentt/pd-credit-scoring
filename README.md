# PD Credit Scoring — модель предсказания дефолта

Итоговый проект по дисциплине «Автоматизация процессов разработки и тестирования
моделей машинного обучения».

Здесь я собрал сквозной пайплайн для PD-модели (probability of default):
от подготовки данных до API и мониторинга дрифта. Датасет —
[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
с UCI: 30 000 клиентов кредитных карт на Тайване. Целевая переменная —
был ли дефолт в следующем месяце (`default.payment.next.month`).

## Что получилось

Лучше всего себя показал **GradientBoostingClassifier** после подбора
гиперпараметров через GridSearchCV:

| Метрика (test) | Значение   |
| -------------- | ---------- |
| ROC-AUC        | **0.7837** |
| F1             | 0.4669     |
| Precision      | 0.6509     |
| Recall         | 0.3640     |
| Accuracy       | 0.8162     |

Всего прогнал 5 экспериментов, все залогированы в MLflow:

| Эксперимент                                         | ROC-AUC | F1     |
| --------------------------------------------------- | ------- | ------ |
| gb_deep_subsample (GradientBoosting, subsample=0.8) | 0.7837  | 0.4669 |
| gb_grid (GradientBoosting)                          | 0.7804  | 0.4752 |
| rf_grid (RandomForest)                              | 0.7765  | 0.4668 |
| logreg_balanced (LogReg, class_weight=balanced)     | 0.7479  | 0.5094 |
| logreg_baseline (LogReg)                            | 0.7458  | 0.4060 |

Из интересного: балансировка классов у логрега поднимает recall почти вдвое
(0.30 -> 0.60), но проседает precision. По ROC-AUC бустинг всё равно впереди,
поэтому его и взял в финальную модель.

## Структура проекта

```
├── .github/workflows/ci.yml   # CI: flake8, black, pytest, валидация данных
├── data/
│   ├── raw/                   # сырые данные (под DVC)
│   └── processed/             # train/test после подготовки (DVC)
├── models/                    # обученная модель model.joblib (DVC)
├── notebooks/01_eda.ipynb     # разведочный анализ
├── reports/
│   ├── figures/               # ROC-кривые и графики из EDA
│   └── metrics.json           # метрики финальной модели
├── src/
│   ├── config.py              # пути и константы
│   ├── data/prepare.py        # загрузка, очистка, сплит (стадия DVC prepare)
│   ├── features/build_features.py  # feature engineering
│   ├── validation/            # валидация данных через Pandera
│   ├── models/train.py        # sklearn Pipeline + GridSearchCV + MLflow
│   ├── models/run_experiments.py   # серия из 5 экспериментов
│   ├── models/predict.py      # инференс
│   ├── api/app.py             # FastAPI: /predict, /health
│   └── monitoring/drift.py    # мониторинг дрифта (PSI)
├── tests/                     # тесты pytest + небольшая фикстура данных
├── dvc.yaml                   # DVC-пайплайн: prepare -> train
├── params.yaml                # параметры стадий
├── Dockerfile                 # образ API-сервиса
├── Dockerfile.train           # образ для воспроизведения обучения
└── scripts/                   # docker_run.py (билд+запуск контейнера), sample_request.json
```

## Как запустить

Сразу после `git clone` всё работает из коробки: сырой датасет и обученная
модель лежат прямо в репозитории (и продублированы в DVC), так что API,
тесты и мониторинг можно запускать без каких-либо `dvc pull`.

Сначала лучше сделать виртуальное окружение, что бы зависимости проекта
не смешивались с остальными пакетами в системе:

```bash
python -m venv .venv

# активация — Windows, PowerShell:
.venv\Scripts\Activate.ps1

# активация — Windows, Git Bash:
source .venv/Scripts/activate

# активация — Linux / macOS:
source .venv/bin/activate
```

Дальше ставим зависимости и запускаем:

```bash
pip install -r requirements.txt

# весь пайплайн целиком (prepare -> train)
dvc repro

# или по шагам
python -m src.data.prepare            # подготовка + валидация данных
python -m src.models.run_experiments  # 5 экспериментов с MLflow
```

### MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Открываем http://localhost:5000, эксперимент `pd-credit-scoring`. По каждому
запуску видны параметры, метрики, ROC-кривая и сама модель.

### Тесты и валидация

```bash
pytest tests/ -v
python -m src.validation.validate data/raw/UCI_Credit_Card.csv
```

### API

Локально, без докера:

```bash
uvicorn src.api.app:app --port 8000
```

В докере — просто две команды. На Windows они те же самые (в PowerShell
или cmd), нужен только запущенный Docker Desktop:

```bash
# собрать образ (выполняется из корня проекта)
docker build -t pd-scoring-api .

# запустить контейнер
docker run -d --name pd-api -p 8000:8000 pd-scoring-api

# проверить, что поднялся
curl http://localhost:8000/health
```

Пример запроса на скоринг:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @scripts/sample_request.json
# -> {"default": 0, "probability": 0.113958, "threshold": 0.5}
```

Остановить и удалить контейнер: `docker rm -f pd-api`.

Можно запустить скрипт — всё это (билд, запуск, проверка /health)
делает `python scripts/docker_run.py`

Swagger живёт на http://localhost:8000/docs, там можно потыкать /predict руками.

### Мониторинг дрифта

```bash
# локальный скоринг части тестовой выборки
python -m src.monitoring.drift --sample-size 2000

# то же самое, но через запущенное API
python -m src.monitoring.drift --api-url http://localhost:8000/predict
```

Скрипт считает PSI (Population Stability Index) по распределению скоров и по
ключевым признакам. Читается так: PSI < 0.1 — всё стабильно, 0.1–0.25 —
умеренный дрифт, > 0.25 — пора переобучаться.

## Feature engineering

Что добавил к исходным признакам:

- `pay_delay_mean`, `pay_delay_max`, `months_delayed` — агрегаты по истории
  просрочек (PAY_0..PAY_6), это самый сильный блок предикторов;
- `utilization_mean` — средняя утилизация кредитного лимита;
- `pay_ratio_mean` — средняя доля погашения счёта;
- `bill_trend` — динамика долга за 6 месяцев;
- `age_group` — биннинг возраста, `limit_per_age` — лимит на год возраста.

## Валидация данных

Схема на Pandera (`src/validation/schema.py`) проверяет сырой файл: типы,
пропуски, диапазоны (возраст 18–100, PAY_x от -2 до 9), допустимые категории
и бинарный таргет. Тест `test_schema_rejects_anomalies` специально подсовывает
битые данные и ждёт, что валидация их завернёт. Так же валидация запускается
в CI на каждый пуш.

## CI/CD

- **CI**: GitHub Actions гоняет flake8, black --check, pytest и валидацию
  данных на каждый push/PR.
- **Версионирование**: данные и модель под DVC, `dvc repro` воспроизводит
  весь пайплайн по `dvc.yaml` и `params.yaml`.
- **Развёртывание**: Docker-образ с FastAPI, собирается двумя командами
  `docker build` / `docker run` (см. раздел API), health-check уже в Dockerfile.
