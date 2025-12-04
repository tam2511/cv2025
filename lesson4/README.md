# Object Detection

Lecture: [link](https://disk.yandex.ru/i/SJKmglkKihEidg)

Seminar: [seminar](./seminar)

## Dataset

**COCO val2017 (Person subset)** - подмножество датасета COCO для детекции объектов класса "person".

- **Количество изображений**: 
  - Train: ~2093 изображений
  - Val: ~300 изображений
  - Test: ~300 изображений
  - Mini train: ~1000 изображений (для быстрого тестирования)
- **Класс**: Person (1 класс)
- **Расположение**: `data/coco/`
- **Структура данных**:
  - `val2017/` - изображения из COCO val2017
  - `annotations/` - аннотации в формате COCO JSON
    - `instances_val2017.json` - оригинальные аннотации
    - `instances_val2017_train.json` - train split
    - `instances_val2017_val.json` - val split
    - `instances_val2017_test.json` - test split
    - `instances_val2017_mini_train.json` - mini train split

**Подготовка датасета:**

**Шаг 1: Скачивание данных**

```bash
cd lesson4/seminar
python tools/download_coco_mini.py
```

Этот скрипт автоматически скачает:
- Изображения COCO val2017 (~780MB)
- Аннотации COCO trainval2017 (~241MB)

**Шаг 2: Подготовка сплитов**

```bash
cd lesson4/seminar
python tools/prepare_coco_mini.py
```

Этот скрипт создаст train/val/test/mini_train сплиты из изображений с классом "person".

**Проверка структуры данных:**

После подготовки убедитесь, что структура данных правильная:

```bash
cd lesson4/seminar
ls -la ../data/coco/val2017/ | head -5
ls -la ../data/coco/annotations/
```

Должны быть:
- Папка `val2017/` с изображениями
- Папка `annotations/` с JSON файлами сплитов

**Примечание:** Если датасет уже подготовлен и находится в правильной структуре (`../data/coco/` относительно `seminar/`), ноутбуки будут работать автоматически без дополнительных действий.

## Модели

В семинаре реализованы две архитектуры для детекции объектов:

1. **DETR (Detection Transformer)** - end-to-end детектор на основе Transformer
   - Использует learnable queries для поиска объектов
   - Не требует NMS (Non-Maximum Suppression)
   - Архитектура: ResNet50 backbone + Transformer encoder/decoder

2. **AnchorDETR** - улучшенная версия DETR с anchor points
   - Использует anchor-based queries вместо learnable queries
   - Более быстрая сходимость обучения
   - Архитектура: ResNet50 backbone + Anchor-based Transformer

## Структура проекта

```
seminar/
├── src/                    # Исходный код
│   ├── models/            # Модели (DETR, AnchorDETR)
│   ├── dataset_coco.py   # Датасет COCO
│   ├── lightning_module.py # PyTorch Lightning модуль
│   └── config.py          # Загрузка конфигов
├── configs/               # Конфигурационные файлы
│   ├── config_coco_detr_person.json
│   └── config_coco_anchor_detr_person.json
├── tools/                 # Утилиты
│   ├── download_coco_mini.py    # Скачивание данных
│   └── prepare_coco_mini.py    # Подготовка сплитов
├── notebooks/             # Ноутбуки семинара
│   ├── train_detr_person.ipynb
│   ├── train_anchor_detr_person.ipynb
│   └── resnet_fpn_to_detr.ipynb
└── requirements.txt       # Зависимости
```

## Использование

1. Установите зависимости:
```bash
cd lesson4/seminar
pip install -r requirements.txt
```

2. Подготовьте датасет (см. раздел Dataset выше)

3. Запустите ноутбуки:
   - `train_detr_person.ipynb` - обучение DETR
   - `train_anchor_detr_person.ipynb` - обучение AnchorDETR
   - `resnet_fpn_to_detr.ipynb` - постепенное построение архитектуры DETR
