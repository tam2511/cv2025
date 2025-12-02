# Сегментация

Lecture: [link](https://disk.yandex.ru/i/3DmzBbk3plzR2A)

Seminar: [seminar](./seminar)

## Dataset

**CamVid** - датасет для семантической сегментации дорожных сцен.

- **Количество изображений**: ~368 train, ~101 val, ~233 test
- **Классы**: 12 классов
  - Sky, Building, Column-Pole, Road, Sidewalk, Tree, Sign-Symbol, Fence, Car, Pedestrian, Bicyclist, Void
- **Расположение**: `data/`
- **Структура данных**:
  - `train/` - изображения для обучения
  - `trainannot/` - аннотации (маски) для обучения
  - `val/` - изображения для валидации
  - `valannot/` - аннотации для валидации
  - `test/` - изображения для тестирования
  - `testannot/` - аннотации для тестирования
  - `train.txt`, `val.txt`, `test.txt` - файлы со списками путей к изображениям и маскам

**Скачивание датасета:**

**Способ 1: Автоматическое скачивание через скрипт**

```bash
cd lesson3/data
wget http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/data/CamVid.zip
unzip CamVid.zip
```

**Способ 2: Ручное скачивание с официального сайта**

1. Посетите официальный сайт: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
2. Скачайте архив `CamVid.zip`
3. Распакуйте архив в директорию `data/`
4. Убедитесь, что структура соответствует ожидаемой (папки `train/`, `trainannot/`, `val/`, `valannot/`, `test/`, `testannot/`)

**Способ 3: Использование альтернативных источников**

Если официальный сайт недоступен, можно использовать зеркала или готовые репозитории:

```bash
cd lesson3/data
# Альтернативная ссылка (если доступна)
wget https://github.com/alexgkendall/SegNet-Tutorial/raw/master/CamVid.zip
unzip CamVid.zip
```

**Проверка структуры данных:**

После скачивания убедитесь, что структура данных правильная:

```bash
cd lesson3/data
ls -la train/ | head -5
ls -la trainannot/ | head -5
ls -la val/ | head -5
```

Должны быть файлы с одинаковыми именами в соответствующих папках (например, `train/0001TP_006690.png` и `trainannot/0001TP_006690.png`).

**Примечание:** Если датасет уже скачан и находится в правильной структуре, ноутбуки будут работать автоматически без дополнительных действий.

