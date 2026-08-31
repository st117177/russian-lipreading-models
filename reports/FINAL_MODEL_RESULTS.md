# Current Model Results

## Задача и данные

Задача — классификация изолированных видеоклипов губ по 10 русским словам.
Dataset v2 содержит 1355 клипов и 8 спикеров. Исходный строгий speaker-based
split имел следующий вид:

| Часть | Клипы | Спикеры |
|---|---:|---|
| Train | 1100 | spk03, spk04, spk05, spk11, spk12 |
| Validation | 105 | spk06 |
| Test | 150 | spk07, spk08 |

После просмотра исходного test был улучшен crop pipeline. Landmark v3 сохраняет
тех же train/validation-спикеров, исключает 15 train-клипов, не прошедших порог
landmark coverage, и намеренно не включает просмотренный test:

| Часть | Клипы | Спикеры |
|---|---:|---|
| Train | 1085 | spk03, spk04, spk05, spk11, spk12 |
| Validation | 105 | spk06 |

Новый финальный test должен состоять из ранее не использованных спикеров.

## Финальная модель

```text
source word clip
-> MediaPipe lip landmarks, eye-line alignment, temporal smoothing
-> 24 grayscale mouth frames
-> frozen LRW-pretrained 3D Conv + ResNet18
-> frame sequence [24, 512]
-> trainable bidirectional GRU
-> linear classifier for 10 words
```

Visual frontend взят из официальной LRW-pretrained модели проекта
[Lipreading using Temporal Convolutional Networks](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks).
Обучаемая часть содержит 495626 параметров. Для train применяется балансировка
по непустым группам `(speaker, word)`. Главная метрика выбора — validation
macro-F1.

## Controlled-сравнения на validation

Все значения — среднее и стандартное отклонение по seed 42, 123 и 2026.

| Visual frontend + BiGRU | Accuracy | Macro-F1 | Balanced accuracy | Классы |
|---|---:|---:|---:|---:|
| Frozen ImageNet ResNet18 | 0.1587 +/- 0.0440 | 0.1416 +/- 0.0458 | 0.1715 +/- 0.0104 | 9.33/10 |
| Frozen LRW 3D Conv + ResNet18, old crop | 0.2857 +/- 0.0530 | 0.2736 +/- 0.0290 | 0.3157 +/- 0.0039 | 9.67/10 |
| Frozen LRW 3D Conv + ResNet18, landmark v3 | **0.5460 +/- 0.0110** | **0.5359 +/- 0.0139** | **0.5787 +/- 0.0108** | **10/10** |

LRW frontend повысил mean macro-F1 на `0.1320` и выиграл у ImageNet frontend
на всех трех seed. Это главный положительный модельный результат работы.

При фиксированных модели и протоколе landmark v3 дополнительно повысил mean
macro-F1 на `0.2622` относительно старого heuristic crop и выиграл на всех трех
paired seed. Это главный положительный preprocessing-результат работы.

## Историческая test-оценка старого crop

Три checkpoint старого crop были заранее выбраны по validation. На исходном
test выполнялся только inference, без обучения и изменения модели.

| Seed | Accuracy | Macro-F1 | Balanced accuracy | Классы |
|---:|---:|---:|---:|---:|
| 42 | 0.1600 | 0.1331 | 0.1612 | 9/10 |
| 123 | 0.2133 | 0.1742 | 0.2087 | 10/10 |
| 2026 | 0.1733 | 0.1515 | 0.1713 | 10/10 |
| **Mean +/- std** | **0.1822 +/- 0.0278** | **0.1530 +/- 0.0206** | **0.1804 +/- 0.0250** | **9.67 +/- 0.58** |

Test majority baseline всегда предсказывает слово `чтобы`, которое занимает
28 из 150 клипов. Его accuracy равна `0.1867`, balanced accuracy — `0.1`, а
macro-F1 — около `0.0315`. Поэтому старая модель с heuristic crop близка к
majority baseline по обычной accuracy, но заметно превосходит его по
class-balanced метрикам и предсказывает почти все классы.

Эти числа остаются корректным результатом старой системы, но `spk07` и `spk08`
уже просмотрены и не могут быть новым независимым test для landmark v3.

## Per-class results на историческом test

Средние значения по трем checkpoint:

| Слово | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| будет | 18 | 0.238 | 0.370 | 0.287 |
| время | 7 | 0.030 | 0.048 | 0.037 |
| есть | 25 | 0.265 | 0.253 | 0.255 |
| значит | 6 | 0.157 | 0.278 | 0.194 |
| когда | 20 | 0.278 | 0.167 | 0.202 |
| может | 10 | 0.094 | 0.233 | 0.134 |
| потом | 3 | 0.000 | 0.000 | 0.000 |
| просто | 25 | 0.288 | 0.080 | 0.120 |
| человек | 8 | 0.139 | 0.292 | 0.188 |
| чтобы | 28 | 0.184 | 0.083 | 0.113 |

## Вывод и ограничения

Lip-reading pretraining существенно улучшил validation-качество относительно
ImageNet pretraining и простой замены temporal backend. Затем landmark alignment
повысил validation macro-F1 с `0.2736` до `0.5359`: стабильная локализация рта и
совместимость preprocessing с LRW оказались важным ограничением старой системы.

Редкие классы исторического test дают нестабильные оценки: `потом` представлен
тремя клипами, `значит` — шестью, `время` — семью. Текущая конфигурация теперь
заморожена. Оставшийся экспериментальный этап — один inference-запуск на новом,
более сбалансированном test минимум из двух ранее не использованных спикеров.
