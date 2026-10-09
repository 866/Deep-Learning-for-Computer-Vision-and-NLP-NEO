# Deep Learning for Computer Vision and NLP

Цей проєкт використовує Python та менеджер пакетів `uv` для швидкого налаштування середовища, роботи з залежностями та запуску ноутбуків.
Зверніть увагу, що деякі ноутбуки були згенеровані за допомогою агентів Claude Code (Opus 5.5) та Pi (Qwen3.8-27B/125B), тому, на жаль, можливі неточності, особливо в україномовних версіях ноутбуків.

## 1. Встановлення `uv` для роботи з ноутбуками

### Linux / macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Після установки перезапустіть термінал або виконайте:

```bash
source $HOME/.cargo/env
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Або через `winget`:

```powershell
winget install --id=astral-sh.uv -e
```

---

## 2. Створення віртуального середовища та установка пакетів

У корені проєкту виконайте:

```bash
uv venv
```

### Активація середовища

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

### Встановлення всіх залежностей проєкту

```bash
uv sync
```

Ця команда зчитує `pyproject.toml` і встановлює всі пакети, які вже описані в проєкті.

### Додавання нового пакета через `uv`

```bash
uv add pandas matplotlib numpy scikit-learn
```

Якщо потрібно встановити пакет тільки в поточне середовище:

```bash
uv pip install jupyterlab
```

### Пакети, які вже використані в проєкті

```toml
ipykernel
matplotlib
numpy
pandas
scikit-learn
torch
```

---

## 3. Як запускати ноутбуки

### Варіант 1: Visual Studio Code

1. Відкрийте папку проєкту в VS Code.
2. Встановіть розширення Python.
3. Виберіть інтерпретатор: `Python: Select Interpreter`.
4. Виберіть `.venv` або середовище, створене через `uv`.
5. Відкрийте файл `.ipynb` у ноутбуці.

Команда для запуску VS Code:

```bash
code .
```

Переваги: зручно для відладки, роботи з файлами проєкту та запуску клітинок.

### Варіант 2: Jupyter Lab

```bash
uv run --with jupyterlab jupyter lab
```

Після запуску відкриється веб-інтерфейс Jupyter Lab у браузері.

### Варіант 3: Jupyter Notebook

```bash
uv run --with notebook jupyter notebook
```

Це класичний інтерфейс Jupyter, який часто використовується для навчальних ноутбуків.

### Варіант 4: Запуск через `uv run`

```bash
uv run python main.py
```

Також можна запускати ноутбуки або скрипти через `uv`, якщо потрібен ізольований середовищний контекст.

---

## 4. Коротка шпаргалка

| Дія | Команда | Примітка |
| --- | --- | --- |
| Встановити `uv` | `curl -LsSf https://astral.sh/uv/install.sh | sh` | Для Linux/macOS |
| Створити віртуальне середовище | `uv venv` | Створює `.venv` |
| Активувати середовище | `source .venv/bin/activate` | Linux/macOS |
| Встановити залежності | `uv sync` | Зчитує `pyproject.toml` |
| Додати пакет | `uv add <package>` | Додає до `pyproject.toml` |
| Запустити Jupyter Lab | `uv run --with jupyterlab jupyter lab` | Веб-інтерфейс |
| Запустити VS Code | `code .` | Найзручніший для проєкту |

---

## 5. Рекомендація

Для цього проєкту найзручніше використовувати:

- VS Code + Python extension для розробки та відладки;
- Jupyter Lab для роботи з ноутбуками та візуалізацією;
- `uv` для керування середовищем і пакетами.

Це дозволяє зберігати стабільне середовище, легко оновлювати залежності та швидко запускати експерименти.
