"""CLI-валидация CSV-файла против схемы Pandera.

Запуск: python -m src.validation.validate <path-to-csv>
Код выхода 0 — данные валидны, 1 — найдены аномалии.
"""

import sys

import pandas as pd

from src.validation.schema import validate_raw


def main() -> int:
    if len(sys.argv) < 2:
        print("Использование: python -m src.validation.validate <csv>")
        return 2
    path = sys.argv[1]
    df = pd.read_csv(path)
    try:
        validate_raw(df)
    except Exception as exc:  # pandera.errors.SchemaErrors
        print(f"ВАЛИДАЦИЯ НЕ ПРОЙДЕНА: {path}")
        print(str(exc)[:3000])
        return 1
    print(f"Валидация пройдена: {path} ({len(df)} строк)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
