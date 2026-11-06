"""Сборка docker-образа и запуск API одной командой.

Кроссплатформенно: работает на Windows, Linux и macOS,
нужен только установленный Docker.

Запуск из корня проекта: python scripts/docker_run.py
"""

import subprocess
import sys
import time
import urllib.request

IMAGE = "pd-scoring-api"
CONTAINER = "pd-api"
PORT = 8000


def main():
    print(f"==> Сборка образа {IMAGE}")
    subprocess.run(["docker", "build", "-t", IMAGE, "."], check=True)

    print("==> Удаляю старый контейнер, если был")
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)

    print(f"==> Запуск контейнера на порту {PORT}")
    subprocess.run(
        ["docker", "run", "-d", "--name", CONTAINER, "-p", f"{PORT}:8000", IMAGE],
        check=True,
    )

    print("==> Жду готовности сервиса...")
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2) as r:
                print("==> API поднялся:", r.read().decode())
                print(f"==> Swagger UI: http://localhost:{PORT}/docs")
                return
        except Exception:
            time.sleep(1)

    sys.exit(f"Сервис не поднялся за 30 секунд, смотрите: docker logs {CONTAINER}")


if __name__ == "__main__":
    main()
