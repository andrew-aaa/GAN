#!/usr/bin/env python3
"""
install_deps.py — Единый скрипт установки зависимостей для Antidote GAN.
Запуск в Colab: !python install_deps.py
Запуск локально: python install_deps.py
"""
import sys
import subprocess
import importlib
import shutil
import os

# === КОНФИГУРАЦИЯ ВЕРСИЙ ===
VERSIONS = {
    "torch": "2.4.0",
    "torchvision": "0.19.0",
    "torchaudio": "2.4.0",
    "fair-esm": "2.0.0",
    "numpy": "1.26.4",      # Критично: <2.0 для openfold
    "pandas": "2.2.2",
    "biopython": "1.84",
    "tqdm": "4.66.4",
    "omegaconf": "2.3.0",   # Совместима с openfold
    "einops": "0.7.0",
    "pytorch-lightning": "2.3.0",
    "modelcif": "1.2",
    "dm-tree": "0.1.8",
}
# Коммит openfold, совместимый с fair-esm==2.0.0
OPENFOLD_COMMIT = "4b41059694619831a7db195b7e0988fc4ff3a307"
OPENFOLD_URL = f"git+https://github.com/aqlaboratory/openfold.git@{OPENFOLD_COMMIT}"
# Индекс для PyTorch с поддержкой CUDA 12.1
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu121"

def print_step(msg: str):
    print(f"\n[+] {msg}")

def print_ok(msg: str):
    print(f"    ✅ {msg}")

def print_err(msg: str):
    print(f"    ❌ {msg}", file=sys.stderr)

def run_cmd(cmd: list[str], desc: str) -> bool:
    """Запускает команду и возвращает True при успехе."""
    print_step(f"{desc}...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=600  # 10 минут на компиляцию
        )
        print_ok("Успешно")
        return True
    except subprocess.CalledProcessError as e:
        print_err(f"Ошибка: {e.stderr.strip()[:200]}")
        return False
    except subprocess.TimeoutExpired:
        print_err("Таймаут (слишком долго)")
        return False

def is_colab() -> bool:
    """Определяет, запущен ли код в Google Colab."""
    return "COLAB_GPU" in os.environ or os.path.exists("/content")

def install_system_deps():
    """Устанавливает системные сборщики (только если их нет)."""
    if shutil.which("ninja") and shutil.which("cmake"):
        print_ok("ninja и cmake уже установлены")
        return True
    
    if is_colab():
        return run_cmd(
            ["apt-get", "update", "-qq"],
            "Обновление apt"
        ) and run_cmd(
            ["apt-get", "install", "-qq", "-y", "ninja-build", "cmake", "build-essential"],
            "Установка ninja/cmake"
        )
    else:
        print_err("Установите ninja и cmake вручную: sudo apt install ninja-build cmake")
        return False

def install_pytorch():
    """Устанавливает PyTorch с правильным индексом."""
    # Проверяем, установлен ли уже совместимый PyTorch
    try:
        import torch
        if torch.__version__.startswith(VERSIONS["torch"]) and "cu121" in torch.version.cuda:
            print_ok(f"PyTorch {VERSIONS['torch']}+cu121 уже установлен")
            return True
    except ImportError:
        pass
    
    cmd = [
        sys.executable, "-m", "pip", "install", "-q",
        f"torch=={VERSIONS['torch']}+cu121",
        f"torchvision=={VERSIONS['torchvision']}+cu121",
        f"torchaudio=={VERSIONS['torchaudio']}+cu121",
        "--index-url", PYTORCH_INDEX,
    ]
    return run_cmd(cmd, f"Установка PyTorch {VERSIONS['torch']}+cu121")

def install_core_deps():
    """Устанавливает основные зависимости из PyPI."""
    packages = [
        f"numpy=={VERSIONS['numpy']}",
        f"pandas=={VERSIONS['pandas']}",
        f"biopython=={VERSIONS['biopython']}",
        f"tqdm=={VERSIONS['tqdm']}",
        f"omegaconf=={VERSIONS['omegaconf']}",
        f"einops=={VERSIONS['einops']}",
        f"pytorch-lightning=={VERSIONS['pytorch-lightning']}",
        f"modelcif=={VERSIONS['modelcif']}",
        f"dm-tree=={VERSIONS['dm-tree']}",
        "ninja", "cmake",  # На случай, если системные не сработали
    ]
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + packages
    return run_cmd(cmd, "Установка core-зависимостей")

def install_openfold():
    """Устанавливает openfold из совместимого коммита."""
    # Проверяем, установлен ли уже
    try:
        import openfold
        print_ok("openfold уже импортируется")
        return True
    except ImportError:
        pass
    
    # --no-deps чтобы не перетянуть несовместимые версии
    cmd = [
        sys.executable, "-m", "pip", "install", "-q",
        f"openfold @ {OPENFOLD_URL}",
        "--no-deps",
    ]
    # Устанавливаем переменные окружения для CUDA
    env = os.environ.copy()
    env["CUDA_HOME"] = "/usr/local/cuda"
    env["PATH"] = f"/usr/local/cuda/bin:{env.get('PATH', '')}"
    
    print_step(f"Установка openfold (коммит {OPENFOLD_COMMIT[:8]}...)")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=env,
            timeout=900  # 15 минут на компиляцию CUDA-ядер
        )
        print_ok("Успешно")
        return True
    except subprocess.CalledProcessError as e:
        print_err(f"Ошибка компиляции openfold")
        print_err(f"Попробуйте вручную: pip install '{OPENFOLD_URL}' --no-deps")
        return False

def install_fair_esm():
    """Устанавливает fair-esm (без [esmfold], так как openfold уже стоит)."""
    try:
        import esm
        if esm.__version__.startswith(VERSIONS["fair-esm"]):
            print_ok(f"fair-esm {VERSIONS['fair-esm']} уже установлен")
            return True
    except ImportError:
        pass
    
    cmd = [
        sys.executable, "-m", "pip", "install", "-q",
        f"fair-esm=={VERSIONS['fair-esm']}",
    ]
    return run_cmd(cmd, f"Установка fair-esm {VERSIONS['fair-esm']}")

def apply_patches():
    """Применяет патчи для совместимости с PyTorch 2.x."""
    print_step("Применение патчей совместимости...")
    
    # Патч для torch._six (удалён в PyTorch 2.0+)
    try:
        import sys
        from types import ModuleType
        if "torch._six" not in sys.modules:
            import torch
            # Создаем фейковый модуль
            torch_six = ModuleType("torch._six")
            torch_six.inf = float("inf")
            sys.modules["torch._six"] = torch_six
        print_ok("Патч torch._six применён")
    except Exception as e:
        print_err(f"Предупреждение: патч torch._six: {e}")
    
    # Патч для numpy 2.0+ совместимости
    try:
        import numpy as np
        if not hasattr(np, "BUFSIZE"):
            np.BUFSIZE = 8192
        print_ok("Патч numpy.BUFSIZE применён")
    except Exception as e:
        print_err(f"Предупреждение: патч numpy: {e}")
    
    return True

def verify_installation():
    """Проверяет, что все пакеты импортируются корректно."""
    print_step("Проверка установки...")
    errors = []
    
    checks = [
        ("torch", "torch", lambda: __import__("torch").__version__),
        ("fair-esm", "esm", lambda: __import__("esm").__version__),
        ("openfold", "openfold", lambda: __import__("openfold")),
        ("numpy", "numpy", lambda: __import__("numpy").__version__),
        ("pandas", "pandas", lambda: __import__("pandas").__version__),
    ]
    
    for name, module, getter in checks:
        try:
            ver = getter()
            print_ok(f"{name}: {ver}")
        except ImportError as e:
            print_err(f"{name}: не импортируется ({e})")
            errors.append(name)
        except Exception as e:
            print_err(f"{name}: ошибка проверки ({e})")
            errors.append(name)
    
    # Специальная проверка для ESMFold
    try:
        import esm
        # Проверяем, что trunk импортируется (это падает, если openfold битый)
        from esm.esmfold.v1.trunk import FoldingTrunk, FoldingTrunkConfig
        print_ok("ESMFold trunk: импортируется")
    except ImportError as e:
        print_err(f"ESMFold trunk: ошибка импорта ({e})")
        errors.append("esmfold-trunk")
    
    return len(errors) == 0

def main():
    print("=" * 60)
    print("🚀 Antidote GAN — Установка зависимостей")
    print("=" * 60)
    
    steps = [
        ("Системные зависимости", install_system_deps),
        ("PyTorch", install_pytorch),
        ("Core-зависимости", install_core_deps),
        ("OpenFold", install_openfold),
        ("fair-esm", install_fair_esm),
        ("Патчи совместимости", apply_patches),
    ]
    
    failed = False
    for name, func in steps:
        if not func():
            print_err(f"Шаг '{name}' не выполнен")
            failed = True
            break
    
    if not failed:
        if verify_installation():
            print("\n" + "=" * 60)
            print("✅ ВСЕ ЗАВИСИМОСТИ УСТАНОВЛЕНЫ УСПЕШНО!")
            print("=" * 60)
            print("\n📋 Следующие шаги:")
            print("1. Перезапустите Runtime: Runtime → Restart session")
            print("2. Запустите предобработку: !python precompute_toxin_embeddings.py")
            print("3. Запустите обучение: !python train.py")
            print("4. Для фолдинга: !python fold_candidates.py")
            return 0
        else:
            print_err("\nУстановка завершена с ошибками. Проверьте логи выше.")
            return 1
    else:
        print_err("\nУстановка прервана из-за ошибки.")
        return 1

if __name__ == "__main__":
    sys.exit(main())