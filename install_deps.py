#!/usr/bin/env python3
"""
install_deps.py — Установка зависимостей для Antidote GAN (openfold ставится вручную).
Запуск в Colab: !python install_deps.py
"""
import sys
import subprocess
import shutil
import os

# === КОНФИГУРАЦИЯ ВЕРСИЙ ===
VERSIONS = {
    "torch": "2.4.0",
    "torchvision": "0.19.0",
    "torchaudio": "2.4.0",
    "fair-esm": "2.0.0",
    "numpy": "1.26.4",
    "pandas": "2.2.2",
    "biopython": "1.84",
    "tqdm": "4.66.4",
    "omegaconf": "2.3.0",
    "einops": "0.7.0",
    "pytorch-lightning": "2.3.0",
    "modelcif": "1.2",
    "dm-tree": "0.1.8",
}
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu121"

def print_step(msg: str): print(f"\n[+] {msg}")
def print_ok(msg: str): print(f"    ✅ {msg}")
def print_warn(msg: str): print(f"    ⚠️  {msg}")
def print_err(msg: str): print(f"    ❌ {msg}", file=sys.stderr)

def run_cmd(cmd: list[str], desc: str) -> bool:
    print_step(f"{desc}...")
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        print_ok("Успешно")
        return True
    except subprocess.CalledProcessError as e:
        print_err(f"Ошибка: {e.stderr.strip()[:200]}")
        return False
    except subprocess.TimeoutExpired:
        print_err("Таймаут")
        return False

def install_build_tools():
    if shutil.which("ninja") and shutil.which("cmake"):
        print_ok("ninja и cmake уже доступны")
        return True
    return run_cmd([sys.executable, "-m", "pip", "install", "-q", "ninja", "cmake"], "Установка ninja/cmake через pip")

def install_pytorch():
    try:
        import torch
        if torch.__version__.startswith(VERSIONS["torch"]) and "cu121" in torch.version.cuda:
            print_ok(f"PyTorch {VERSIONS['torch']}+cu121 уже установлен")
            return True
    except ImportError: pass
    cmd = [sys.executable, "-m", "pip", "install", "-q",
           f"torch=={VERSIONS['torch']}+cu121", f"torchvision=={VERSIONS['torchvision']}+cu121",
           f"torchaudio=={VERSIONS['torchaudio']}+cu121", "--index-url", PYTORCH_INDEX]
    return run_cmd(cmd, f"Установка PyTorch {VERSIONS['torch']}+cu121")

def install_core_deps():
    packages = [f"numpy=={VERSIONS['numpy']}", f"pandas=={VERSIONS['pandas']}",
                f"biopython=={VERSIONS['biopython']}", f"tqdm=={VERSIONS['tqdm']}",
                f"omegaconf=={VERSIONS['omegaconf']}", f"einops=={VERSIONS['einops']}",
                f"pytorch-lightning=={VERSIONS['pytorch-lightning']}",
                f"modelcif=={VERSIONS['modelcif']}", f"dm-tree=={VERSIONS['dm-tree']}"]
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + packages
    return run_cmd(cmd, "Установка core-зависимостей")

def install_fair_esm():
    try:
        import esm
        if esm.__version__.startswith(VERSIONS["fair-esm"]):
            print_ok(f"fair-esm {VERSIONS['fair-esm']} уже установлен")
            return True
    except ImportError: pass
    cmd = [sys.executable, "-m", "pip", "install", "-q", f"fair-esm=={VERSIONS['fair-esm']}"]
    return run_cmd(cmd, f"Установка fair-esm {VERSIONS['fair-esm']}")

def apply_patches():
    print_step("Применение патчей совместимости...")
    try:
        from types import ModuleType
        if "torch._six" not in sys.modules:
            torch_six = ModuleType("torch._six")
            torch_six.inf = float("inf")
            sys.modules["torch._six"] = torch_six
        print_ok("Патч torch._six применён")
    except Exception as e: print_warn(f"Патч torch._six: {e}")

    try:
        import numpy as np
        if not hasattr(np, "BUFSIZE"): np.BUFSIZE = 8192
        print_ok("Патч numpy.BUFSIZE применён")
    except Exception as e: print_warn(f"Патч numpy: {e}")
    return True

def verify_installation():
    print_step("Проверка установки...")
    errors = []
    checks = [
        ("torch", "torch", lambda: __import__("torch").__version__),
        ("fair-esm", "esm", lambda: __import__("esm").__version__),
        ("numpy", "numpy", lambda: __import__("numpy").__version__),
        ("pandas", "pandas", lambda: __import__("pandas").__version__),
    ]
    for name, module, getter in checks:
        try:
            print_ok(f"{name}: {getter()}")
        except Exception as e:
            print_err(f"{name}: не импортируется ({e})")
            errors.append(name)

    print_warn("openfold: НЕ установлен (см. инструкцию ниже)")
    return len(errors) == 0

def main():
    print("=" * 60)
    print("🚀 Antidote GAN — Установка зависимостей (без openfold)")
    print("=" * 60)

    steps = [
        ("Build tools", install_build_tools),
        ("PyTorch", install_pytorch),
        ("Core-зависимости", install_core_deps),
        ("fair-esm", install_fair_esm),
        ("Патчи совместимости", apply_patches),
    ]

    failed = False
    for name, func in steps:
        if not func():
            print_err(f"Шаг '{name}' не выполнен")
            failed = True
            break

    if not failed and verify_installation():
        print("\n" + "=" * 60)
        print("✅ ОСНОВНЫЕ ЗАВИСИМОСТИ УСТАНОВЛЕНЫ УСПЕШНО!")
        print("=" * 60)
        print("\n📋 ДАЛЕЕ (ВЫПОЛНИТЕ ПОСЛЕ РЕСТАРТА RUNTIME):")
        print("1. Runtime → Restart session")
        print("2. Установите openfold вручную (pre-build wheel):")
        print("   !pip install --no-deps openfold -f https://storage.googleapis.com/jax-releases/cuda12/cuda12.html -q")
        print("3. Проверьте: !python verify_env.py")
        print("4. Запустите пайплайн: !python precompute_toxin_embeddings.py && !python train.py")
        return 0
    else:
        print_err("\nУстановка завершена с ошибками. Проверьте логи выше.")
        return 1

if __name__ == "__main__":
    sys.exit(main())