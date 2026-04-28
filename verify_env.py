import sys
import torch
import numpy as np

print("🔍 Проверка окружения...\n")
status = "✅"

def check(name, module_str, expected_prefix=None):
    global status
    try:
        __import__(module_str)
        mod = sys.modules[module_str]
        ver = getattr(mod, '__version__', 'N/A')
        print(f"  {name}: {ver}")
        if expected_prefix and not str(ver).startswith(expected_prefix):
            print(f"    ⚠️  Ожидается {expected_prefix}.*, найдено {ver}")
    except Exception as e:
        print(f"  {name}: ❌ ОШИБКА - {e}")
        status = "❌"

print("📦 ML Framework:")
check("PyTorch", "torch", "2.4.0")
print(f"  CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")

print("\n📦 Data & Utils:")
check("NumPy", "numpy", "1.26.4")
check("Pandas", "pandas", "2.2.2")
check("TQDM", "tqdm", "4.66.4")
check("BioPython", "Bio", "1.84")
check("modelcif", "modelcif", "1.2")

print("\n📦 ESM & ESMFold:")
try:
    import esm
    print(f"  fair-esm: {esm.__version__}")
except Exception as e:
    print(f"  fair-esm: ❌ {e}")
    status = "❌"

try:
    import openfold
    print("  openfold: ✅ импортируется")
except Exception as e:
    print(f"  openfold: ❌ {e}")
    status = "❌"

print("\n🧪 Быстрый тест ESM-2 (8M):")
try:
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    print("  ✅ ESM-2 загрузился")
except Exception as e:
    print(f"  ❌ ESM-2: {e}")
    status = "❌"

print("\n" + "="*40)
if status == "✅":
    print("✅ ВСЕ ЗАВИСИМОСТИ УСТАНОВЛЕНЫ КОРРЕКТНО!")
else:
    print("❌ ОБНАРУЖЕНЫ ОШИБКИ. Проверьте логи выше.")