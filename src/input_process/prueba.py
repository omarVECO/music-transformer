import os
import sys

# 1. TRUCO DE INGENIERÍA: Encontrar la raíz del proyecto (2 niveles arriba)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))
os.chdir(BASE_DIR)

from input_process.pipeline import run_input_pipeline

# CONFIGURACIÓN
archivo_en_temp = "CScale.wav"
ruta_con_carpetas = os.path.join("data", "input_temp", archivo_en_temp)
resultado = run_input_pipeline(ruta_con_carpetas)

print("--- [TEST LEDESMAsexo] ---")

# Ejecutamos el pipeline (que ya sabe buscar en data/input_temp)
resultado = run_input_pipeline(archivo_en_temp)

if resultado:
    print(f"✅ ÉXITO TOTAL: MIDI generado en {resultado}")
else:
    print("❌ ERROR: Revisa que el audio esté en data/input_temp/")