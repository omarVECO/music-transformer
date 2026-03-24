import os
import sys

# Encontrar la raíz del proyecto (2 niveles arriba de src/input_process/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))
os.chdir(BASE_DIR)

from input_process.pipeline import run_input_pipeline

# CONFIGURACIÓN
archivo_en_temp = "CScale.wav"
ruta_audio = os.path.join("data", "input_temp", archivo_en_temp)

print("--- [TEST LEDESMAcoito] ---")

resultado = run_input_pipeline(ruta_audio)

if resultado:
    print(f" ÉXITO TOTAL: MIDI listo en {resultado}")
else:
    print(" ERROR: Revisa que el audio esté en data/input_temp/")