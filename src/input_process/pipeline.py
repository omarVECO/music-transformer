import os
from .validator import validate_audio
from .transcriber import audio_to_midi

def run_input_pipeline(wav_path):
    
    # Verificación de seguridad
    if not os.path.exists(wav_path):
        print(f"ERROR: El archivo no existe en: {wav_path}")
        return None
    
    #1. Validamos
    valido, instrumento = validate_audio(wav_path)
    
    if not valido:
        print(f"ERROR: El instrumento {instrumento} no es soportado.")
        return None

    # 2. Transcribimos
    ruta_midi = audio_to_midi(wav_path)
    
    print(f"SUCCESS: Pipeline completado. Archivo generado: {ruta_midi}")
    return ruta_midi