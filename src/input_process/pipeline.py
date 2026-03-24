import os
from .validator import validate_audio
from .transcriber import audio_to_midi

def run_input_pipeline(wav_path):
    
    # Verificación de seguridad
    if not os.path.exists(wav_path):
        print(f"ERROR: El archivo no existe en: {wav_path}")
        return None
    
    # 1. Validamos
    instrumento, es_valido = validate_audio(wav_path)
    
    if not es_valido:
        print(f"ERROR: El instrumento '{instrumento}' no es soportado.")
        print(f"       Solo se aceptan: piano, guitar, bass.")
        return None

    # 2. Transcribimos
    midi_path = audio_to_midi(wav_path)
    
    print(f"SUCCESS: Pipeline completado. MIDI listo para tokenizar: {midi_path}")
    return midi_path