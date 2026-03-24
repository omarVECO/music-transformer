#import torch

INSTRUMENTOS_VALIDOS = {"piano", "guitar", "bass"}

def validate_audio(audio_path):
    """
    Usa el fine-tuning de CNN14 o CNN14 (checamos) para clasificar el instrumento del audio.
    Solo piano, guitarra y bajo son válidos para generar acompañamiento.

    Retorna (instrumento_detectado, es_valido).
    """
    print(f"DEBUG: Validando instrumento en {audio_path}")
    
    # --- MOCK: reemplazar esta línea con la inferencia real de CNN14 ---
    instrumento = "guitar"
    # ------------------------------------------------------------------

    es_valido = instrumento in INSTRUMENTOS_VALIDOS

    if es_valido:
        print(f"DEBUG: Instrumento detectado: '{instrumento}' válido")
    else:
        print(f"DEBUG: Instrumento detectado: '{instrumento}' no soportado")

    return instrumento, es_valido