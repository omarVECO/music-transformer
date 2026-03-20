#import torch

def validate_audio(audio_path):
    """
    Usa la arquitectura CNN14 para clasificar el instrumento.
    """
    # Lógica de TT1 (Aquí cargarás tus pesos más adelante)
    # Por ahora, simulamos que detecta una guitarra
    print(f"DEBUG: Validando instrumento en {audio_path}")
    
    es_valido = True
    instrumento = "guitar" 
    
    return es_valido, instrumento