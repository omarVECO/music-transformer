import os
from basic_pitch.inference import predict

def audio_to_midi(audio_path):
    """
    Convierte el audio a MIDI y lo guarda en la carpeta de datos.
    """
    print(f"DEBUG: Transcribiendo {audio_path} a MIDI...")
    
    # Inferencia de Basic Pitch
    model_output, midi_data, note_events = predict(audio_path)

    # Definir ruta de salida profesional
    output_dir = os.path.join("data", "processed", "input_midis")
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(audio_path).rsplit('.', 1)[0]
    midi_path = os.path.join(output_dir, f"{base_name}.mid")
    
    midi_data.write(midi_path)
    return midi_path