import os
from basic_pitch.inference import predict

def audio_to_midi(audio_path):
    """
    Convierte el audio a MIDI usando Basic Pitch.
    Guarda el .mid en data/processed/input_midis/ y retorna su ruta.
    El .mid es lo que se consume con pretty_midi + midi_tokenizer.
    """
    print(f"DEBUG: Transcribiendo {audio_path} a MIDI...")

    model_output, midi_data, note_events = predict(audio_path)

    output_dir = os.path.join("data", "processed", "input_midis")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(audio_path).rsplit('.', 1)[0]
    midi_path = os.path.join(output_dir, f"{base_name}.mid")

    midi_data.write(midi_path)

    print(f"DEBUG: MIDI guardado en {midi_path}")
    return midi_path