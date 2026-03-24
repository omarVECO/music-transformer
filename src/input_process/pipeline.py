import os
from .validator import validate_audio
from .transcriber import audio_to_midi

def run_input_pipeline(wav_path, genre="FUNK", mood="HAPPY", instrument="BASS"):
    """
    Orquesta el flujo completo para un solo audio.
    
    Parámetros
    ----------
    wav_path   : ruta al archivo de audio
    genre      : ROCK | POP | FUNK | JAZZ | LATIN | CLASSICAL | ELECTRONIC
    mood       : HAPPY | SAD | DARK | RELAXED | TENSE
    instrument : BASS | PIANO | GUITAR  (instrumento de acompañamiento a generar)

    Retorna
    -------
    dict con claves:
        "midi_path"  → ruta del .mid generado  (None si falló)
        "genre"      → género recibido
        "mood"       → mood recibido
        "instrument" → instrumento recibido
        "error"      → mensaje de error (None si fue exitoso)
    """
    resultado = {
        "midi_path":  None,
        "genre":      genre,
        "mood":       mood,
        "instrument": instrument,
        "error":      None,
    }

    # Verificación de existencia
    if not os.path.exists(wav_path):
        resultado["error"] = f"Archivo no encontrado: {wav_path}"
        return resultado

    # 1. Validamos instrumento de entrada (CNN14 de Omar)
    instrumento_detectado, es_valido = validate_audio(wav_path)
    if not es_valido:
        resultado["error"] = (
            f"Instrumento detectado '{instrumento_detectado}' no soportado. "
            f"Solo se aceptan: piano, guitar, bass."
        )
        return resultado

    # 2. Transcribimos audio → MIDI (Basic Pitch)
    midi_path = audio_to_midi(wav_path)
    resultado["midi_path"] = midi_path

    # A partir de aquí elOMisexo toma el .mid con:
    #   pm = pretty_midi.PrettyMIDI(midi_path)
    #   melody_inst, _ = select_tracks(pm)
    #   enc_tokens = notes_to_token_sequence(
    #       melody_inst, pm, tempo_bpm, key_token,
    #       genre, mood, energy, inst_to_token(melody_inst), is_encoder=True
    #   )

    print(f"SUCCESS: MIDI listo → {midi_path}  [{genre} / {mood} / {instrument}]")
    return resultado