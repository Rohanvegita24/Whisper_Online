import gradio as gr
from whisper_online import OnlineASRProcessor, FasterWhisperASR,WhisperTimestampedASR
from faster_whisper import WhisperModel
import numpy as np

# Source and target languages (same for ASR)
src_lan = "en"
tgt_lan = "en"

# Load the Whisper model (replace "medium" with a suitable size)
asr = FasterWhisperASR(src_lan, "large-v3")

# Create the processor object
model = OnlineASRProcessor(asr)

# Function for continuous live transcription
def transcription(audio_data):
    transcription_text = ""
    asr.set_translate_task()
    asr.use_vad()
    for chunk in audio_data:
        # Convert audio chunk to float tensor
        audio_tensor = np.array(chunk).astype(float)
        # Insert audio chunk and process
        model.insert_audio_chunk(audio_tensor)
        transcription_text += model.process_iter()

    # Finalize transcription
    transcription_text += model.finish()
    return transcription_text

# Gradio interface for live audio input
iface = gr.Interface(
    fn=transcription,
    inputs=gr.Audio(sources="upload",type="numpy"),
    outputs="text",
    title="Live Transcription",
    
)

# Launch the Gradio interface
iface.launch()
