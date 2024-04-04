from whisper_online import FasterWhisperASR, OnlineASRProcessor, load_audio
from faster_whisper import WhisperModel
import gradio as gr


src_lan = "en"
tgt_lan = "en"

# Load the Whisper model (replace "medium" with a suitable size)
asr = FasterWhisperASR(src_lan, model_dir="large-v2")
asr.use_vad()
min_chunk=0.01
final = []
# Create the processor object
online = OnlineASRProcessor(asr)


def live(input_audio):
    #seg=" "
    audio_data = load_audio(input_audio)
    online.insert_audio_chunk(audio_data)
    asr.transcribe(input_audio)
    transcription = online.process_iter()
    transcription=online.finish()
    print(transcription)
    #for segment in seg:
        
        #seg +="%s" %(segment)
    final.append(transcription[-1])
    output_string = ' '.join(final)
    return  output_string
tran = gr.Interface(
    fn=live,
    inputs=gr.Audio(sources="microphone", type="filepath", streaming=True),
    outputs="text",
    title="Live Transcription",
    live=True,
)

tran.launch()