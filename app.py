import gradio as gr
import numpy as np
from bark import SAMPLE_RATE, generate_audio, preload_models
from IPython.display import Audio
from bark.generation import (
    generate_text_semantic,
)
from bark.api import semantic_to_waveform
import nltk  # we'll use this to split into sentences

# preload_models()

history = []


def text_to_speech(text, index):
    if index is None or index.strip() == "":
        today_str = str(np.datetime64("today", "D"))
        audio_file = f"output_{today_str}_{len(history)}.wav"
        SPEAKER = "v2/en_speaker_6"
        GEN_TEMP = 0.6

        pieces = []
        silence = np.zeros(int(0.25 * SAMPLE_RATE))
        sentences = nltk.sent_tokenize(text)
        for line in sentences:
            semantic_tokens = generate_text_semantic(
                line,
                history_prompt=SPEAKER,
                temp=GEN_TEMP,
                min_eos_p=0.05,  # this controls how likely the generation is to end
            )
            audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER)
            pieces += [audio_array, silence.copy()]

        genAudio = Audio(np.concatenate(pieces), rate=SAMPLE_RATE)
        with open(audio_file, "wb") as file:
            file.write(genAudio.data)
        history.append(audio_file)
    else:
        audio_file = history[int(index) - 1]
    return audio_file


# Create the Gradio interface
gr_interface = gr.Interface(
    fn=text_to_speech,  # The function to execute
    inputs=[
        gr.Textbox(label="Input text"),  # Text input
        gr.Textbox(label="Input history index"),
    ],  # The input type (text)
    outputs="audio"  # The output type (audio)
)

# Launch the Gradio app
gr_interface.launch()
