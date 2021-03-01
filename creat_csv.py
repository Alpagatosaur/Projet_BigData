# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 15:23:00 2021

@author: tanch
"""

#https://medium.com/@klintcho/creating-an-open-speech-recognition-dataset-for-almost-any-language-c532fb2bc0cf



# import required libraries 
#from pydub import AudioSegment 
#from pydub.playback import play 

# Import an audio file 
# Format parameter only 
# for readability 
#wav_file = AudioSegment.from_file(file = 'chouette-hulotte-chant-et-cris1.wav', format = "wav") 

#https://medium.com/@alexandro.ramr777/audio-files-to-dataset-by-feature-extraction-with-librosa-d87adafe5b64

from glob import glob

audio_files = glob('Sons/'+'*.wav')
print(f'Number of normal audios : {len(audio_files)}')