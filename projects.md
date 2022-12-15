# Projects
## 1. Stanford NLP Lecture Transcription using OpenAI's Whisper
  Whisper is an automatic speech recognition (ASR) model trained on hours of multilingual and multitask supervised data. It is implemented as an encoder-decoder transformer architecture where audio are splitted into 30 seconds of chunks, converted into a log-Mel spectrogram, and then passed into an encoder. The decoder is trained to predict the corresponding text caption, intermixed with special tokens that direct the single model to perform tasks such as language identification, phrase-level timestamps, multilingual speech transcription, and to-English speech translation. For more info about whisper, read [here](https://openai.com/blog/whisper/).
  
   I used whisper model to transcribe Stanford NLP lectures into corresponding text captions. [Here](http://3.14.28.154/) is the result of the transcribed lectures. This web app is build using Flask and deployed on AWS EC2 instance. You can find transcribed audio file in the form of text [here](https://github.com/nepalprabin/whisper-webapp/blob/main/Stanford_NLP_lecture_transcripts.zip).
   
