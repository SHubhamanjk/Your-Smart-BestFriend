from gtts import gTTS
import speech_recognition as sr
import re
import langchain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.schema import AIMessage, HumanMessage
import pickle
import numpy as np
import nltk
import re
from nltk.stem import PorterStemmer
import tempfile
import uuid
from pathlib import Path
import time
import pyttsx3 
from googletrans import Translator

nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

with open('logistic_regression.pkl', 'rb') as lr_file:
    lr = pickle.load(lr_file)

with open('tfidfvectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('label_encoder.pkl', 'rb') as label_encoder_file:
    lb = pickle.load(label_encoder_file)



def clean_data(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ",text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

def predict_emotion(input_text):
    cleaned_text = clean_data(input_text)
    input_vectorizer = vectorizer.transform([cleaned_text])
    
    predicted_label = lr.predict(input_vectorizer)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    label = np.max(lr.predict(input_vectorizer))
    
    return predicted_emotion


groq_api_key = "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"

memory = ConversationBufferMemory(
    memory_key="chat_history",
    human_prefix="User",
    ai_prefix="AI",
    return_messages=False  
)

llm = ChatGroq(api_key=groq_api_key, model_name="gemma2-9b-it")

mental_health_prompt = PromptTemplate(
    input_variables=["chat_history", "user_input", "user_mood"],
    template="""
     You are an advanced AI assistant and a highly skilled mental health professional.
Your expertise includes cognitive behavioral therapy (CBT), mindfulness techniques, emotional intelligence, and personalized mood-based guidance. 
You provide empathetic, structured, and insightful responses, tailored to the user's emotions and situation also keep the context in your mind.  
    
As a mental health counselor, provide empathetic support following these rules:

1. Respond in one friendly paragraph (100-200 words)
2. No emojis, markdown, or formatting
3. Adjust tone based on mood and input text: calm for anger, gentle for sadness and try to make it fun appropriately.
4. Acknowledge feelings first, then offer practical steps (CBT/mindfulness)
5. Use simple language with Indian context examples
6. Ask follow-up questions to engage user
7. If the user is asking in Hindi, reply in Hindi; if Hinglish, then Hinglish, and maintain this till the end , Make responses relatable to Indian users, even starting with a Hinglish sentence if the user asks in English.Speak Hinglish like urban guy.
8. Use only your knowledge - say "I don't know" if unsure
9. Speak like a human expert, not an AI bot. Responses should feel natural, warm, and engaging ,Use creativity & engagement – You can use motivational quotes, humor, sarcasm, real-life examples, and even jokes where appropriate. 
10. Provide actionable steps - instead of just sympathizing, always offer practical advice, exercises,or solutions to help the user.



Conversation History:
{chat_history}

Current Message: "{user_input}"
Detected Mood: "{user_mood}"

Keep tone friendly and make it like a conversation between two best friends.""",
)

llm_chain = LLMChain(
    llm=llm,
    prompt=mental_health_prompt,
    verbose=False
)

def clean_response(response_text: str):
    """Remove any AI prefixes from response"""
    prefixes = ["AI:", "Response:", "Assistant:", "**"]
    for prefix in prefixes:
        if response_text.startswith(prefix):
            response_text = response_text[len(prefix):].strip()
    return response_text.strip('"').strip()

def get_ai_response(user_input: str, user_mood: str):
    history_data = memory.load_memory_variables({})
    chat_history = history_data.get("chat_history", "")
    
    response = llm_chain.invoke({
        "chat_history": chat_history,
        "user_input": user_input,
        "user_mood": user_mood
    })
    
    cleaned_response = clean_response(response["text"])
    
    memory.save_context(
        {"input": user_input},
        {"output": cleaned_response}
    )
    
    return cleaned_response


def speech_to_text(source_type='microphone', file_path=None, silence_duration=3):
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 150  
    recognizer.dynamic_energy_threshold = True

    if source_type == 'microphone':
        with sr.Microphone() as source:
            print("Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source, duration=1)

            audio_data = []
            last_speech_time = time.time()

            while True:
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    audio_data.append(audio)
                    last_speech_time = time.time()  

                except sr.WaitTimeoutError:
                    pass 

                if time.time() - last_speech_time > silence_duration:
                    print("\nSilence detected. Processing final transcription...\n")
                    break

            combined_audio = sr.AudioData(
                b"".join(a.frame_data for a in audio_data),
                source.SAMPLE_RATE,
                source.SAMPLE_WIDTH
            )

            try:
                final_text = recognizer.recognize_google(combined_audio)
                print("Final Transcription:", final_text)
                return final_text
            except sr.UnknownValueError:
                print("Could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

    elif source_type == 'file' and file_path:
        with sr.AudioFile(file_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio)
                print("Text:", text)
                return text
            except sr.UnknownValueError:
                print("Could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

    else:
        print("Invalid source type or file path not provided.")



def text_to_speech(text, lang='en', gender='male', speed=180):
    try:
        engine = pyttsx3.init()
        
        voices = engine.getProperty('voices')
        if gender.lower() == 'female':
            engine.setProperty('voice', voices[1].id)  
        else:
            engine.setProperty('voice', voices[0].id)  

        engine.setProperty('rate', speed) 

        temp_dir = tempfile.gettempdir()
        filename = f"speech_{uuid.uuid4().hex}.mp3"
        file_path = Path(temp_dir) / filename

        engine.save_to_file(text, str(file_path))
        engine.runAndWait() 

        return str(file_path)
    
    except Exception as e:
        print(f"Error in text_to_speech: {str(e)}")
        return None






def talk():
    """Voice-based conversation system with emotional intelligence"""
    
    # Initialize TTS engine first to catch errors early
    try:
        engine = pyttsx3.init()
    except Exception as e:
        print(f"Failed to initialize TTS engine: {str(e)}")
        return

    # Greeting templates
    GREETINGS = [
        "Hey buddy! Long time no chat... How's life treating you?",
        "Yo! What's up? Ready for our daily dose of real talk?",
        "Namaste dost! Kaise ho aaj? Let's vibe together...",
        "Ah, my favorite human! What's cooking in that beautiful mind today?",
        "Hey partner in crime! Ready to unpack some thoughts?"
    ]
    
    # Farewell templates
    GOODBYES = [
        "Catch you later, partner! Remember: You're stronger than you think.",
        "Alright, time to bounce... But I'm always here if you need!",
        "Chalo, phir milenge! Take care of yourself, yaar.",
        "Signing off for now... Don't forget to hydrate!",
        "Peace out! You've got this – whatever 'this' is today."
    ]

    # Random greeting selection
    import random
    greeting = random.choice(GREETINGS)
    engine.say(greeting)
    engine.runAndWait()  # This blocks until speech is done
    
    while True:
        try:
            # Get voice input
            print("\n" + "="*40 + "\nListening... (Say 'bye' to exit)")
            user_speech = speech_to_text()
            
            if not user_speech:
                engine.say("Hey, I think my ears glitched. Mind repeating that?")
                engine.runAndWait()
                continue
                
            # Check for exit conditions
            if any(word in user_speech.lower() for word in ['bye', 'stop', 'quit', 'enough']):
                farewell = random.choice(GOODBYES)
                engine.say(farewell)
                engine.runAndWait()
                break
                
            # Process input
            print(f"\nUser said: {user_speech}")
            mood = predict_emotion(user_speech)
            print(f"Detected mood: {mood}")
            
            # Generate response
            ai_response = get_ai_response(user_speech, mood)
            print(f"\nAI Response: {ai_response}")
            
            # Convert response to speech
            engine.say(ai_response)
            engine.runAndWait()
            
        except KeyboardInterrupt:
            engine.say("Whoops! Seems like someone's in a hurry. Catch you later!")
            engine.runAndWait()
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            engine.say("Yikes! My circuits glitched. Let's try that again...")
            engine.runAndWait()

async def translate_to_english(text):
    translator = Translator()
    detection = await translator.detect(text)  # Await the coroutine
    if detection.lang != "en":
        translation = await translator.translate(text, src=detection.lang, dest="en")
        return translation.text
    return text

# talk()
