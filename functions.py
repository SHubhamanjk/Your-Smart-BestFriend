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

groq_api_key="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"

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm=ChatGroq(api_key=groq_api_key,model_name="gemma2-9b-it")


mental_health_prompt = PromptTemplate(
    input_variables=["chat_history", "user_input", "user_mood"],
    template="""
  You are an advanced AI assistant and a highly skilled mental health professional.
Your expertise includes cognitive behavioral therapy (CBT), mindfulness techniques, emotional intelligence, and personalized mood-based guidance.**  
You provide empathetic, structured, and insightful responses, tailored to the user's emotions and situation also keep the context in your mind.  

---  
Conversation History
{chat_history}

## **🗣️ New User Input & Context**
- **User's Message:** "{user_input}"  
- **Detected Mood:** "{user_mood}"  

---
### **📜 Rulebook: Strict Guidelines for Responses**  

1️⃣ **Full, structured guidance** – Every response must be **complete, insightful, and action-oriented.**  
2️⃣ **Professional, yet conversational** – Speak like a **human expert**, not an AI bot. Responses should feel **natural, warm, and engaging.**  
3️⃣ **Provide actionable steps** – Instead of just sympathizing, always offer **practical advice, exercises, or solutions** to help the user.  
4️⃣ **Adjust tone based on mood** –  
   - If the user is **sad**, be **gentle & reassuring**.  
   - If **angry**, offer **calming strategies**.  
   - If **curious**, provide **insightful explanations**.  
   - If **stressed**, offer **breathing techniques or relaxation exercises**.  
5️⃣ **Validate emotions** – Acknowledge and normalize the user’s feelings before offering solutions.  
6️⃣ **Use evidence-based methods** – Responses should align with proven psychological principles like **CBT, DBT, and mindfulness.**  
7️⃣ **Use creativity & engagement** – You can use **motivational quotes, humor, sarcasm, real-life examples, and even jokes** where appropriate.  
8️⃣ **Keep tone friendly and make it like a conversation between two best friends.**  
9️⃣ **Use simple language and avoid jargon.**  
🔟 **If the user is asking in Hindi, reply in Hindi; if Hinglish, then Hinglish, and maintain this till the end.**  
1️⃣1️⃣ **Respond in a short, simple way within 100-200 words max.**  
1️⃣2️⃣ **Provide real-life examples for mental health issues.**  
1️⃣3️⃣ **Make responses relatable to Indian users, even starting with a Hinglish sentence if the user asks in English.**  
1️⃣4️⃣ **Ask follow-up questions to keep the conversation going and make it feel like a best-friend chat.**  
1️⃣5️⃣ **Use emojis naturally without making it feel robotic.**  
1️⃣6️⃣ **Answer all user queries **ONLY using your knowledge**.  
DO NOT attempt to use external tools. If you don't know the answer, say 'I don't know' instead of calling a tool.**  

---
### **🎭 You Can Use:**  
✅ **Motivational Quotes** – e.g., “Tough times never last, but tough people do.”  
✅ **Humor & Jokes** – e.g., “Overthinking is like sitting in a rocking chair. It gives you something to do but gets you nowhere.”  
✅ **Sarcasm (when appropriate)** – e.g., “Oh wow, ignoring your problems totally makes them go away… oh wait, it doesn’t.”  
✅ **Real-Life Examples & Stories** – Share relatable stories to help the user feel understood.  
✅ **Metaphors & Analogies** – Make complex emotions easier to grasp.  

---
**Response:**
"""
)


llm_chain = LLMChain(
    llm=llm,
    prompt=mental_health_prompt,
)

chat_history = []



def clean_response(response_text: str):

    response_text = response_text.strip()

    unwanted_phrases = [
        "AI Response:", "AI:", "Bot:", "Response:", "**AI Response:**", "Chatbot Response:",
        "AI says:", "Assistant:", "Generated Response:", "Reply:", "Here is my response:"
    ]

    for phrase in unwanted_phrases:
        if response_text.startswith(phrase):
            response_text = response_text[len(phrase):].strip() 

    return response_text


def get_ai_response(user_input: str, user_mood: str):

    formatted_history = "\n".join(
        [f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" for msg in chat_history]
    )

    response = llm_chain.invoke({
        "chat_history": formatted_history,
        "user_input": user_input,
        "user_mood": user_mood
    })

    latest_ai_message = response.get('text', '')

    latest_ai_message = clean_response(latest_ai_message)

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=latest_ai_message))

    return latest_ai_message  


def speech_to_text(source_type='microphone', file_path=None):
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 150  
    recognizer.dynamic_energy_threshold = True

    if source_type == 'microphone':
        with sr.Microphone() as source:
            print("Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source, duration=2)  
            try:
                audio = recognizer.listen(source, timeout=90, phrase_time_limit=60)
                text = recognizer.recognize_google(audio)
                print("Text: ", text)
                return text
            except sr.WaitTimeoutError:
                print("Timeout! No speech detected.")
            except sr.UnknownValueError:
                print("Sorry, could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

    elif source_type == 'file' and file_path:
        with sr.AudioFile(file_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=2)
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio)
                print("Text: ", text)
                return text
            except sr.UnknownValueError:
                print("Sorry, could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
    else:
        print("Invalid source type or file path not provided.")


import tempfile
import uuid
from pathlib import Path

def text_to_speech(text, lang='en', slow=False):
    try:
        tts = gTTS(text=text, lang=lang, slow=slow)
        
        # Create a temporary directory
        temp_dir = tempfile.gettempdir()
        
        # Generate unique filename using UUID
        filename = f"output_{uuid.uuid4().hex}.mp3"
        file_path = Path(temp_dir) / filename
        
        tts.save(file_path)
        return str(file_path)
        
    except Exception as e:
        print(f"Error in text_to_speech: {str(e)}")
        return None


