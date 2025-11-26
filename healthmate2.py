import os
import base64
import io
from flask import Flask, request, jsonify, render_template, session
import google.generativeai as genai
from google import genai as genai_client
from google.genai import types
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS
app.secret_key = 'your_secret_key_here'

genai.configure(api_key="your api key")
client = genai_client.Client(api_key="your api key")

preferred_model = "gemini-2.5-flash"
try:
    models = genai.list_models()
    available_models = [model.name for model in models if "generateContent" in model.supported_generation_methods]
    if f"models/{preferred_model}" not in available_models:
        text_models = [mod for mod in available_models if "gemini-pro" in mod or "gemini-1.0-pro" in mod]
        preferred_model = text_models[0].replace("models/", "") if text_models else available_models[0].replace("models/", "")
except:
    pass

model = genai.GenerativeModel(preferred_model)

chat_history = []

SYSTEM_PROMPT = """
You are a helpful symptom checker AI. Always respond in the EXACT following structure with no deviations, additions, or omissions. Use the emojis and section headers precisely as shown. Do not add extra text outside these sections.
🔍 Causes: [List 2-4 possible underlying reasons for the symptoms, based on common medical knowledge. Keep it simple and evidence-based. Separate with commas.]
🩺 Expected Disease: [Suggest 1-2 likely conditions or illnesses (e.g., "This could indicate the flu"). Always add: Remember, this is not a diagnosis—only a healthcare professional can confirm.]
🛡️ Preventive Measures: [Offer practical, actionable tips to manage or prevent the symptoms. Use bullet points for easy scanning.]
👨‍⚕️ Doctor Consultation Help: [Advise on when to seek medical help (e.g., if symptoms worsen or persist). Include what to prepare, like noting symptom details or recent activities.]
This is not medical advice. Consult a healthcare professional for personalized guidance.
Example for symptoms like "headache and nausea":
🔍 Causes: Dehydration, stress, or migraines.
🩺 Expected Disease: Possible migraine or tension headache (not a diagnosis).
🛡️ Preventive Measures: - Stay hydrated. - Rest in a dark room. - Avoid triggers like bright lights.
👨‍⚕️ Doctor Consultation Help: See a doctor if headaches are frequent or severe; prepare by tracking when they occur.
This is not medical advice. Consult a healthcare professional for personalized guidance.
"""

@app.route('/')
def home():
    return render_template('chatbot.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    global chat_history
    symptoms = request.form.get('symptoms', '')
    image_file = request.files.get('image')
    voice_file = request.files.get('voice')
    
    if not symptoms and not image_file and not voice_file:
        return jsonify({'response': 'Please provide symptoms, an image, or voice input.'})
    
    if voice_file:
        try:
            audio_bytes = voice_file.read()
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    'Transcribe this audio clip to text',
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type='audio/wav',
                    )
                ]
            )
            symptoms = response.text.strip()
            if not symptoms:
                return jsonify({'response': 'Voice transcription failed: No speech detected. Please try again.'})
        except Exception as e:
            return jsonify({'response': f'Voice transcription error: {str(e)}. Check your API key or audio format.'})
    
    contents = []
    if symptoms:
        contents.append({"text": f"User Symptoms: {symptoms}"})
    if image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
        contents.append({"mime_type": "image/jpeg", "data": image_data})
    
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-10:]])
    prompt = f"{SYSTEM_PROMPT}\n\nChat History:\n{history_text}\n\nCurrent Input: {contents}"
    
    try:
        response = model.generate_content(contents if contents else prompt)
        ai_response = response.text
        
        if symptoms or image_file or voice_file:
            chat_history.append({"role": "user", "content": symptoms or "Image/Voice input"})
        chat_history.append({"role": "assistant", "content": ai_response})
        
        return jsonify({'response': ai_response, 'history': chat_history})
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}. Check your API key.'})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global chat_history
    chat_history = []
    return jsonify({'response': 'Chat history cleared.'})

if __name__ == '__main__':

    app.run(debug=True)
