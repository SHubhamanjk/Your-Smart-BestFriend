import flet as ft
import random
import time
import threading
import uuid
from pathlib import Path
from functions import *

# ========== UI Configuration ==========
PRIMARY_COLOR = ft.colors.DEEP_PURPLE_400
SECONDARY_COLOR = ft.colors.WHITE
BG_COLOR = ft.colors.BLACK

# ========== App States ==========
class AppState:
    def __init__(self):
        self.listening = False
        self.volume_mode = "speaker"
        self.current_voice = "male"  # 'male' or 'female'
        self.speech_speed = 180     # 100-300 WPM
        self.chat_history = ft.ListView()
        self.active_thread = None
        self.audio_player = None
        self.exit_flag = False

# ========== Utility Functions ==========
def get_temp_file(extension=".mp3"):
    temp_dir = Path.cwd() / "temp"
    temp_dir.mkdir(exist_ok=True)
    return temp_dir / f"{uuid.uuid4().hex}{extension}"

def cleanup_audio_file(file_path):
    try:
        time.sleep(5)  # Wait for audio to finish playing
        Path(file_path).unlink(missing_ok=True)
    except Exception as e:
        print(f"Error cleaning up audio file: {str(e)}")

# ========== Main App ==========
def main(page: ft.Page):
    page.title = "Your Smart BestFriend"
    page.bgcolor = BG_COLOR
    page.window_width = 400
    page.window_height = 700
    state = AppState()

    # ========== UI Components ==========
    title = ft.Text(
        "Your Smart BestFriend",
        size=20,
        color=PRIMARY_COLOR,
        weight=ft.FontWeight.BOLD,
        animate_rotation=ft.animation.Animation(300, "bounceOut"),
    )

    gif = ft.Image(
        src="sphere-4646_256.gif",
        width=200,
        height=200,
        fit=ft.ImageFit.CONTAIN,
    )

    close_btn = ft.IconButton(
        icon=ft.icons.CLOSE,
        icon_color=SECONDARY_COLOR,
        on_click=lambda e: exit_app(page, state),
    )

    voice_toggle = ft.IconButton(
        icon=ft.icons.MIC_OFF,
        icon_color=SECONDARY_COLOR,
        on_click=lambda e: toggle_listening(e, state, page),
    )

    # Voice and Speed Controls
    voice_dropdown = ft.Dropdown(
        width=120,
        value="male",
        options=[
            ft.dropdown.Option("male", "Ashish"),
            ft.dropdown.Option("female", "Nidhi"),
        ],
        on_change=lambda e: change_voice(e.control.value, page, state)
    )

    speed_slider = ft.Slider(
        width=150,
        min=100,
        max=300,
        divisions=20,
        value=180,
        label="{value} WPM",
        on_change=lambda e: change_speed(e.control.value, page, state)
    )

    controls_row = ft.Row(
        [
            close_btn,
            ft.Container(expand=True),
            voice_dropdown,
            speed_slider,
            voice_toggle,
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.CENTER
    )

    # ========== App Layout ==========
    page.add(
        ft.Stack(
            [
                ft.Container(gif, alignment=ft.alignment.center),
                ft.Column(
                    [
                        ft.Container(title, alignment=ft.alignment.top_center),
                        ft.Container(
                            state.chat_history,
                            expand=True,
                            padding=20,
                        ),
                        controls_row,
                    ],
                    expand=True,
                ),
            ],
            expand=True,
        )
    )

    # ========== Core Functions ==========
    def add_message(text, sender, state):
        icon = "🤖" if sender == "ai" else "👤"
        text = f"{icon} {text}"
        message = ft.Text(
            text,
            color=SECONDARY_COLOR if sender == "ai" else PRIMARY_COLOR,
            size=14,
        )
        state.chat_history.controls.append(message)
        state.chat_history.update()


    def speak(text, page, state):
        try:
            audio_file = text_to_speech(
                text,
                gender=state.current_voice,
                speed=state.speech_speed
            )
            if not audio_file:
                return

            if state.audio_player:
                page.overlay.remove(state.audio_player)

            state.audio_player = ft.Audio(
                src=audio_file,
                autoplay=True,
                volume=1.0 if state.volume_mode == "speaker" else 0.5,
            )
            page.overlay.append(state.audio_player)
            page.update()
            
            threading.Thread(
                target=lambda: (time.sleep(5), Path(audio_file).unlink(missing_ok=True)),
                daemon=True
            ).start()
            
        except Exception as e:
            print(f"Speak Error: {e}")

    def toggle_listening(e, state, page):
        state.listening = not state.listening
        e.control.icon = ft.icons.MIC if state.listening else ft.icons.MIC_OFF
        e.control.update()
        
        if state.listening:
            state.active_thread = threading.Thread(
                target=lambda: continuous_listening(page, state),
                daemon=True
            )
            state.active_thread.start()

    def continuous_listening(page, state):
        while state.listening and not state.exit_flag:
            try:
                user_input = speech_to_text()
                if user_input and user_input.lower() not in ["no speech detected", "error"]:
                    add_message(user_input, "user", state)
                    emotion = predict_emotion(user_input)
                    response = get_ai_response(user_input, emotion)
                    add_message(response, "ai", state)
                    speak(response, page, state)
                else:
                    add_message("Hey, I think my ears glitched. Mind repeating that?", "ai", state)
                    speak("Hey, I think my ears glitched. Mind repeating that?", page, state)    
                time.sleep(1)
                    
                    
                    
        
            except Exception as e:
                print(f"Error in continuous_listening: {str(e)}")
                state.listening = False
                page.update()

    def change_voice(voice, page, state):
            state.current_voice = voice
            page.snack_bar = ft.SnackBar(
                content=ft.Text(f"Voice set to {'Nidhi' if voice == 'female' else 'Ashish'}"),
                bgcolor=PRIMARY_COLOR
            )
            page.snack_bar.open = True
            page.update()

    def change_speed(speed, page, state):
                state.speech_speed = speed
                page.snack_bar = ft.SnackBar(
                    content=ft.Text(f"Speed set to {speed} WPM"),
                    bgcolor=PRIMARY_COLOR
                )
                page.snack_bar.open = True
                page.update()


    def exit_app(page, state):
        state.exit_flag = True
        state.listening = False
        if state.active_thread and state.active_thread.is_alive():
            state.active_thread.join(timeout=2)
        
        goodbyes = [
            "See you soon! Take care!",
            "Goodbye! Remember to smile!",
            "Until next time! Stay awesome!",
            "Catch you later, partner! Remember: You're stronger than you think.",
            "Alright, time to bounce... But I'm always here if you need!",
            "Chalo, phir milenge! Take care of yourself, yaar.",
            "Signing off for now... Don't forget to hydrate!",
            "Peace out! You've got this – whatever 'this' is today."
        ]
        farewell = random.choice(goodbyes)
        speak(farewell, page, state)
        time.sleep(2)
        page.window_destroy()

    # ========== Startup Greeting ==========
    def initial_greeting():
        if state.exit_flag: return
        
        greetings = [
            "Hey there! How can I help you today?",
            "Hello friend! What's on your mind?",
            "Hi! Ready to chat?",
            "Good to see you! How can I assist?",
            "Hey buddy! Long time no chat... How's life treating you?",
            "Yo! What's up? Ready for our daily dose of real talk?",
            "Namaste dost! Kaise ho aaj? Let's vibe together...",
            "Ah, my favorite human! What's cooking in that beautiful mind today?",
            "Hey partner in crime! Ready to unpack some thoughts?"
        ]
        response = random.choice(greetings)
        add_message(response, "ai", state)
        speak(response, page, state)

    threading.Thread(target=initial_greeting, daemon=True).start()

if __name__ == "__main__":
    ft.app(target=main)
