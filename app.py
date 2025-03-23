import flet as ft
import random
import time
import threading
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
        self.current_voice = "en"
        self.chat_history = ft.ListView()
        self.active_thread = None
        self.audio_player = None

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
        animate_scale=ft.animation.Animation(1000, "bounceOut"),
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

    settings_menu = ft.PopupMenuButton(
        icon=ft.icons.SETTINGS,
        icon_color=SECONDARY_COLOR,
        items=[
            ft.PopupMenuItem(
                text="Change Voice",
                on_click=lambda e: change_voice(page, state)
            ),
            ft.PopupMenuItem(
                text="Volume Settings",
                on_click=lambda e: change_volume_mode(page, state)
            ),
        ],
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
                        ft.Row(
                            [close_btn, ft.Container(expand=True), voice_toggle, settings_menu],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            vertical_alignment=ft.CrossAxisAlignment.END,
                        ),
                    ],
                    expand=True,
                ),
            ],
            expand=True,
        )
    )

    # ========== Startup Greeting ==========
    def initial_greeting():
        greetings = [
            "Hey there! How can I help you today?",
            "Hello friend! What's on your mind?",
            "Hi! Ready to chat?",
            "Good to see you! How can I assist?",
        ]
        response = random.choice(greetings)
        add_message(response, "ai", state)
        speak(response, page, state)

    threading.Thread(target=initial_greeting, daemon=True).start()

# ========== Core Functions ==========
def add_message(text, sender, state):
    message = ft.Text(
        text,
        color=SECONDARY_COLOR if sender == "ai" else PRIMARY_COLOR,
        size=14,
    )
    state.chat_history.controls.append(message)
    state.chat_history.update()

def speak(text, page, state):
    try:
        audio_file = text_to_speech(text, lang=state.current_voice)
        if not audio_file:
            return

        # Clean up previous audio player
        if state.audio_player:
            page.overlay.remove(state.audio_player)
            state.audio_player = None

        state.audio_player = ft.Audio(
            src=audio_file,
            autoplay=True,
            volume=1.0 if state.volume_mode == "speaker" else 0.5,
            on_loaded=lambda _: threading.Thread(
                target=cleanup_audio_file,
                args=(audio_file,),
                daemon=True
            ).start()
        )
        page.overlay.append(state.audio_player)
        page.update()
        
    except Exception as e:
        print(f"Error in speak: {str(e)}")

def cleanup_audio_file(file_path):
    try:
        time.sleep(5)  # Wait for audio to finish playing
        Path(file_path).unlink(missing_ok=True)
    except Exception as e:
        print(f"Error cleaning up audio file: {str(e)}")
def toggle_listening(e, state, page):
    state.listening = not state.listening
    e.control.icon = ft.icons.MIC if state.listening else ft.icons.MIC_OFF
    e.control.update()
    
    if state.listening:
        state.active_thread = threading.Thread(
            target=continuous_listening,
            args=(page, state),
            daemon=True
        )
        state.active_thread.start()

def continuous_listening(page, state):
    while state.listening:
        try:
            user_input = speech_to_text()
            if user_input and user_input.lower() not in ["no speech detected", "error"]:
                add_message(user_input, "user", state)
                emotion = predict_emotion(user_input)
                response = get_ai_response(user_input, emotion)
                add_message(response, "ai", state)
                speak(response, page, state)
            time.sleep(1)
            
        except Exception as e:
            print(f"Error in continuous_listening: {str(e)}")
            state.listening = False
            page.update()

def change_voice(page, state):
    state.current_voice = "hi" if state.current_voice == "en" else "en"
    page.snack_bar = ft.SnackBar(
        content=ft.Text(f"Voice changed to {state.current_voice.upper()}"),
        bgcolor=PRIMARY_COLOR
    )
    page.snack_bar.open = True
    page.update()

def change_volume_mode(page, state):
    state.volume_mode = "headphone" if state.volume_mode == "speaker" else "speaker"
    page.snack_bar = ft.SnackBar(
        content=ft.Text(f"Volume mode: {state.volume_mode.capitalize()}"),
        bgcolor=PRIMARY_COLOR
    )
    page.snack_bar.open = True
    page.update()

def exit_app(page, state):
    state.listening = False
    if state.active_thread and state.active_thread.is_alive():
        state.active_thread.join(timeout=2)
    
    goodbyes = [
        "See you soon! Take care!",
        "Goodbye! Remember to smile!",
        "Until next time! Stay awesome!",
    ]
    farewell = random.choice(goodbyes)
    speak(farewell, page, state)
    time.sleep(2)
    page.window_destroy()

if __name__ == "__main__":
    ft.app(target=main)