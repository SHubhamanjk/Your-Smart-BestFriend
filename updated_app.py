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
        self.current_view = "chat"  
        self.current_voice = "male"
        self.speech_speed = 180
        self.chat_history = ft.ListView(expand=True, spacing=10)
        self.active_thread = None
        self.audio_player = None
        self.exit_flag = False
        self.advanced_mode = False
        self.beast_mode = False
        self.tasks = []

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
    )

    gif = ft.Image(
        src="assets/sphere-4646_256.gif",
        width=200,
        height=200,
        fit=ft.ImageFit.CONTAIN,
    )

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

    # Text input and send button
    text_input = ft.TextField(
        hint_text="Type your message here...",
        expand=True,
        border_color=PRIMARY_COLOR,
        color=SECONDARY_COLOR,
        on_submit=lambda e: handle_text_input(text_input, state, page)
    )

    send_button = ft.IconButton(
        icon=ft.icons.SEND,
        icon_color=PRIMARY_COLOR,
        on_click=lambda e: handle_text_input(text_input, state, page)
    )

    chat_controls_row = ft.Row(
        [
            ft.IconButton(
                icon=ft.icons.CLOSE,
                icon_color=SECONDARY_COLOR,
                on_click=lambda e: exit_app(page, state)
            ),
            ft.IconButton(
                icon=ft.icons.SEARCH_OFF,
                icon_color=SECONDARY_COLOR,
                on_click=lambda e: toggle_advanced_mode(e, state, page)
            ),
            ft.IconButton(
                icon=ft.icons.SETTINGS_INPUT_ANTENNA,
                icon_color=SECONDARY_COLOR,
                on_click=lambda e: toggle_beast_mode(e, state, page)
            ),
            ft.Container(expand=True),
            voice_dropdown,
            speed_slider,
            ft.IconButton(
                icon=ft.icons.MIC_OFF,
                icon_color=SECONDARY_COLOR,
                on_click=lambda e: toggle_listening(e, state, page)
            ),
            ft.IconButton(
                icon=ft.icons.CALENDAR_TODAY,
                icon_color=SECONDARY_COLOR,
                on_click=lambda e: switch_view("todo", state, page)
            )
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.CENTER
    )

    input_row = ft.Row(
        [text_input, send_button],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN
    )

    task_input = ft.TextField(hint_text="Add new task", expand=True)
    tasks_list = ft.ListView(expand=True)
    
    todo_view = ft.Column(
        [
            ft.Row(
                [
                    ft.IconButton(
                        icon=ft.icons.ARROW_BACK,
                        icon_color=SECONDARY_COLOR,
                        on_click=lambda e: switch_view("chat", state, page)
                    ),
                    ft.Text("Todo List", size=20, color=PRIMARY_COLOR),
                ],
                alignment=ft.MainAxisAlignment.START
            ),
            ft.Divider(height=20, color=ft.colors.TRANSPARENT),
            ft.Row(
                [
                    task_input,
                    ft.IconButton(
                        icon=ft.icons.ADD_CIRCLE,
                        icon_color=PRIMARY_COLOR,
                        on_click=lambda e: add_task(state, page)
                    )
                ]
            ),
            tasks_list
        ],
        visible=False
    )

    chat_view = ft.Stack(
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
                    chat_controls_row,
                    input_row
                ],
                expand=True,
            )
        ],
        expand=True
    )

    page.add(
        ft.Column(
            [
                chat_view,
                todo_view
            ],
            expand=True
        )
    )

    # ========== Helper Functions ==========
    def handle_text_input(text_field, state, page):
        user_input = text_field.value.strip()
        if user_input:
            add_message(user_input, "user", state)
            text_field.value = ""
            text_field.update()
            process_user_input(user_input, state, page)

    def switch_view(target_view, state, page):
        state.current_view = target_view
        chat_view.visible = (target_view == "chat")
        todo_view.visible = (target_view == "todo")
        page.update()

    def add_task(state, page):
        if task_input.value.strip():
            state.tasks.append(task_input.value.strip())
            tasks_list.controls.append(
                ft.Row(
                    [
                        ft.Checkbox(label=task_input.value.strip()),
                        ft.IconButton(
                            icon=ft.icons.DELETE,
                            icon_color=ft.colors.RED_400,
                            on_click=lambda e: delete_task(e.control.data, state, page),
                            data=task_input.value.strip()
                        )
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                )
            )
            task_input.value = ""
            page.update()

    def delete_task(task, state, page):
        state.tasks.remove(task)
        tasks_list.controls = [row for row in tasks_list.controls if row.controls[0].label != task]
        page.update()

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

    def process_user_input(user_input, state, page):
        try:
            if state.beast_mode:
                response = automation_work(user_input)
                add_message(response, "ai", state)
                speak(response, page, state)
            else:
                if state.advanced_mode:
                    response = advance_search(user_input)
                else:
                    emotion = predict_emotion(user_input)
                    response = get_ai_response(user_input, emotion)
                
                add_message(response, "ai", state)
                speak(response, page, state)
        except Exception as e:
            print(f"Processing error: {str(e)}")
            add_message("Oops! Something went wrong. Let's try that again.", "ai", state)

    def toggle_advanced_mode(e, state, page):
        state.advanced_mode = not state.advanced_mode
        e.control.icon = ft.icons.SEARCH if state.advanced_mode else ft.icons.SEARCH_OFF
        e.control.icon_color = PRIMARY_COLOR if state.advanced_mode else SECONDARY_COLOR
        e.control.update()
        status = "ON" if state.advanced_mode else "OFF"
        page.snack_bar = ft.SnackBar(
            content=ft.Text(f"Advanced Search Mode {status}"),
            bgcolor=PRIMARY_COLOR
        )
        page.snack_bar.open = True
        page.update()

    def toggle_beast_mode(e, state, page):
        state.beast_mode = not state.beast_mode
        e.control.icon_color = ft.colors.AMBER if state.beast_mode else SECONDARY_COLOR
        e.control.update()
        status = "ON" if state.beast_mode else "OFF"
        page.snack_bar = ft.SnackBar(
            content=ft.Text(f"Beast Mode {status}"),
            bgcolor=PRIMARY_COLOR
        )
        page.snack_bar.open = True
        page.update()

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
                volume=1.0,
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
        text_input.disabled = state.listening
        e.control.update()
        
        if state.listening:
            state.active_thread = threading.Thread(
                target=lambda: continuous_listening(page, state),
                daemon=True
            )
            state.active_thread.start()
        page.update()

    def continuous_listening(page, state):
        while state.listening and not state.exit_flag:
            try:
                user_input = speech_to_text()
                if user_input and user_input.lower() not in ["no speech detected", "error"]:
                    add_message(user_input, "user", state)
                    process_user_input(user_input, state, page)
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
