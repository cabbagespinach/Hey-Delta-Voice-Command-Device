"""Spoken texts shared by generate_synthetic_commands.py and clone_fleurs_voices.py (no dependencies)."""

# How each prompt is spoken (numbers and times in words, so every voice says the same thing).
SPOKEN = {
    "dim_lights_20": "Dim lights to twenty percent", "dim_lights_50": "Dim lights to fifty percent",
    "dim_lights_80": "Dim lights to eighty percent",
    "set_timer_1": "Set a timer for one minute", "set_timer_5": "Set a timer for five minutes",
    "set_timer_10": "Set a timer for ten minutes",
    "set_alarm_6am": "Set an alarm at six A M", "set_alarm_7am": "Set an alarm at seven A M",
    "set_alarm_9pm": "Set an alarm at nine P M",
    "set_temperature_18": "Set temperature to eighteen degrees", "set_temperature_22": "Set temperature to twenty two degrees",
    "set_temperature_26": "Set temperature to twenty six degrees",
}
UNKNOWN_SENTENCES = [
    # everyday speech
    "How are you today", "I'm going to the store later", "What's for dinner tonight", "Can you pass me the remote",
    "I think it's going to rain", "Let's watch a movie", "Where did I put my keys", "That was really funny",
    "I need to charge my phone", "Good morning everyone", "Thank you so much", "Never mind",
    "Wait a second", "Okay I'm done", "Sorry wrong one", "Hello", "Yes please", "No thanks",
    # near-miss commands (similar words, not on the list)
    "Turn on the TV", "Open the window", "Call Mom", "Text John", "Remind me to buy milk", "Play a video",
    "What day is it", "Set the table", "Turn off the fan", "Dim the screen", "Set a reminder for tomorrow",
    "What's the news", "Volume is fine", "Stop talking", "Next week", "Pause for a moment", "Call me later",
    "Set an alarm for tomorrow morning", "What's the temperature outside", "Lights look nice",
]
