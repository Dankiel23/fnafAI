"""
Game constants extracted for easy reference and modification.

All values match the original FNAF 1 game mechanics as documented
in research/how-the-game-works.md
"""

# Camera identifiers
CAMERAS = ['1A', '1B', '1C', '2A', '2B', '3', '4A', '4B', '5', '6', '7']
POSITIONS = CAMERAS + ['office']

CAMERA_NAMES = {
    '1A': 'Show stage',
    '1B': 'Dining area',
    '1C': 'Pirate cove',
    '2A': 'West hall',
    '2B': 'W. hall corner',
    '3': 'Supply closet',
    '4A': 'East hall',
    '4B': 'E. hall corner',
    '5': 'Backstage',
    '6': 'Kitchen',
    '7': 'Restrooms',
}

# Night timing
NIGHT_DURATION_SECONDS = 535
TWELVE_AM_DURATION = 90  # seconds
OTHER_HOUR_DURATION = 89  # seconds each

# AI increase timings (in game seconds)
AI_INCREASE_2AM = 179
AI_INCREASE_3AM = 268
AI_INCREASE_4AM = 357

# Movement opportunity intervals (seconds)
FREDDY_INTERVAL = 3.02
BONNIE_INTERVAL = 4.97
CHICA_INTERVAL = 4.98
FOXY_INTERVAL = 5.01

# Starting AI levels per night [unused, night1, night2, ..., night7/custom]
FREDDY_AI_LEVELS = [None, 0, 0, 1, None, 3, 4, 20]  # Night 4 is random 1 or 2
BONNIE_AI_LEVELS = [None, 0, 3, 0, 2, 5, 10, 20]
CHICA_AI_LEVELS = [None, 0, 1, 5, 4, 7, 12, 20]
FOXY_AI_LEVELS = [None, 0, 1, 2, 6, 5, 16, 20]

# Power drain spacing per night
POWER_DRAIN_SPACING = [0, 9.6, 6, 5, 4, 3, 3, 3]

# Starting power
STARTING_POWER = 99.0

# Max power usage multiplier
MAX_POWER_MULTIPLIER = 4

# Freddy's fixed path
FREDDY_PATH = {'1A': '1B', '1B': '7', '7': '6', '6': '4A', '4A': '4B'}

# Freddy wait time formula: max(0, 1000 - AI * 100) frames at 60fps
FREDDY_WAIT_BASE_FRAMES = 1000
FREDDY_WAIT_AI_MULTIPLIER = 100
GAME_FPS = 60

# Foxy cooldown range after cameras go off (seconds)
FOXY_COOLDOWN_MIN = 0.83
FOXY_COOLDOWN_MAX = 16.67

# Foxy attack countdown (seconds)
FOXY_ATTACK_COUNTDOWN = 25.0

# Foxy running delay when triggered by camera 2A (seconds)
FOXY_RUN_DELAY = 1.87

# Foxy steps needed to leave Pirate Cove
FOXY_STEPS_TO_LEAVE = 3

# Bonnie/Chica in-office jumpscare timer (seconds)
OFFICE_JUMPSCARE_TIMEOUT = 30.0

# Freddy in-office jumpscare chance per second
FREDDY_OFFICE_JUMPSCARE_CHANCE = 0.25

# Power outage timing
POWER_OUTAGE_CHECK_INTERVAL_ARRIVAL = 5.0  # seconds
POWER_OUTAGE_CHECK_INTERVAL_SONG = 5.0
POWER_OUTAGE_CHECK_INTERVAL_DARKNESS = 2.0
POWER_OUTAGE_CHANCE = 0.2
POWER_OUTAGE_MAX_ATTEMPTS = 4

# Bonnie's possible positions
BONNIE_POSITIONS = ['1B', '3', '5', '2A', '2B']

# Chica's adjacent room map
CHICA_ADJACENCY = {
    '1A': ['1B'],
    '1B': ['4A', '6', '7'],
    '6': ['1B', '7'],
    '7': ['1B', '6'],
    '4A': ['1B', '4B'],
}

# In-game time conversion factor
TIME_CONVERSION_FACTOR = 0.6741573033707866
