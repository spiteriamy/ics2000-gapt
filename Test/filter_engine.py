from PIL import Image
import random
import time

# cache to store last used Filters and timestamps
_last_filters = {}

# sad Filters
sad_face = Image.open('Filters/sad/sad_face.png').resize((20, 20))
single_teardrop = Image.open('Filters/sad/single_teardrop.png').resize((50, 70))
teardrop_left = Image.open('Filters/sad/teardrop_left.png').resize((20, 40))
teardrop_right = Image.open('Filters/sad/teardrop_right.png').resize((20, 40))

# angry Filters
angry_face = Image.open('Filters/Angry/angry_face.png').resize((200, 200))
horn_left = Image.open('Filters/Angry/horn_left.png').resize((30, 50))
horn_right = Image.open('Filters/Angry/horn_right.png').resize((30, 50))

# disgust Filters
disgust_face = Image.open('Filters/Disgust/disgust_face.png').resize((200, 200))

# fear Filters
scared_face_1 = Image.open('Filters/Fear/scared_face_1.png').resize((150, 150))
scared_face_2 = Image.open('Filters/Fear/scared_face_2.png').resize((150, 150))

# happy Filters
happy_face = Image.open('Filters/Happy/happy_face.png').resize((200, 200))

# surprised Filters
surprise_face = Image.open('Filters/Surprised/surprise_face.png').resize((200, 200))

# define filter options + their landmark placements and offsets
sad_filters = [
    {
        "filter": sad_face,
        "landmark": 46,  # right eye lower lid
        "offset": (100, 80)
    },
    {
        "filter": single_teardrop,
        "landmark": 36,  # left eye outer corner
        "offset": (60, 80)
    },
    {
        "filter": [teardrop_left, teardrop_right],  # two teardrops (left and right eye)
        "landmark": [41, 47],  # left lower lid, right lower lid
        "offset": [(60, 60), (90, 60)]
    }

]

angry_filters = [
    {
        "filter": [horn_left, horn_right],  # two horns
        "landmark": [18, 25],  # left lower lid, right lower lid
        "offset": [(60, 0), (120, 0)]
    },
    {
        "filter": angry_face,
        "landmark": 11,  # near ear
        "offset": (-10, -20)
    }
]

disgust_filters = [
    {
        "filter": disgust_face,
        "landmark": 11,  # near jawline
        # "offset": (-20, -70)
        "offset": (-20, -50)
    }
]

scared_filters = [
    {
        "filter": scared_face_1,
        "landmark": 0,  # near ear
        "offset": (30, 0)
    },
    {
        "filter": scared_face_2,
        "landmark": 11,  # near ear
        "offset": (0, 0)
    }
]

happy_filters = [
    {
        "filter": happy_face,
        "landmark": 11,  # near jawline
        "offset": (-10, -30)
    }
]

surprise_filters = [
    {
        "filter": surprise_face,
        "landmark": 11,  # near jawline
        "offset": (-10, -20)
    }
]


def apply_filter(img, selected_filter, landmarks, scale_factor):
    """helper to apply a single or multiple filter elements to image based on landmarks and offsets."""
    if isinstance(selected_filter["filter"], list):
        # multiple elements (left and right horns or teardrops)
        for i in range(len(selected_filter["filter"])):
            img_filter = selected_filter["filter"][i].resize(
                (int(selected_filter["filter"][i].width * scale_factor),
                 int(selected_filter["filter"][i].height * scale_factor))
            )
            x_offset = landmarks.part(selected_filter["landmark"][i]).x + int(
                selected_filter["offset"][i][0] * scale_factor)
            y_offset = landmarks.part(selected_filter["landmark"][i]).y + int(
                selected_filter["offset"][i][1] * scale_factor)
            img.paste(img_filter, (x_offset, y_offset), img_filter)
    else:
        # one element (all other Filters)
        img_filter = selected_filter["filter"].resize(
            (int(selected_filter["filter"].width * scale_factor),
             int(selected_filter["filter"].height * scale_factor))
        )
        x_offset = landmarks.part(selected_filter["landmark"]).x + int(
            selected_filter["offset"][0] * scale_factor)
        y_offset = landmarks.part(selected_filter["landmark"]).y + int(
            selected_filter["offset"][1] * scale_factor)
        img.paste(img_filter, (x_offset, y_offset), img_filter)

    return img


def draw_filter(img, landmarks, emotion, face_size, change_interval=5):

    """applies a random filter based on the emotion."""
    try:
        w, h = face_size
        scale_factor = w / 60

        emotion_filters = {
            'sad': sad_filters,
            'angry': angry_filters,
            'disgust': disgust_filters,
            'fear': scared_filters,
            'happy': happy_filters,
            'surprise': surprise_filters
        }

        if emotion not in emotion_filters:
            return img

        current_time = time.time()
        last = _last_filters.get(emotion, {"time": 0, "filter": None})

        if (current_time - last["time"]) > change_interval or last["filter"] is None:
            selected_filter = random.choice(emotion_filters[emotion])
            _last_filters[emotion] = {"time": current_time, "filter": selected_filter}
        else:
            selected_filter = last["filter"]

        img = apply_filter(img, selected_filter, landmarks, scale_factor)

    except Exception as e:
        print(f"Error loading filter: {e}")

    return img
