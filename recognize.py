import dlib
import face_recognition
import pickle
import re
import cv2
import numpy as np

from collections import defaultdict

import pyttsx3 as pyttsx
import time
import glob
import subprocess
import os
import datetime

from gtts import gTTS
from vlc import MediaPlayer


video_capture = cv2.VideoCapture(0)

known_face_encodings = []
known_face_names = []

greetings = {}

PREFIX = "Ola"
LANGUAGE = "pt"

CONSECUTIVE_FRAME_FILTER= 5

p = MediaPlayer()
cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_NORMAL)


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


lastUpdate = 0
lastSave = 0
saidTime = 0
saidName = ""
saveCount = 0
BUFFER = 10


def save_state():
    statedict = {'enc': known_face_encodings, 'names': known_face_names}
    with open("state.pickle", 'wb') as f:
        pickle.dump(statedict, f)


def load_state():
    with open("state.pickle", 'rb') as f:
        state = pickle.load(f)
    return state["enc"], state['names']


def write_greetings(file):
    with open(file, 'w') as f:
        for name, greet in greetings.items():
            f.write("{0}:{1}".format(name, greet))


def load_greetings(file):
    with open(file, 'r') as f:
        for line in f:
            tokens = line.split(":")
            name = tokens[0]
            greet = ":".join(tokens[1:])

            if name not in greetings:
                greetings[name] = greet
                if not os.path.isfile(os.path.join("/tmp", name+".mp3")):
                    write_audio(name)
            elif greetings[name]!=greet:
                greetings[name] = greet
                write_audio(name)


def load_person(name, encoding):
    known_face_encodings.append(encoding)
    known_face_names.append(name)

    if name not in greetings:
        greetings[name] = PREFIX+"{name}"
        write_audio(name=name)
        write_greetings("Faces/greetings.txt")


def write_audio(name):
    tts = gTTS(text=greetings[name].format(name=name),
               lang=LANGUAGE,
               )
    tts.save('/tmp/'+name+'.mp3')


def update_people(folder):
    for im in glob.glob(os.path.join(folder, "*.jpg")):
        name = os.path.split(im)[-1].split('.')[0]
        if name not in known_face_names:
            image = cv2.imread(im)
            encoding = face_recognition.face_encodings(image)[0]
            load_person(name, encoding)


known_face_encodings, known_face_names = load_state()

last_faceNames = {}

faces_count = defaultdict(int)

while True:
    if time.time() - lastUpdate > 3:
        update_people("Faces")
        save_state()
        load_greetings("Faces/greetings.txt")
        lastUpdate = time.time()
        # Grab a single frame of video

    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodinhhs in the current frame of video
        face_locations = face_recognition.face_locations(
            rgb_small_frame, model='cnn')
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations, num_jitters=3)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                condition = time.time()-saidTime > 5
                condition = condition or (
                    name != saidName and time.time()-saidTime > 1)
                condition = condition and not p.is_playing()
                condition = condition and faces_count[name] > CONSECUTIVE_FRAME_FILTER
                if condition:

                    horas = datetime.datetime.now().hour
                    # text = re.escape(text)
                    # # args = ['gtts-cli',
                    #         '"{0}"'.format(text),
                    #         "-l",
                    #         "pt",
                    #         '| play -t mp3 - &']
                    # subprocess.call(args, shell=True)
                    p = MediaPlayer('/tmp/'+name+'.mp3')
                    p.play()

                    saidTime = time.time()
                    saidName = name

            face_names.append(name)

        for face in face_names:
            if face in last_faceNames:
                faces_count[name] += 1
            else:
                faces_count[name] = 0

        last_faceNames = set(face_names)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size

        top = max(top-50, 0)
        left = max(left-50, 0)
        right = min(right+50, small_frame.shape[1])
        bottom = min(bottom+50, small_frame.shape[0])

        top *= 1
        right *= 1
        bottom *= 1
        left *= 1

        if name == "Unknown" and time.time()-lastSave > 0.7:
            lastSave = time.time()
            saveCount += 1
            save = frame[top:bottom, left:right]
            save = cv2.resize(save, None, fx=2, fy=2)
            cv2.imwrite("Unknown"+str(saveCount)+".jpg", save)
            if saveCount > BUFFER:
                saveCount = 0

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
write_greetings("Faces/greetings.txt")
save_state()
video_capture.release()
cv2.destroyAllWindows()
