import face_recognition
import re
import cv2
import numpy as np

import time
import glob
import subprocess
import os
import datetime

from gtts import gTTS
from vlc import MediaPlayer

VIDEOPATH="/home/store/Video/Processar/MVI_1403.MOV"
video_capture = cv2.VideoCapture(VIDEOPATH)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')

known_face_encodings = []
known_face_names = []

tempfile = '/tmp/audioRF.mp3'


def get_people(folder):
    print(glob.glob(os.path.join(folder, "*.jpg")))
    for im in glob.glob(os.path.join(folder, "*.jpg")):
        image = cv2.imread(im)
        name = os.path.split(im)[-1].split('.')[0]
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)

        prefix = "Olá"
        text = " ".join([prefix, name, "!"])
        tts = gTTS(text=text, lang='pt')
        tts.save('/tmp/'+name+'.mp3')


get_people("Faces")
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


def update_people(folder):
    for im in glob.glob(os.path.join(folder, "*.jpg")):
        name = os.path.split(im)[-1].split('.')[0]
        if name not in known_face_names:
            image = cv2.imread(im)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(name)
            prefix = "Olá"
            text = " ".join([prefix, name, "!"])
            tts = gTTS(text=text, lang='pt')
            tts.save('/tmp/'+name+'.mp3')


ret, frame = video_capture.read()
print(frame.shape[:2])
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame.shape[:2][::-1]), True)
while True:
    if time.time() - lastUpdate > 3:
        update_people("Faces")
        lastUpdate = time.time()
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame, model='cnn')
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations,
             num_jitters=30)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size

        top=max(top-10,0)
        left=max(left-10,0)
        right=min(right+10, small_frame.shape[1])
        bottom=min(bottom+10, small_frame.shape[0])

        top *= 1
        right *= 1
        bottom *= 1
        left *= 1

        if name == "Unknown" and time.time()-lastSave > 0.7:
            lastSave = time.time()
            saveCount += 1
            save = frame[top:bottom, left:right]
            save = cv2.resize(save,None,fx=2,fy=2)
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
    out.write(frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
out.release()
video_capture.release()
cv2.destroyAllWindows()
