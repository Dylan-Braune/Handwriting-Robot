"""
main.py -- the three modes of operation from the proposal's functional
description: train, classify, and write (reproduce). Run this file and
type a command.

    train <image> <text> <user_id>     e.g. train page1.jpg abcdef 0
    classify <image>
    write <text> <user_id>
    quit
"""
import numpy as np

import capture
import conditioning
import segmentation
import network
import trajectory
import storage

CHAR_SIZE = 20                    # every character is resized to this many pixels square
INPUT_SIZE = CHAR_SIZE * CHAR_SIZE
HIDDEN_SIZE = 64
N_USERS = 10                      # per proposal requirement 6: minimum 10 users
CHARSET = "abcdefghijklmnopqrstuvwxyz"

char_net = network.init_weights(INPUT_SIZE, HIDDEN_SIZE, len(CHARSET))
writer_net = network.init_weights(INPUT_SIZE, HIDDEN_SIZE, N_USERS)


def process_page(image_path=None):
    """FU 1.1/1.2/2.1/2.2: get an image (camera or file), condition
    it, and separate it into individual character images."""
    gray = capture.load_page(image_path) if image_path else capture.capture_page()
    binary = conditioning.binarize(gray)
    binary = conditioning.deskew(binary)
    boxes = segmentation.find_characters(binary)
    characters = [segmentation.crop_and_resize(binary, box, CHAR_SIZE) for box in boxes]
    return characters


def train(image_path, expected_text, user_id):
    """FU 2.4/2.5/2.6: adjusts both networks' weights against the
    known text/user, and stores each character's trajectory map."""
    characters = process_page(image_path)
    for char_image, expected_char in zip(characters, expected_text):
        x = char_image.flatten()[None, :].astype(np.float64)
        network.train_step(char_net, x, np.array([CHARSET.index(expected_char)]))
        network.train_step(writer_net, x, np.array([user_id]))
        points = trajectory.image_to_trajectory(char_image)
        storage.save_trajectory(f"user{user_id}", expected_char, points)


def classify(image_path):
    """FU 2.3: reads back the text and identifies the writer, one
    character at a time, then takes the most common writer guess
    across all characters on the page as the page's writer."""
    characters = process_page(image_path)
    text = ""
    writer_votes = np.zeros(N_USERS)
    for char_image in characters:
        x = char_image.flatten()[None, :].astype(np.float64)
        char_index = network.predict(char_net, x)[0]
        writer_index = network.predict(writer_net, x)[0]
        text += CHARSET[char_index]
        writer_votes[writer_index] += 1
    return text, int(writer_votes.argmax())


def write_text(text, user_id, start_x_mm=10.0, start_y_mm=10.0, char_spacing_mm=8.0):
    """FU 2.7/2.8/3.x: looks up each character's stored trajectory and
    drives the gantry through it."""
    import motor
    motor.connect()
    x_mm = start_x_mm
    for character in text:
        points = storage.load_trajectory(f"user{user_id}", character)
        if points is None:
            print(f"No trajectory stored for '{character}', skipping")
        else:
            motor.write_trajectory(points, x_mm, start_y_mm)
        x_mm += char_spacing_mm


if __name__ == "__main__":
    print(__doc__)
    while True:
        parts = input("> ").split()
        if not parts:
            continue
        if parts[0] == "train":
            train(parts[1], parts[2], int(parts[3]))
        elif parts[0] == "classify":
            text, writer = classify(parts[1])
            print(f"Text: {text}   Writer: user{writer}")
        elif parts[0] == "write":
            write_text(parts[1], int(parts[2]))
        elif parts[0] == "quit":
            break
        else:
            print(__doc__)
