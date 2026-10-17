YOLO11 Heatmap and People Counting

Overview

This project uses Ultralytics YOLO11 to track people in a video, generate a movement heatmap, and count crossings of a horizontal line.

Project files

main.py: tracks people and generates a heatmap image and an annotated video.

count.py: counts detected people crossing a horizontal line.

input.mp4: input video required to run the scripts.

yolo11n.pt: YOLO11 model weights required by the scripts.

requirements.txt: Python dependencies.

Requirements

Python 3.11 recommended

A virtual environment

The packages listed in requirements.txt

Installation

Create and activate a virtual environment:

py -3.11 -m venv .venv .\.venv\Scripts\Activate.ps1 

Install the dependencies:

python -m pip install --upgrade pip python -m pip install -r requirements.txt 

Ensure that input.mp4 and yolo11n.pt are available in the project root.

Run the heatmap and tracking script

python main.py 

Expected outputs:

output/tracked.mp4

output/store_heatmap.jpg

Run the people counting script

python count.py 

The program displays the counts in the video window and prints the final statistics in the terminal.

Press q to stop video processing early.

Notes

The scripts use the default YOLO person class and process the provided input video. Counting accuracy depends on the video, tracking quality, and the position of the counting line.