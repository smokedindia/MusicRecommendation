import sys
import signal
import os
from PySide2 import QtGui
from PySide2.QtCore import Qt, QUrl
from PySide2.QtWidgets import (
  QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QApplication,
  QPushButton, QLabel, QMainWindow, QStyle, QSlider, QSizePolicy
)
from PySide2.QtMultimedia import QMediaContent, QMediaPlayer
from PySide2.QtMultimediaWidgets import QVideoWidget
from moviepy import editor
import ffmpeg # ffmpeg-python, NOT ffmpeg, NOT python-ffmpeg!!!
import matplotlib.pyplot as plt
import librosa
from datetime import datetime

sys.path.append(".")
from qrangeslider import RangeSlider

gtzan = {'format': 's16le', 'acodec': 'pcm_s16le',
         'ac': '1', 'ar': '22050'}

class TrimWidget(QWidget):
  def __init__(self, parent = None):
    super().__init__()
    self.filename = None
    self.duration = 0
    self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
    self.vWid = QVideoWidget()

    self.playButton = QPushButton()
    self.playButton.setEnabled(False)
    self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
    self.playButton.clicked.connect(self.play)

    self.importButton = QPushButton("Replace video...")
    self.importButton.clicked.connect(self.open)

    self.doneButton = QPushButton("Trim and extract audio")
    self.doneButton.clicked.connect(self.trim)

    self.positionSlider = QSlider(Qt.Horizontal)
    self.positionSlider.setRange(0, 0)
    self.positionSlider.sliderMoved.connect(self.setPosition)

    self.trimSlider = RangeSlider()

    self.trimSlider.mouseMoved.connect(self.updateRange)

    self.waveform = QLabel()
    self.tmpwav = ".tmp.wav"

    controlLayout = QHBoxLayout()
    controlLayout.setContentsMargins(0, 0, 0, 0)
    controlLayout.addWidget(self.playButton)
    controlLayout.addWidget(self.importButton)
    controlLayout.addWidget(self.doneButton)

    self.lo = QVBoxLayout()
    self.lo.addWidget(self.vWid)
    self.lo.addWidget(self.positionSlider)
    self.lo.addWidget(self.waveform)
    self.lo.addWidget(self.trimSlider)
    self.lo.addLayout(controlLayout)
    self.setLayout (self.lo)

    self.lo = QVBoxLayout(self)

    self.mediaPlayer.setVideoOutput(self.vWid)
    self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
    self.mediaPlayer.positionChanged.connect(self.positionChanged)
    self.mediaPlayer.durationChanged.connect(self.durationChanged)

  def open(self):
    self.filename = QFileDialog.getOpenFileName(
      self,
      "Open Video File",
      os.getcwd(),
      filter = "Video Files [*.mp4, *.mov, *.avi, *.mkv, *.3gp, *.webm]\
       (*.mp4 *.mov *.avi *.mkv *.3gp *.webm)"
      )[0]
    if self.filename is not None and self.filename != '':
      self.setFilename(self.filename)

  def updateRange(self, a):
    (start, end) = self.trimSlider.getRange()
    curpos = round(self.positionSlider.value()/1000)
    # print ("{}, {}, {}".format(curpos, start, end))
    if curpos < start:
      self.setPosition(1000*start)
    if curpos > end:
      self.setPosition(1000*end)

  def play(self):
    if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
      self.mediaPlayer.pause()
    else:
      self.mediaPlayer.play()

  def trim(self):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    (start, end) = self.trimSlider.getRange()
    orig_stream = ffmpeg.input(self.filename)

    aud = (
      orig_stream.audio
      .filter_('atrim', start = start, end = end)
      .filter_('asetpts', 'PTS-STARTPTS')
    )

    ffmpeg.output(aud,
                  "trimmed-({}-{})-{}.wav".format(start, end, current_time),
                  acodec = gtzan['acodec'],
                  ac = gtzan['ac'],
                  ar = gtzan['ar']).run()
    return

  def mediaStateChanged(self, state):
    if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
      self.playButton.setIcon(
            self.style().standardIcon(QStyle.SP_MediaPause))
    else:
      self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))

  def positionChanged(self, position):
    (start, end) = self.trimSlider.getRange()
    start = 1000 * start
    if start > 0:
      start -= 500
    end = 1000 * end + 500
    end = min (end, self.duration * 1000)
    position = max(position, start)
    if position > end:
      position = end
      if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
        self.mediaPlayer.pause()

    self.positionSlider.setValue(position)

  def setPosition(self, position):
    self.mediaPlayer.setPosition(position)

  def durationChanged(self, duration):
    self.positionSlider.setRange(0, duration)

  def setFilename(self, filename):
    self.filename = filename
    self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(filename)))
    self.playButton.setEnabled(True)
    self.duration = editor.VideoFileClip(filename).duration
    self.trimSlider.setRangeLimit(0, int(self.duration))
    self.trimSlider.setRange(0, int(self.duration))
    orig_stream = ffmpeg.input(self.filename)
    aud = (orig_stream.audio)

    ffmpeg.output(aud, self.tmpwav,
                  acodec = gtzan['acodec'],
                  ac = gtzan['ac'],
                  ar = gtzan['ar'],
                  ).overwrite_output().run()
    y, sr = librosa.load(self.tmpwav)
    fig = plt.figure(
      1, figsize=(13, 0.5), 
      dpi = 100, facecolor = "black", frameon = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.plot(y, "g")
    plt.savefig(".tmp.png")

    pm = QtGui.QPixmap(".tmp.png")
    self.waveform.setPixmap(pm)
    self.waveform.show()


class MainWindow(QMainWindow):
  def __init__(self, parent = None):
    super().__init__()

    self.importWid = QWidget(self)
    self.trimVidWid = TrimWidget()
    
    self.setWindowTitle("Import Video")
    
    self.importLabel = QLabel(
      "Welcome! To start, click 'Import Video...' button below",
      alignment=Qt.AlignCenter
      )
    self.importButton = QPushButton("Import video...")
    self.importButton.clicked.connect(self.call_file_dialog)

    self.filename = None

    self.setCentralWidget(self.importWid)
    layout = QVBoxLayout(self)
    layout.addWidget(self.importLabel)
    layout.addWidget(self.importButton)
    self.importWid.setLayout(layout)
    self.importWid.show()


  def proceedToTrim(self):
    self.setWindowTitle("Trim Video")
    self.importWid.setVisible(False)
    self.setCentralWidget(self.trimVidWid)
    self.trimVidWid.show()



  def call_file_dialog(self):
    self.filename = QFileDialog.getOpenFileName(
      self,
      "Open Video File",
      os.getcwd(),
      filter = "Video Files [*.mp4, *.mov, *.avi, *.mkv, *.3gp, *.webm]\
       (*.mp4 *.mov *.avi *.mkv *.3gp *.webm)"
      )[0]
    if self.filename is not None and self.filename != '':
      self.trimVidWid.setFilename(self.filename)
      self.proceedToTrim()
      


def main():
  app = QApplication([])
  signal.signal(signal.SIGINT, signal.SIG_DFL)


  wind = MainWindow()
  wind.resize(1200, 700)
  wind.show()

  sys.exit(app.exec_())

if __name__ == '__main__':
  main()