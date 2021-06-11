import sys
import signal
import os
from PySide2 import QtGui
from PySide2.QtCore import Qt, QUrl, QTimer
from PySide2.QtWidgets import (
  QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QApplication,
  QPushButton, QLabel, QMainWindow, QStyle, QSlider, QSizePolicy,
  QDialog, QLineEdit
)
from PySide2.QtMultimedia import QMediaContent, QMediaPlayer
from PySide2.QtMultimediaWidgets import QVideoWidget
from moviepy import editor
import ffmpeg # ffmpeg-python, NOT ffmpeg, NOT python-ffmpeg!!!
import matplotlib.pyplot as plt
import librosa
from datetime import datetime
from skimage import io
from skimage.transform import resize
import glob
import random
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time

sys.path.append(".")
from qrangeslider import RangeSlider

gtzan = {'format': 's16le', 'acodec': 'pcm_s16le',
         'ac': '1', 'ar': '22050'}

class TrimWidget(QWidget):
  def __init__(self, parent = None):
    super().__init__()
    self.parent = parent
    self.filename = None
    self.duration = 0
    self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
    self.vWid = QVideoWidget()

    self.playButton = QPushButton()
    self.playButton.setEnabled(False)
    self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
    self.playButton.clicked.connect(self.play)

    self.restartButton = QPushButton("Start over")
    self.restartButton.clicked.connect(self.restart)

    self.doneButton = QPushButton("Trim and extract audio")
    self.doneButton.clicked.connect(self.trim)

    self.positionSlider = QSlider(Qt.Horizontal)
    self.positionSlider.setRange(0, 0)
    self.positionSlider.sliderMoved.connect(self.setPosition)

    self.playBackTime = QLabel()
    self.trimTime = QLabel(alignment = Qt.AlignCenter)

    self.trimSlider = RangeSlider()

    self.trimSlider.mouseMoved.connect(self.updateRange)

    self.waveform = QLabel()
    self.tmpwav = ".tmp.wav"

    self.controlLayout = QHBoxLayout()
    self.controlLayout.setContentsMargins(0, 0, 0, 0)
    self.controlLayout.addWidget(self.playButton)
    self.controlLayout.addWidget(self.restartButton)
    self.controlLayout.addWidget(self.doneButton)

    self.lo = QVBoxLayout()
    self.lo.addWidget(self.vWid)
    self.lo.addWidget(self.positionSlider)
    self.lo.addWidget(self.playBackTime)
    self.lo.addWidget(self.waveform)
    self.lo.addWidget(self.trimSlider)
    self.lo.addWidget(self.trimTime)
    self.lo.addLayout(self.controlLayout)
    self.setLayout (self.lo)

    self.lo = QVBoxLayout(self)

    self.mediaPlayer.setVideoOutput(self.vWid)
    self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
    self.mediaPlayer.positionChanged.connect(self.positionChanged)
    self.mediaPlayer.durationChanged.connect(self.durationChanged)

  def restart(self):
    if self.filename is not None and self.filename != '':
      self.trimSlider.setRangeLimit(0, int(self.duration))
      self.trimSlider.setRange(0, int(self.duration))
      self.setTrimTimeLabel()
    return

  def updateRange(self, a = 0):
    self.setTrimTimeLabel()
    (start, end) = self.trimSlider.getRange()
    curpos = round(self.positionSlider.value()/1000)
    if curpos < start:
      self.setPosition(1000*start)
    if curpos > end:
      self.setPosition(1000*end)

  def play(self):
    if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
      self.mediaPlayer.pause()
    else:
      (start, end) = self.trimSlider.getRange()
      position = self.mediaPlayer.position()
      if position > 1000 * end:
        self.setPosition(1000*start)
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

    if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
      self.mediaPlayer.pause()
    self.parent.proceedToOut()
    return

  def mediaStateChanged(self, state):
    if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
      self.playButton.setIcon(
            self.style().standardIcon(QStyle.SP_MediaPause))
    else:
      self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))

  def pos_to_hms(self, position):
    total_sec = position // 1000
    s = total_sec % 60
    total_sec = total_sec // 60
    m = total_sec % 60
    total_sec = total_sec // 60
    h = total_sec
    return (h, m, s)

  def setTrimTimeLabel(self):
    (start, end) = self.trimSlider.getRange()
    (h,m,s) = self.pos_to_hms(1000*start)
    (ht, mt, st) = self.pos_to_hms(1000*end)
    self.trimTime.setText(
      "Selected Interval: %02d:%02d:%02d - %02d:%02d:%02d"%(h,m,s,ht,mt,st)
    )

  def positionChanged(self, position):
    (h,m,s) = self.pos_to_hms(position)
    (ht, mt, st) = self.pos_to_hms(1000*self.duration)
    self.playBackTime.setText("%02d:%02d:%02d/%02d:%02d:%02d"%(h,m,s,ht,mt,st))
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
      # self.setPosition (position-1)

    self.positionSlider.setValue(position)

  def setPosition(self, position):
    (start, end) = self.trimSlider.getRange()
    start *= 1000
    end *= 1000
    if position < start:
      position = start
    if position > end:
      position = end
    self.mediaPlayer.setPosition(position)

  def durationChanged(self, duration):
    self.setTrimTimeLabel()
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

  def getTrimPointsHms(self):
    (start, end) = self.trimSlider.getRange()
    return (self.pos_to_hms(1000*start), self.pos_to_hms(1000*end))


class Login(QDialog):
  def __init__(self, parent=None):
    super(Login, self).__init__(parent)
    self.intro = QLabel("Please enter gmail credentials")
    self.textName = QLineEdit(self)
    self.textPass = QLineEdit(self)
    self.textPass.setEchoMode(QLineEdit.Password)
    self.buttonLogin = QPushButton('Login', self)
    self.buttonLogin.clicked.connect(parent.handleLogin)
    layout = QVBoxLayout(self)
    layout.addWidget(self.intro)
    layout.addWidget(self.textName)
    layout.addWidget(self.textPass)
    layout.addWidget(self.buttonLogin)

class OutWidget(QWidget):
  def __init__(self, parent = None, filename = None):
    super().__init__()
    self.parent = parent
    self.filename = filename
    self.thumbnailPath = None
    self.thumbnail = QLabel()
    self.positionsLabel = None
    self.predictionLabel = None
    self.lo = QVBoxLayout()
    self.outs = None
    self.loginButton = QPushButton(
      "Click here to log in to YouTube and get similar music recommendations"
      )
    self.loginDialog = None
    self.gtzan_genre_map = {0: "blues", 1: "classical", 
        2:"country", 3:"disco", 4:"hiphop", 5: "jazz", 6:"metal",
        7:"pop", 8: "reggae", 9: "rock"}
    self.yt_genre_map = {0: "alternative, punk", 1: "ambient",
     2:"children's", 3:"cinematic", 4:"classical", 5: "country, folk",
     6: "dance, electronic", 7: "hiphop, rap", 8: "holiday", 9:"jazz, blues",
     10:"pop", 11:"RnB, soul", 12: "reggae", 13: "rock"}
    self.gtzan_to_yt = {0:9, 1:4, 2:5, 3:6, 4:7, 5:9, 6:13, 7:10, 8:12, 9:13}
    self.loadText = "Please wait while YT audiolibrary loads"
    self.errText = "something went wrong. Please try again"

  def login(self):
    self.loginDialog = Login(parent = self)
    self.loginDialog.show()

  def handleLogin(self):
    l = self.loginDialog.textName
    p = self.loginDialog.textPass
    self.loginDialog.setVisible(False)
    self.outs.setText(self.loadText)
    self.outs.show()
    sys.path.append("./chromedriver")
    browser = webdriver.Chrome()
    # browser.minimize_window()
    browser.get("https://youtube.com/audiolibrary")
    try:
      self.loginButton.setVisible(False)
      login = browser.find_element_by_id("identifierId")
      login.send_keys(l)
      login.send_keys(Keys.RETURN)
      browser.implicitly_wait(5)
      pas = browser.find_element_by_name("password")
      pas.send_keys(p)
      pas.send_keys(Keys.RETURN)

      browser.implicitly_wait(5)
      time.sleep(1)
    except:
      self.loginButton.setVisible(True)
      print ("already logged in")

    try:
      filter = browser.find_element_by_id("filter-icon")
      filter.click()
      noAttr = browser.find_element_by_id("text-item-8")
      noAttr.click()

      filter = browser.find_element_by_id("filter-icon")
      filter.click()
      genreButton = browser.find_element_by_id("text-item-3")
      genreButton.click()
      prediction = self.prediction
      prediction = self.gtzan_to_yt[prediction]
      predGenre = browser.find_element_by_id("checkbox-"+str(prediction))
      predGenre.click()
      appl = browser.find_element_by_id("apply-button")
      appl.click()
      time.sleep(2)

      titles = []
      titleElems = browser.find_elements_by_css_selector("div#title")
      for title in titleElems[:3]:
        titles.append(title.text)

      self.outs.setText(
        "Recommended songs are: {}\nin genre: {}".format(
          ", ".join(titles), self.yt_genre_map[prediction]))
      browser.quit()
    except:
      self.loginButton.setVisible(True)
      self.outs.setText(self.errText)
      browser.quit()

  def onShow (self):
    self.thumbnailPath = self.createThumbnail()
    self.im = io.imread(self.thumbnailPath)
    self.im = self.downscale(self.im)
    io.imsave(self.thumbnailPath, self.im)

    pm = QtGui.QPixmap(self.thumbnailPath)
    self.thumbnail.setPixmap(pm)
    self.thumbnail.show()
    (st, en) = self.parent.trimVidWid.getTrimPointsHms()
    self.positionsLabel = QLabel("Audio at %02d:%02d:%02d - %02d:%02d:%02d"%(
        st[0],st[1],st[2],en[0],en[1],en[2]
        ))

    self.prediction = self.getModelPrediction()
    prediction = self.gtzan_genre_map[self.prediction]
    self.predictionLabel = QLabel()
    self.predictionLabel.setText("Predicted genre: {}".format(prediction))
    self.outs = QLabel("")

    self.lo.addWidget(self.thumbnail)
    self.lo.addWidget(self.positionsLabel)
    self.lo.addWidget(self.predictionLabel)
    self.lo.addWidget(self.loginButton)
    self.lo.addWidget(self.outs)
    self.lo.setAlignment(Qt.AlignCenter)
    self.setLayout(self.lo)
    self.loginButton.clicked.connect(self.login)

  def setFilename(self, filename):
    self.filename = filename


  def createThumbnail(self):
    orig_stream = ffmpeg.input(self.filename)
    orig_stream.output('.thumbnail.png', vframes=1).overwrite_output().run()
    return '.thumbnail.png'

  def downscale(self, image):
    return resize(image, (90, 160), anti_aliasing=False)

  def getModelPrediction(self):
    return random.randint(0, 9)

class MainWindow(QMainWindow):
  def __init__(self, parent = None):
    super().__init__()

    self.importWid = QWidget(self)
    self.trimVidWid = TrimWidget(parent = self)
    
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
    self.outWid = OutWidget(parent = self)


  def proceedToTrim(self):
    self.setWindowTitle("Trim Video")
    if self.importWid and self.importWid is not None and self.importWid.isVisible():
      self.importWid.setVisible(False)
    self.setCentralWidget(self.trimVidWid)
    self.trimVidWid.show()

  def proceedToOut(self):
    self.setWindowTitle("Results")
    if self.trimVidWid and self.trimVidWid.isVisible():
      self.trimVidWid.setVisible(False)
    self.setCentralWidget(self.outWid)
    self.outWid.onShow()
    self.outWid.show()

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
      self.outWid.setFilename(self.filename)
      self.proceedToTrim()

  def cleanup(self):
    filelist = glob.glob(".tmp.*") + \
      glob.glob("trimmed-*") + [".thumbnail.png"]
    for file in filelist:
      try:
        os.remove(file)
      except:
        continue
    QApplication.quit()
  
  def sigint_handler(self, signum, frame):
    self.cleanup()
    print()
    


class UI():
  def __init__(self):
    app = QApplication([])

    self.wind = MainWindow()
    signal.signal(signal.SIGINT, self.wind.sigint_handler)
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)
    self.wind.resize(1200, 700)
    self.wind.show()
    app.lastWindowClosed.connect(self.exitHandler)
    app.aboutToQuit.connect(self.exitHandler)

    sys.exit(app.exec_())

  def exitHandler(self):
    self.wind.cleanup()

def main():
  ui = UI()


if __name__ == '__main__':
  main()