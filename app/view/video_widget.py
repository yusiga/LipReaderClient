# coding:utf-8
from PyQt5.QtCore import Qt, pyqtSignal, QUrl, QSizeF, QTimer
from PyQt5.QtGui import QPainter
from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt5.QtWidgets import QWidget, QGraphicsView, QVBoxLayout, QGraphicsScene
from icecream import ic
from qfluentwidgets import FluentStyleSheet
from qfluentwidgets.multimedia import StandardMediaPlayBar, SimpleMediaPlayBar

from ..common.config import isWin11


# from ..common.style_sheet import FluentStyleSheet
# from .media_play_bar import StandardMediaPlayBar


class GraphicsVideoItem(QGraphicsVideoItem):
    """ Graphics video item """

    def paint(self, painter: QPainter, option, widget):
        # Windows 11 修复：移除 CompositionMode_Difference，使用正常绘制模式
        # CompositionMode_Difference 在 Windows 11 上会导致视频无法正常显示（白板问题）
        if isWin11():
            # Windows 11: 使用正常绘制模式，确保视频画面正常显示
            super().paint(painter, option, widget)
        else:
            # Windows 10 及其他系统: 保持原有行为
            painter.setCompositionMode(QPainter.CompositionMode_Difference)
            super().paint(painter, option, widget)


class VideoWidget(QGraphicsView):
    """ Video widget """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.isHover = False
        self.timer = QTimer(self)

        self.vBoxLayout = QVBoxLayout(self)
        self.videoItem = QGraphicsVideoItem()
        self.graphicsScene = QGraphicsScene(self)
        self.playBar = SimpleMediaPlayBar(self)

        self.setMouseTracking(True)
        self.setScene(self.graphicsScene)
        self.graphicsScene.addItem(self.videoItem)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        self.player.setVideoOutput(self.videoItem)
        FluentStyleSheet.MEDIA_PLAYER.apply(self)

        self.timer.timeout.connect(self._onHideTimeOut)

        # add video status timer
        self.videoStatusTimer = QTimer(self)
        self.videoStatusTimer.timeout.connect(self.__onVideoStatusChangeEvent)
        self.videoStatusTimer.start(1000)

    def __onVideoStatusChangeEvent(self):
        if not self.playBar.player.isPlaying():
            self.playBar.player.play()
            self.playBar.player.pause()

    def setVideo(self, url: QUrl):
        """ set the video to play """
        self.player.setSource(url)
        self.fitInView(self.videoItem, Qt.KeepAspectRatio)

    def hideEvent(self, e):
        self.pause()
        e.accept()

    def wheelEvent(self, e):
        return

    def enterEvent(self, e):
        self.isHover = True
        self.playBar.fadeIn()

    def leaveEvent(self, e):
        self.isHover = False
        self.timer.start(3000)

    def _onHideTimeOut(self):
        if not self.isHover:
            self.playBar.fadeOut()

    def play(self):
        self.playBar.play()

    def pause(self):
        self.playBar.pause()

    def stop(self):
        self.playBar.stop()

    def togglePlayState(self):
        """ toggle play state """
        if self.player.isPlaying():
            self.pause()
        else:
            self.play()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.videoItem.setSize(QSizeF(self.size()))
        self.fitInView(self.videoItem, Qt.KeepAspectRatio)
        self.playBar.move(0, 0)
        self.playBar.setFixedSize(self.width(), self.playBar.height())

    @property
    def player(self):
        return self.playBar.player
