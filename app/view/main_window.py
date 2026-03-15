# coding: utf-8
from PyQt5.QtCore import QUrl, QSize, QTimer
from PyQt5.QtGui import QIcon, QDesktopServices, QColor
from PyQt5.QtWidgets import QApplication, QWidget

from qfluentwidgets import (
    NavigationAvatarWidget,
    NavigationItemPosition,
    MessageBox,
    FluentWindow,
    SplashScreen,
    SystemThemeListener,
    isDarkTheme,
)
from qfluentwidgets import FluentIcon as FIF

from .gallery_interface import GalleryInterface

# from ..recycled.home_interface import HomeInterface

from .setting_interface import SettingInterface

from .lipreading_interface import LipReadingInterface
from .visual_kws_interface import VisualKwsInterface

from ..common.config import ZH_SUPPORT_URL, EN_SUPPORT_URL, cfg
from ..common.icon import Icon
from ..common.signal_bus import signalBus
from ..common.translator import Translator
from ..common import resource


class MainWindow(FluentWindow):

    def __init__(self):
        super().__init__()
        self.initWindow()

        # create system theme listener
        self.themeListener = SystemThemeListener(self)

        # create sub interface（句子级第一页，关键字检测第二页）
        self.lipReadingInterface = LipReadingInterface(self)
        self.visualKwsInterface = VisualKwsInterface(self)
        self.settingInterface = SettingInterface(self)

        # enable acrylic effect
        self.navigationInterface.setAcrylicEnabled(True)

        self.connectSignalToSlot()

        # add items to navigation interface
        self.initNavigation()
        self.splashScreen.finish()

        # start theme listener
        self.themeListener.start()

    def connectSignalToSlot(self):
        signalBus.micaEnableChanged.connect(self.setMicaEffectEnabled)
        signalBus.switchToSampleCard.connect(self.switchToSample)
        signalBus.supportSignal.connect(self.onSupport)

    def initNavigation(self):
        # add navigation items
        t = Translator()
        # 左侧导航栏-1
        # self.addSubInterface(self.homeInterface, FIF.HOME, self.tr("Home"))  # 主页
        self.navigationInterface.addSeparator()  # 左侧导航栏之间的分割线
        self.addSubInterface(self.lipReadingInterface, FIF.MESSAGE, "句子级唇语识别")
        self.addSubInterface(self.visualKwsInterface, Icon.TEXT, "唇语关键字检测")

        # why not show
        self.addSubInterface(
            self.settingInterface, FIF.SETTING, self.tr("设置"), NavigationItemPosition.BOTTOM
        )

    def initWindow(self):
        self.resize(1200, 800)  # (w, h)
        self.setMinimumWidth(760)
        self.setWindowIcon(QIcon(":/gallery/images/logo.png"))
        self.setWindowTitle("MARS实验室中文唇语识别系统")

        # create splash screen
        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(106, 106))
        self.splashScreen.raise_()

        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)
        self.show()
        QApplication.processEvents()

    def onSupport(self):
        language = cfg.get(cfg.language).value
        if language.name() == "zh_CN":
            QDesktopServices.openUrl(QUrl(ZH_SUPPORT_URL))
        else:
            QDesktopServices.openUrl(QUrl(EN_SUPPORT_URL))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, "splashScreen"):
            self.splashScreen.resize(self.size())

    def closeEvent(self, e):
        self.themeListener.terminate()
        self.themeListener.deleteLater()
        super().closeEvent(e)

    def _onThemeChangedFinished(self):
        super()._onThemeChangedFinished()

        # retry
        if self.isMicaEffectEnabled():
            QTimer.singleShot(
                100, lambda: self.windowEffect.setMicaEffect(self.winId(), isDarkTheme())
            )

    def switchToSample(self, routeKey, index):
        """switch to sample"""
        interfaces = self.findChildren(GalleryInterface)
        for w in interfaces:
            if w.objectName() == routeKey:
                self.stackedWidget.setCurrentWidget(w, False)
                w.scrollToCard(index)
