# coding:utf-8
import os

from qfluentwidgets import (
    SettingCardGroup,
    SwitchSettingCard,
    FolderListSettingCard,
    OptionsSettingCard,
    PushSettingCard,
    HyperlinkCard,
    PrimaryPushSettingCard,
    ScrollArea,
    ComboBoxSettingCard,
    ExpandLayout,
    Theme,
    CustomColorSettingCard,
    setTheme,
    setThemeColor,
    RangeSettingCard,
    isDarkTheme,
    ConfigItem,
    OptionsConfigItem,
    OptionsValidator,
    RangeConfigItem,
    RangeValidator,
    FolderListValidator,
    qconfig,
)
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import InfoBar
from PyQt5.QtCore import Qt, pyqtSignal, QUrl, QStandardPaths
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QWidget, QLabel, QFileDialog

from ..view.gallery_interface import GalleryInterface
from ..common.config import cfg, HELP_URL, FEEDBACK_URL, AUTHOR, VERSION, YEAR, isWin11
from ..common.signal_bus import signalBus
from ..common.style_sheet import StyleSheet
from ..view.folder_setting_card import FolderSettingCard

from icecream import ic


class SettingInterfaceV2(ScrollArea):
    """Setting interface"""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.scrollWidget = QWidget()
        self.expandLayout = ExpandLayout(self.scrollWidget)

        # setting label
        self.settingLabel = QLabel("Lip Reader", self)

        # 准备设置
        self.prepareSettingGroup = SettingCardGroup("准备设置", self.scrollWidget)

        self.selectFolderCard = FolderSettingCard(
            ConfigItem("", "LocalMusic", "app/lipreader", FolderListValidator()),
            "select a lip video or Folder",
            # directory=QStandardPaths.writableLocation(QStandardPaths.MusicLocation),
            parent=self.prepareSettingGroup,
        )
        self.selectFolderCard.folderChanged.connect(self.__select_video_file)

        self.hardwareCard = OptionsSettingCard(
            # cfg.themeMode,
            OptionsConfigItem("", "hardware", "CPU", OptionsValidator(["CPU", "GPU"])),
            FIF.BRUSH,
            "选择推理硬件",
            "默认使用 CPU 推理",
            texts=[
                "CPU",
                "GPU",
            ],
            parent=self.prepareSettingGroup,
        )
        self.hardwareCard.optionChanged.connect(self.__hardwareCardEvent)

        self.languageCard = ComboBoxSettingCard(
            # cfg.language,
            OptionsConfigItem(
                "", "language", "中文", OptionsValidator(["中文", "拼音", "English"])
            ),
            FIF.LANGUAGE,
            "目标语言",
            "选择唇语识别的目标语言",
            texts=["中文", "拼音", "English"],
            parent=self.prepareSettingGroup,
        )
        self.languageCard.comboBox.currentIndexChanged.connect(self.__languageCardEvent)

        # material
        self.hyperParameterGroup = SettingCardGroup("超参数设置", self.scrollWidget)
        self.clipRadiusCard = RangeSettingCard(
            RangeConfigItem("", "clipRadius", 6, RangeValidator(0, 10)),
            FIF.ALBUM,
            "clip radius",
            "根据说话人的语速调节clip参数",
            self.hyperParameterGroup,
        )
        self.clipRadiusCard.valueChanged.connect(self.__clipRadiusCardEvent)

        # update software
        self.updateSoftwareGroup = SettingCardGroup("Software update", self.scrollWidget)
        self.updateOnStartUpCard = SwitchSettingCard(
            FIF.UPDATE,
            "Check for updates when the application starts",
            "The new version will be more stable and have more features",
            configItem=cfg.checkUpdateAtStartUp,
            parent=self.updateSoftwareGroup,
        )

        self.__initWidget()

    def __initWidget(self):
        self.resize(1000, 800)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setViewportMargins(0, 80, 0, 20)
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)
        self.setObjectName("settingInterface")

        # initialize style sheet
        self.scrollWidget.setObjectName("scrollWidget")
        self.settingLabel.setObjectName("settingLabel")
        StyleSheet.SETTING_INTERFACE.apply(self)

        # initialize layout
        self.__initLayout()

    def __initLayout(self):
        self.settingLabel.move(36, 30)

        # add cards to group
        self.prepareSettingGroup.addSettingCard(self.selectFolderCard)
        # self.musicInThisPCGroup.addSettingCard(self.downloadFolderCard)

        # self.personalGroup.addSettingCard(self.micaCard)
        self.prepareSettingGroup.addSettingCard(self.hardwareCard)
        # self.personalGroup.addSettingCard(self.themeColorCard)
        # self.personalGroup.addSettingCard(self.zoomCard)
        self.prepareSettingGroup.addSettingCard(self.languageCard)

        self.hyperParameterGroup.addSettingCard(self.clipRadiusCard)

        # self.updateSoftwareGroup.addSettingCard(self.updateOnStartUpCard)
        #
        # self.aboutGroup.addSettingCard(self.helpCard)
        # self.aboutGroup.addSettingCard(self.feedbackCard)
        # self.aboutGroup.addSettingCard(self.aboutCard)

        # add setting card group to layout
        self.expandLayout.setSpacing(28)
        self.expandLayout.setContentsMargins(36, 10, 36, 0)
        self.expandLayout.addWidget(self.prepareSettingGroup)
        # self.expandLayout.addWidget(self.personalGroup)
        self.expandLayout.addWidget(self.hyperParameterGroup)
        # self.expandLayout.addWidget(self.updateSoftwareGroup)
        # self.expandLayout.addWidget(self.aboutGroup)

        # def __showRestartTooltip(self):
        """ show restart tooltip """
        # InfoBar.success(
        #     self.tr('Updated successfully'),
        #     self.tr('Configuration takes effect after restart'),
        #     duration=1500,
        #     parent=self
        # )

        # def __onDownloadFolderCardClicked(self):
        #     """ download folder card clicked slot """
        #     folder = QFileDialog.getExistingDirectory(
        #         self, self.tr("Choose folder"), "./")
        #     if not folder or cfg.get(cfg.downloadFolder) == folder:
        #         return
        #
        #     cfg.set(cfg.downloadFolder, folder)
        #     self.downloadFolderCard.setContent(folder)

        # def __connectSignalToSlot(self):
        """ connect signal to slot """
        # cfg.appRestartSig.connect(self.__showRestartTooltip)

        # music in the pc
        # self.downloadFolderCard.clicked.connect(
        #     self.__onDownloadFolderCardClicked)

        # personalization
        # cfg.themeChanged.connect(setTheme)
        # self.themeColorCard.colorChanged.connect(lambda c: setThemeColor(c))
        # self.micaCard.checkedChanged.connect(signalBus.micaEnableChanged)

        # about
        # self.feedbackCard.clicked.connect(
        #     lambda: QDesktopServices.openUrl(QUrl(FEEDBACK_URL)))

    def __hardwareCardEvent(self, hardware):
        # todo print selected value
        ic(hardware._ConfigItem__value)
        pass

    def __languageCardEvent(self, language):
        # todo
        ic(self.languageCard.comboBox.currentText())
        ic(self.languageCard.comboBox.currentIndex())

    def __clipRadiusCardEvent(self, value):
        # todo
        ic(value)

    def __select_video_file(self, folder):
        ic(folder)
