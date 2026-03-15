# coding:utf-8
import os
from ctypes import pythonapi

from PyQt5.QtCore import Qt, QUrl, QEvent, QStandardPaths, QSizeF, right, QThread
from PyQt5.QtWidgets import (
    QListWidgetItem,
    QFrame,
    QTreeWidgetItem,
    QHBoxLayout,
    QTreeWidgetItemIterator,
    QTableWidgetItem,
    QWidget,
    QVBoxLayout,
    QFileDialog,
    QLabel,
    QSizePolicy,
)
from PyQt5.QtGui import QDesktopServices, QFont
from fontTools.fontBuilder import FontBuilder
from qfluentwidgets import (
    TreeWidget,
    TableWidget,
    ListWidget,
    PrimaryPushButton,
    LineEdit,
    HorizontalFlipView,
    IconWidget,
    StrongBodyLabel,
    BodyLabel,
    ScrollArea,
    FolderListSettingCard,
    SettingCardGroup,
    SwitchSettingCard,
    RangeSettingCard,
    RangeConfigItem,
    RangeValidator,
    ExpandLayout,
    FolderListValidator,
    ConfigItem,
    OptionsSettingCard,
    OptionsValidator,
    OptionsConfigItem,
    ComboBoxSettingCard,
    TitleLabel,
    PrimaryPushSettingCard,
    BoolValidator,
    ProgressRing,
    IndeterminateProgressRing,
    ProgressBar,
    IndeterminateProgressBar,
)
from qfluentwidgets import FluentIcon as FIF
from torch.distributed.elastic.agent.server import Worker

# from qfluentwidgets.multimedia import VideoWidget, SimpleMediaPlayBar

from ..view.folder_setting_card import FolderSettingCard, FileSettingCard

# from .gallery_interface import GalleryInterface
from ..view.gallery_clear_interface import GalleryInterface
from .status_info_interface import ProgressWidget
from ..common.translator import Translator
from ..common.style_sheet import StyleSheet
from ..lipreader.utils.util import VideoProcessUtil, ModelInferenceUtil
from ..common.config import cfg, HELP_URL, FEEDBACK_URL, AUTHOR, VERSION, YEAR, isWin11
from ..lipreader.video_process import read_lips

from icecream import ic
from ..view.video_widget import VideoWidget
import torch
import numpy as np


class ShowInterface(GalleryInterface):
    """Lip Reader interface"""

    def __init__(self, parent=None):
        super().__init__(title="Lip Reader", subtitle="", parent=parent)
        self.setObjectName("showInterface")
        self.scrollWidget = QWidget()
        self.__initPrepareGroup()
        self.__initHyperGroup()
        self.__initResultGroup()
        self.__initVideoGroup()
        self.__initProcessGroup()
        self.__initTableGroup()

    def __initPrepareGroup(self):
        # file select
        titleLabel = TitleLabel("Show Demo", self)
        self.addExampleCard("", titleLabel, alignment=Qt.AlignHCenter)

        self.selectFileCard = FileSettingCard(
            ConfigItem("", "file", "", FolderListValidator()), "lip video file", "选择唇语视频文件"
        )
        self.selectFileCard.folderChanged.connect(self.__select_video_file)

        self.showOrHideCard = SwitchSettingCard(
            FIF.UPDATE,
            "显示/隐藏 设置",
            "",
            ConfigItem("Update", "CheckUpdateAtStartUp", False, BoolValidator()),
        )
        self.showOrHideCard.checkedChanged.connect(self.__showOrHideCardEvent)

        self.prepareSettingGroup = SettingCardGroup("准备设置", self.scrollWidget)
        self.prepareSettingGroup.addSettingCard(self.selectFileCard)
        self.prepareSettingGroup.addSettingCard(self.showOrHideCard)
        self.addExampleCard("", self.prepareSettingGroup, 1, Qt.AlignTop)

    def __initShowOrHideGroup(self):
        self.showOrHideCard = SwitchSettingCard(
            FIF.UPDATE,
            "显示/隐藏 设置",
            "",
            ConfigItem("Update", "CheckUpdateAtStartUp", False, BoolValidator()),
        )
        self.showOrHideCard.checkedChanged.connect(self.__showOrHideCardEvent)
        self.addExampleCard("", self.showOrHideCard, 1, Qt.AlignTop)

    def __initVideoGroup(self):
        self.videoWidget = VideoWidget()
        self.videoWidget.setFixedHeight(400)
        self.videoWidget.setFixedWidth(400)
        self.addExampleCard("", self.videoWidget, 0, Qt.AlignHCenter)
        self.videoWidget.hide()

    def __initHyperGroup(self):
        self.hardwareCard = OptionsSettingCard(
            OptionsConfigItem("", "hardware", "cuda", OptionsValidator(["cpu", "cuda"])),
            FIF.BRUSH,
            "选择推理硬件",
            "默认使用 GPU 推理",
            texts=[
                "CPU",
                "GPU",
            ],
        )
        self.hardwareCard.optionChanged.connect(self.__hardwareCardEvent)

        self.languageCard = ComboBoxSettingCard(
            OptionsConfigItem("", "language", "hz", OptionsValidator(["hz", "py", "english"])),
            FIF.LANGUAGE,
            "目标语言",
            "选择唇语识别的目标语言",
            texts=["中文", "拼音", "English"],
        )
        self.languageCard.comboBox.currentIndexChanged.connect(self.__languageCardEvent)

        self.hyperParameterGroup = SettingCardGroup("超参数设置", self.scrollWidget)
        self.hyperParameterGroup.addSettingCard(self.hardwareCard)
        self.hyperParameterGroup.addSettingCard(self.languageCard)

        self.addExampleCard("", self.hyperParameterGroup, 1, Qt.AlignTop)

        if not self.showOrHideCard.configItem._ConfigItem__value:
            self.hyperParameterGroup.hide()

    def __initResultGroup(self):
        resultLabel = TitleLabel("Inference Result", self)
        self.addExampleCard("", resultLabel, alignment=Qt.AlignHCenter)

        # start lipreading
        self.startCard = PrimaryPushSettingCard(
            "开始",  # right button name
            FIF.UPDATE,
            "开始识别",
            "句子级唇语识别",
        )
        self.startCard.clicked.connect(self.__asyncStartEvent)
        self.addExampleCard("", self.startCard, 1, Qt.AlignTop)

    def __initProcessGroup(self):
        self.ring = ProgressRing(self)
        self.ring.setFixedSize(70, 70)
        self.ring.setStrokeWidth(10)
        self.ring.setTextVisible(True)
        self.addExampleCard("", self.ring)
        self.ring.hide()

    def __initTableGroup(self):
        self.table = TableFrame(self)
        self.addExampleCard("", self.table)
        self.table.hide()

    def __select_video_file(self):
        if len(self.selectFileCard.folders) == 0:
            return
        video_file = self.selectFileCard.folders[0]
        self.videoWidget.setVideo(QUrl.fromLocalFile(video_file))
        self.videoWidget.show()
        pass

    def __hardwareCardEvent(self):
        # ic(self.hardwareCard.configItem._ConfigItem__value)
        pass

    def __languageCardEvent(self):
        # ic(self.languageCard.configItem._ConfigItem__value)
        pass

    def __showOrHideCardEvent(self):
        if self.showOrHideCard.configItem._ConfigItem__value:
            self.hyperParameterGroup.show()
        else:
            self.hyperParameterGroup.hide()
        pass

    def __asyncStartEvent(self):
        self.worker = QThread()
        self.worker.run = self.__asyncRunningFunc
        self.worker.start()
        self.worker.finished.connect(self.__asyncFinishedFunc)

    def __asyncFinishedFunc(self):
        self.ring.hide()
        self.table.show()
        self.table.showData(self.predict_text[0])

    def __asyncRunningFunc(self):
        self.ring.show()
        # 1 prepare
        device = self.hardwareCard.configItem._ConfigItem__value
        language = self.languageCard.configItem._ConfigItem__value
        # get video file
        video_file_list = [file for file in self.selectFileCard.folders if file.endswith(".mp4")]
        self.ring.setValue(10)

        # 2 lip
        frame_list = [VideoProcessUtil.get_frame(file) for file in video_file_list]
        ic("get frame list")
        self.ring.setValue(20)
        landmark_list = [VideoProcessUtil.get_landmarks(frame) for frame in frame_list]
        ic("get landmark list")
        self.ring.setValue(50)
        lip_list = [
            VideoProcessUtil.get_lips(frame, landmark)
            for frame, landmark in zip(frame_list, landmark_list)
        ]
        ic("get lip list")
        self.ring.setValue(60)

        # 3 mask
        video_lip, video_mask = ModelInferenceUtil.prepare_data(
            lip_list
        )  # (B, C, T, H, W) | (B, T)
        self.ring.setValue(80)

        # 4 inference
        predict_text = ModelInferenceUtil.lipreading_inference(
            video_lip, video_mask, device, language
        )
        self.ring.setValue(100)

        # 5 show table
        ic(predict_text)
        self.predict_text = predict_text
        # table = TableFrame(self)
        # self.addExampleCard("", table)
        # todo
        # self.ring.hide()
        # self.table.show()
        # self.table.showData(predict_text[0])


class TableFrame(TableWidget):
    def showData(self, predict_text):
        self.verticalHeader().hide()
        self.horizontalHeader().hide()
        self.setBorderRadius(10)
        self.setBorderVisible(True)

        self.setColumnCount(1)
        self.setRowCount(1)

        self.setColumnWidth(0, 800)
        self.setRowHeight(0, 70)
        self.setFixedSize(800, 70)

        item = QTableWidgetItem(predict_text)
        item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        font = QFont()
        font.setPointSize(16)
        font.setFamily("KaiTi")  # KaiTi | SimSun | FangSong | Microsoft YaHei | SimHei
        item.setFont(font)
        self.setItem(0, 0, item)
