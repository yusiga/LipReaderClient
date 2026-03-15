# coding:utf-8
import os

from PyQt5.QtCore import Qt, QUrl, QEvent, QStandardPaths
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
)
from PyQt5.QtGui import QDesktopServices, QFont
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
)
from qfluentwidgets import FluentIcon as FIF
from ..view.folder_setting_card import FolderSettingCard, FileSettingCard

# from .gallery_interface import GalleryInterface
from ..view.gallery_clear_interface import GalleryInterface
from ..common.translator import Translator
from ..common.style_sheet import StyleSheet
from ..lipreader.utils.util import VideoProcessUtil, ModelInferenceUtil
from ..common.config import cfg, HELP_URL, FEEDBACK_URL, AUTHOR, VERSION, YEAR, isWin11
from ..lipreader.video_process import read_lips

from icecream import ic
import torch
import numpy as np


class LipReaderInterface(GalleryInterface):
    """Lip Reader interface"""

    def __init__(self, parent=None):
        # define attribute
        # self.device = "cpu"
        # self.language = "hz"
        # t = Translator()
        super().__init__(title="Lip Reader", subtitle="", parent=parent)
        self.setObjectName("lipReaderInterface")

        # file select
        titleLabel = TitleLabel("Lip Reader", self)
        self.addExampleCard("", titleLabel, alignment=Qt.AlignHCenter)
        # font = QFont()
        # font.setPointSize(16)
        # titleLabel.setFont(font)

        self.selectFileCard = FileSettingCard(
            ConfigItem("", "folder", "", FolderListValidator()),
            "lip video file",
            "选择唇语视频文件",
            # directory=QStandardPaths.writableLocation(QStandardPaths.MusicLocation),
        )
        self.selectFileCard.folderChanged.connect(self.__select_video_file)

        self.selectFolderCard = FolderSettingCard(
            ConfigItem("", "folder", "", FolderListValidator()),
            "lip video folder",
            "选择唇语视频文件夹",
            # directory=QStandard   Paths.writableLocation(QStandardPaths.MusicLocation),
        )
        self.selectFolderCard.folderChanged.connect(self.__select_video_folder)

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
            # cfg.language,
            OptionsConfigItem("", "language", "hz", OptionsValidator(["hz", "py", "english"])),
            FIF.LANGUAGE,
            "目标语言",
            "选择唇语识别的目标语言",
            texts=["中文", "拼音", "English"],
            # parent=self.prepareSettingGroup
        )
        self.languageCard.comboBox.currentIndexChanged.connect(self.__languageCardEvent)

        self.scrollWidget = QWidget()
        self.prepareSettingGroup = SettingCardGroup("准备设置", self.scrollWidget)
        self.prepareSettingGroup.addSettingCard(self.selectFileCard)
        self.prepareSettingGroup.addSettingCard(self.selectFolderCard)
        self.prepareSettingGroup.addSettingCard(self.hardwareCard)
        self.prepareSettingGroup.addSettingCard(self.languageCard)
        self.addExampleCard("", self.prepareSettingGroup, 1, Qt.AlignTop)

        # hyper parameter
        self.clipRadiusCard = RangeSettingCard(
            RangeConfigItem("", "clipRadius", 6, RangeValidator(0, 10)),
            FIF.ALBUM,
            "clip",
            "根据说话人的语速调节clip参数",
            # self.hyperParameterGroup
        )
        self.clipRadiusCard.valueChanged.connect(self.__clipRadiusCardEvent)
        self.hyperParameterGroup = SettingCardGroup("超参数设置", self.scrollWidget)
        self.hyperParameterGroup.addSettingCard(self.clipRadiusCard)
        self.addExampleCard("", self.hyperParameterGroup, 1, Qt.AlignTop)

        resultLabel = TitleLabel("Inference Result", self)
        self.addExampleCard("", resultLabel, alignment=Qt.AlignHCenter)

        # start lipreading
        self.startCard = PrimaryPushSettingCard(
            "开始",  # right button name
            FIF.UPDATE,
            "开始识别",
            "句子级唇语识别",
        )
        self.startCard.clicked.connect(self.__startCardEvent)
        self.addExampleCard("", self.startCard, 1, Qt.AlignTop)

        # text show
        # self.lineEdit = LineEdit(self)
        # self.lineEdit.setText("decode text!")
        # self.lineEdit.setClearButtonEnabled(True)
        # self.addExampleCard(
        #     title=self.tr("Decoded Text:"),
        #     widget=self.lineEdit,
        #     stretch=1,
        #     alignment=Qt.AlignTop
        # )

        # table view
        # self.lipreadingTable = TableFrame(self)
        # self.addExampleCard(title="", widget=self.lipreadingTable)

    def __select_video_file(self):
        ic(self.selectFileCard.folders)
        pass

    def __select_video_folder(self):
        ic(self.selectFolderCard.folders)
        pass

    def __hardwareCardEvent(self):
        # ic(self.hardwareCard.configItem._ConfigItem__value)
        pass

    def __languageCardEvent(self):
        # ic(self.languageCard.configItem._ConfigItem__value)
        pass

    def __clipRadiusCardEvent(self, value):
        # ic(value)
        pass

    def __startCardEvent(self):
        # 1 prepare
        device = self.hardwareCard.configItem._ConfigItem__value
        language = self.languageCard.configItem._ConfigItem__value
        # get video file
        file_1 = [file for file in self.selectFileCard.folders if file.endswith(".mp4")]
        file_2 = [
            os.path.join(root, file_name)
            for folder in self.selectFolderCard.folders
            for root, dirs, files in os.walk(folder)
            for file_name in files
            if file_name.endswith(".mp4")
        ]
        video_file_list = set(file_1 + file_2)

        # 2 lip
        frame_list = [VideoProcessUtil.get_frame(file) for file in video_file_list]
        ic("get frame list")
        landmark_list = [VideoProcessUtil.get_landmarks(frame) for frame in frame_list]
        ic("get landmark list")
        lip_list = [
            VideoProcessUtil.get_lips(frame, landmark)
            for frame, landmark in zip(frame_list, landmark_list)
        ]
        ic("get lip list")

        # 3 mask
        video_lip, video_mask = ModelInferenceUtil.prepare_data(
            lip_list
        )  # (B, C, T, H, W) | (B, T)

        # 4 inference
        predict_text = ModelInferenceUtil.lipreading_inference(
            video_lip, video_mask, device, language
        )

        # 5 show table
        ic(predict_text)
        table = TableFrame(self)
        self.addExampleCard("", table)
        table.showData(video_file_list, predict_text)


class TableFrame(TableWidget):
    def showData(self, video_file_list, predict_text):
        self.verticalHeader().hide()
        self.setBorderRadius(10)
        self.setBorderVisible(True)

        self.setColumnCount(2)
        rowCount = len(video_file_list)
        self.setRowCount(rowCount)
        self.setHorizontalHeaderLabels(["video file", "predict text"])

        for i in range(2):
            self.setColumnWidth(i, 400)
        for i in range(rowCount):
            self.setRowHeight(i, 50)

        self.setFixedSize(800, 50 * (rowCount + 1))
        self.setRowCount(len(video_file_list))
        for i, (file, predict) in enumerate(zip(video_file_list, predict_text)):
            self.setItem(i, 0, QTableWidgetItem(file))
            self.setItem(i, 1, QTableWidgetItem(predict))
