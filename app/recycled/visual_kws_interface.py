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
from transformers.models.layoutlmv2.tokenization_layoutlmv2 import table

from app.view.folder_setting_card import FolderSettingCard, FileSettingCard

# from .gallery_interface import GalleryInterface
from app.view.gallery_clear_interface import GalleryInterface
from app.common.translator import Translator
from app.common.style_sheet import StyleSheet
from app.lipreader.utils.util import VideoProcessUtil, ModelInferenceUtil
from app.common.config import cfg, HELP_URL, FEEDBACK_URL, AUTHOR, VERSION, YEAR, isWin11
from app.lipreader.video_process import read_lips

from icecream import ic
import torch
import numpy as np


class VisualKwsInterface(GalleryInterface):
    """Lip Reader interface"""

    def __init__(self, parent=None):
        # define attribute
        super().__init__(title="Visual Keyword Spotting", subtitle="", parent=parent)
        self.setObjectName("visualKwsInterface")

        # file select
        titleLabel = TitleLabel("Visual Keyword Spotting", self)
        self.addExampleCard("", titleLabel, alignment=Qt.AlignHCenter)

        self.selectFileCard = FileSettingCard(
            ConfigItem("", "folder", "", FolderListValidator()),
            "kws video file",
            "选择视觉关键词检测视频文件",
        )
        self.selectFileCard.folderChanged.connect(self.__select_video_file)

        self.selectFolderCard = FolderSettingCard(
            ConfigItem("", "folder", "", FolderListValidator()),
            "kws video folder",
            "选择唇语视频文件夹",
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
            OptionsConfigItem("", "language", "hz", OptionsValidator(["hz", "py", "english"])),
            FIF.LANGUAGE,
            "目标语言",
            "选择唇语识别的目标语言",
            texts=["中文", "拼音", "English"],
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
        self.topKRadiusCard = RangeSettingCard(
            RangeConfigItem("", "topKRadius", 5, RangeValidator(1, 10)),
            FIF.ALBUM,
            "Top K",
            "根据需要调节 topK 参数",
            # self.hyperParameterGroup
        )
        self.topKRadiusCard.valueChanged.connect(self.__topRadiusCardEvent)
        self.hyperParameterGroup = SettingCardGroup("超参数设置", self.scrollWidget)
        self.hyperParameterGroup.addSettingCard(self.clipRadiusCard)
        self.hyperParameterGroup.addSettingCard(self.topKRadiusCard)
        self.addExampleCard("", self.hyperParameterGroup, 1, Qt.AlignTop)

        resultLabel = TitleLabel("Inference Result", self)
        self.addExampleCard("", resultLabel, alignment=Qt.AlignHCenter)

        # start lipreading
        self.startCard = PrimaryPushSettingCard(
            "开始",  # right button name
            FIF.UPDATE,
            "开始检测",
            "视觉关键词检测",
        )
        self.startCard.clicked.connect(self.__startCardEvent)
        self.addExampleCard("", self.startCard, 1, Qt.AlignTop)
        self.resetCard = PrimaryPushSettingCard(
            "重置",  # right button name
            FIF.UPDATE,
            "清除上次检测数据",
            "视觉关键词检测",
        )
        self.resetCard.clicked.connect(self.__resetCardEvent)
        self.addExampleCard("", self.resetCard, 1, Qt.AlignTop)

        # table view
        self.kwsTables = []

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

    def __clipRadiusCardEvent(self):
        ic(self.clipRadiusCard.configItem._ConfigItem__value)
        pass

    def __topRadiusCardEvent(self):
        ic(self.topKRadiusCard.configItem._ConfigItem__value)
        pass

    def __resetCardEvent(self):
        for table in self.kwsTables:
            # table = TableFrame(self)
            # self.kwsTables.append(table)
            self.removeExampleCard("", table)
            # table.showData(predict_text)
        pass

    def __startCardEvent(self):
        # 1 prepare
        device = self.hardwareCard.configItem._ConfigItem__value
        language = self.languageCard.configItem._ConfigItem__value
        clip = self.clipRadiusCard.configItem._ConfigItem__value
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

        # 4 inference (B, clip_group, top_k)
        predict_text_list = ModelInferenceUtil.kws_inference(
            video_lip, video_mask, device, language, clip
        )

        # 5 show table
        # todo table view
        ic(predict_text_list)
        for predict_text in predict_text_list:
            table = TableFrame(self)
            self.kwsTables.append(table)
            self.addExampleCard("", table)
            table.showData(predict_text)


class TableFrame(TableWidget):
    def showData(self, predict_text):
        self.verticalHeader().hide()
        self.setBorderRadius(10)
        self.setBorderVisible(True)

        clip_group = len(predict_text)
        top_k = len(predict_text[0])
        self.setColumnCount(clip_group)
        self.setRowCount(top_k)
        self.setHorizontalHeaderLabels([str(i) for i in range(1, clip_group + 1)])
        for i in range(clip_group):
            self.setColumnWidth(i, 50)
        for i in range(top_k):
            self.setRowHeight(i, 50)
        width = 800
        if 50 * clip_group < 800:
            width = 50 * clip_group

        self.setFixedSize(width, 50 * (top_k + 1))
        for j, top in enumerate(predict_text):
            for i, item in enumerate(top):
                self.setItem(i, j, QTableWidgetItem(item))
