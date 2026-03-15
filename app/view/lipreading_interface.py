# coding:utf-8
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

from PyQt5.QtCore import Qt, QUrl, QThread
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QHBoxLayout, QTableWidgetItem, QWidget, QVBoxLayout, QFileDialog
from icecream import ic
from qfluentwidgets import FluentIcon as FIF

from qfluentwidgets import (
    TableWidget,
    ScrollArea,
    SettingCardGroup,
    SwitchSettingCard,
    RangeSettingCard,
    RangeConfigItem,
    RangeValidator,
    FolderListValidator,
    ConfigItem,
    OptionsValidator,
    OptionsConfigItem,
    ComboBoxSettingCard,
    TitleLabel,
    PrimaryPushSettingCard,
    BoolValidator,
)

from .folder_setting_card import FolderSettingCard, FileSettingCard
from .video_widget import VideoWidget
from ..common.style_sheet import StyleSheet
from ..lipreader.api import lipreading_inference


class GalleryInterface(ScrollArea):
    """Gallery interface"""

    def __init__(self, parent=None):
        """
        Parameters
        ----------
        parent: QWidget
            parent widget
        """
        super().__init__(parent=parent)
        self.view = QWidget(self)
        self.vBoxLayout = QVBoxLayout(self.view)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidget(self.view)
        self.setWidgetResizable(True)

        self.vBoxLayout.setSpacing(30)
        self.vBoxLayout.setAlignment(Qt.AlignTop)
        self.vBoxLayout.setContentsMargins(36, 20, 36, 20)

        self.view.setObjectName("view")
        StyleSheet.GALLERY_INTERFACE.apply(self)

    def scrollToCard(self, index: int):
        """scroll to example card"""
        w = self.vBoxLayout.itemAt(index).widget()
        self.verticalScrollBar().setValue(w.y())

    def resizeEvent(self, e):
        super().resizeEvent(e)


class LipReadingInterface(GalleryInterface):
    """Lip Reaing Interface"""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("lipReadingInterface")
        self.__initLayout()
        self.__initTitleLayout()
        self.__initLeftLayout()
        self.__initRightLayout()
        self.__initDownLayout()

    def __initLayout(self):
        self.titleLayout = QHBoxLayout()
        self.upLayout = QHBoxLayout()
        self.downLayout = QVBoxLayout()
        self.leftLayout = QVBoxLayout()
        self.rightLayout = QVBoxLayout()

        self.upLayout.addLayout(self.leftLayout)
        self.upLayout.addLayout(self.rightLayout)
        self.upLayout.setStretch(0, 1)
        self.upLayout.setStretch(1, 1)

        self.vBoxLayout.addLayout(self.titleLayout)
        self.vBoxLayout.addLayout(self.upLayout)
        self.vBoxLayout.addLayout(self.downLayout)

        self.video_file_list = []
        self.result_table = TableFrame(self)
        self.predict_text_list = []

    def __initTitleLayout(self):
        self.titleLabel = TitleLabel("句子级唇语识别", None)
        self.titleLayout.addWidget(self.titleLabel, 1, Qt.AlignHCenter)

    def __initLeftLayout(self):
        self.prepareSettingGroup = SettingCardGroup("准备设置")
        self.__initLeftShowLayout()
        self.__initLeftHideLayout()
        self.leftLayout.addWidget(self.prepareSettingGroup, 1, Qt.AlignTop)

    def __initLeftShowLayout(self):
        self.selectFileCard = FileSettingCard(
            ConfigItem("", "folder", "", FolderListValidator()),
            "选择文件",
            "选择唇语识别视频文件",
        )
        self.selectFileCard.folderChanged.connect(self.__select_video_file)

        self.startCard = PrimaryPushSettingCard(
            "开始",  # right button name
            FIF.UPDATE,
            "识别",
            "句子级唇语识别",
        )
        self.startCard.clicked.connect(self.__asyncStartEvent)

        self.showOrHideCard = SwitchSettingCard(
            FIF.UPDATE,
            "显示/隐藏 配置信息",
            "",
            ConfigItem("Properties", "ShowOrHideProperties", False, BoolValidator()),
        )
        self.showOrHideCard.checkedChanged.connect(self.__showOrHideCardEvent)

        self.prepareSettingGroup.addSettingCard(self.selectFileCard)
        self.prepareSettingGroup.addSettingCard(self.startCard)
        self.prepareSettingGroup.addSettingCard(self.showOrHideCard)

    def __initLeftHideLayout(self):
        self.selectFolderCard = FolderSettingCard(
            ConfigItem("", "folder", "", FolderListValidator()),
            "选择文件夹",
            "选择唇语视频文件夹",
        )
        self.selectFolderCard.folderChanged.connect(self.__select_video_folder)

        self.hardwareCard = ComboBoxSettingCard(
            OptionsConfigItem("", "hardware", "cpu", OptionsValidator(["cpu", "cuda"])),
            FIF.BRUSH,
            "推理硬件",
            "默认使用 GPU 推理",
            texts=[
                "CPU",
                "GPU",
            ],
        )
        self.hardwareCard.comboBox.currentIndexChanged.connect(self.__hardwareCardEvent)

        self.languageCard = ComboBoxSettingCard(
            OptionsConfigItem("", "language", "hz", OptionsValidator(["hz", "py", "english"])),
            FIF.LANGUAGE,
            "目标语言",
            "选择唇语识别的目标语言",
            texts=["中文", "拼音", "English"],
        )
        self.languageCard.comboBox.currentIndexChanged.connect(self.__languageCardEvent)

        self.outputModeCard = PrimaryPushSettingCard(
            "导出csv文件",  # right button name
            FIF.UPDATE,
            "导出",
            "将检测结果导出至csv文件中",
        )
        self.outputModeCard.clicked.connect(self.__outputModeCardEvent)

        self.prepareSettingGroup.addSettingCard(self.selectFolderCard)
        self.prepareSettingGroup.addSettingCard(self.hardwareCard)
        self.prepareSettingGroup.addSettingCard(self.languageCard)
        self.prepareSettingGroup.addSettingCard(self.outputModeCard)

        self.selectFolderCard.hide()
        self.hardwareCard.hide()
        self.languageCard.hide()
        self.outputModeCard.hide()

    def __initRightLayout(self):
        self.videoWidget = VideoWidget(self.view)
        self.videoWidget.setFixedHeight(400)
        self.videoWidget.setFixedWidth(400)
        self.rightLayout.addWidget(self.videoWidget, 1, Qt.AlignHCenter | Qt.AlignTop)
        # self.videoWidget.hide()

    def __initDownLayout(self):
        pass

    def __select_video_file(self):
        if len(self.selectFileCard.folders) == 0:
            return
        video_file = self.selectFileCard.folders[0]
        ic(video_file)
        self.videoWidget.setVideo(QUrl.fromLocalFile(video_file))
        self.videoWidget.show()

    def __showOrHideCardEvent(self):
        if self.showOrHideCard.configItem.value:
            self.selectFolderCard.show()
            self.hardwareCard.show()
            self.languageCard.show()
            self.outputModeCard.show()
        else:
            self.selectFolderCard.hide()
            self.hardwareCard.hide()
            self.languageCard.hide()
            self.outputModeCard.hide()

    def __select_video_folder(self):
        ic(self.selectFolderCard.folders)
        pass

    def __hardwareCardEvent(self):
        # ic(self.hardwareCard.configItem.value)
        pass

    def __languageCardEvent(self):
        # ic(self.languageCard.configItem.value)
        pass

    def __outputModeCardEvent(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹", "", options=options)

        if not folder:
            return
        if not self.video_file_list:
            return
        if not self.predict_text_list:
            return
        data = [[a, b] for a, b in zip(self.video_file_list, self.predict_text_list)]
        # 拼接完整文件路径
        # filepath = Path(folder) / f"lipreading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"
        filepath = Path(folder) / f"lipreading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # 写入 CSV 文件
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")

        print(f"✅ 数据已保存至：{os.path.abspath(filepath)}")

    def __asyncStartEvent(self):
        self.worker = QThread()
        self.worker.run = self.__asyncRunningFunc
        self.worker.start()
        self.worker.finished.connect(self.__asyncFinishedFunc)

    def __asyncFinishedFunc(self):
        while self.downLayout.count():
            item = self.downLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 创建新表格并添加
        self.result_table = TableFrame(self)
        self.result_table.showData(self.video_file_list, self.predict_text_list)
        self.downLayout.addWidget(
            self.result_table, 1, Qt.AlignCenter | Qt.AlignTop
        )  # 使用AlignTop对齐

    def __asyncRunningFunc(self):
        # 0 清除原有的表格 table
        self.downLayout.removeWidget(self.result_table)
        self.result_table.deleteLater()  # 销毁 table 对象
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
        self.video_file_list = set(file_1 + file_2)

        self.predict_text_list = lipreading_inference(self.video_file_list, device, language)

        # show
        ic(self.predict_text_list)


class TableFrame(TableWidget):
    def showData(self, video_file_list, predict_text):
        self.verticalHeader().hide()
        self.setBorderRadius(10)
        self.setBorderVisible(True)

        self.setColumnCount(2)
        rowCount = len(video_file_list)
        self.setRowCount(rowCount)
        self.setHorizontalHeaderLabels(["唇语视频文件", "模型预测文本"])

        for i in range(2):
            self.setColumnWidth(i, 400)
        for i in range(rowCount):
            self.setRowHeight(i, 50)

        self.setFixedSize(800, 50 * (rowCount + 1))
        self.setRowCount(len(video_file_list))

        for i, (file, predict) in enumerate(zip(video_file_list, predict_text)):
            self.setItem(i, 0, QTableWidgetItem(file))
            self.setItem(i, 1, QTableWidgetItem(predict))
