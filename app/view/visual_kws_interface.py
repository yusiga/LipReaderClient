# coding:utf-8
import os
from pathlib import Path
import pandas as pd

from PyQt5.QtCore import Qt, QUrl, QThread
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QHBoxLayout, QTableWidgetItem, QWidget, QVBoxLayout, QFileDialog
from icecream import ic
from qfluentwidgets import FluentIcon as FIF

# from qfluentwidgets import qconfig
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

from ..lipreader.api import kws_inference


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
        self.vBoxLayout.setContentsMargins(36, 20, 36, 36)

        self.view.setObjectName("view")
        StyleSheet.GALLERY_INTERFACE.apply(self)

    def scrollToCard(self, index: int):
        """scroll to example card"""
        w = self.vBoxLayout.itemAt(index).widget()
        self.verticalScrollBar().setValue(w.y())

    def resizeEvent(self, e):
        super().resizeEvent(e)


class VisualKwsInterface(GalleryInterface):
    """Visual Ketword Spotting Interface"""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("visualKwsInterface")
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
        # self.vBoxLayout.setStretch(0, 1)
        # self.vBoxLayout.setStretch(1, 6)
        # self.vBoxLayout.setStretch(2, 6)

        self.video_file_list = []
        self.kws_tables = []
        self.predict_text_list = []

    def __initTitleLayout(self):
        self.titleLabel = TitleLabel("唇语关键字检测", None)
        self.titleLayout.addWidget(self.titleLabel, 1, Qt.AlignHCenter)  # TODO

    def __initLeftLayout(self):
        self.prepareSettingGroup = SettingCardGroup("准备设置")
        self.__initLeftShowLayout()
        self.__initLeftHideLayout()
        self.leftLayout.addWidget(self.prepareSettingGroup, 1, Qt.AlignTop)

    def __initLeftShowLayout(self):
        self.selectFileCard = FileSettingCard(
            ConfigItem("", "folder", "", FolderListValidator()),
            "选择文件",
            "选择唇语关键词检测视频文件",
        )
        self.selectFileCard.folderChanged.connect(self.__select_video_file)

        self.clipRadiusCard = RangeSettingCard(
            RangeConfigItem("", "clipRadius", 5, RangeValidator(1, 10)),
            FIF.ALBUM,
            "clip",
            "根据说话人的语速调节clip参数",
        )
        self.clipRadiusCard.valueChanged.connect(self.__clipRadiusCardEvent)
        self.topKRadiusCard = RangeSettingCard(
            RangeConfigItem("", "topKRadius", 5, RangeValidator(1, 10)),
            FIF.ALBUM,
            "Top K",
            "根据需要调节 topK 参数",
        )
        self.topKRadiusCard.valueChanged.connect(self.__topRadiusCardEvent)

        self.startCard = PrimaryPushSettingCard(
            "开始",  # right button name
            FIF.UPDATE,
            "检测",
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
        self.prepareSettingGroup.addSettingCard(self.clipRadiusCard)
        self.prepareSettingGroup.addSettingCard(self.topKRadiusCard)
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

        self.strategyCard = ComboBoxSettingCard(
            OptionsConfigItem("", "strategy", "Manu", OptionsValidator(["Manu", "Auto"])),
            FIF.LANGUAGE,
            "切换策略",
            "切换中文关键字检测策略",
            texts=["Manu", "Auto"],
        )
        self.strategyCard.comboBox.currentIndexChanged.connect(self.__strategyCardEvent)

        self.outputModeCard = PrimaryPushSettingCard(
            "导出csv文件",  # right button name
            FIF.UPDATE,
            "导出",
            "将检测结果导出至csv文件中",
        )
        self.outputModeCard.clicked.connect(self.__outputModeCardEvent)

        self.prepareSettingGroup.addSettingCard(self.selectFolderCard)
        self.prepareSettingGroup.addSettingCard(self.hardwareCard)
        self.prepareSettingGroup.addSettingCard(self.strategyCard)
        self.prepareSettingGroup.addSettingCard(self.outputModeCard)

        self.selectFolderCard.hide()
        self.hardwareCard.hide()
        self.strategyCard.hide()
        self.outputModeCard.hide()

    def __initRightLayout(self):

        self.videoWidget = VideoWidget(self.view)
        self.videoWidget.setFixedHeight(400)
        self.videoWidget.setFixedWidth(400)
        self.rightLayout.addWidget(self.videoWidget, 1, Qt.AlignHCenter | Qt.AlignTop)
        # self.videoWidget.hide()

    def __initDownLayout(self):
        # self.table = TableFrame(self)
        # self.downLayout.addWidget(self.table, 1, Qt.AlignTop)
        # self.table.hide()
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
            self.strategyCard.show()
            self.outputModeCard.show()
            # self.resetCard.show()
            # self.showOrHideLabel.show()
        else:
            self.selectFolderCard.hide()
            self.hardwareCard.hide()
            self.strategyCard.hide()
            self.outputModeCard.hide()
            # self.resetCard.hide()
            # self.showOrHideLabel.hide()

    def __select_video_folder(self):
        ic(self.selectFolderCard.folders)
        pass

    def __hardwareCardEvent(self):
        # ic(self.hardwareCard.configItem.value)
        pass

    def __strategyCardEvent(self):
        # ic(self.strategyCard.configItem.value)
        if self.strategyCard.configItem.value == "Auto":
            self.clipRadiusCard.hide()
        if self.strategyCard.configItem.value == "Manu":
            self.clipRadiusCard.show()

    def __clipRadiusCardEvent(self):
        ic(self.clipRadiusCard.configItem.value)
        pass

    def __topRadiusCardEvent(self):
        ic(self.topKRadiusCard.configItem.value)
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
        for video_file, predict_text in zip(self.video_file_list, self.predict_text_list):
            pred_text = [list(text) for text in zip(*predict_text)]  # 转置
            # 拼接完整文件路径
            filepath = Path(folder) / ("visualKws_" + Path(video_file).stem + ".csv")
            # 写入 CSV 文件
            df = pd.DataFrame(pred_text)
            df.to_csv(filepath, index=False, encoding="utf-8-sig")

            print(f"✅ 数据已保存至：{os.path.abspath(filepath)}")

    def __asyncStartEvent(self):
        self.worker = QThread()
        self.worker.run = self.__asyncRunningFunc
        self.worker.start()
        self.worker.finished.connect(self.__asyncFinishedFunc)

    def __asyncFinishedFunc(self):
        for predict_text in self.predict_text_list:
            table = TableFrame(self)
            self.kws_tables.append(table)
            self.downLayout.addWidget(table, 1, Qt.AlignCenter | Qt.AlignTop)
            table.showData(predict_text)

    def __asyncRunningFunc(self):
        # 0 清除原有的表格 table
        for table in self.kws_tables[:]:  # 使用切片复制列表，避免迭代时修改列表
            self.downLayout.removeWidget(table)
            table.deleteLater()  # 销毁 table 对象
            self.kws_tables.remove(table)  # 从列表中移除 table
        # 1 prepare
        device = self.hardwareCard.configItem.value
        strategy = self.strategyCard.configItem.value
        clip = self.clipRadiusCard.configItem.value
        top_k = self.topKRadiusCard.configItem.value
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

        self.predict_text_list = kws_inference(self.video_file_list, device, strategy, top_k, clip)

        # 5 show table
        ic(self.predict_text_list)
        flattened_list = sum(self.predict_text_list[0], [])
        ic(flattened_list)


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


class TableText(TableWidget):
    def showData(self, predict_text):
        self.verticalHeader().hide()
        self.horizontalHeader().hide()
        self.setBorderRadius(10)
        self.setBorderVisible(True)

        self.setColumnCount(1)
        self.setRowCount(1)

        self.setColumnWidth(0, 400)
        self.setRowHeight(0, 50)
        self.setFixedSize(400, 50)

        item = QTableWidgetItem(predict_text)
        item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        font = QFont()
        font.setPointSize(16)
        font.setFamily("KaiTi")  # KaiTi | SimSun | FangSong | Microsoft YaHei | SimHei
        item.setFont(font)
        self.setItem(0, 0, item)
