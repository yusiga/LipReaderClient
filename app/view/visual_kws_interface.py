# coding:utf-8
import os
from pathlib import Path
import pandas as pd

from PyQt5.QtCore import Qt, QUrl, QThread
from PyQt5.QtWidgets import (
    QHBoxLayout, QTableWidgetItem, QWidget, QVBoxLayout, QFileDialog,
    QFrame, QLabel, QSizePolicy, QSpinBox,
)
from icecream import ic
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import (
    TableWidget,
    ScrollArea,
    TitleLabel,
    CaptionLabel,
    BodyLabel,
    PushButton,
    PrimaryPushButton,
    ComboBox,
    SwitchButton,
    StrongBodyLabel,
)

from .video_widget import VideoWidget
from .folder_list_dialog import FolderListDialog
from ..common.style_sheet import StyleSheet
from ..lipreader.api import kws_inference


class GalleryInterface(ScrollArea):
    """Gallery interface - 基类保留以兼容 main_window"""

    def __init__(self, parent=None):
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

    def scrollToCard(self, index: int):
        w = self.vBoxLayout.itemAt(index).widget()
        if w:
            self.verticalScrollBar().setValue(w.y())


class VisualKwsInterface(GalleryInterface):
    """唇语关键字检测 - 新布局：顶部大视频 + 横向操作栏 + 可展开配置 + 全宽结果区"""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("visualKwsInterface")
        self.view.setObjectName("lipreaderView")
        StyleSheet.LIPREADER_INTERFACE.apply(self)
        self.vBoxLayout.setContentsMargins(32, 24, 32, 24)
        self.vBoxLayout.setSpacing(20)

        self._selected_files = []
        self._selected_folders = []
        self._device = "cpu"
        self._strategy = "Manu"
        self._clip = 5
        self._top_k = 5
        self.video_file_list = []
        self.predict_text_list = []
        self.kws_tables = []

        self.__initLayout()
        self.__buildHeader()
        self.__buildVideoSection()
        self.__buildToolbar()
        self.__buildConfigRow()
        self.__buildResultSection()
        self.__connectSignals()

    def __initLayout(self):
        self.vBoxLayout.setSpacing(24)

    def __buildHeader(self):
        header = QWidget(self.view)
        header.setObjectName("lipreaderHeader")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(4)
        self.titleLabel = TitleLabel("唇语关键字检测", header)
        self.titleLabel.setObjectName("lipreaderTitle")
        self.subtitleLabel = CaptionLabel("选择视频并运行检测，关键字结果将显示在下方", header)
        self.subtitleLabel.setObjectName("lipreaderSubtitle")
        layout.addWidget(self.titleLabel)
        layout.addWidget(self.subtitleLabel)
        self.vBoxLayout.addWidget(header, 0, Qt.AlignLeft | Qt.AlignTop)

    def __buildVideoSection(self):
        videoContainer = QFrame(self.view)
        videoContainer.setObjectName("lipreaderVideoContainer")
        videoContainer.setMinimumHeight(340)
        videoContainer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QVBoxLayout(videoContainer)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)
        self.videoWidget = VideoWidget(videoContainer)
        self.videoWidget.setFixedSize(640, 360)
        self.videoWidget.setObjectName("videoWidget")
        layout.addWidget(self.videoWidget, 0, Qt.AlignCenter)
        self.vBoxLayout.addWidget(videoContainer)

    def __buildToolbar(self):
        toolbar = QFrame(self.view)
        toolbar.setObjectName("lipreaderToolbar")
        toolbar.setMinimumHeight(56)
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(16)

        self.btnSelectVideo = PushButton("选择视频", toolbar, FIF.FOLDER)
        self.btnStart = PrimaryPushButton("开始检测", toolbar, FIF.PLAY)
        self.fileHintLabel = BodyLabel("未选择视频", toolbar)
        self.fileHintLabel.setStyleSheet("color: gray;")
        self.advancedSwitch = SwitchButton(toolbar)
        self.advancedSwitch.setText("高级选项")

        layout.addWidget(self.btnSelectVideo)
        layout.addWidget(self.btnStart)
        layout.addWidget(self.fileHintLabel, 1)
        layout.addWidget(self.advancedSwitch, 0, Qt.AlignRight)
        self.vBoxLayout.addWidget(toolbar)

    def __buildConfigRow(self):
        self.configRow = QFrame(self.view)
        self.configRow.setObjectName("lipreaderConfigRow")
        self.configRow.setVisible(False)
        layout = QHBoxLayout(self.configRow)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(24)

        layout.addWidget(StrongBodyLabel("批量文件夹:", self.configRow))
        self.btnAddFolder = PushButton("添加文件夹", self.configRow, FIF.FOLDER_ADD)
        self.btnViewFolders = PushButton("查看已选文件夹 (0)", self.configRow, FIF.FOLDER)
        layout.addWidget(self.btnAddFolder)
        layout.addWidget(self.btnViewFolders)

        layout.addWidget(StrongBodyLabel("推理硬件:", self.configRow))
        self.hardwareCombo = ComboBox(self.configRow)
        self.hardwareCombo.addItems(["CPU", "GPU"])
        layout.addWidget(self.hardwareCombo)

        layout.addWidget(StrongBodyLabel("策略:", self.configRow))
        self.strategyCombo = ComboBox(self.configRow)
        self.strategyCombo.addItems(["Manu", "Auto"])
        self.strategyCombo.currentIndexChanged.connect(self.__onStrategyChanged)
        layout.addWidget(self.strategyCombo)

        self.clipLabel = StrongBodyLabel("Clip:", self.configRow)
        self.clipSpin = QSpinBox(self.configRow)
        self.clipSpin.setRange(1, 10)
        self.clipSpin.setValue(5)
        self.clipSpin.setMinimumWidth(60)
        layout.addWidget(self.clipLabel)
        layout.addWidget(self.clipSpin)

        layout.addWidget(StrongBodyLabel("Top K:", self.configRow))
        self.topKSpin = QSpinBox(self.configRow)
        self.topKSpin.setRange(1, 10)
        self.topKSpin.setValue(5)
        self.topKSpin.setMinimumWidth(60)
        layout.addWidget(self.topKSpin)

        layout.addStretch(1)
        self.btnExport = PushButton("导出 CSV", self.configRow, FIF.SAVE)
        layout.addWidget(self.btnExport)
        self.vBoxLayout.addWidget(self.configRow)

    def __onStrategyChanged(self, index):
        self._strategy = "Auto" if index == 1 else "Manu"
        self.clipLabel.setVisible(self._strategy == "Manu")
        self.clipSpin.setVisible(self._strategy == "Manu")

    def __connectSignals(self):
        self.btnSelectVideo.clicked.connect(self.__onSelectVideo)
        self.btnStart.clicked.connect(self.__asyncStartEvent)
        self.advancedSwitch.checkedChanged.connect(lambda c: self.configRow.setVisible(c))
        self.btnAddFolder.clicked.connect(self.__onAddFolder)
        self.btnViewFolders.clicked.connect(self.__onViewFolders)
        self.hardwareCombo.currentIndexChanged.connect(
            lambda i: setattr(self, "_device", "cuda" if i == 1 else "cpu")
        )
        self.clipSpin.valueChanged.connect(lambda v: setattr(self, "_clip", v))
        self.topKSpin.valueChanged.connect(lambda v: setattr(self, "_top_k", v))
        self.btnExport.clicked.connect(self.__outputModeCardEvent)

    def __onSelectVideo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择唇语检测视频", "", "视频 (*.mp4);;所有 (*.*)", None, QFileDialog.Options()
        )
        if not path:
            return
        self._selected_files = [path] if path.endswith(".mp4") else []
        if self._selected_files:
            self.videoWidget.setVideo(QUrl.fromLocalFile(self._selected_files[0]))
            self.videoWidget.show()
            name = Path(path).name
            self.fileHintLabel.setText(f"已选: {name}" if len(name) <= 30 else f"已选: ...{name[-27:]}")
        else:
            self.fileHintLabel.setText("请选择 .mp4 文件")

    def __onAddFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择唇语视频文件夹", "", QFileDialog.Options())
        if folder and folder not in self._selected_folders:
            self._selected_folders.append(folder)
            self.__refreshFolderButtonText()

    def __refreshFolderButtonText(self):
        n = len(self._selected_folders)
        self.btnViewFolders.setText(f"查看已选文件夹 ({n})")

    def __onViewFolders(self):
        d = FolderListDialog(self._selected_folders, self)
        d.exec_()
        self.__refreshFolderButtonText()

    def __outputModeCardEvent(self):
        folder = QFileDialog.getExistingDirectory(self, "选择保存位置", "", QFileDialog.Options())
        if not folder or not self.video_file_list or not self.predict_text_list:
            return
        for video_file, predict_text in zip(self.video_file_list, self.predict_text_list):
            pred_text = [list(col) for col in zip(*predict_text)]
            filepath = Path(folder) / ("visualKws_" + Path(video_file).stem + ".csv")
            pd.DataFrame(pred_text).to_csv(filepath, index=False, encoding="utf-8-sig")
            print(f"✅ 数据已保存至：{os.path.abspath(filepath)}")

    def __asyncStartEvent(self):
        self.worker = QThread()
        self.worker.run = self.__asyncRunningFunc
        self.worker.start()
        self.worker.finished.connect(self.__asyncFinishedFunc)

    def __asyncFinishedFunc(self):
        for t in self.kws_tables:
            self.resultLayout.removeWidget(t)
            t.deleteLater()
        self.kws_tables.clear()
        try:
            self.resultLayout.removeWidget(self.resultPlaceholder)
        except Exception:
            pass
        if not self.predict_text_list:
            self.resultLayout.addWidget(self.resultPlaceholder)
            return
        for predict_text in self.predict_text_list:
            table = TableFrame(self.resultContainer)
            self.kws_tables.append(table)
            self.resultLayout.addWidget(table, 0, Qt.AlignTop)
            table.showData(predict_text)

    def __asyncRunningFunc(self):
        for t in self.kws_tables[:]:
            self.resultLayout.removeWidget(t)
            t.deleteLater()
            self.kws_tables.remove(t)
        file_1 = [f for f in self._selected_files if f.endswith(".mp4")]
        file_2 = [
            os.path.join(root, name)
            for folder in self._selected_folders
            for root, dirs, files in os.walk(folder)
            for name in files
            if name.endswith(".mp4")
        ]
        self.video_file_list = sorted(set(file_1 + file_2))
        if not self.video_file_list:
            return
        self._clip = self.clipSpin.value()
        self._top_k = self.topKSpin.value()
        self.predict_text_list = kws_inference(
            self.video_file_list, self._device, self._strategy, self._top_k, self._clip
        )
        ic(self.predict_text_list)

    def __buildResultSection(self):
        self.resultContainer = QFrame(self.view)
        self.resultContainer.setObjectName("lipreaderResultContainer")
        self.resultContainer.setMinimumHeight(120)
        self.resultLayout = QVBoxLayout(self.resultContainer)
        self.resultLayout.setContentsMargins(0, 12, 0, 0)
        self.resultLayout.setAlignment(Qt.AlignTop)
        self.resultPlaceholder = BodyLabel("检测结果将显示在此处", self.resultContainer)
        self.resultPlaceholder.setObjectName("lipreaderResultPlaceholder")
        self.resultLayout.addWidget(self.resultPlaceholder)
        self.vBoxLayout.addWidget(self.resultContainer, 1)


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
        width = min(800, 50 * clip_group)
        self.setFixedSize(width, 50 * (top_k + 1))
        for j, top in enumerate(predict_text):
            for i, item in enumerate(top):
                self.setItem(i, j, QTableWidgetItem(item))
