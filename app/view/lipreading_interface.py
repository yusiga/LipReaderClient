# coding:utf-8
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

from PyQt5.QtCore import Qt, QUrl, QThread
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QHBoxLayout, QWidget, QVBoxLayout, QFileDialog,
    QFrame, QLabel, QSizePolicy, QScrollArea,
)
from icecream import ic
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import (
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
from ..lipreader.api import lipreading_inference


class GalleryInterface(ScrollArea):
    """Gallery interface - 基类保留以兼容 main_window 的 stackedWidget"""

    def __init__(self, parent=None):
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

    def scrollToCard(self, index: int):
        w = self.vBoxLayout.itemAt(index).widget()
        if w:
            self.verticalScrollBar().setValue(w.y())


class LipReadingInterface(GalleryInterface):
    """句子级唇语识别 - 新布局：顶部大视频 + 横向操作栏 + 可展开配置 + 全宽结果区"""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("lipReadingInterface")
        self.view.setObjectName("lipreaderView")
        StyleSheet.LIPREADER_INTERFACE.apply(self)
        self.vBoxLayout.setContentsMargins(32, 24, 32, 24)
        self.vBoxLayout.setSpacing(20)

        self._selected_files = []
        self._selected_folders = []
        self._device = "cpu"
        self._language = "hz"
        self.video_file_list = []
        self.predict_text_list = []
        self.result_cards = None

        self.__initLayout()
        self.__buildHeader()
        self.__buildVideoSection()
        self.__buildToolbar()
        self.__buildConfigRow()
        self.__buildResultSection()
        self.__connectSignals()

    def __initLayout(self):
        self.vBoxLayout.setSpacing(24)
        self.contentMargins = (32, 24, 32, 24)

    def __buildHeader(self):
        """左上角标题区"""
        header = QWidget(self.view)
        header.setObjectName("lipreaderHeader")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(4)
        self.titleLabel = TitleLabel("句子级唇语识别", header)
        self.titleLabel.setObjectName("lipreaderTitle")
        self.subtitleLabel = CaptionLabel("选择视频并运行识别，结果将显示在下方", header)
        self.subtitleLabel.setObjectName("lipreaderSubtitle")
        layout.addWidget(self.titleLabel)
        layout.addWidget(self.subtitleLabel)
        self.vBoxLayout.addWidget(header, 0, Qt.AlignLeft | Qt.AlignTop)

    def __buildVideoSection(self):
        """中央大视频区域"""
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
        """横向操作栏：选择视频 | 开始识别 | 高级选项"""
        toolbar = QFrame(self.view)
        toolbar.setObjectName("lipreaderToolbar")
        toolbar.setMinimumHeight(56)
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(16)

        self.btnSelectVideo = PushButton("选择视频", toolbar, FIF.FOLDER)
        self.btnSelectVideo.setObjectName("lipreaderBtnSelect")
        self.btnStart = PrimaryPushButton("开始识别", toolbar, FIF.PLAY)
        self.btnStart.setObjectName("lipreaderBtnStart")
        self.fileHintLabel = BodyLabel("未选择视频", toolbar)
        self.fileHintLabel.setObjectName("lipreaderFileHint")
        self.fileHintLabel.setStyleSheet("color: gray;")
        self.advancedSwitch = SwitchButton(toolbar)
        self.advancedSwitch.setText("高级选项")
        self.advancedSwitch.setObjectName("lipreaderAdvancedSwitch")

        layout.addWidget(self.btnSelectVideo)
        layout.addWidget(self.btnStart)
        layout.addWidget(self.fileHintLabel, 1)
        layout.addWidget(self.advancedSwitch, 0, Qt.AlignRight)
        self.vBoxLayout.addWidget(toolbar)

    def __buildConfigRow(self):
        """高级选项行：文件夹、硬件、语言、导出（默认隐藏）"""
        self.configRow = QFrame(self.view)
        self.configRow.setObjectName("lipreaderConfigRow")
        self.configRow.setVisible(False)
        layout = QHBoxLayout(self.configRow)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(24)

        layout.addWidget(StrongBodyLabel("批量文件夹:", self.configRow))
        self.btnAddFolder = PushButton("添加文件夹", self.configRow, FIF.FOLDER_ADD)
        self.btnViewFolders = PushButton("查看已选文件夹 (0)", self.configRow, FIF.FOLDER)
        self.btnViewFolders.setObjectName("btnViewFolders")
        layout.addWidget(self.btnAddFolder)
        layout.addWidget(self.btnViewFolders)

        layout.addWidget(StrongBodyLabel("推理硬件:", self.configRow))
        self.hardwareCombo = ComboBox(self.configRow)
        self.hardwareCombo.addItems(["CPU", "GPU"])
        self.hardwareCombo.setCurrentIndex(0)
        layout.addWidget(self.hardwareCombo)

        layout.addWidget(StrongBodyLabel("目标语言:", self.configRow))
        self.languageCombo = ComboBox(self.configRow)
        self.languageCombo.addItems(["中文", "拼音", "English"])
        self.languageCombo.setCurrentIndex(0)
        layout.addWidget(self.languageCombo)

        layout.addStretch(1)
        self.btnExport = PushButton("导出 CSV", self.configRow, FIF.SAVE)
        layout.addWidget(self.btnExport)
        self.vBoxLayout.addWidget(self.configRow)

    def __buildResultSection(self):
        """结果区占位，表格在识别完成后动态添加"""
        self.resultContainer = QFrame(self.view)
        self.resultContainer.setObjectName("lipreaderResultContainer")
        self.resultContainer.setMinimumHeight(120)
        self.resultLayout = QVBoxLayout(self.resultContainer)
        self.resultLayout.setContentsMargins(0, 12, 0, 0)
        self.resultLayout.setAlignment(Qt.AlignTop)
        self.resultPlaceholder = BodyLabel("识别结果将显示在此处", self.resultContainer)
        self.resultPlaceholder.setObjectName("lipreaderResultPlaceholder")
        self.resultLayout.addWidget(self.resultPlaceholder)
        self.vBoxLayout.addWidget(self.resultContainer, 1)

    def __connectSignals(self):
        self.btnSelectVideo.clicked.connect(self.__onSelectVideo)
        self.btnStart.clicked.connect(self.__asyncStartEvent)
        self.advancedSwitch.checkedChanged.connect(self.__onAdvancedToggled)
        self.btnAddFolder.clicked.connect(self.__onAddFolder)
        self.btnViewFolders.clicked.connect(self.__onViewFolders)
        self.hardwareCombo.currentIndexChanged.connect(self.__onHardwareChanged)
        self.languageCombo.currentIndexChanged.connect(self.__onLanguageChanged)
        self.btnExport.clicked.connect(self.__outputModeCardEvent)

    def __onSelectVideo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择唇语识别视频", "", "视频 (*.mp4);;所有 (*.*)", None, QFileDialog.Options()
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

    def __onAdvancedToggled(self, checked):
        self.configRow.setVisible(checked)

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

    def __onHardwareChanged(self, index):
        self._device = "cuda" if index == 1 else "cpu"

    def __onLanguageChanged(self, index):
        self._language = ["hz", "py", "english"][index]

    def __select_video_file(self):
        pass

    def __outputModeCardEvent(self):
        folder = QFileDialog.getExistingDirectory(self, "选择保存位置", "", QFileDialog.Options())
        if not folder or not self.video_file_list or not self.predict_text_list:
            return
        filepath = Path(folder) / f"lipreading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame([[a, b] for a, b in zip(self.video_file_list, self.predict_text_list)])
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"✅ 数据已保存至：{os.path.abspath(filepath)}")

    def __asyncStartEvent(self):
        self.worker = QThread()
        self.worker.run = self.__asyncRunningFunc
        self.worker.start()
        self.worker.finished.connect(self.__asyncFinishedFunc)

    def __asyncFinishedFunc(self):
        if getattr(self, "result_cards", None):
            self.resultLayout.removeWidget(self.result_cards)
            self.result_cards.deleteLater()
            self.result_cards = None
        try:
            self.resultLayout.removeWidget(self.resultPlaceholder)
        except Exception:
            pass
        if not self.video_file_list:
            self.resultLayout.addWidget(self.resultPlaceholder)
            return
        self.result_cards = ResultCardsWidget(self.resultContainer)
        self.result_cards.showData(self.video_file_list, self.predict_text_list)
        self.resultLayout.addWidget(self.result_cards, 1, Qt.AlignTop)

    def __asyncRunningFunc(self):
        if getattr(self, "result_cards", None):
            self.resultLayout.removeWidget(self.result_cards)
            self.result_cards.deleteLater()
            self.result_cards = None
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
        self.predict_text_list = lipreading_inference(
            self.video_file_list, self._device, self._language
        )
        ic(self.predict_text_list)


class ResultCard(QFrame):
    """单条识别结果卡片：视频名 + 识别结果大号展示"""

    def __init__(self, video_path: str, result_text: str, index: int, parent=None):
        super().__init__(parent)
        self.setObjectName("resultCard")
        self.setMinimumHeight(100)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)
        name = Path(video_path).name
        fileLabel = CaptionLabel(f"视频 {index}: {name}", self)
        fileLabel.setObjectName("resultCardFile")
        layout.addWidget(fileLabel)
        titleLabel = StrongBodyLabel("识别结果", self)
        titleLabel.setObjectName("resultCardTitle")
        layout.addWidget(titleLabel)
        resultLabel = QLabel(result_text or "(无识别结果)", self)
        resultLabel.setObjectName("resultCardText")
        resultLabel.setWordWrap(True)
        resultLabel.setFont(QFont("Microsoft YaHei UI", 16, QFont.Bold))
        resultLabel.setMinimumHeight(44)
        layout.addWidget(resultLabel)


class ResultCardsWidget(QScrollArea):
    """识别结果卡片列表：每条结果一张卡片，识别结果大号显眼"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("resultCardsWidget")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        self._content = QWidget(self)
        self._layout = QVBoxLayout(self._content)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(16)
        self._layout.setAlignment(Qt.AlignTop)
        self.setWidget(self._content)

    def showData(self, video_file_list, predict_text_list):
        for i, (path, text) in enumerate(zip(video_file_list, predict_text_list), 1):
            card = ResultCard(path, text, i, self._content)
            self._layout.addWidget(card)
