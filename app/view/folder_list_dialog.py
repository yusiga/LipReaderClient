# coding: utf-8
"""已选文件夹管理弹窗：查看文件夹名称、文件夹内视频文件列表，支持添加/移除文件夹"""
import os
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QFrame, QSplitter, QFileDialog, QAbstractItemView,
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, BodyLabel, StrongBodyLabel,
    FluentIcon as FIF,
)


class FolderListDialog(QDialog):
    """已选文件夹列表弹窗：左侧文件夹列表，右侧显示选中文件夹内的视频文件"""

    def __init__(self, folder_list: list, parent=None):
        super().__init__(parent)
        self._folder_list = folder_list  # 使用引用，修改会同步到调用方
        self.setWindowTitle("已选文件夹")
        self.setMinimumSize(640, 420)
        self.resize(700, 480)
        self.__buildUi()

    def __buildUi(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题与说明
        title = StrongBodyLabel("已选文件夹（可查看名称与其中视频文件）", self)
        layout.addWidget(title)

        splitter = QSplitter(Qt.Horizontal)

        # 左侧：文件夹列表
        left = QFrame(self)
        left.setObjectName("folderDialogLeft")
        leftLayout = QVBoxLayout(left)
        leftLayout.setContentsMargins(0, 0, 0, 0)
        leftLayout.addWidget(BodyLabel("文件夹列表:", self))
        self.folderListWidget = QListWidget(self)
        self.folderListWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.folderListWidget.setMinimumWidth(280)
        for path in self._folder_list:
            self.folderListWidget.addItem(path)
        self.folderListWidget.currentRowChanged.connect(self.__onFolderSelectionChanged)
        leftLayout.addWidget(self.folderListWidget)
        btnRow = QHBoxLayout()
        self.btnAdd = PushButton("添加文件夹", self, FIF.FOLDER_ADD)
        self.btnRemove = PushButton("移除选中", self, FIF.DELETE)
        btnRow.addWidget(self.btnAdd)
        btnRow.addWidget(self.btnRemove)
        btnRow.addStretch(1)
        leftLayout.addLayout(btnRow)
        splitter.addWidget(left)

        # 右侧：选中文件夹内的视频文件
        right = QFrame(self)
        right.setObjectName("folderDialogRight")
        rightLayout = QVBoxLayout(right)
        rightLayout.setContentsMargins(0, 0, 0, 0)
        self.filesLabel = StrongBodyLabel("选中文件夹中的视频文件:", self)
        rightLayout.addWidget(self.filesLabel)
        self.fileListWidget = QListWidget(self)
        self.fileListWidget.setMinimumWidth(280)
        rightLayout.addWidget(self.fileListWidget)
        hint = BodyLabel("在左侧选择文件夹可查看其中的 .mp4 文件", self)
        hint.setStyleSheet("color: gray; font-size: 12px;")
        rightLayout.addWidget(hint)
        splitter.addWidget(right)

        splitter.setSizes([320, 360])
        layout.addWidget(splitter)

        closeBtn = PrimaryPushButton("关闭", self, FIF.CLOSE)
        closeBtn.clicked.connect(self.accept)
        layout.addWidget(closeBtn, 0, Qt.AlignRight)

        self.btnAdd.clicked.connect(self.__onAddFolder)
        self.btnRemove.clicked.connect(self.__onRemoveSelected)
        self.__onFolderSelectionChanged(self.folderListWidget.currentRow())

    def __onAddFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择要添加的文件夹", "", QFileDialog.Options())
        if not folder:
            return
        if folder in self._folder_list:
            return
        self._folder_list.append(folder)
        self.folderListWidget.addItem(folder)

    def __onRemoveSelected(self):
        row = self.folderListWidget.currentRow()
        if row < 0 or row >= len(self._folder_list):
            return
        self._folder_list.pop(row)
        self.folderListWidget.takeItem(row)
        self.fileListWidget.clear()

    def __onFolderSelectionChanged(self, row):
        self.fileListWidget.clear()
        if row < 0 or row >= len(self._folder_list):
            self.filesLabel.setText("选中文件夹中的视频文件: （请先选择左侧文件夹）")
            return
        folder = self._folder_list[row]
        self.filesLabel.setText(f"选中文件夹中的视频文件: {Path(folder).name}")
        try:
            for root, dirs, files in os.walk(folder):
                for name in files:
                    if name.lower().endswith(".mp4"):
                        self.fileListWidget.addItem(os.path.join(root, name))
        except Exception as e:
            self.fileListWidget.addItem(f"(无法读取: {e})")
