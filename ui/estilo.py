"""Identidade visual do aplicativo Qt, sem recursos de navegador."""
from pathlib import Path

STYLE = '''
QMainWindow, QDialog { background: #f3f6fa; }
QWidget { font-family: "Inter", "Noto Sans", "DejaVu Sans"; font-size: 13px; color: #233448; }
QWidget#sidebar { background: #122437; }
QLabel#brand { color: #ffffff; font-size: 26px; font-weight: 800; }
QLabel#brandSub { color: #91a5bb; font-size: 11px; }
QLabel#navGroup { color: #6f8ca6; font-size: 10px; font-weight: 700; padding-top: 10px; }
QListWidget#navigation { background: transparent; border: none; outline: none; color: #bdcbd8; }
QListWidget#navigation::item { padding: 12px 14px; margin: 2px 0; border-radius: 7px; }
QListWidget#navigation::item:hover { background: #1c354b; }
QListWidget#navigation::item:selected { background: #214459; color: #7ee8dd; border-left: 3px solid #57d7c4; }
QFrame#selector { background: white; border-bottom: 1px solid #dde5ef; }
QLabel#eyebrow { font-size: 10px; font-weight: 700; color: #73869b; }
QLabel#pageTitle { font-size: 28px; font-weight: 750; color: #162c41; }
QLabel#muted { color: #72849a; }
QLabel#description { color: #586d82; font-size: 13px; }
QLabel#legendTrain { color: #138f82; font-weight: 600; font-size: 11px; padding: 0 6px; }
QLabel#legendValidation { color: #7e72c4; font-weight: 600; font-size: 11px; padding: 0 6px; }
QLabel#chip { background: #e1f4ee; color: #257763; padding: 6px 10px; border-radius: 10px; font-size: 11px; }
QLabel#notice { background: #e9f1fa; color: #395b7d; border: 1px solid #d4e3f2; border-radius: 8px; padding: 13px; }
QLabel#warning { background: #fff5e4; color: #825d24; border: 1px solid #f0dfbf; border-radius: 8px; padding: 13px; }
QFrame#card { background: white; border: 1px solid #e0e7ef; border-radius: 10px; }
QFrame#historyCard { background: white; border: 1px solid #dce6ee; border-radius: 12px; }
QTabWidget#experimentSections::pane { border: none; padding-top: 12px; }
QTabBar::tab { background: transparent; color: #71849a; padding: 11px 15px; margin-right: 5px; border-bottom: 3px solid transparent; font-weight: 600; }
QTabBar::tab:selected { color: #167f79; border-bottom-color: #23998d; }
QTabBar::tab:hover:!selected { background: #e8f0f6; border-radius: 5px; color: #35576f; }
QLabel#cardTitle { font-size: 14px; font-weight: 650; color: #223b51; }
QLabel#stat { font-size: 28px; font-weight: 750; color: #19394d; }
QPushButton { background: white; border: 1px solid #d5dfe9; padding: 8px 13px; border-radius: 6px; font-weight: 550; }
QPushButton:hover { background: #eff6fb; border-color: #9bb9cf; }
QPushButton:pressed { background: #deebf4; }
QPushButton:disabled { color: #9ba9b9; background: #f2f4f7; border-color: #e4e9ef; }
QPushButton#primary { background: #167f79; border-color: #167f79; color: white; }
QPushButton#primary:hover { background: #106b66; }
QPushButton#danger { color: #ae4848; border-color: #e2bcbc; }
QComboBox, QLineEdit, QSpinBox { background: white; border: 1px solid #d7e1eb; border-radius: 6px; padding: 7px 9px; min-height: 19px; selection-background-color: #167f79; }
QComboBox:focus, QLineEdit:focus, QSpinBox:focus { border: 1px solid #298b89; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow { image: url(@CHEVRON@); width: 12px; height: 8px; }
QComboBox QAbstractItemView { background: white; color: #233448; selection-background-color: #dcefea; selection-color: #176a63; }
QScrollArea, QStackedWidget { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { border: none; background: transparent; width: 9px; margin: 2px; }
QScrollBar::handle:vertical { background: #c4d0dd; min-height: 28px; border-radius: 3px; }
QScrollBar::handle:vertical:hover { background: #8ca8bd; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QTableWidget { background: white; alternate-background-color: #f6f9fc; border: 1px solid #e1e8f0; border-radius: 7px; gridline-color: #edf1f6; selection-background-color: #def0eb; selection-color: #233448; }
QHeaderView::section { background: #edf3f8; color: #61778d; padding: 10px; border: none; font-weight: 600; }
QTableWidget::item { padding: 8px; }
QPlainTextEdit { background: #152639; color: #bdcedd; border: none; border-radius: 8px; padding: 12px; font-family: "DejaVu Sans Mono"; font-size: 11px; }
QProgressBar { background: #e3eaf1; border: none; border-radius: 2px; height: 4px; }
QProgressBar::chunk { background: #28a397; border-radius: 2px; }
QStatusBar { background: #eaf0f6; color: #62798e; font-size: 11px; }
QToolBar { border: none; background: #f6f9fc; spacing: 6px; }
QToolTip { background: #203c52; color: white; border: none; padding: 6px; }
QSplitter::handle { background: #e3eaf1; }
'''.replace('@CHEVRON@', (Path(__file__).parent / 'assets/chevron.svg').as_posix())
