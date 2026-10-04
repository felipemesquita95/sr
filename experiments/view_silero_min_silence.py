#!/usr/bin/env python3
"""Native, local-only viewer of prepared Silero audit artifacts."""
import argparse
import json
from pathlib import Path
import sys
import os
from datetime import datetime, timezone
from PySide6.QtCore import Qt, QTimer
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox


class Viewer(QWidget):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.items = json.loads((root/'items.json').read_text())
        assert len(self.items)==100
        self.index = 0
        self.playing = True
        self.setWindowTitle('Silero 6.2.3 — auditoria 100 / 200 / 300 ms')
        self.resize(1550, 950)
        layout=QVBoxLayout(self)
        heading=QLabel('Silero 6.2.3 · comparação controlada de silêncio mínimo')
        heading.setStyleSheet('font-size:24px;font-weight:bold')
        layout.addWidget(heading)
        controls=QHBoxLayout()
        for text, action in [('← Anterior', lambda:self.step(-1)), ('Próxima →', lambda:self.step(1))]:
            button=QPushButton(text)
            button.clicked.connect(action)
            controls.addWidget(button)
        self.pause=QPushButton('Pausar')
        self.pause.clicked.connect(self.toggle)
        controls.addWidget(self.pause)
        self.counter=QLabel()
        self.counter.setStyleSheet('font-size:19px;font-weight:bold')
        controls.addWidget(self.counter)
        self.countdown=QLabel()
        controls.addWidget(self.countdown)
        controls.addStretch()
        layout.addLayout(controls)
        self.title=QLabel()
        self.title.setStyleSheet('font-size:21px;font-weight:bold')
        layout.addWidget(self.title)
        self.svg=QSvgWidget()
        layout.addWidget(self.svg,stretch=1)
        summaries=QHBoxLayout()
        self.labels=[]
        for ms in (100,200,300):
            group=QGroupBox(f'min_silence_duration_ms = {ms}')
            box=QVBoxLayout(group)
            label=QLabel()
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.RichText)
            box.addWidget(label)
            self.labels.append(label)
            summaries.addWidget(group)
        layout.addLayout(summaries)
        footer=QLabel('16 kHz · entrada 0,50 · saída 0,35 · fala mínima 250 ms · margem 30 ms\n'
                     '+1 verde: atividade · −1 vermelho: não atividade. Sem pré-ênfase ou novo recorte RMS. Áudios separados disponíveis na pasta de cada amostra.')
        layout.addWidget(footer)
        self.paths=QLabel()
        self.paths.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.paths)
        self.timer=QTimer(self)
        self.timer.setInterval(3000)
        self.timer.timeout.connect(lambda:self.step(1))
        self.clock=QTimer(self)
        self.clock.setInterval(250)
        self.clock.timeout.connect(self.tick)
        self.clock.start()
        self.show_sample()
        self.timer.start()
        self.setStyleSheet('QWidget{background:#f5f7fa;color:#18344a;font-size:15px} QPushButton{background:white;padding:10px;border:1px solid #afbdca;border-radius:5px} QGroupBox{font-weight:bold;border:1px solid #afbdca;border-radius:6px;margin-top:12px;padding:12px} QGroupBox::title{subcontrol-origin:margin;left:10px} QLabel{border:0}')

    def tick(self):
        seconds=max(0,(self.timer.remainingTime()+999)//1000)
        self.countdown.setText(f'Próxima em {seconds//60}:{seconds%60:02d} · loop contínuo' if self.playing else 'Troca automática pausada')
        (self.root/'viewer_state.json').write_text(json.dumps(dict(index=self.index+1,count=100,
            name=self.items[self.index]['name'],automatic=self.playing,interval_ms=3000,
            remaining_seconds=seconds,pid=os.getpid(),heartbeat=datetime.now(timezone.utc).isoformat()),
            ensure_ascii=False,indent=2)+'\n')

    def show_sample(self):
        item=self.items[self.index]
        self.counter.setText(f'Amostra {self.index+1}/100')
        self.title.setText(item['name'])
        folder=self.root/item['folder']
        self.svg.load(str(folder/'comparacao.svg'))
        for label,c in zip(self.labels,item['conditions']):
            label.setText(f"<b>{c['activity_segments']} segmentos · {c['activity_percent']:.2f}% atividade</b><br>"
                f"Atividade: {c['activity_seconds']:.3f} s · Não atividade: {c['non_activity_seconds']:.3f} s<br>"
                f"Segmentos de atividade — média: {c['mean_segment_seconds']:.3f} s · menor: {c['min_segment_seconds']:.3f} s · maior: {c['max_segment_seconds']:.3f} s")
        self.paths.setText('Áudios e intervalos da amostra: '+str(folder))
        (self.root/'viewer_state.json').write_text(json.dumps(dict(index=self.index+1,count=100,
            name=item['name'],automatic=self.playing,interval_ms=3000),ensure_ascii=False,indent=2)+'\n')
        self.tick()

    def step(self,delta):
        self.index=(self.index+delta)%len(self.items)
        if self.playing:
            self.timer.start()
        self.show_sample()

    def toggle(self):
        self.playing=not self.playing
        self.pause.setText('Pausar' if self.playing else 'Continuar')
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()
        self.show_sample()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    parser.add_argument('--verify-offscreen',action='store_true')
    args=parser.parse_args()
    app=QApplication(sys.argv[:1])
    window=Viewer(args.results.resolve())
    window.show()
    if args.verify_offscreen:
        app.processEvents()
        assert window.timer.interval()==3000 and window.timer.isActive()
        window.step(-1)
        assert window.index==99
        window.step(1)
        assert window.index==0
        window.toggle()
        assert not window.timer.isActive()
        window.toggle()
        assert window.timer.isActive()
        window.timer.timeout.emit()
        assert window.index==1
        window.step(-1)
        app.processEvents()
        window.grab().save(str(args.results/'viewer_preview.png'))
        print('Verified: 100 samples, 3000 ms, loop, manual reset, pause/resume, automatic advance')
        window.close()
        return
    sys.exit(app.exec())


if __name__=='__main__':
    main()
