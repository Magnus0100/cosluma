"""Exercise real Qt prompt switches with captured tasks and no network."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import qt_workbench as qt
from generation_prompt import DEFAULT_PROMPT, compose_generation_prompt


def main():
    assert qt.QT_AVAILABLE
    app = qt.QApplication.instance() or qt.QApplication([])
    class CapturedTasks:
        def __init__(self): self.tasks=[]
        def start(self, task): self.tasks.append(task)
    class Window(qt.Workbench):
        def refresh(self): pass
    temporary_workspace = tempfile.TemporaryDirectory(prefix='cosplay_prompt_ui_')
    window = Window(Path(temporary_workspace.name), 'http://127.0.0.1:1')
    tasks = CapturedTasks(); window.pool = tasks
    assert window.prompt.toPlainText()==DEFAULT_PROMPT
    custom = 'Keep my dry floor and use restrained violet scene illumination.'
    window.prompt.setPlainText(custom)
    window.preset.setCurrentIndex(window.preset.findData('night_scene'))
    night = window.preset_rows['night_scene']['prompt']
    assert window.prompt.toPlainText()==compose_generation_prompt(custom, night)
    window.preset.setCurrentIndex(window.preset.findData('cinematic_backlight'))
    back = window.preset_rows['cinematic_backlight']['prompt']
    assert window.prompt.toPlainText()==compose_generation_prompt(custom, back)
    assert night not in window.prompt.toPlainText()
    window.recommended_styles.clear()
    window.recommended_styles.addItem('fixture', {'id':'night_scene','english_direction':'Use soft violet environmental fill.'})
    window.apply_recommended_style()
    assert window.prompt.toPlainText()==compose_generation_prompt(custom, 'Use soft violet environmental fill.')
    window.preset.setCurrentIndex(window.preset.findData(''))
    assert window.prompt.toPlainText()==custom
    window.restore_generation_prompt()
    assert window.prompt.toPlainText()==DEFAULT_PROMPT
    assert not tasks.tasks
    window.close(); app.processEvents()
    temporary_workspace.cleanup()
    print(json.dumps({'ok':True,'real_Qt_style_switches':True,'custom_brief_preserved':True,
        'one_managed_style':True,'recommended_style_preserved_brief':True,'restore_current_default':True,'network_calls':0}))


if __name__=='__main__':
    main()
