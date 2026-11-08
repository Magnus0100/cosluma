"""One local project owns a photo queue, creative history and production batches."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from image_io import save_json


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class Project:
    def __init__(self, path, data):
        self.path = Path(path).expanduser().resolve()
        self.data = data

    @classmethod
    def create(cls, directory, name, *, workspace=None, creative=None, batches=None):
        name = name.strip()
        if not name:
            raise ValueError('请填写项目名称')
        directory = Path(directory).expanduser().resolve()
        path = directory/'project.json'
        if path.exists():
            raise FileExistsError('此文件夹已有项目，请打开它，或选择其他位置')
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('请选择新的或空的项目文件夹')
        directory.mkdir(parents=True, exist_ok=True)
        now = timestamp()
        state = {'sources':[], 'results':[], 'character':'', 'brief':'', 'page':0,
            'output_root':str(directory/'work'), 'output_edge':''}
        state.update(workspace or {})
        project = cls(path, {'format':'cosluma-project', 'version':1, 'id':uuid.uuid4().hex,
            'name':name, 'created_at':now, 'updated_at':now,
            'workspace':state, 'creative':deepcopy(creative or {}), 'batches':list(batches or [])})
        save_json(path, project.data)
        return project

    @classmethod
    def open(cls, path):
        path = Path(path).expanduser().resolve()
        if path.is_dir():
            path = path/'project.json'
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if data['format'] != 'cosluma-project' or data['version'] != 1:
            raise ValueError('此文件不是支持的 CosLuma 项目')
        if not data['id'] or not data['name'].strip():
            raise ValueError('项目缺少名称或标识')
        if not isinstance(data['workspace'], dict) or not isinstance(data['creative'], dict) or not isinstance(data['batches'], list):
            raise ValueError('项目的照片、创作或批次记录格式错误')
        state = data['workspace']
        if not isinstance(state['sources'], list) or not isinstance(state['results'], list) or not state['output_root']:
            raise ValueError('项目缺少照片列表、成片列表或保存位置')
        if state.get('page', 0) not in (0, 1, 2):
            raise ValueError('项目的工作页面记录错误')
        return cls(path, data)

    def save(self, workspace, creative, batches, *, delivery=None):
        values = {'workspace':workspace, 'creative':creative, 'batches':batches}
        if delivery is not None:
            values['delivery'] = delivery
        if all(self.data.get(key) == value for key, value in values.items()):
            return
        updated = {**self.data, **deepcopy(values), 'updated_at':timestamp()}
        save_json(self.path, updated)
        self.data = updated

    def save_delivery(self, delivery):
        self.save(self.data['workspace'],self.data['creative'],self.data['batches'],delivery=delivery)

    def rename(self, name):
        name = name.strip()
        if not name:
            raise ValueError('请填写项目名称')
        updated = {**self.data, 'name':name, 'updated_at':timestamp()}
        save_json(self.path, updated)
        self.data = updated


def recent_projects(preferences):
    return json.loads(preferences.value('projects/recent', '[]', type=str))


def remember_project(preferences, project):
    paths = [path for path in recent_projects(preferences)
        if Path(path).resolve() != project.path]
    preferences.setValue('projects/recent', json.dumps([str(project.path), *paths], ensure_ascii=False))
    preferences.sync()
