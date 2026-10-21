"""Interactive Qt canvas for Cos-specific production nodes and templates."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QPointF, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QBrush
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView,
    QGraphicsScene, QGraphicsRectItem, QGraphicsPathItem, QGraphicsItem,
    QLabel, QPushButton, QComboBox, QCheckBox, QSlider, QSplitter, QFileDialog)

from cos_workflow_graph import (build_graph, template_from_graph, validate_template,
                                preset, PRESETS)

STATUS = {'complete':'已准备', 'partial':'部分完成', 'pending':'待准备',
          'failed':'需处理', 'needs_attention':'需处理', 'unsupported':'待接入', 'not_requested':'本次无需',
          'running':'处理中', 'cancelled':'已取消'}
PORT_LABELS = {'DISPLAY_IMAGE':'显像 RGB', 'SUBJECT_PAIR':'原片人物与 alpha',
    'REFERENCE_IMAGE':'参考图', 'APPEARANCE_LAYER':'仿色图层', 'BACKGROUND_LAYER':'背景图层',
    'BOUNDARY_LAYER':'边缘受光层', 'EFFECT_LAYER':'前景光效层', 'COMPOSITE_IMAGE':'合成 RGB',
    'PHOTO_SOURCE':'原始照片与身份基准','OBSERVED_DESIGN':'观察后的具体设计',
    'GENERATED_EDIT':'完整生成编辑图','FROZEN_APPEARANCE':'冻结的生成外观',
    'SAVED_APPEARANCE':'已保存成片与母版','INDEPENDENT_REVIEW':'独立作品审查记录',
    'REVIEWED_BEST':'当前受审片支持的完整画面','EDIT_DECISION':'模型决定的下一项操作'}
PORT_LABELS['PHOTO_APPEARANCE']='冻结外观（原生与生成来源）'


class _Node(QGraphicsRectItem):
    def __init__(self, node, owner):
        super().__init__(0,0,230,130)
        self.node=node;self.owner=owner
        self.setFlags(QGraphicsItem.ItemIsMovable|QGraphicsItem.ItemIsSelectable|
                      QGraphicsItem.ItemSendsGeometryChanges)
        if owner.graph.get('readonly'):
            self.setFlag(QGraphicsItem.ItemIsMovable,False)
        self.setPos(*node['position']); self.setZValue(2)
        self.setToolTip(node['detail'])

    def paint(self,painter,option,widget=None):
        selected=self.isSelected();enabled=self.node['parameters']['enabled']
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor('#263343' if enabled else '#222832'))
        painter.setPen(QPen(QColor('#75cbbf' if selected else '#506173'),2 if selected else 1))
        painter.drawRoundedRect(self.rect(),10,10)
        painter.setPen(QColor('#eef2f5'));font=self.owner.font();font.setBold(True);font.setPointSize(12)
        painter.setFont(font);painter.drawText(14,29,self.node['label'])
        font.setBold(False);font.setPointSize(9);painter.setFont(font)
        painter.setPen(QColor('#b8c9d8'));painter.drawText(14,56,self.node['description'])
        state=STATUS.get(self.node['status'],self.node['status'])
        if self.node['editable'] and not enabled:state+=' · 已旁路'
        painter.setPen(QColor('#89d9c2' if self.node['status']=='complete' else '#e5c18e'))
        painter.drawText(14,86,state)
        if self.node['editable']:
            painter.setPen(QColor('#aabccc'));painter.drawText(14,112,f"强度 {self.node['parameters']['strength']*100:.0f}%")
        painter.setBrush(QColor('#75cbbf'));painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(0,65),5,5);painter.drawEllipse(QPointF(230,65),5,5)

    def itemChange(self,change,value):
        result=super().itemChange(change,value)
        if change==QGraphicsItem.ItemPositionHasChanged:
            self.owner._edges()
            if not self.owner.loading:
                self.node['position']=[self.pos().x(),self.pos().y()]
                self.owner._layout_changed()
        return result


class _View(QGraphicsView):
    def wheelEvent(self,event):
        if event.modifiers() & Qt.ControlModifier:
            scale=1.15 if event.angleDelta().y()>0 else 1/1.15
            current=self.transform().m11()
            if .25<=current*scale<=3:self.scale(scale,scale)
            event.accept()
        else:super().wheelEvent(event)


class CosWorkflowCanvas(QWidget):
    actionRequested=Signal(str)
    templateChanged=Signal(dict)

    def __init__(self,parent=None):
        super().__init__(parent)
        self.loading=True;self.graph=None;self.node_items={};self.connections=[]
        layout=QVBoxLayout(self);toolbar=QHBoxLayout()
        self.presets=QComboBox()
        for key,value in PRESETS.items():self.presets.addItem(value['name'],key)
        toolbar.addWidget(QLabel('Cos 工作流'));toolbar.addWidget(self.presets)
        apply=QPushButton('采用预设');apply.clicked.connect(self._preset);toolbar.addWidget(apply)
        self.template_buttons=[apply]
        for label,callback in (('打开模板',self._import),('保存模板',self._export),('适合画布',self._fit)):
            button=QPushButton(label);button.clicked.connect(callback);toolbar.addWidget(button)
            if label!='适合画布':self.template_buttons.append(button)
        toolbar.addStretch();layout.addLayout(toolbar)
        splitter=QSplitter(Qt.Horizontal);layout.addWidget(splitter,1)
        self.scene=QGraphicsScene(self);self.scene.setBackgroundBrush(QBrush(QColor('#17212c')))
        self.view=_View(self.scene);self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag);self.view.setMinimumHeight(440)
        splitter.addWidget(self.view)
        inspector=QWidget();panel=QVBoxLayout(inspector)
        self.title=QLabel('选择节点');self.title.setStyleSheet('font-size:18px;font-weight:600');panel.addWidget(self.title)
        self.detail=QLabel('点击查看模块，拖动整理画布。Ctrl + 滚轮缩放。');self.detail.setWordWrap(True);panel.addWidget(self.detail)
        self.ports=QLabel();self.ports.setWordWrap(True);panel.addWidget(self.ports)
        self.enabled=QCheckBox('使用此节点');self.enabled.toggled.connect(self._controls);panel.addWidget(self.enabled)
        self.strength_label=QLabel('强度 100%');panel.addWidget(self.strength_label)
        self.strength=QSlider(Qt.Horizontal);self.strength.setRange(0,100);self.strength.setValue(100)
        self.strength.valueChanged.connect(self._controls);panel.addWidget(self.strength)
        self.action=QPushButton('打开模块');self.action.clicked.connect(self._action);panel.addWidget(self.action)
        self.note=QLabel('连接由后台节点契约生成，当前使用预设连线。开关与强度控制已准备图层；没有结果的节点需先准备。')
        self.note.setWordWrap(True);panel.addWidget(self.note);panel.addStretch()
        inspector.setMinimumWidth(245);inspector.setMaximumWidth(340);splitter.addWidget(inspector)
        splitter.setStretchFactor(0,1)
        self.scene.selectionChanged.connect(self._selection)
        self.selected_id=None;self.loading=False;self._needs_fit=True
        self.set_run({})

    def set_run(self,run,report=None,template=None):
        previous=self.graph
        selected=self.selected_id
        positions={key:[item.pos().x(),item.pos().y()] for key,item in self.node_items.items()}
        transform=self.view.transform();center=self.view.mapToScene(self.view.viewport().rect().center())
        graph=build_graph(run,report,template=template)
        same=previous is not None and previous.get('run_id')==graph.get('run_id')
        stored=template or ((run.get('options') or {}).get('appearance_revision') or {}).get('workflow_graph') or {}
        if same:
            for node in graph['nodes']:
                if node['id'] in positions and node['id'] not in (stored.get('layout') or {}):
                    node['position']=positions[node['id']]
        self.loading=True;self.graph=graph;self.scene.blockSignals(True)
        readonly=bool(graph.get('readonly'))
        self.presets.setEnabled(not readonly)
        self.presets.setVisible(not readonly)
        for button in self.template_buttons:
            button.setEnabled(not readonly);button.setVisible(not readonly)
        for control in (self.enabled,self.strength,self.strength_label,self.action):
            control.setVisible(not readonly)
        self.note.setText(graph.get('notice') or ('本次按制作范围显示实际生产步骤；只读展示执行事实，不提供旧图层滑条。'+
            (' 原执行合同按历史签名保留。' if graph.get('historical_execution_graph_id') else '') if readonly else
            '连接由后台节点契约生成，当前使用预设连线。开关与强度控制已准备图层；没有结果的节点需先准备。'))
        self.connections=[];self.node_items={};self.scene.clear();self.selected_id=None
        for node in self.graph['nodes']:
            item=_Node(node,self);self.node_items[node['id']]=item;self.scene.addItem(item)
        for edge in self.graph['connections']:
            item=QGraphicsPathItem();item.setPen(QPen(QColor('#586f87'),2));item.setZValue(0)
            item.setToolTip(f"{edge['from']}.{edge['output']} → {edge['to']}.{edge['input']}\n{edge['type']}")
            self.scene.addItem(item);self.connections.append((edge,item))
        self._edges()
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-60,-60,60,60))
        if same and selected in self.node_items:self.node_items[selected].setSelected(True)
        self.scene.blockSignals(False);self.loading=False;self._selection()
        if same:
            self.view.setTransform(transform);self.view.centerOn(center)
            # Qt lays out scrollbars after the scene changes. Restore the center
            # once that layout settles; an older refresh cannot move a newer view.
            QTimer.singleShot(0,lambda:self.view.centerOn(center) if self.graph is graph else None)
        else:
            self._needs_fit=True
            if self.isVisible():self._fit()

    def _edges(self):
        for edge,item in self.connections:
            source=self.node_items[edge['from']];target=self.node_items[edge['to']]
            a=source.pos()+QPointF(230,65);b=target.pos()+QPointF(0,65)
            path=QPainterPath(a);span=max(70,abs(b.x()-a.x())*.5)
            path.cubicTo(a+QPointF(span,0),b-QPointF(span,0),b);item.setPath(path)

    def _selection(self):
        items=[item for item in self.scene.selectedItems() if isinstance(item,_Node)]
        self.selected_id=items[0].node['id'] if items else None
        node=items[0].node if items else None;self.loading=True
        self.title.setText(node['label'] if node else '选择节点')
        self.detail.setText(node['detail'] if node else self.graph.get('notice') or '点击查看模块，拖动整理画布。Ctrl + 滚轮缩放。')
        contract=node['contract'] if node else None
        if contract:
            inputs='、'.join(dict.fromkeys(PORT_LABELS.get(p['type'],p['type']) for p in contract['inputs'].values())) or '导入文件'
            outputs='、'.join(dict.fromkeys(PORT_LABELS.get(p['type'],p['type']) for p in contract['outputs'].values())) or '保存文件'
            alpha=('不使用源人物 alpha；按冻结外观坐标与责任区处理' if contract['alpha_role']=='not_applicable' else
                '生成原片人物 alpha' if contract['alpha_role']=='produce' else '保留原片人物 alpha')
            self.ports.setText(f'输入：{inputs}\n输出：{outputs}\n{alpha}')
            self.ports.setToolTip(json.dumps(contract,ensure_ascii=False,indent=2))
        else:self.ports.clear()
        self.enabled.setEnabled(bool(node and node['editable']))
        self.strength.setEnabled(bool(node and node['editable']))
        self.action.setEnabled(not self.graph.get('readonly') and
            (bool(node and self.graph.get('run_id')) or bool(node and node['id']=='source')))
        if node:
            self.enabled.setChecked(node['parameters']['enabled']);self.strength.setValue(round(node['parameters']['strength']*100))
            self.action.setText(node['action'])
        self.strength_label.setText(f'强度 {self.strength.value()}%');self.loading=False

    def _controls(self,*_):
        self.strength_label.setText(f'强度 {self.strength.value()}%')
        if self.loading or not self.selected_id or self.graph.get('readonly'):return
        item=self.node_items[self.selected_id]
        item.node['parameters']={'enabled':self.enabled.isChecked(),'strength':self.strength.value()/100}
        item.update();self.templateChanged.emit(self.template())

    def _layout_changed(self):
        if self.graph and not self.graph.get('readonly'):self.templateChanged.emit(self.template())

    def template(self):
        return template_from_graph(self.graph)

    def _action(self):
        if self.selected_id and not self.graph.get('readonly'):self.actionRequested.emit(self.selected_id)

    def _preset(self):
        value=preset(self.presets.currentData());current=self.template()
        value['layout']=current['layout'];self._apply_template(value)

    def _apply_template(self,value):
        if self.graph.get('readonly'):
            raise ValueError('本次生成外观流程只读，不能采用源图图层模板')
        value=validate_template(value);self.loading=True
        for key,params in value['nodes'].items():
            self.node_items[key].node['parameters']=copy.deepcopy(params);self.node_items[key].update()
        for key,position in value['layout'].items():
            item=self.node_items[key];item.setPos(*position);item.node['position']=list(position)
        self.loading=False;self._selection();self.templateChanged.emit(self.template())

    def _import(self):
        path,_=QFileDialog.getOpenFileName(self,'打开 Cos 工作流模板','','Cos 工作流 (*.json)')
        if not path:return
        try:self._apply_template(json.loads(Path(path).read_text(encoding='utf8')))
        except (OSError,ValueError,KeyError) as error:self.detail.setText('模板无法使用：'+str(error))

    def _export(self):
        path,_=QFileDialog.getSaveFileName(self,'保存 Cos 工作流模板','cos-workflow.json','Cos 工作流 (*.json)')
        if path:Path(path).write_text(json.dumps(self.template(),ensure_ascii=False,indent=2)+'\n',encoding='utf8')

    def _fit(self):
        if self.scene.items():
            self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-60,-60,60,60))
            self.view.fitInView(self.scene.sceneRect(),Qt.KeepAspectRatio)
            self._needs_fit=False

    def set_busy(self,busy):
        self.setEnabled(not busy)

    def showEvent(self,event):
        super().showEvent(event)
        if self._needs_fit:QTimer.singleShot(0,self._fit)
