#!/usr/bin/env python3
"""Desktop PySide6 workbench for backend_server.py; no image algorithms."""
from __future__ import annotations
import argparse, copy, ctypes, hashlib, html, json, os, subprocess, sys, threading, urllib.request, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Iterable
from urllib.parse import urlparse
from runtime_resources import configure_cpu_runtime

configure_cpu_runtime()

FROZEN_QT_DIAGNOSTIC = {}
CRASH_TRACE_FILE = None


def post_completion_message(run):
    if run.get('status') in {'failed', 'blocked', 'cancelled', 'render_failed', 'needs_rerender'}:
        return '制作未完成，输入和可用结果已保留；请查看任务详情后继续。'
    completion = run.get('system_post_completion') or {}
    if (completion.get('status') == 'completed_with_findings' or completion.get('failed_checks')
            or completion.get('unknown_checks') or completion.get('unfulfilled_independent_effects')):
        return '后期处理已结束，仍有未通过或待确认的审片项；请在审片页查看。'
    effects = next((row for row in completion.get('stages', []) if row.get('id') == 'independent_effects'), {})
    if effects.get('status') == 'skipped':
        return '后期处理已结束；本次未新增独立特效，请审片确认参考中的效果。'
    return '后期处理已结束；请在审片页核对已保存版本并批准或继续调整。'


def _enable_crash_trace():
    """Keep an opt-in native crash trace open for frozen Windows diagnostics."""
    global CRASH_TRACE_FILE
    destination = os.environ.get("COSPLAY_CRASH_TRACE")
    if destination:
        import faulthandler
        target = Path(destination).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        CRASH_TRACE_FILE = target.open("a", encoding="utf-8")
        faulthandler.enable(file=CRASH_TRACE_FILE, all_threads=True)


_enable_crash_trace()


def _prepare_frozen_qt_runtime():
    """Make PySide6's Qt DLLs discoverable in a PyInstaller one-folder build."""
    if not getattr(sys, "frozen", False):
        return
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    FROZEN_QT_DIAGNOSTIC["meipass"] = str(bundle_root)
    # Qt's ICU and MSVC support DLLs are collected at the bundle root while
    # the PySide6 extension modules live in subdirectories, so all three
    # locations must participate in Windows DLL resolution.
    candidates = [bundle_root, bundle_root / "PySide6", bundle_root / "shiboken6"]
    for directory in candidates:
        FROZEN_QT_DIAGNOSTIC[str(directory)] = directory.is_dir()
        if directory.is_dir():
            try:
                os.add_dll_directory(str(directory))
            except (AttributeError, OSError):
                pass
    existing_path = os.environ.get("PATH", "")
    qt_paths = os.pathsep.join(str(p) for p in candidates if p.is_dir())
    if qt_paths:
        os.environ["PATH"] = qt_paths + os.pathsep + existing_path
    # Some Windows installations still ignore the added DLL directory for a
    # transitive Qt dependency. Preload the three libraries imported by this
    # module so QtCore.pyd resolves consistently from a one-folder bundle.
    for library in ("Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll"):
        path = bundle_root / "PySide6" / library
        if path.is_file():
            try:
                ctypes.WinDLL(str(path))
            except OSError as exc:
                FROZEN_QT_DIAGNOSTIC[library] = repr(exc)
            else:
                FROZEN_QT_DIAGNOSTIC[library] = "loaded"


_prepare_frozen_qt_runtime()

try:
    from PySide6.QtCore import QEvent, QObject, QRunnable, QThreadPool, QTimer, QRect, QSize, QPointF, Qt, Signal, Slot
    from PySide6.QtGui import QColor, QIcon, QImage, QImageReader, QPainter, QPixmap, QPolygonF
    from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QColorDialog, QFormLayout, QGroupBox, QHBoxLayout,
        QLabel, QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem, QMainWindow, QDialog, QMessageBox, QPushButton, QPlainTextEdit,
        QScrollArea, QSlider, QSpinBox, QSplitter, QTabWidget, QToolButton, QSizePolicy, QLineEdit, QTextEdit, QDoubleSpinBox,
        QVBoxLayout, QWidget, QFileDialog, QGroupBox, QCheckBox, QColorDialog, QInputDialog)
    QT_AVAILABLE = True
except ImportError as exc:
    QT_AVAILABLE = False
    QT_IMPORT_ERROR = repr(exc)
    if FROZEN_QT_DIAGNOSTIC:
        QT_IMPORT_ERROR += " diagnostics=" + repr(FROZEN_QT_DIAGNOSTIC)

def _default_repo_root() -> Path:
    """Choose a usable project root for source and frozen launches."""
    from workspace_location import resolve_workspace
    fallback = (Path.home() / "CosplayWorkbench/workspace" if getattr(sys, "frozen", False)
                else Path(__file__).resolve().parents[1])
    return resolve_workspace(fallback, argv=sys.argv[1:])


REPO_ROOT = _default_repo_root()

VISUAL_CHECK_KEYS = (
    "identity", "face_eyes", "hands_fingers", "hair_halo",
    "outline_light", "garment_texture", "background_light_effects",
    "overall_impression",
)
VISUAL_CHECK_LABELS = {
    "identity": "人物身份/构图",
    "face_eyes": "脸部与眼睛",
    "hands_fingers": "手部与手指",
    "hair_halo": "头发与轮廓光",
    "outline_light": "外轮廓与光效",
    "garment_texture": "服装与纹理",
    "background_light_effects": "背景与光效",
    "overall_impression": "整体观感",
}

CREATIVE_TASK_MODES = {
    'local_portrait': '自然原片后期',
    'existing_reference': '已有参考并完成后期（默认零生图）',
    'create_reference': '创作参考并完成后期',
}

# New workspaces prefer the full-precision local quality workflow. A saved
# or explicit user choice still wins; an unavailable model is reported.
PREFERRED_LOCAL_GENERATION_PROVIDER = 'qwen_image_2_1'


def generation_readiness_text(row):
    """Keep configuration evidence separate from service and model readiness."""
    configuration = '配置已检查' if row.get('ready') else {
        'missing_models': '模型文件未齐', 'missing_workflow': '需配置工作流',
        'missing_endpoint': '需配置地址', 'missing_key': '需配置凭据',
        'disabled': '已停用'}.get(row.get('status'), '待配置')
    if str(row.get('id') or row.get('registry_id') or '').startswith('qwen_image_') or str(row.get('model_env') or '').startswith('COSPLAY_QWEN_'):
        service = ('服务可连接' if row.get('service_ready') is True else '服务未连接') if row.get('network_checked') is True else '服务未检测'
        return configuration + ' · ' + service + ' · 模型加载未验证'
    return configuration


def image_submission_budget_text(budget, *, autofinish=False, source_native=False):
    """Count every image-model stage, including a local regional continuation."""
    names = (('initial_reference', '首次编辑额度' if source_native else '全局参考'), ('reference_correction', '参考纠错'),
             ('independent_effect', '独立效果'), ('regional_refinement', '局部精修'))
    if 'editing_refinement' in budget: names += (('editing_refinement', '条件自动纠偏' if autofinish else '审片改进'),)
    total = sum(int(budget.get(key) or 0) for key, _ in names)
    return ('新增图像模型提交总上限：' + str(total) + '；' + ' / '.join(
        label + ' ' + str(budget.get(key) or 0) for key, label in names))


def creative_task_plan_text(plan: dict) -> str:
    """Present the backend's checked facts without inventing readiness or prices."""
    iteration=plan.get('editing_iteration') or {}
    shaping=plan.get('shaping_request') or {}
    lines = [(('恢复形体调整前版本' if shaping.get('action')=='restore' else '生成可撤销形体候选') if shaping else
              ('按用户反馈改进当前保存版本' if iteration.get('user_feedback') else '按独立审片改进当前保存版本') if iteration else
              str(plan.get('label') or CREATIVE_TASK_MODES.get(plan.get('mode'), '制作计划'))),
             '计划检查通过；开始时会重新核验输入与配置。' if plan.get('ready') is True else '尚不能开始；请处理以下缺口。']
    lines += ['缺口：' + str(reason) for reason in plan.get('blocking_reasons', [])]
    if shaping:
        names = {'face':'脸部轮廓', 'waist':'腰身', 'arms':'手臂', 'legs':'腿部轮廓'}
        lines += [('明确授权：'+ '、'.join(names.get(key,key)+' '+str(value)
                    for key,value in shaping.get('parts',{}).items())) if shaping.get('action')=='apply' else
                  '恢复完整调整前版本；不调用模型，也会撤去此后基于形体版本所做的其它修改。',
                  '预览只产生保存候选；明确采用后才切换当前版本，不等于人工批准。']
    if iteration.get('user_feedback'):
        lines += ['本轮用户反馈：' + iteration['user_feedback'],
                  '父版本自动审片作为历史基线保留；不代表本轮反馈目标已满足。']
    scope = plan.get('editing_scope') or {}
    if scope:
        from photo_editing import WORK_TYPES, STYLES
        lines += ['制作范围：' + WORK_TYPES.get(scope.get('work_type'), '未知')
                  + ' · 风格表达：' + STYLES.get(scope.get('style'), '未知'),
                  '场景目标：' + (str(scope.get('scene_target') or '原场地内调整')
                      if scope.get('work_type') == 'full_composite' else '保留原场地及室内外关系'),
                  '室内外目标：' + {'same': '保持原片', 'indoor': '明确改为室内', 'outdoor': '明确改为室外'}.get(scope.get('target_setting'), '未知'),
                  ('本地精修不向外部视觉服务发送原片；制作范围不等于已通过审片。'
                   if plan.get('mode') == 'local_portrait' else
                   '从同一冻结调整前版本开始；修改强度不在上次结果上叠加。' if shaping else
                   '复用父版本已冻结的原图观察与范围；本轮只执行决策、选定工具与独立复审。' if iteration else
                   '创作设计在开始后实际观察原片；计划检查不等于已设计、已生成或已通过审片。')]
    direction = plan.get('creative_direction') or {}
    role=(((plan.get('intent') or {}).get('constraints') or {}).get('reference_role')
          or (plan.get('reference_contract') or {}).get('role'))
    if role == 'same_photo_finish':
        lines += ['参考用途：已冻结同照精修匹配意图；尚未实际配准，执行时验证同照关系与有效范围。',
                  '原片拥有几何和原生纹理；参考仅为光色与修饰标尺，不直接复制参考 RGB。新增生图 0 次。']
    elif role == 'source_conditioned_design':
        lines += ['参考用途：原图单独创作设计参考；不需要人工精修图作为输入。' if plan.get('mode')=='create_reference'
                  else '参考用途：复用已核验同源生成的设计参考，不重新生图。',
                  '原片拥有几何和原生细节；执行时核验生成来源与配准，再将设计光色映射到原尺寸。']
    if direction:
        lines += ['冻结角色：' + str((direction.get('character') or {}).get('name') or '未指定'),
                  '场景：' + str(direction.get('scene') or ''),
                  '主效：' + str((direction.get('main_effect') or {}).get('description') or '未指定'),
                  ('依据：内置推荐与原始请求分别冻结；没有自动研究或批准。' if direction.get('recommendation')
                   else '依据：用户要求离线编译；没有自动研究或批准。')]
    recommendation = direction.get('recommendation') or {}
    if recommendation:
        lines += ['产品推荐目标：原景精修 v' + str(recommendation.get('version'))
                  + ' · 内容 ' + str(recommendation.get('profile_sha256') or '')[:12],
                  '推荐来自内置产品目标，用户原始作品要求保持独立；暖光、保留彩灯、无特效及分域禁止优先。',
                  '肤色随本人和所选照明，不统一漂白；实际完成域与质量须看保存结果，不等于已批准。']
        if plan.get('refinement_inheritance'):
            lines.append('推荐继承自已保存的 checked 目标；同原片、参考与版本证明已冻结，新增参考图 0 次。')
    inputs = plan.get('inputs') or {}
    for key, label in (('source', '原片'), ('reference', '参考')):
        node = inputs.get(key) or {}
        if node.get('path'):
            lines.append(label + '：' + Path(str(node['path'])).name + ' · ' + str(node.get('sha256') or '')[:12])
    budget = plan.get('image_budget') or {}
    auto=(plan.get('editing_autofinish') or {}).get('maximum_improvements',0)
    zero_finish=(plan.get('editing_autofinish') or {}).get('maximum_zero_image_finishes',0)
    lines.append(image_submission_budget_text(budget,autofinish=bool(auto),source_native=bool(plan.get('editing_initialization'))))
    if plan.get('editing_initialization'):
        lines += ['从原图原生精度基线开始，由模型选择首次局部或整体编辑；图像额度不代表必定整图重画。',
            f'首次编辑后最多再改 {auto} 轮，另有 {zero_finish} 轮零生图收尾；候选共享额度，实际调用见回执。' if auto else
            '本次只执行首次编辑；未启用额外纠偏。',
            '基线保存不代表完成修改；未审稿与自动审片结果、人工批准分别记录。']
    elif plan.get('editing_autofinish') is not None:
        lines.append('历史任务按已保存事实只读展示；新制作须重新检查当前任务。')
    if budget.get('regional_refinement'):
        lines.append('局部精修为本地 GPU 图像模型推理，占用本机计算时间；不是外部付费图像请求。人物安全范围仍由后台核验。')
    if plan.get('regional_candidate_reuse_authorization'):
        lines.append('复用已返回局部候选，仅本地光度对齐与合成；新增生图 0 次。原模型画布与本次 CPU 区域分别冻结。')
    stages = [str(row.get('label') or row.get('id') or '') for row in plan.get('stages', []) if isinstance(row, dict)]
    capabilities = plan.get('capabilities') or {}
    missing = [str(row.get('label') or row.get('key')) for row in capabilities.get('models', [])
               if isinstance(row, dict) and row.get('available') is not True]
    preflight = capabilities.get('model_preflight') or {}
    model_status = ('缺少 ' + '、'.join(missing) if missing else
                    '计划检查通过' if preflight.get('ready') is True else '请查看模型预检缺口')
    lines.append('分析模型：' + model_status)
    lines.append('模型检查：模型文件/小文件身份与配置检查；权重内容和推理在执行验证。')
    services = plan.get('services') or {}
    if services.get('text_vision'):
        lines.append('文本/视觉服务：' + str(services.get('provider') or '待配置') +
                     (' · ' + str(services['endpoint']) if services.get('endpoint') else ''))
        lines.append('将发送：' + '、'.join(str(value) for value in services.get('sent_content', [])))
        count = services.get('text_vision_requests_upper_bound')
        lines.append('文本/视觉请求上限：' + (str(count) if count is not None else '尚无法估计；按有限阶段执行，实际用量见回执'))
        lines.append('费用：' + str(services.get('price') or '价格未知'))
    else:
        lines.append('文本/视觉服务：不使用；不向该服务发送图片或创作要求。')
    lines.append(('独立审片：' if scope else '自动审片与诊断修复：') + (
        ('已启用，额外发送原片与成片，费用与用量见服务回执；不等于人工批准。' if scope else
         '已启用，会额外发送成片与局部审片窗口。')
        if services.get('visual_review') else '未启用，保存后由人工审片。'))
    seconds = plan.get('estimated_seconds')
    lines.append('耗时：' + (str(seconds) + ' 秒' if seconds is not None else '未知；' + str(plan.get('estimate_basis') or '尚无当前硬件统计')))
    lines.append('观察边界：整图与匹配原尺寸窗口分别审查；局部通过不认证全图原生细节，人工批准另行记录。')
    if stages: lines.append('处理步骤：' + ' → '.join(stages))
    lines += ['范围：' + str(value) for value in plan.get('limitations', [])]
    return '\n'.join(lines)

def api_call(base: str, method: str, path: str, body: Optional[dict] = None) -> dict:
    payload = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base.rstrip("/") + path, data=payload, headers=headers, method=method)
    # Preparation includes vision, cited web research and a second image plan.
    # It runs in ApiTask, so its network budget does not block the GUI thread.
    timeout = 600 if path.endswith(( "/auto-color", "/reconstruct-design", "/native-detail-preview")) else 120
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            value = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        detail = str(error)
        try:
            failure = json.loads(error.read(65536).decode('utf8'))
            reported = failure.get('error') if isinstance(failure, dict) else None
            if isinstance(reported, dict): reported = reported.get('message')
            if isinstance(reported, str) and reported.strip(): detail = reported[:1500]
        except (OSError, UnicodeError, ValueError):
            pass
        finally:
            error.close()
        raise RuntimeError(f'HTTP {error.code}: {detail}') from error
    return value if isinstance(value, dict) else {"value": value}

def api_stream(base: str, path: str, on_event) -> None:
    """Consume a dependency-free SSE endpoint until the server closes it."""
    req = urllib.request.Request(
        base.rstrip("/") + path,
        headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=75) as response:
        event_name = "message"; event_id = None; data_lines = []
        for raw in response:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                if data_lines:
                    raw_data = "\n".join(data_lines)
                    try:
                        data = json.loads(raw_data)
                    except json.JSONDecodeError:
                        data = {"raw": raw_data}
                    on_event({"event": event_name, "id": event_id, "data": data})
                event_name = "message"; event_id = None; data_lines = []
            elif line.startswith(":"):
                continue
            elif line.startswith("event:"):
                event_name = line[6:].strip() or "message"
            elif line.startswith("id:"):
                event_id = line[3:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

def asset_path(node: Any, root: Path) -> Optional[Path]:
    if not isinstance(node, dict) or not node.get("path"):
        return None
    path = Path(str(node["path"]))
    return path if path.is_absolute() else (root / path).resolve()


def native_runtime_smoke(root: Path) -> dict:
    """Load the shipped native image runtime as well as the Qt UI runtime."""
    result = {}
    try:
        import numpy
        result["numpy"] = {"ok": True, "version": numpy.__version__, "array_result": (numpy.asarray([1, 2, 3]) * 2).tolist()}
    except Exception as exc:
        result["numpy"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        import rawpy
        result["rawpy"] = {"ok": True, "version": rawpy.__version__}
    except Exception as exc:
        result["rawpy"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        if getattr(sys, "frozen", False):
            import onnxruntime
        else:
            from source_alpha import _runtime
            onnxruntime = _runtime(root)
        result["onnxruntime"] = {"ok": True, "version": onnxruntime.__version__}
    except Exception as exc:
        result["onnxruntime"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        from design_reconstruction_pipeline import production_code_report
        result["production_code"] = production_code_report()
    except Exception as exc:
        result["production_code"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        from face_contour_deform import producer_identity
        result['face_contour_runtime']={'ok':True,'producer':producer_identity(),
            'detector_executed':False,'image_generation_submissions':0}
    except Exception as exc:
        result['face_contour_runtime']={'ok':False,'error':f'{type(exc).__name__}: {exc}'}
    result["ok"] = all(result[module]["ok"] for module in ("numpy", "rawpy", "onnxruntime", "production_code",'face_contour_runtime'))
    return result


def capability_declaration() -> dict[str, Any]:
    """Declare the current production contract exposed to product actions."""
    native = {
        "render_mode": "native_source_design",
        "read_only_compatibility": False,
        "layers": ["source", "reference", "final", "alpha", "silhouette", "risk"],
        "alignment_revision_supported": False,
        "appearance_transfer_supported": False,
        "appearance_policies": [],
        "candidate_person_rgb_used": False,
        "source_colour_match_contract": "reference_appearance_stack.v2",
        "source_colour_match_supported_contracts": ["reference_appearance_stack.v2"],
        "source_colour_match_preparation": "reference_pipeline.prepare",
        "reference_appearance_stack_supported": True,
        "reference_appearance_strength_range": [0, 1],
        "independent_transfer_blend_scene_controls": False,
        "independent_colour_lightness_controls": True,
        "independent_colour_lightness_requires": "reference_appearance_stack.v2",
        "reference_region_compatibility_report": "reference_support.v1",
        "reference_working_domain": "display_linear_sdr",
        "reference_display_transform": "clip_sdr",
        "workflow": ["source_retouch", "original_and_reference_colour_match", "composition", "independent_effects", "comprehensive_review"],
        "complete_automatic_visual_result_claimed": False,
        "visual_quality_inferred_from_worker_success": False,
        "saved_native_images_viewable": True,
        "physical_relighting_supported": False,
        "explicit_complete_post_supported": True,
        "automatic_native_retouch_supported": True,
        "automatic_material_finish_supported": True,
        "independent_rgba_effect_supported": True,
        "comprehensive_native_review_supported": True,
        "system_model_choices": ["current"],
        "existing_design_reference_import_supported": True,
        "creative_task_batch_supported": True,
        "creative_task_batch_modes": list(CREATIVE_TASK_MODES),
        "creative_task_batch_geometry_shared": False,
        "creative_task_batch_failed_item_retry": True,
        "creative_task_batch_restart_recovery": True,
        "existing_reference_image_budget": 0,
        "saved_version_batch_export_supported": True,
        "saved_version_batch_export_rerenders": False,
        "local_reconstruction_preparation_durable": True,
        "local_reconstruction_cancel_mode": "cooperative_subprocess",
        "prepared_reconstruction_requires_explicit_child_save": True,
        "manual_layer_controls_supported": True,
        "manual_layer_names": ["background", "particles", "outline_light", "garment_texture", "clone_retouch", "tone_curves", "hsl_adjust", "local_colour_brush"],
        "local_colour_brush_supported": True,
        "local_colour_brush_scopes": ["foreground", "background", "composite"],
        "alpha_revision_supported": False,
        "native_detail_supported": True,
        "render_engine": "source-float-v5-explicit-working-domain",
        "source_precision_modes": ["raw16_linear", "raster8_srgb"],
        "export_formats": ["JPEG", "PNG", "TIFF", "TIFF16", "TIFF32_LINEAR"],
        "export_actual_bit_depth": {"JPEG": 8, "PNG": 8, "TIFF": 8, "TIFF16": 16, "TIFF32_LINEAR": 32},
    }
    return {
        "workflow_mode": "single_source",
        "default_render_mode": "native_source_design",
        "project_cleanup_supported": True,
        "project_cleanup_scope": "unreferenced_stopped_workspace_records",
        "native": native,
    }

if QT_AVAILABLE:
    class Signals(QObject):
        result = Signal(object)
        error = Signal(str)
        finished = Signal()

    class ApiTask(QRunnable):
        def __init__(self, base: str, method: str, path: str, body: Optional[dict], label: str):
            super().__init__(); self.base, self.method, self.path, self.body, self.label = base, method, path, body, label; self.signals = Signals()
        @Slot()
        def run(self):
            try: self.signals.result.emit(api_call(self.base, self.method, self.path, self.body))
            except Exception as exc: self.signals.error.emit(f"{self.label}: {exc}")
            finally: self.signals.finished.emit()

    class ShapingInspectionTask(QRunnable):
        """Local read-only byte/numeric inspection, including a cold first read."""
        def __init__(self, root, kind, run, label, *, parent_run=None, parent_identity=None, request=None):
            super().__init__(); self.root, self.kind, self.run_record, self.label = root, kind, run, label
            self.parent_run, self.parent_identity, self.request = parent_run, parent_identity, request
            self.cancel = threading.Event(); self.signals = Signals()
        @Slot()
        def run(self):
            try:
                from shaping_inspection import inspect_parent, inspect_candidate
                if self.kind == 'parent':
                    value = inspect_parent(self.root, self.run_record, cancel=self.cancel.is_set)
                else:
                    value = inspect_candidate(self.root, self.run_record, parent_run=self.parent_run,
                        parent_identity=self.parent_identity, request=self.request, cancel=self.cancel.is_set)
                self.signals.result.emit(value)
            except Exception as error:
                from shaping_inspection import observe_run
                self.signals.result.emit({'ok':False, 'reason':'只读检查失败：'+str(error),
                    'snapshot':observe_run(self.root, self.run_record), 'watch_paths':[]})
            finally: self.signals.finished.emit()

    class SourcePreviewTask(QRunnable):
        """Decode RAW and cache its viewing proxy away from the GUI thread."""
        def __init__(self, root: Path, source: Path):
            super().__init__(); self.root, self.source = root, source; self.label = "source-preview"; self.signals = Signals()
        @Slot()
        def run(self):
            try:
                from cache_store import ContentCache
                from raw_support import open_rgb, storage_info, RGB_DECODE_VERSION
                cache = ContentCache(self.root); proxy = cache.find_latest_proxy(self.source, params={"max_edge": 2048, "decode_version": RGB_DECODE_VERSION})
                # Use the stat-bound metadata memo for both raster and RAW
                # inputs.  A proxy hit must not reopen a camera container via
                # LibRaw merely to rediscover its oriented dimensions.
                facts = storage_info(self.source)
                width, height = facts['oriented_size']
                if not proxy:
                    image = open_rgb(self.source); image.thumbnail((2048, 2048))
                    proxy = Path(cache.put_proxy(self.source, image, params={"max_edge": 2048, "decode_version": RGB_DECODE_VERSION}, dimensions=[width, height])["path"])
                self.signals.result.emit({"source": str(self.source), "proxy_path": str(proxy), "width": width, "height": height})
            except Exception as exc: self.signals.error.emit(str(exc))
            finally: self.signals.finished.emit()

    class StartupServiceTask(QRunnable):
        """Prepare the selected owned service, never a photo/model inference."""
        def __init__(self,provider,endpoint,cancel):
            super().__init__();self.provider,self.endpoint,self.cancel=provider,endpoint,cancel
            self.signals=Signals()
        @Slot()
        def run(self):
            try:
                if self.cancel.is_set():return
                from start_qwen_image import ensure_running
                result=ensure_running(self.provider,self.endpoint,cancel=self.cancel.is_set)
                if not isinstance(result,dict) or result.get('managed') is not True:
                    raise RuntimeError('The selected Qwen service has no verified managed identity')
                if not self.cancel.is_set():
                    self.signals.result.emit({'state':'service_ready','service':result,
                        'model_load_verified':False,'model_warming_performed':False,
                        'model_requests':0,'image_submissions':0})
            except Exception as error:
                if not self.cancel.is_set():self.signals.error.emit(str(error))
            finally:self.signals.finished.emit()

    class JobStreamTask(QRunnable):
        """Long-lived SSE reader; normal API requests remain QRunnable based."""
        def __init__(self, base: str, path: str):
            super().__init__(); self.base, self.path = base, path; self.signals = Signals()
        @Slot()
        def run(self):
            try:
                api_stream(self.base, self.path, self.signals.result.emit)
            except Exception as exc:
                self.signals.error.emit(str(exc))
            finally:
                self.signals.finished.emit()

    class TaskRelay(QObject):
        """Give worker callbacks an explicit receiver on the GUI thread."""
        def __init__(self, owner, label, observation, stream=False):
            super().__init__(owner)
            self.owner, self.label, self.observation, self.stream = owner, label, observation, stream
            self.backend_generation = owner.backend_generation
            self.model_setup_observation = owner.model_setup_observation
        def current(self):
            return (self.backend_generation == self.owner.backend_generation and
                    (self.label not in {'model-setup', 'model-setup-saved'} or self.model_setup_observation == self.owner.model_setup_observation) and
                    (self.label not in {"job-status", "job-events", "job-run-loaded", "job-stream"} or self.observation == self.owner.job_observation))
        @Slot(object)
        def result(self, value):
            if self.current():
                self.owner.result(self.label, value)
                self.owner.refresh_cos_workspace()
        @Slot(str)
        def error(self, error):
            if not self.current(): return
            if self.current():
                if self.stream: self.owner.job_stream_error(error)
                else: self.owner.error(error, self.label)
                self.owner.refresh_cos_workspace()
        @Slot()
        def finished(self):
            if self.stream and self.current(): self.owner.job_stream_finished()
            self.owner.pending_tasks.pop(self, None)
            self.deleteLater()

    class ToneCurveCanvas(QWidget):
        curveEdited=Signal(object)
        def __init__(self):
            super().__init__();self.points=[[0.,0.],[1.,1.]];self.channel='RGB';self.drag_index=None;self.setMinimumHeight(245)
        def set_curve(self,points,channel):
            self.points=json.loads(json.dumps(points));self.channel=channel;self.drag_index=None;self.update()
        def graph_rect(self):
            return QRect(28,12,max(1,self.width()-42),max(1,self.height()-40))
        def graph_point(self,point):
            rect=self.graph_rect();return QPointF(rect.left()+point[0]*rect.width(),rect.bottom()-point[1]*rect.height())
        def mousePressEvent(self,event):
            rect=self.graph_rect();x=min(1.,max(0.,(event.position().x()-rect.left())/rect.width()));y=min(1.,max(0.,(rect.bottom()-event.position().y())/rect.height()))
            distances=[(self.graph_point(point)-event.position()).manhattanLength() for point in self.points];index=min(range(len(distances)),key=distances.__getitem__)
            if event.button()==Qt.RightButton:
                if distances[index]<18 and 0<index<len(self.points)-1:self.points.pop(index);self.curveEdited.emit(self.points);self.update()
                return
            if event.button()!=Qt.LeftButton:return
            if distances[index]>18:
                if x<.005:index=0
                elif x>.995:index=len(self.points)-1
                else:
                    self.points.append([x,y]);self.points.sort(key=lambda point:point[0]);index=next(i for i,point in enumerate(self.points) if point[0]==x)
            self.drag_index=index;self.move_point(x,y);self.update()
        def move_point(self,x,y):
            index=self.drag_index
            if index==0:x=0.
            elif index==len(self.points)-1:x=1.
            else:x=max(self.points[index-1][0]+.003,min(self.points[index+1][0]-.003,x))
            self.points[index]=[float(x),float(y)]
        def mouseMoveEvent(self,event):
            if self.drag_index is None:return
            rect=self.graph_rect();x=min(1.,max(0.,(event.position().x()-rect.left())/rect.width()));y=min(1.,max(0.,(rect.bottom()-event.position().y())/rect.height()))
            self.move_point(x,y);self.update()
        def mouseReleaseEvent(self,event):
            if event.button()==Qt.LeftButton and self.drag_index is not None:
                self.drag_index=None;self.curveEdited.emit(json.loads(json.dumps(self.points)));self.update()
        def paintEvent(self,event):
            import numpy as np
            from tone_curves import curve_values
            painter=QPainter(self);painter.setRenderHint(QPainter.Antialiasing);rect=self.graph_rect();painter.fillRect(rect,QColor('#181b24'))
            painter.setPen(QColor('#353946'))
            for t in (.0,.25,.5,.75,1.):
                painter.drawLine(self.graph_point([t,0]),self.graph_point([t,1]));painter.drawLine(self.graph_point([0,t]),self.graph_point([1,t]))
            painter.setPen(QColor('#596171'));painter.drawLine(self.graph_point([0,0]),self.graph_point([1,1]))
            color={'RGB':'#ded8ff','R':'#ff7c8c','G':'#7dda9e','B':'#84adff'}[self.channel];painter.setPen(QColor(color))
            x=np.linspace(0,1,129,dtype=np.float32);y=curve_values(x,self.points)
            for i in range(len(x)-1):painter.drawLine(self.graph_point([x[i],y[i]]),self.graph_point([x[i+1],y[i+1]]))
            painter.setBrush(QColor(color))
            for point in self.points:painter.drawEllipse(self.graph_point(point),4.5,4.5)
            painter.setPen(QColor('#acafbd'));painter.drawText(10,rect.bottom()+18,'0');painter.drawText(rect.right()-20,rect.bottom()+18,'255');painter.drawText(1,rect.top()+10,'255')
            painter.end()

    class Preview(QWidget):
        colorStrokeFinished = Signal(object)
        cloneSourcePicked = Signal(object)
        cloneStrokeFinished = Signal(object)
        localColourStrokeFinished = Signal(object)
        nativeDetailRequested = Signal(object)
        receiverGeometryEdited = Signal(object)
        receiverDraftChanged = Signal(object)
        def __init__(self, name: str):
            super().__init__(); self.path = None; self.proxy_path = None; self.zoom = 100; self._pan_origin = None; self._pan_start = None; self.region_rect = None
            self.light_fields = []; self.selected_light_id = ""; self.show_light_fields = False
            self.native_detail_enabled = False
            self.receiver_active = False; self.receiver_mode = 'pan'; self.receiver_canvas_size = None
            self.receiver_geometry = {'receiver_region': None, 'receiver_plane': None, 'foot_anchors': {}}
            self.receiver_visible = False; self._receiver_draft = []; self._receiver_drag = None
            self.local_brush_active=False; self.local_brush_mode='paint'; self.local_brush_radius=.01; self._local_brush_stroke=None; self._local_brush_hover=None;self.selected_local_brush=None
            self.matte_strokes=[]
            self.clone_active=False; self.clone_source=None; self.clone_radius=.01; self._clone_stroke=None; self._clone_hover=None
            self.color_selection = None; self.show_color_selection = False; self.color_brush_mode = "pan"; self.color_brush_radius = .02; self._color_stroke = None; self._base_image = QImage()
            layout = QVBoxLayout(self); self.path_label = QLabel(name)
            controls = QHBoxLayout(); controls.addWidget(QLabel("缩放"))
            self.zoom_slider = QSlider(Qt.Horizontal); self.zoom_slider.setRange(5, 200); self.zoom_slider.setValue(100)
            self.zoom_value = QLabel("100%")
            self.fit_button = QPushButton("适屏")
            controls.addWidget(self.zoom_slider, 1); controls.addWidget(self.zoom_value); controls.addWidget(self.fit_button)
            self.image = QLabel("暂无文件"); self.image.setAlignment(Qt.AlignCenter)
            self.scroll = QScrollArea(); self.scroll.setWidgetResizable(False); self.scroll.setWidget(self.image)
            self.image.installEventFilter(self); self.image.setMouseTracking(True)
            self.image.setFocusPolicy(Qt.ClickFocus)
            self.scroll.setAlignment(Qt.AlignCenter)
            self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded); self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.scroll.viewport().installEventFilter(self); self.scroll.viewport().setCursor(Qt.OpenHandCursor)
            layout.addWidget(self.path_label); layout.addLayout(controls); layout.addWidget(self.scroll, 1)
            self.zoom_slider.valueChanged.connect(self.set_zoom); self.fit_button.clicked.connect(self.fit_view)
        def showEvent(self, event):
            super().showEvent(event)
            QTimer.singleShot(0, self.fit_view)
        def eventFilter(self, watched, event):
            if watched is self.image and self.receiver_active:
                if event.type() == QEvent.KeyPress:
                    if event.key() == Qt.Key_Escape:
                        self.cancel_receiver_draft(); return True
                    if event.key() in {Qt.Key_Return, Qt.Key_Enter}:
                        self.commit_receiver_draft(); return True
                if event.type() in {QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick}:
                    point = self.receiver_source_point(event.position())
                    if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                        self.image.setFocus()
                        handle = self.receiver_handle_at(event.position()) if not self._receiver_draft else None
                        if handle is not None:
                            self._receiver_drag = {'handle': handle, 'geometry': json.loads(json.dumps(self.receiver_geometry))}
                        elif self.receiver_mode in {'region', 'plane'}:
                            limit = 4 if self.receiver_mode == 'plane' else 64
                            if len(self._receiver_draft) < limit:
                                self._receiver_draft.append(point); self.receiverDraftChanged.emit(list(self._receiver_draft))
                        elif self.receiver_mode in {'left', 'right'}:
                            geometry = json.loads(json.dumps(self.receiver_geometry)); geometry['foot_anchors'][self.receiver_mode] = point
                            self.receiverGeometryEdited.emit(geometry)
                        self.render_overlays(); return True
                    if event.type() == QEvent.MouseMove and self._receiver_drag is not None:
                        self.move_receiver_handle(self._receiver_drag['geometry'], self._receiver_drag['handle'], point)
                        self.render_overlays(); return True
                    if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                        if self._receiver_drag is not None:
                            self.move_receiver_handle(self._receiver_drag['geometry'], self._receiver_drag['handle'], point)
                            geometry = self._receiver_drag['geometry']; self._receiver_drag = None
                            self.receiverGeometryEdited.emit(geometry)
                        self.render_overlays(); return True
                    if event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
                        return True
            if watched is self.image and self.receiver_visible and self.receiver_mode != 'pan' and not self.receiver_active and event.type() in {QEvent.MouseButtonPress,QEvent.MouseButtonDblClick}:
                return True
            if watched is self.image and self.local_brush_active:
                if event.type() in {QEvent.MouseButtonPress,QEvent.MouseMove,QEvent.MouseButtonRelease}:
                    point=event.position();normalized=[min(1.,max(0.,point.x()/max(1,self.image.width()-1))),min(1.,max(0.,point.y()/max(1,self.image.height()-1)))]
                    self._local_brush_hover=normalized
                    if event.type()==QEvent.MouseButtonPress and event.button()==Qt.LeftButton:
                        self._local_brush_stroke={'mode':self.local_brush_mode,'points':[normalized]};self.render_overlays();return True
                    if event.type()==QEvent.MouseMove and self._local_brush_stroke is not None:
                        self._local_brush_stroke['points'].append(normalized);self.render_overlays();return True
                    if event.type()==QEvent.MouseButtonRelease and event.button()==Qt.LeftButton:
                        if self._local_brush_stroke is not None:
                            self._local_brush_stroke['points'].append(normalized);stroke=self._local_brush_stroke;self._local_brush_stroke=None;self.localColourStrokeFinished.emit(stroke)
                        self.render_overlays();return True
            if watched is self.image and self.clone_active:
                if event.type() in {QEvent.MouseButtonPress,QEvent.MouseMove,QEvent.MouseButtonRelease}:
                    point=event.position(); normalized=[min(1.,max(0.,point.x()/max(1,self.image.width()-1))),min(1.,max(0.,point.y()/max(1,self.image.height()-1)))]
                    self._clone_hover=normalized
                    if event.type()==QEvent.MouseButtonPress and event.button()==Qt.LeftButton:
                        if event.modifiers() & Qt.AltModifier:
                            self.clone_source=normalized;self.cloneSourcePicked.emit(normalized);self.render_overlays();return True
                        if self.clone_source is not None:
                            self._clone_stroke={'source_anchor':list(self.clone_source),'points':[normalized]}
                        self.render_overlays();return True
                    if event.type()==QEvent.MouseMove and self._clone_stroke is not None:
                        self._clone_stroke['points'].append(normalized);self.render_overlays();return True
                    if event.type()==QEvent.MouseButtonRelease and event.button()==Qt.LeftButton:
                        if self._clone_stroke is not None:
                            self._clone_stroke['points'].append(normalized);stroke=self._clone_stroke;self._clone_stroke=None;self.cloneStrokeFinished.emit(stroke)
                        self.render_overlays();return True
            if watched is self.image and self.native_detail_enabled and not self.receiver_active and not self.clone_active and not self.local_brush_active and event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
                point = event.position()
                self.nativeDetailRequested.emit([min(1.,max(0.,point.x()/max(1,self.image.width()))),
                                                 min(1.,max(0.,point.y()/max(1,self.image.height())))])
                return True
            if watched is self.image and self.color_selection is not None and self.color_brush_mode in {"add", "erase"}:
                if event.type() in {QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease}:
                    point = event.position()
                    normalized = [min(1., max(0., point.x() / max(1, self.image.width() - 1))), min(1., max(0., point.y() / max(1, self.image.height() - 1)))]
                    if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                        self._color_stroke = {"mode": self.color_brush_mode, "radius": self.color_brush_radius, "points": [normalized]}; self.render_overlays(); return True
                    if event.type() == QEvent.MouseMove and self._color_stroke is not None:
                        self._color_stroke["points"].append(normalized); self.render_overlays(); return True
                    if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self._color_stroke is not None:
                        self._color_stroke["points"].append(normalized); stroke = self._color_stroke; self._color_stroke = None
                        self.colorStrokeFinished.emit(stroke); self.render_overlays(); return True
            if watched is self.scroll.viewport():
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    self._pan_origin = event.position().toPoint()
                    self._pan_start = (self.scroll.horizontalScrollBar().value(), self.scroll.verticalScrollBar().value())
                    watched.setCursor(Qt.ClosedHandCursor); return True
                if event.type() == QEvent.MouseMove and self._pan_origin is not None:
                    point = event.position().toPoint(); delta = point - self._pan_origin
                    self.scroll.horizontalScrollBar().setValue(self._pan_start[0] - delta.x())
                    self.scroll.verticalScrollBar().setValue(self._pan_start[1] - delta.y()); return True
                if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                    self._pan_origin = None; self._pan_start = None; watched.setCursor(Qt.OpenHandCursor); return True
            return super().eventFilter(watched, event)

        def receiver_source_point(self, displayed):
            """Map a point in the whole-image display to oriented source pixels."""
            return [min(1., max(0., coordinate / max(1, display - 1))) * (source - 1)
                for coordinate, display, source in zip((displayed.x(), displayed.y()),
                    (self.image.width(), self.image.height()), self.receiver_canvas_size)]

        def receiver_display_point(self, point):
            return QPointF(*[coordinate / max(1, source - 1) * (display - 1)
                for coordinate, source, display in zip(point, self.receiver_canvas_size,
                    (self._base_image.width(), self._base_image.height()))])

        def receiver_handle_at(self, displayed):
            candidates = []
            for component, mode in (('receiver_region', 'region'), ('receiver_plane', 'plane')):
                if self.receiver_mode in {mode, 'move'}:
                    candidates.extend((component, index, point) for index, point in enumerate(self.receiver_geometry[component] or []))
            if self.receiver_mode in {'left', 'right', 'move'}:
                candidates.extend(('foot_anchors', side, point) for side, point in self.receiver_geometry['foot_anchors'].items()
                    if self.receiver_mode in {side, 'move'})
            nearest = None; distance = 101.
            for component, index, point in candidates:
                delta = self.receiver_display_point(point) - displayed; candidate_distance = delta.x()**2 + delta.y()**2
                if candidate_distance <= 100 and candidate_distance < distance:
                    nearest, distance = (component, index), candidate_distance
            return nearest

        @staticmethod
        def move_receiver_handle(geometry, handle, point):
            geometry[handle[0]][handle[1]] = point

        def cancel_receiver_draft(self):
            self._receiver_draft = []; self._receiver_drag = None
            self.receiverDraftChanged.emit([]); self.render_overlays()

        def commit_receiver_draft(self):
            count = len(self._receiver_draft)
            if (self.receiver_mode == 'region' and count >= 3) or (self.receiver_mode == 'plane' and count == 4):
                geometry = json.loads(json.dumps(self.receiver_geometry))
                geometry['receiver_' + self.receiver_mode] = list(self._receiver_draft)
                self._receiver_draft = []; self.receiverDraftChanged.emit([])
                self.receiverGeometryEdited.emit(geometry); self.render_overlays()
        def set_path(self, path: Optional[Path], proxy_path: Optional[Path] = None):
            self.path = path; self.proxy_path = proxy_path if proxy_path and proxy_path.is_file() else None
            self.path_label.setText(path.name if path else "暂无图像"); self.path_label.setToolTip(str(path) if path else ""); self.reload()
        def set_zoom(self, value: int):
            self.zoom = int(value); self.zoom_value.setText(f"{self.zoom}%"); self.reload()
        def fit_view(self):
            if not self.path or not self.path.is_file(): return
            reader = QImageReader(str(self.proxy_path or self.path)); reader.setAutoTransform(True); size = reader.size()
            if not size.isValid(): return
            viewport = self.scroll.viewport().size()
            ratio = min(viewport.width() / max(1, size.width()), viewport.height() / max(1, size.height()))
            # A tall RAW needs less than 5% when the editor is narrow. Keep
            # the scaled image inside the actual viewport at fit-to-screen.
            self.zoom_slider.setMinimum(1)
            value = max(1, min(200, int(ratio * 100)))
            self.zoom_slider.blockSignals(True); self.zoom_slider.setValue(value); self.zoom_slider.blockSignals(False)
            self.zoom = max(1., min(200., ratio * 100)); self.zoom_value.setText(f"{self.zoom:.1f}%"); self.reload()
        def reload(self):
            if not self.path or not self.path.is_file():
                self._base_image = QImage(); self.image.setPixmap(QPixmap()); self.image.setText("暂无文件"); self.image.adjustSize(); return
            display_path = self.proxy_path if self.proxy_path and (self.zoom <= 100 or self.path.suffix.casefold() in {".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".rw2", ".orf"}) else self.path
            reader = QImageReader(str(display_path)); reader.setAutoTransform(True)
            size = reader.size()
            if size.isValid():
                size.setWidth(max(1, round(size.width() * self.zoom / 100)))
                size.setHeight(max(1, round(size.height() * self.zoom / 100)))
                reader.setScaledSize(size)
            img = reader.read()
            if img.isNull(): self._base_image = QImage(); self.image.setText("无法读取文件"); self.image.adjustSize(); return
            self._base_image = img; self.render_overlays()

        def render_overlays(self):
            img = self._base_image
            if img.isNull(): return
            pixmap = QPixmap.fromImage(img)
            if self.region_rect:
                x0, y0, x1, y1 = self.region_rect
                painter = QPainter(pixmap); painter.setPen(QColor("#ffd166"))
                painter.drawRect(QRect(round(min(x0, x1) * img.width()), round(min(y0, y1) * img.height()), round(abs(x1 - x0) * img.width()), round(abs(y1 - y0) * img.height())))
                painter.end()
            if self.show_light_fields and self.light_fields:
                painter = QPainter(pixmap); painter.setRenderHint(QPainter.Antialiasing)
                for field in self.light_fields:
                    if not field.get("enabled", True): continue
                    cx, cy = field.get("center", [.5, .5]); rx, ry = field.get("radius", [.2, .2])
                    selected = field.get("id") == self.selected_light_id
                    painter.save(); painter.translate(float(cx) * img.width(), float(cy) * img.height())
                    painter.rotate(float(field.get("rotation_deg", 0)))
                    pen = painter.pen(); pen.setColor(QColor("#d3b6ff" if selected else "#7889a5")); pen.setWidth(2 if selected else 1); pen.setStyle(Qt.DashLine); painter.setPen(pen)
                    painter.drawEllipse(QRect(round(-float(rx) * img.width()), round(-float(ry) * img.height()), round(2 * float(rx) * img.width()), round(2 * float(ry) * img.height())))
                    painter.restore()
                painter.end()
            if self.show_color_selection and self.color_selection is not None:
                from source_color import selection_mask
                selection = json.loads(json.dumps(self.color_selection))
                if self._color_stroke is not None: selection.setdefault("strokes", []).append(self._color_stroke)
                scale = min(1., 1200. / max(img.width(), img.height()))
                size = (max(1, round(img.width() * scale)), max(1, round(img.height() * scale)))
                mask = selection_mask(size, selection)
                from PIL import Image
                colored = Image.new("RGBA", size, (184, 131, 255, 0)); colored.putalpha(mask.point(lambda value: round(value * .38)))
                payload = colored.tobytes(); overlay = QImage(payload, size[0], size[1], size[0] * 4, QImage.Format_RGBA8888).copy()
                painter = QPainter(pixmap); painter.drawImage(QRect(0, 0, img.width(), img.height()), overlay); painter.end()
            if self.selected_local_brush is not None:
                stroke=self.selected_local_brush;points=stroke.get('points',[])
                painter=QPainter(pixmap);painter.setRenderHint(QPainter.Antialiasing)
                pen=painter.pen();pen.setColor(QColor('#cfb6ff'));pen.setWidth(2);pen.setStyle(Qt.DashLine);painter.setPen(pen)
                radius=stroke['radius_px']*min(img.width(),img.height())/min(stroke['native_canvas_size'])
                for point in points[::max(1,len(points)//12)]:
                    painter.drawEllipse(QPointF(point[0]*(img.width()-1),point[1]*(img.height()-1)),radius,radius)
                painter.end()
            if self.local_brush_active and self._local_brush_hover is not None:
                painter=QPainter(pixmap);painter.setRenderHint(QPainter.Antialiasing)
                radius=max(1,round(self.local_brush_radius*min(img.width(),img.height())))
                cx,cy=[round(v*d) for v,d in zip(self._local_brush_hover,(img.width()-1,img.height()-1))]
                painter.setPen(QColor({'erase':'#ffb5a5','foreground':'#73e5ba','background':'#ff8e93'}.get(self.local_brush_mode,'#a9caff')))
                painter.drawEllipse(cx-radius,cy-radius,radius*2,radius*2)
                if self._local_brush_stroke is not None:
                    points=self._local_brush_stroke['points']
                    pen=painter.pen();pen.setStyle(Qt.DotLine);painter.setPen(pen)
                    for a,b in zip(points[:-1],points[1:]):
                        painter.drawLine(QPointF(a[0]*(img.width()-1),a[1]*(img.height()-1)),QPointF(b[0]*(img.width()-1),b[1]*(img.height()-1)))
                painter.end()
            if self.matte_strokes:
                painter=QPainter(pixmap);painter.setRenderHint(QPainter.Antialiasing)
                for stroke in self.matte_strokes:
                    native=stroke['native_canvas_size'];radius=stroke['radius_px']*min(img.width(),img.height())/min(native)
                    pen=painter.pen();pen.setColor(QColor('#73e5ba' if stroke['mode']=='foreground' else '#ff8e93'));pen.setWidthF(max(1,2*radius));pen.setCapStyle(Qt.RoundCap);painter.setPen(pen)
                    points=stroke['points']
                    if len(points)==1:painter.drawPoint(QPointF(points[0][0]*(img.width()-1),points[0][1]*(img.height()-1)))
                    for a,b in zip(points[:-1],points[1:]):painter.drawLine(QPointF(a[0]*(img.width()-1),a[1]*(img.height()-1)),QPointF(b[0]*(img.width()-1),b[1]*(img.height()-1)))
                painter.end()
            if self.clone_active:
                painter=QPainter(pixmap);painter.setRenderHint(QPainter.Antialiasing)
                if self.clone_source is not None:
                    donor=self.clone_source
                    if self._clone_stroke is not None:
                        donor=[a+b-c for a,b,c in zip(self._clone_stroke['source_anchor'],self._clone_stroke['points'][-1],self._clone_stroke['points'][0])]
                    cx,cy=[round(v*d) for v,d in zip(donor,(img.width()-1,img.height()-1))]
                    painter.setPen(QColor('#73e5ba'));painter.drawLine(cx-8,cy,cx+8,cy);painter.drawLine(cx,cy-8,cx,cy+8)
                if self._clone_hover is not None:
                    cx,cy=[round(v*d) for v,d in zip(self._clone_hover,(img.width()-1,img.height()-1))]
                    radius=max(1,round(self.clone_radius*min(img.width(),img.height())))
                    painter.setPen(QColor('#ebe4ff'));painter.drawEllipse(cx-radius,cy-radius,radius*2,radius*2)
                painter.end()
            if self.receiver_visible and self.receiver_canvas_size:
                geometry = self._receiver_drag['geometry'] if self._receiver_drag else self.receiver_geometry
                painter = QPainter(pixmap); painter.setRenderHint(QPainter.Antialiasing)
                for component, color in (('receiver_region', '#6bdcc4'), ('receiver_plane', '#ffd166')):
                    points = geometry[component] or []
                    if not points: continue
                    pen = painter.pen(); pen.setColor(QColor(color)); pen.setWidth(2); painter.setPen(pen)
                    fill = QColor(color); fill.setAlpha(22); painter.setBrush(fill)
                    displayed = [self.receiver_display_point(point) for point in points]
                    painter.drawPolygon(QPolygonF(displayed)); painter.setBrush(QColor(color))
                    for index, point in enumerate(displayed):
                        painter.drawEllipse(point, 4., 4.)
                        if component == 'receiver_plane': painter.drawText(point + QPointF(7, -7), str(index + 1))
                painter.setBrush(Qt.NoBrush)
                for side, point in geometry['foot_anchors'].items():
                    displayed = self.receiver_display_point(point); painter.setPen(QColor('#a6c8ff' if side == 'left' else '#f4a6d8'))
                    painter.drawEllipse(displayed, 7., 7.); painter.drawLine(displayed + QPointF(-10, 0), displayed + QPointF(10, 0))
                    painter.drawLine(displayed + QPointF(0, -10), displayed + QPointF(0, 10))
                    painter.drawText(displayed + QPointF(12, -8), '左脚' if side == 'left' else '右脚')
                if self._receiver_draft:
                    pen = painter.pen(); pen.setColor(QColor('#ffffff')); pen.setStyle(Qt.DashLine); pen.setWidth(2); painter.setPen(pen)
                    displayed = [self.receiver_display_point(point) for point in self._receiver_draft]
                    for a, b in zip(displayed[:-1], displayed[1:]): painter.drawLine(a, b)
                    for index, point in enumerate(displayed):
                        painter.drawEllipse(point, 4., 4.); painter.drawText(point + QPointF(7, -7), str(index + 1))
                painter.end()
            self.image.setPixmap(pixmap); self.image.resize(pixmap.size())

        def replace_preview_path(self, path):
            displayed = self.image.width()
            positions = [(bar.value() / max(1, bar.maximum())) for bar in (self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar())]
            reader = QImageReader(str(path)); reader.setAutoTransform(True); size = reader.size()
            if size.isValid():
                self.zoom = displayed / max(1, size.width()) * 100
                self.zoom_slider.setMaximum(max(200, int(self.zoom) + 1)); self.zoom_slider.blockSignals(True); self.zoom_slider.setValue(round(self.zoom)); self.zoom_slider.blockSignals(False)
                self.zoom_value.setText(f"{self.zoom:.1f}%")
            self.set_path(path)
            def restore_position():
                for bar, position in zip((self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar()), positions): bar.setValue(round(position * bar.maximum()))
            QTimer.singleShot(0, restore_position)

    class AlphaCanvas(QWidget):
        """Small non-destructive alpha editor with trust/structure overlays.

        The canvas still edits only the grayscale alpha.  The optional
        structure image is a review heatmap and is never used as a write-back
        mask; this keeps the overlay useful for finding an over-wide trust
        region without silently expanding it.
        """
        editsChanged = Signal(object)
        def __init__(self):
            super().__init__(); self.source = None; self.structure_path = None; self.image = QImage(); self.structure = QImage(); self.edits = []; self._history = []; self._history_index = -1
            self.mode = "paint"; self.radius = 0.025; self.opacity = 1.0; self.overlay_mode = "trust"; self._stroke_id = 0; self._last_point = None; self._display_cache = None
            self.setMinimumHeight(240)
        def set_path(self, path: Optional[Path], structure_path: Optional[Path] = None):
            self.source = path; self.structure_path = structure_path; self.edits = []; self.image = QImage(); self.structure = QImage(); self._display_cache = None
            if path and path.is_file():
                reader = QImageReader(str(path)); reader.setAutoTransform(True); self.image = reader.read().convertToFormat(QImage.Format_Grayscale8)
            if structure_path and structure_path.is_file():
                reader = QImageReader(str(structure_path)); reader.setAutoTransform(True); self.structure = reader.read().convertToFormat(QImage.Format_RGB32)
            self._history = [(self.image.copy(), [])]; self._history_index = 0
            self.update(); self.editsChanged.emit([])
        def set_tool(self, mode: str): self.mode = mode
        def set_overlay_mode(self, mode: str): self.overlay_mode = str(mode or "trust"); self._display_cache = None; self.update()
        def set_radius(self, value: int): self.radius = max(1, value) / 1000.0
        def set_opacity(self, value: int): self.opacity = max(0, min(100, value)) / 100.0
        def reset(self):
            if self.source: self.set_path(self.source, self.structure_path)
        def _record_history(self):
            self._history = self._history[:self._history_index + 1]
            self._history.append((self.image.copy(), list(self.edits))); self._history_index += 1
        def _restore_history(self):
            image, edits = self._history[self._history_index]
            self.image = image.copy(); self.edits = list(edits); self.update(); self.editsChanged.emit(list(self.edits))
        def undo(self):
            if self._history_index > 0:
                self._history_index -= 1; self._restore_history()
        def redo(self):
            if self._history_index + 1 < len(self._history):
                self._history_index += 1; self._restore_history()
        def target_rect(self):
            if self.image.isNull(): return self.rect()
            scale = min(self.width() / max(1, self.image.width()), self.height() / max(1, self.image.height()))
            w = max(1, round(self.image.width() * scale)); h = max(1, round(self.image.height() * scale))
            return QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)
        def event_point(self, event):
            rect = self.target_rect()
            if not rect.contains(event.position().toPoint()) or self.image.isNull(): return None
            pos = event.position().toPoint(); return ((pos.x() - rect.x()) / rect.width(), (pos.y() - rect.y()) / rect.height())
        def stroke(self, event):
            point = self.event_point(event)
            if point is None: return
            x, y = point; value = 255 if self.mode == "paint" else 0
            # Keep both normalized geometry (portable between proxy/full size)
            # and the concrete brush width used by the renderer.  The region
            # is review metadata so an operator can see exactly what changed.
            # brush_px is the full diameter consumed by the renderer; the UI
            # radius is half of that diameter.
            brush_px = max(1, round(self.radius * min(self.image.width(), self.image.height()) * 2.0))
            previous = self._last_point
            points = ([list(previous), [round(x, 6), round(y, 6)]] if previous is not None else [[round(x, 6), round(y, 6)]])
            xs = [x] + ([previous[0]] if previous is not None else []); ys = [y] + ([previous[1]] if previous is not None else [])
            region = [round(max(0.0, min(xs) - self.radius), 6), round(max(0.0, min(ys) - self.radius), 6),
                      round(min(1.0, max(xs) + self.radius), 6), round(min(1.0, max(ys) + self.radius), 6)]
            edit = {"op": self.mode, "mode": self.mode, "x": round(x, 6), "y": round(y, 6),
                    "points": points, "radius": round(self.radius, 6),
                    "brush_px": brush_px, "region": region, "stroke_id": self._stroke_id,
                    "value": value, "opacity": round(self.opacity, 4)}
            self.edits.append(edit)
            color = QColor(value, value, value); painter = QPainter(self.image); painter.setOpacity(self.opacity); painter.setBrush(color); painter.setPen(color)
            radius = max(1, round(self.radius * min(self.image.width(), self.image.height())))
            painter.drawEllipse(round(x * self.image.width() - radius), round(y * self.image.height() - radius), radius * 2, radius * 2); painter.end()
            self._last_point = (round(x, 6), round(y, 6))
            self._record_history(); self.update(); self.editsChanged.emit(list(self.edits))
        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                self._stroke_id += 1
                self.stroke(event)
        def mouseMoveEvent(self, event):
            if event.buttons() & Qt.LeftButton: self.stroke(event)
        def mouseReleaseEvent(self, event):
            if event.button() == Qt.LeftButton: self._last_point = None
        def paintEvent(self, event):
            painter = QPainter(self); painter.fillRect(self.rect(), QColor("#252525"))
            if self.image.isNull(): painter.setPen(QColor("#bbbbbb")); painter.drawText(self.rect(), Qt.AlignCenter, "没有可编辑的 alpha")
            else:
                target = self.target_rect()
                # Trust mode uses a compact colour key: blue=low confidence,
                # amber=transition, green=high confidence.  It is rendered
                # at preview size so a large RAW never causes a huge loop.
                if self.overlay_mode == "trust":
                    scaled = self.image.scaled(target.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                    colored = QImage(scaled.size(), QImage.Format_RGB32)
                    for yy in range(scaled.height()):
                        for xx in range(scaled.width()):
                            a = scaled.pixelColor(xx, yy).red()
                            if a < 96: color = QColor(55, 85, 145)
                            elif a < 200: color = QColor(210, 170, 55)
                            else: color = QColor(65, 165, 105)
                            colored.setPixelColor(xx, yy, color)
                    painter.drawImage(target, colored)
                    painter.setOpacity(0.32); painter.drawImage(target, scaled); painter.setOpacity(1.0)
                else:
                    painter.drawImage(target, self.image)
                if self.overlay_mode in {"structure", "trust+structure"} and not self.structure.isNull():
                    painter.setOpacity(0.55 if self.overlay_mode == "structure" else 0.38)
                    painter.drawImage(target, self.structure.scaled(target.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
                    painter.setOpacity(1.0)
                painter.setPen(QColor("#777777")); painter.drawRect(target)
            painter.end()

    class Workbench(QMainWindow):
        backendLog = Signal(str)
        def __init__(self, root: Path, url: str, embedded_backend: bool = False):
            super().__init__()
            # A frozen one-folder build cannot spawn ``scripts/backend_server.py``
            # because source files are inside the archive.  Use the bundled
            # backend in that case even when the flag was omitted; source
            # interpreter runs retain the separate-process default.
            if getattr(sys, "frozen", False):
                embedded_backend = True
            self.root_path, self.url = root.resolve(), url; self.embedded_backend = embedded_backend; self.pool = QThreadPool.globalInstance()
            self.backend_generation = 0; self.model_setup_status = {}; self.model_setup_dialog = None
            self.model_setup_submission = None
            self.model_auto_discovery_attempted = False
            self.model_setup_observation = 0
            self.process = None; self.embedded_server = None; self.rows = {}; self.previews = {}; self.source_run_id = ""; self.source_sample = ""; self.source_revision = 0; self.source_status = ""; self.render_pending = False; self.active_job_id = ""; self.job_stream_active = False; self.risk_crop_paths = {}; self.job_rows = {}; self.provider_rows = {}; self.provider_settings = {}; self.appearance_controls = []; self.appearance_extra_controls = []; self.job_observation = 0
            self.provider_settings_loaded = False; self.provider_default = None; self.pending_tasks = {}; self.source_run_source = ""; self.source_run_reference = ""
            self.generation_model_user_selected = False
            self.generation_model_save_error = False
            self.generation_model_save_pending = None; self.generation_model_save_active = None
            self.generation_model_save_timer = QTimer(self); self.generation_model_save_timer.setSingleShot(True)
            self.generation_model_save_timer.setInterval(250)
            self.generation_model_save_timer.timeout.connect(self.persist_generation_model)
            self.job_poll_terminal_observation = None
            self.job_poll_timer = QTimer(self); self.job_poll_timer.setSingleShot(True)
            self.job_poll_timer.timeout.connect(self.poll_active_job)
            self.product_result_panels = []; self.product_stage_report = {}; self.product_run = {}
            self.shaping_panels = []; self.shaping_parts = {}; self.shaping_revision = 0
            self.shaping_execution = None; self.shaping_candidate = None
            self.shaping_parent_fact = None; self.shaping_inspections = {}; self.shaping_adoption = None
            self.shaping_inspection_closed = False
            self._startup_result_requested = False; self._startup_result_context = None
            self.workflow_graph = None; self._workflow_cache_state = None; self.professional_mode = False
            self.simple_creation = None; self.simple_comparing = False; self._workflow_signature = None
            self.startup_readiness = 'ready'
            self._startup_service_enabled=False;self._startup_service_closed=False
            self._startup_service_context=None;self._startup_service_records={};self._startup_service_suppressed=False
            self._startup_service_last_target=None
            self._startup_recent_loaded=False
            self.startup_service_timer=QTimer(self);self.startup_service_timer.setSingleShot(True)
            self.startup_service_timer.setInterval(200);self.startup_service_timer.timeout.connect(self.start_selected_service)
            self._quick_progress_key = None; self._quick_progress_value = 0.0
            self.simple_cancel_submissions = set()
            self.creative_task_plan = None; self.creative_task_plan_request = None
            self.active_checked_task = None
            self.source_generated_appearance = False
            self._workflow_canvas_available = True
            # The bundled source-refinement profile is an explicit option. A
            # fresh task starts from the user's brief so ordinary users do not
            # inherit extra skin/material operators they did not request.
            self.refinement_profile_choice = None
            self.creative_task_revision = 0; self.creative_task_start_pending = False
            self.creative_task_execution_label = None
            self.creative_task_retry_batch_id = None
            self.regional_enabled = False; self.regional_core_box = None
            self.same_photo_finish_enabled = False
            self.source_conditioned_reference = False
            self.creative_task_policies = {mode: {'text_vision': mode == 'create_reference', 'visual_review': mode == 'create_reference',
                'background_mode': 'preserve' if mode == 'local_portrait' else 'enhance_source',
                'preserve_background': mode == 'local_portrait', 'no_effects': mode == 'local_portrait'}
                for mode in CREATIVE_TASK_MODES}
            self.editing_requests = {mode: {'work_type': 'retouch' if mode == 'local_portrait' else 'partial_composite',
                'style': 'expressive', 'target_setting': 'same', 'scene_target': '', 'language': 'zh'}
                for mode in CREATIVE_TASK_MODES}
            self.editing_scope_panels = []
            self.editing_autofinish_enabled = True
            self.editing_autofinish_maximum = 2
            self.editing_autofinish_zero_finishes = 1
            self.editing_improve_buttons = []
            self._session_binding = None; self._session_baseline = None; self._session_written = None
            self._session_loading = False; self._session_dirty = False; self._session_submission = None
            self._session_checkpoint_failed = False
            self._session_commit_child_id = None
            self.source_light_plan = None; self.source_light_version = 0; self.source_light_sequence = 0
            self.source_light_requests = {}; self.source_light_loading = False; self.source_light_draft_path = None
            self.source_light_design = None; self.source_light_origin = "manual"
            self.source_color_plan = None; self.source_color_design = None; self.source_color_origin = "manual"; self.source_color_loading = False; self.source_color_auto_request = None
            self.source_color_provenance = None
            self.design_reconstruction_plan = None; self.design_reconstruction_loading = False; self.design_reconstruction_request = None
            self.design_reconstruction_prepared = False
            self.native_reconstruction_job_id = ''; self.native_reconstruction_result_seen = ''
            self.native_reconstruction_preparation_kind = 'reference_appearance'
            self.portrait_beauty = None; self.portrait_beauty_faces = []; self.portrait_beauty_loading = False
            self.portrait_beauty_request = None; self.portrait_beauty_job_id = ''; self.portrait_beauty_result_seen = ''
            from scene_receiver_intent import normalize_receiver_intent
            self.receiver_intent = normalize_receiver_intent(); self.receiver_intent_loading = False
            self.receiver_source_canvas_size = None
            self.receiver_history = []; self.receiver_history_index = -1
            from authored_receiver_shadow import normalize_receiver_shadow_plan
            self.receiver_shadow_plan = normalize_receiver_shadow_plan(); self.receiver_shadow_loading = False
            self.receiver_shadow_result = None; self.receiver_shadow_preview_requests = {}
            self.source_color_history = []; self.source_color_history_index = -1
            self.clone_history=[];self.clone_history_index=0;self.clone_native_size=[1,1];self.clone_source=None
            self.curve_history=[];self.curve_history_index=-1;self.curve_loading=False
            self.hsl_history=[];self.hsl_history_index=-1;self.hsl_loading=False
            self.local_brush_history=[];self.local_brush_history_index=-1;self.local_brush_loading=False
            self.local_brush_selected_id=None;self.local_brush_selected_index=-1;self.local_brush_selection_loading=False
            self.native_matte_strokes=[];self.native_matte_history=[[]];self.native_matte_history_index=0
            self.native_matte_base_plan=None;self.native_matte_cache_key=None;self.native_matte_version=0;self.native_matte_preview_version=-1;self.native_matte_request=None
            self.native_matte_timer=QTimer(self);self.native_matte_timer.setSingleShot(True);self.native_matte_timer.setInterval(600);self.native_matte_timer.timeout.connect(self.preview_source_lights)
            self.source_light_timer = QTimer(self); self.source_light_timer.setSingleShot(True); self.source_light_timer.setInterval(350); self.source_light_timer.timeout.connect(self.preview_source_lights)

            self.backendLog.connect(self.log_text)
            self.setWindowTitle("Cosplay 二次元后期工作台"); self.resize(1260, 820); self.build()
            if not self.embedded_backend: self.refresh()
            self.session_timer = QTimer(self); self.session_timer.setInterval(1000)
            self.session_timer.timeout.connect(self._checkpoint_session); self.session_timer.start()
        def build(self):
            from workbench_theme import apply_theme
            apply_theme(QApplication.instance())
            self.setWindowTitle("CosLuma · 二次元摄影后期"); self.resize(1480, 960)
            self.project_images = {}; self.project_proxies = {}; self.project_dimensions = {}
            root = QWidget(); self.setCentralWidget(root); outer = QVBoxLayout(root); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
            header = QWidget(); header.setObjectName("brandBar"); bar = QHBoxLayout(header); bar.setContentsMargins(22, 12, 18, 12)
            brand = QLabel("COSLUMA"); brand.setObjectName("brandName"); bar.addWidget(brand)
            self.project_label = QLabel(self.root_path.name); self.project_label.setObjectName("projectLabel"); bar.addWidget(self.project_label); bar.addStretch()
            self.operation_state = QLabel("就绪"); self.operation_state.setObjectName("statusPill"); self.operation_state.setMaximumWidth(420); self.operation_state.setWordWrap(True); bar.addWidget(self.operation_state)
            self.mode_button = QPushButton('专业模式'); self.mode_button.setCheckable(True)
            self.mode_button.hide()
            self.mode_button.toggled.connect(self.set_professional_mode); bar.addWidget(self.mode_button)
            self.start = QPushButton("启动后台"); self.refresh_btn = QPushButton("刷新")
            bar.addWidget(self.refresh_btn)
            settings_button = QPushButton("设置"); settings_button.setObjectName("secondaryButton"); bar.addWidget(settings_button); outer.addWidget(header)
            self.models_button = QPushButton('模型管理'); self.models_button.clicked.connect(self.open_model_setup)
            bar.insertWidget(bar.count()-1, self.models_button)
            connection = QWidget(); connection_form = QFormLayout(connection); connection_form.setContentsMargins(22, 10, 22, 10)
            self.root_edit = QLineEdit(str(self.root_path)); self.url_edit = QLineEdit(self.url)
            self.root_edit.setReadOnly(self.embedded_backend); self.url_edit.setReadOnly(self.embedded_backend)
            if self.embedded_backend:
                self.root_edit.setToolTip('每个窗口绑定一个工程；使用 --root 打开另一工程的新窗口。')
                self.url_edit.setToolTip('内嵌后台的实际地址；端口占用时自动选择可用端口。')
            connection_form.addRow("项目目录", self.root_edit); connection_form.addRow("本地后台地址", self.url_edit); connection_form.addRow(self.start)
            self.model_setup_summary = QLabel('分析模型尚未检查'); self.model_setup_summary.setWordWrap(True)
            model_buttons = QWidget(); model_layout = QHBoxLayout(model_buttons); model_layout.setContentsMargins(0, 0, 0, 0)
            self.model_configure_button = QPushButton('模型管理'); self.model_inspect_button = QPushButton('检查安装路径')
            self.model_configure_button.clicked.connect(self.open_model_setup)
            self.model_inspect_button.clicked.connect(lambda: self.task('GET', '/api/model-setup', None, 'model-setup'))
            model_layout.addWidget(self.model_configure_button); model_layout.addWidget(self.model_inspect_button)
            connection_form.addRow('分析模型', self.model_setup_summary); connection_form.addRow(model_buttons)
            connection.setVisible(False); settings_button.setCheckable(True); settings_button.toggled.connect(connection.setVisible); outer.addWidget(connection)
            body = QHBoxLayout(); body.setContentsMargins(0, 0, 0, 0); body.setSpacing(0)
            self.navigation = QListWidget(); self.navigation.setObjectName("navRail"); self.navigation.setFixedWidth(168)
            self.navigation.addItems(["工作区", "候选版本", "精修与审片", "导出交付", "任务中心", "工作流画布", "快速制作"]); body.addWidget(self.navigation)
            self.tabs = QTabWidget(); self.tabs.tabBar().hide(); self.tabs.setDocumentMode(True)
            self.tabs.addTab(self.project_page(), "项目"); self.tabs.addTab(self.scrollable_page(self.candidate_page()), "候选")
            self.tabs.addTab(self.scrollable_page(self.review_page()), "审片"); self.tabs.addTab(self.scrollable_page(self.delivery_page()), "交付")
            self.tabs.addTab(self.scrollable_page(self.jobs_page()), "任务")
            from cos_workflow_canvas import CosWorkflowCanvas
            from simple_cos_workspace import SimpleCosWorkspace
            self.workflow_canvas = CosWorkflowCanvas(); self.workflow_index = self.tabs.addTab(self.workflow_canvas, '工作流画布')
            self.simple_workspace = SimpleCosWorkspace(); self.simple_index = self.tabs.addTab(self.simple_workspace, '快速制作')
            quick_form = self.simple_workspace.task_mode.parentWidget().layout()
            quick_form.insertWidget(quick_form.indexOf(self.simple_workspace.task_mode) + 1,
                                    self.build_editing_scope_panel())
            quick_form.insertWidget(quick_form.indexOf(self.simple_workspace.native_review_button) + 1,
                                    self.build_editing_improve_button())
            self.workflow_canvas.actionRequested.connect(self.workflow_node_action)
            self.workflow_canvas.templateChanged.connect(self.workflow_template_changed)
            self.simple_workspace.sourceChosen.connect(self.choose_source)
            self.simple_workspace.sourcePathChanged.connect(self.set_simple_source)
            self.simple_workspace.strengthChanged.connect(self.portrait_beauty_strength.setValue)
            self.simple_workspace.briefChanged.connect(self.creative_task_brief.setPlainText)
            self.simple_workspace.actionRequested.connect(self.simple_action)
            self.simple_workspace.generationModelChanged.connect(self.select_generation_model)
            self.simple_workspace.characterChanged.connect(self.character_name.setText)
            self.simple_workspace.characterNotesChanged.connect(self.character_notes.setText)
            self.simple_workspace.backgroundModeChanged.connect(self.set_background_mode)
            self.simple_workspace.noEffectsChanged.connect(self.creative_no_effects.setChecked)
            self.simple_workspace.regionalEnabledChanged.connect(self.set_regional_enabled)
            self.simple_workspace.samePhotoFinishChanged.connect(self.set_same_photo_finish)
            self.simple_workspace.regionalBoxChanged.connect(self.set_regional_box)
            self.simple_workspace.normalizedRegionSelected.connect(self.set_normalized_regional_box)
            self.simple_workspace.nativeDetailRequested.connect(self.request_native_detail)
            self.simple_workspace.taskModeChanged.connect(lambda mode: self.creative_task_mode.setCurrentIndex(self.creative_task_mode.findData(mode)))
            self.creative_task_mode.setCurrentIndex(self.creative_task_mode.findData('create_reference'))
            self.sync_editing_scope_controls()
            self.model.textChanged.connect(self.refresh_generation_models)
            body.addWidget(self.tabs, 1); outer.addLayout(body, 1)
            self.navigation.currentRowChanged.connect(self.tabs.setCurrentIndex); self.tabs.currentChanged.connect(self.navigation.setCurrentRow)
            self.navigation.setCurrentRow(self.simple_index); self.navigation.hide()
            self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(150)
            log_panel, self.log_toggle = self.accordion("运行日志", self.log, False); outer.addWidget(log_panel)
            self.start.clicked.connect(self.start_backend); self.refresh_btn.clicked.connect(self.refresh)
            for edit in (self.source_edit, self.character_name, self.character_notes, self.model): edit.textChanged.connect(self.clear_generation_design_view)
            self.prompt.textChanged.connect(self.clear_generation_design_view); self.preset.currentIndexChanged.connect(self.clear_generation_design_view); self.provider.currentTextChanged.connect(self.clear_generation_design_view)
            for edit in (self.source_edit, self.character_name, self.character_notes, self.model): edit.textChanged.connect(self.invalidate_creative_task_plan)
            self.prompt.textChanged.connect(self.invalidate_creative_task_plan)
            self.preset.currentIndexChanged.connect(self.invalidate_creative_task_plan)
            self.provider.currentTextChanged.connect(self.invalidate_creative_task_plan)
            self.variant_count.valueChanged.connect(self.invalidate_creative_task_plan)
            self.root_edit.textChanged.connect(self.invalidate_creative_task_plan)
            self.url_edit.textChanged.connect(self.invalidate_creative_task_plan)
            self.creative_task_brief.textChanged.connect(self.clear_generation_design_view)
            self.character_name.textChanged.connect(self.refresh_cos_workspace)
            self.character_notes.textChanged.connect(self.refresh_cos_workspace)
            self.source_edit.textChanged.connect(self.refresh_cos_workspace)
            self.update_creative_service_actions()
            self.refresh_cos_workspace()

        def set_professional_mode(self, enabled):
            self.professional_mode = bool(enabled)
            self.mode_button.blockSignals(True); self.mode_button.setChecked(self.professional_mode); self.mode_button.blockSignals(False)
            self.mode_button.setText('返回快速制作' if enabled else '专业模式')
            self.navigation.setVisible(self.professional_mode)
            self.mode_button.setVisible(self.professional_mode)
            self.tabs.setCurrentIndex(self.workflow_index if enabled else self.simple_index)
            self.refresh_cos_workspace()

        def set_simple_source(self, path):
            if self.render_pending: return
            self.simple_creation = None; self.simple_comparing = False; self._reset_quick_progress()
            self.regional_enabled = False; self.regional_core_box = None
            self.same_photo_finish_enabled = False
            self.source_conditioned_reference = False
            self.refinement_profile_choice = ({'id': 'source_refinement', 'version': 1}
                if self.preset.currentData() == 'source_refinement' else None)
            self.source_edit.setText(path); self.project_view_selector.setCurrentIndex(0)
            self.request_source_preview(path); self.refresh_cos_workspace()

        def _cos_draft_run(self):
            run = copy.deepcopy(self.product_run)
            if self.source_generated_appearance: return run
            options = run.setdefault('options', {})
            revision = options.setdefault('appearance_revision', {})
            self.add_local_plans_to_revision(revision)
            return run

        def _reset_quick_progress(self):
            """Start a new quick-route observation at its first milestone."""
            self._quick_progress_key = None
            self._quick_route_key = None
            self._quick_progress_value = 0.0
            self._quick_local_progress_value = None
            self._quick_timing_cursor = None

        def refresh_cos_workspace(self, *_args):
            if not hasattr(self, 'simple_workspace'): return
            source = self.source_edit.text().strip()
            if source != getattr(self, '_regional_source_path', source):
                self.regional_enabled = False; self.regional_core_box = None
                self.same_photo_finish_enabled = False
                self.source_conditioned_reference = False
            self._regional_source_path = source
            same = bool(self.source_run_id and source and self.source_run_source
                        and Path(source).resolve() == Path(self.source_run_source).resolve())
            awaiting_variant = bool(self.simple_creation and self.simple_creation.get('stage') == 'awaiting_variant')
            busy = bool(self.render_pending or self.creative_task_start_pending or (self.simple_creation and not awaiting_variant))
            proxy = self.project_proxies.get('source')
            original = proxy or self.project_images.get('source')
            result = (getattr(self, 'source_light_preview_path', None) or self.source_light_draft_path) if same else None
            if self.simple_comparing and same: result = original
            progress_active = bool(busy or awaiting_variant)
            checked_plan=(self.creative_task_plan or {}).get('plan') or {}
            scoped_ready=(checked_plan.get('ready') is True
                and (checked_plan.get('editing_scope') or {}).get('work_type') in {'partial_composite','full_composite'})
            self.simple_workspace.set_state(source, same, busy, bool(result), same and self.portrait_beauty is not None,
                progress_active=progress_active, readiness='ready' if scoped_ready else self.startup_readiness)
            self.sync_editing_scope_controls(busy=busy)
            for control, value in ((self.simple_workspace.character_name, self.character_name.text()),
                                   (self.simple_workspace.character_notes, self.character_notes.text())):
                control.blockSignals(True); control.setText(value); control.blockSignals(False)
            background = self.creative_background_mode.currentData()
            self.simple_workspace.task_mode.blockSignals(True)
            self.simple_workspace.task_mode.setCurrentIndex(self.simple_workspace.task_mode.findData(self.creative_task_mode.currentData()))
            self.simple_workspace.task_mode.blockSignals(False)
            self.simple_workspace.background_mode.blockSignals(True)
            self.simple_workspace.background_mode.setCurrentIndex(self.simple_workspace.background_mode.findData(background))
            self.simple_workspace.background_mode.blockSignals(False)
            self.simple_workspace.no_effects.blockSignals(True)
            self.simple_workspace.no_effects.setChecked(self.creative_no_effects.isChecked())
            self.simple_workspace.no_effects.blockSignals(False)
            active_checked = self.active_checked_task or {}
            active_intent = (active_checked.get('request') or {}).get('creative_task') or {}
            same_photo = (((active_intent.get('constraints') or {}).get('reference_role') == 'same_photo_finish')
                          if active_checked else self.same_photo_finish_enabled)
            same_photo_available = self.same_photo_finish_available() and not busy and not active_checked
            for control in (self.creative_same_photo_finish, self.simple_workspace.same_photo_finish):
                control.blockSignals(True); control.setChecked(same_photo); control.blockSignals(False)
                control.setEnabled(same_photo_available)
            active_regional = active_intent.get('regional_refinement') or {}
            active_maximum = int(active_regional.get('maximum_submissions') or 0)
            active_regions = active_regional.get('regions') or []
            displayed_box = (active_regions[0].get('core_box') if active_regions else None) if active_checked else self.regional_core_box
            displayed_enabled = active_maximum > 0 if active_checked else self.regional_enabled
            regional_available = self.creative_task_mode.currentData() == 'existing_reference' and background == 'enhance_source' and not busy and not active_checked and not same_photo
            cpu_reuse = regional_available and any(action.get('regional_candidate_reuse')
                for action in (self.product_stage_report.get('whole_work') or {}).get('recovery_actions', []))
            size = self.source_native_size()
            self.regional_controls.sync(size, displayed_enabled, displayed_box, available=regional_available, cpu_reuse=cpu_reuse)
            self.simple_workspace.regional.sync(size, displayed_enabled, displayed_box, available=regional_available, cpu_reuse=cpu_reuse)
            if active_checked:
                note = (image_submission_budget_text(active_checked.get('image_budget') or {})
                    + '；已冻结，处理中不能修改。'
                    + (' 局部精修使用本地 GPU 推理，不是外部付费请求。' if active_maximum else ''))
                for controls in (self.regional_controls, self.simple_workspace.regional): controls.note.setText(note)
            self.simple_workspace.select_region.setEnabled(regional_available and (self.regional_enabled or cpu_reuse) and bool(size))
            self.simple_workspace.source_image.selection = ([self.regional_core_box[0]/size[0], self.regional_core_box[1]/size[1],
                self.regional_core_box[2]/size[0], self.regional_core_box[3]/size[1]] if size and self.regional_core_box else None)
            saved_direction = self.product_run.get('creative_direction') or (self.product_run.get('options') or {}).get('creative_direction') or {}
            direction = active_checked.get('creative_direction') or ((self.creative_task_plan or {}).get('plan') or {}).get('creative_direction') or saved_direction
            self.simple_workspace.direction_summary.setText((('本次制作已冻结方向' if active_checked else '已冻结方向') + ' · 角色：' + str((direction.get('character') or {}).get('name') or '未指定')
                + ' · ' + str(direction.get('scene') or '') + '\n主效：' + str((direction.get('main_effect') or {}).get('description') or '未指定'))
                if direction else '本次方向尚未冻结；旧版本不会补写角色或场景权限。请检查新任务计划。')
            observed = (self.product_run.get('generation_design') or {}).get('observed_design') or {}
            if same and observed and not active_checked:
                self.simple_workspace.direction_summary.setText('已保存版本设计：' + str(observed.get('summary') or '')
                    + '\n统一光照：' + str(observed.get('lighting') or '')
                    + '\n编辑：' + '；'.join(str(row.get('instruction') or '') for row in observed.get('changes', [])[:3]))
            if same_photo:
                self.simple_workspace.direction_summary.setText(self.simple_workspace.direction_summary.text()
                    + '\n同照精修：' + ('匹配意图已冻结；执行时实际核验配准。' if active_checked or self.creative_task_plan
                                          else '尚未实际配准；请先检查计划。')
                    + ' 原片保有几何与原生纹理；新增生图 0 次。')
            if direction.get('recommendation'):
                self.simple_workspace.direction_summary.setText(self.simple_workspace.direction_summary.text()
                    + '\n产品推荐：原景精修；用户作品要求、暖光和彩灯保留项优先。实际完成范围仍须看保存结果。')
            status_label = {'needs_visual_review': '已保存 · 待用户审片', 'delivered': '已有批准记录', 'accepted': '已有批准记录'}.get(self.source_status, self.source_status or '尚未保存')
            dimensions = ' × '.join(map(str, size)) if same and size else '尺寸尚未载入'
            self.simple_workspace.version_summary.setText((f'r{self.source_revision} · {dimensions} · {status_label} · {self.source_run_id}'
                + (' · 当前修改未保存' if self._session_dirty else '')) if same else '尚未打开此原片的保存版本')
            self.simple_workspace.native_review_button.setEnabled(same and self.native_detail_button.isEnabled() and not busy)
            self.simple_workspace.start_plan_button.setEnabled(not busy and bool(self.creative_task_plan)
                and self.creative_task_plan['plan'].get('ready') is True)
            if self.startup_readiness == 'degraded' and not busy and not same:
                self.simple_workspace.set_message('启动预热未确认；选择制作方式后会检查本次所需能力。')
            self.refresh_generation_models(locked=busy or awaiting_variant or self._session_loading)
            run_variants = self.product_run.get('variants') if isinstance(self.product_run, dict) else None
            pending_variants = run_variants if (isinstance(run_variants, list)
                and same and self.source_status == 'awaiting_reference') else []
            self.simple_workspace.set_candidate_pending(bool(pending_variants), len(pending_variants))
            self.simple_workspace.set_images(original, result)
            self.simple_workspace.set_strength(round(float((self.portrait_beauty or {}).get('strength', 1.))*100))
            can_continue_post = bool(same and self.source_run_has_reference and self.design_reconstruction_plan
                                    and self.source_status == 'needs_visual_review'
                                    and not self.render_pending)
            self.simple_workspace.set_post_continuation(can_continue_post, dirty=self._session_dirty)
            self.simple_workspace.set_preparation_state(self.preparation_version_review.text() if same else '')
            self.simple_workspace.set_compare(self.simple_comparing and same)
            if busy or same:
                message = self.operation_state.text()
                if self._session_dirty and not busy:
                    message = '当前效果尚未保存，确认后点击“保存高清版本”。'
                self.simple_workspace.set_message(message)
            if busy or awaiting_variant:
                job = self.job_rows.get(self.active_job_id, {}) if self.active_job_id else {}
                checked_job = (self.active_checked_task or {}).get('job_id')
                if self.active_checked_task and (not checked_job or checked_job != self.active_job_id):
                    job = {}  # An older selected job is not this pending task.
                if self.creative_task_plan_request or (self.creative_task_start_pending and not job):
                    progress, stage, local_progress, stage_index = None, '检查制作计划' if self.creative_task_plan_request else '等待任务提交', None, None
                    message = '正在检查输入与配置；尚未启动制作。' if self.creative_task_plan_request else '开始请求已发送，等待本次任务编号。'
                else:
                    progress, stage, message, local_progress, stage_index = self._quick_progress(job.get('progress'),
                        job.get('stage', ''), job.get('kind', ''),
                        timing=job.get('timing') or (job.get('data') or {}).get('timing'))
                self.simple_workspace.set_progress_stages(self._quick_progress_stages())
                self.simple_workspace.set_job_progress(progress, stage, job.get('message') or message,
                    local_progress, stage_index)
            else:
                self.simple_workspace.set_progress_stages(self._quick_progress_stages())
            self.simple_workspace.export_button.setEnabled(same and bool(result) and not busy and not self._session_dirty)
            self.simple_workspace.brief.blockSignals(True)
            self.simple_workspace.brief.setText(self.creative_task_brief.toPlainText())
            self.simple_workspace.brief.blockSignals(False)
            intent = self.simple_creation or {}
            self.simple_workspace.cancel_button.setEnabled(bool(self.simple_creation or (self.render_pending and self.active_job_id
                and self.active_job_id != intent.get('initial_job'))))
            run = self._cos_draft_run()
            signature = json.dumps([run.get('run_id'), run.get('options'), self.product_stage_report], sort_keys=True, ensure_ascii=False)
            if signature != self._workflow_signature:
                self._workflow_signature = signature
                from cos_workflow_graph import route_for_run
                try:
                    route_for_run(run)
                except ValueError as error:
                    # Saved pixels can remain readable without a current editable
                    # workflow contract. Never invent that contract from UI defaults.
                    self.workflow_canvas.set_run({})
                    self._workflow_canvas_available = False
                    self.workflow_canvas.setToolTip('此保存版本没有当前可编辑工作流；已保存画面仍可查看、审片与导出。')
                    self.log_text('[workflow canvas] ' + str(error))
                else:
                    self.workflow_canvas.set_run(run,self.product_stage_report,
                        template=None if self.source_generated_appearance else self.workflow_graph)
                    self._workflow_canvas_available = True
                    if self.source_generated_appearance and not self.workflow_canvas.graph.get('readonly'):
                        # An older record without the scoped contract cannot be
                        # displayed as an editable source-layer production graph.
                        self.workflow_canvas.set_run({})
                        self._workflow_canvas_available = False
                    self.workflow_canvas.setToolTip('可选择节点查看本次保存事实；生成外观流程只读，原片像素工具不适用于该版本。'
                        if self.source_generated_appearance else '')
            self.workflow_canvas.set_busy(busy)
            self.workflow_canvas.setEnabled(not busy and self._workflow_canvas_available)
            self.update_generated_appearance_actions()
            self.update_editing_improvement_actions()
            self.update_shaping_actions()

        def _quick_progress_stages(self):
            """The checked route is display data, never an execution authority."""
            active = getattr(self, 'active_checked_task', None) or {}
            context = getattr(self, 'creative_task_plan', None) or {}
            intent = self.simple_creation or {}
            stages = (active.get('progress_stages') or (context.get('plan') or {}).get('stages')
                      or intent.get('progress_stages'))
            if stages:
                return stages
            if getattr(self, 'creative_task_plan_request', None):
                return []
            # Only the old explicitly sequenced editing route has these coarse
            # UI phases. Never infer local source-first order from a model name.
            if intent.get('mode') == 'beauty':
                return ['原片准备', '人像美化', '高清保存']
            if intent.get('mode') in {'create_cos', 'existing_reference'} and intent.get('stage') != 'production':
                return [('复用参考' if intent['mode'] == 'existing_reference' else '候选设计与选择'),
                        '人像美化', '仿色与边界融合', '高清保存与审片']
            return []

        def _show_quick_job_progress(self, job):
            """Polling, SSE and preparation callbacks share one presentation."""
            kind = job.get('kind') or self.job_rows.get(self.active_job_id, {}).get('kind', '')
            progress, stage, message, local, index = self._quick_progress(job.get('progress'),
                job.get('stage', ''), kind,
                timing=job.get('timing') or (job.get('data') or {}).get('timing'))
            self.simple_workspace.set_progress_stages(self._quick_progress_stages())
            self.simple_workspace.set_job_progress(progress, stage, job.get('message') or message, local, index)

        def _quick_progress(self, progress=None, stage='', kind='', *, timing=None):
            """Map each worker's local fraction onto the quick route.

            Worker progress is deliberately local to its preparation contract.
            The quick UI presents one monotonic route so starting BeautyGRPO or
            the reconstruction worker never makes the bar jump back to zero.
            Values are coarse milestones, not time estimates.
            """
            intent = self.simple_creation or {}
            mode = intent.get('mode')
            current = intent.get('stage')
            expected_kind = {'base': 'one_click', 'selecting_variant': 'select_reference_variant',
                'production': 'one_click',
                'beauty': 'prepare_portrait_beauty', 'colour': 'native_design_reconstruction',
                'boundary': 'native_design_reconstruction',
                'post': 'one_click'}.get(current)
            if kind and expected_kind and kind != expected_kind:
                progress = None
                timing = None
            observed_label = str((timing or {}).get('module_label') or '') if isinstance(timing, dict) else ''
            if observed_label:
                try: elapsed = float(timing.get('elapsed_seconds'))
                except (TypeError, ValueError): elapsed = None
                if elapsed is not None and 0 <= elapsed < float('inf'):
                    timing_key = (str(intent.get('source') or ''), str(intent.get('mode') or ''),
                        str(getattr(self, 'active_job_id', '') or ''), str(timing.get('request_id') or ''))
                    previous = getattr(self, '_quick_timing_cursor', None)
                    if previous and previous[0] == timing_key and elapsed < previous[1]:
                        observed_label = previous[2]
                    else:
                        self._quick_timing_cursor = (timing_key, elapsed, observed_label)
            try:
                local = None if progress is None else min(1., max(0., float(progress)))
            except (TypeError, ValueError):
                local = None
            active=getattr(self,'active_checked_task',None) or {}
            checked=getattr(self,'creative_task_plan',None) or {}
            if isinstance(active.get('request'),dict):scope=active['request'].get('editing_request') or {}
            else:scope=((checked.get('plan') or {}).get('editing_scope')
                or ((getattr(self,'product_run',None) or {}).get('task_plan') or {}).get('editing_scope') or {})
            if scope.get('work_type') in {'partial_composite','full_composite'} and current in {None,'production','base'}:
                return None, observed_label or stage or '制作执行中', '当前模块与用时来自实际记录；总体进度尚无法可靠估计。', None, None
            if not intent:
                return local, observed_label or stage, '', None, None
            if current == 'awaiting_variant':
                return .40, 'awaiting_reference', '候选设计已生成；请选择一张后继续。', None, None
            if mode == 'create_cos':
                spans = {
                    'base': (.05, .35, '正在生成角色风格候选。'),
                    'selecting_variant': (.40, .02, '正在准备所选候选的高清原片结果。'),
                    'beauty': (.42, .13, '正在准备 BeautyGRPO 人像美化。'),
                    'colour': (.55, .14, '正在准备仿色、材质与背景融合。'),
                    'boundary': (.69, .10, '正在自动补充人物边界受光。'),
                    'post_save_wait': (.79, .04, '正在保存高清融合版本，随后进入特效与完整后期。'),
                    'post': (.83, .16, '正在完成已授权的成片处理与审片。'),
                }
            else:
                spans = {
                    'base': (.05, .30, '正在准备原片。'),
                    'beauty': (.35, .35, '正在准备 BeautyGRPO 人像美化。'),
                    'colour': (.70, .10, '正在准备本地融合。'),
                    'boundary': (.80, .08, '正在自动补充人物边界受光。'),
                    'post_save_wait': (.88, .03, '正在保存高清版本，随后进入后期。'),
                    'post': (.91, .08, '正在完成后期审片。'),
                }
            spans['production'] = (.05, .94, '正在按已检查计划制作；当前模块以实际任务事件为准。')
            start, width, fallback = spans.get(current, (.05, .90, '正在处理当前阶段。'))
            shown = start if local is None else start + width * local
            # Worker events can arrive through both SSE and polling.  Keep a
            # single route monotonic when an older event arrives after a newer
            # one, while allowing a new stage to advance from its own start.
            key = (str(intent.get('source') or ''), str(intent.get('mode') or ''), str(current or ''),
                   str(getattr(self, 'active_job_id', '') or ''))
            if key != getattr(self, '_quick_progress_key', None):
                self._quick_progress_key = key
                route_key = (str(intent.get('source') or ''), str(intent.get('mode') or ''))
                if route_key != getattr(self, '_quick_route_key', None):
                    self._quick_progress_value = 0.0
                    self._quick_route_key = route_key
                self._quick_local_progress_value = None
            previous_local = getattr(self, '_quick_local_progress_value', None)
            if previous_local is not None:
                local = previous_local if local is None else max(previous_local, local)
            self._quick_local_progress_value = local
            shown = max(float(getattr(self, '_quick_progress_value', 0.0)), float(shown))
            self._quick_progress_value = min(.99, shown)
            stage_index = {'base': 0, 'selecting_variant': 0, 'beauty': 1, 'colour': 2,
                           'boundary': 2, 'post_save_wait': 3, 'post': 3}.get(current, 0)
            checked_job = current == 'production' or bool(getattr(self, 'active_checked_task', None))
            label = observed_label or ('完整制作' if current == 'production' else stage if checked_job and stage else current)
            # Source-job fractions describe whole-job milestones. A module's
            # elapsed time does not make that fraction a local phase percent.
            return self._quick_progress_value, label, fallback, None if checked_job else local, None if checked_job else stage_index

        def workflow_template_changed(self, template):
            if self.source_generated_appearance: return
            from cos_workflow_graph import revision_for_template, validate_template, compile_template, route_for_run
            from cos_workflow_cache import compare as compare_workflow_cache
            previous_template = self.workflow_graph
            comparison_run = self._cos_draft_run()
            previous_compiled = (compile_template(previous_template, route_for_run(comparison_run))
                                 if previous_template else None)
            self.workflow_graph = validate_template(template)
            current_compiled = compile_template(self.workflow_graph, route_for_run(comparison_run))
            self._workflow_cache_state = compare_workflow_cache(previous_compiled, current_compiled)
            patch = revision_for_template(self._cos_draft_run(), self.workflow_graph)
            pixels_changed = (('portrait_beauty' in patch and patch['portrait_beauty'] != self.portrait_beauty)
                or ('design_reconstruction_plan' in patch and patch['design_reconstruction_plan'] != self.design_reconstruction_plan))
            # Do not rebuild a scene while a node is being dragged or adjusted.
            if patch.get('portrait_beauty') is not None:
                self.load_portrait_beauty(patch['portrait_beauty'], self.portrait_beauty_faces)
            if patch.get('design_reconstruction_plan') is not None:
                self.load_design_reconstruction(patch['design_reconstruction_plan'])
            draft = self._cos_draft_run()
            self._workflow_signature = json.dumps([draft.get('run_id'), draft.get('options'), self.product_stage_report], sort_keys=True, ensure_ascii=False)
            if pixels_changed:
                self.source_light_version += 1; self.simple_comparing = False
                invalidated = self._workflow_cache_state.get('invalidated', [])
                if invalidated:
                    self.source_light_state.setText('工作流变更，将重算：' + ' → '.join(invalidated) + '。正在准备成片预览…')
            self._checkpoint_session()
            if pixels_changed and self.source_run_id and not self.render_pending: self.source_light_timer.start()
            self.refresh_cos_workspace()

        def workflow_node_action(self, node):
            if node == 'source': self.choose_source(); return
            if node == 'beauty': self.prepare_portrait_beauty(); return
            if node == 'boundary': self.prepare_boundary_light_coordination(); return
            if node == 'composite': self.save_source_lights(); return
            if node == 'export': self.export_source_run(); return
            self.set_professional_mode(True)
            if node == 'reference': self.tabs.setCurrentIndex(0); self.use_existing_design_reference(); return
            self.tabs.setCurrentIndex(2)
            if node == 'matte': self.local_editor_tabs.setCurrentIndex(7)
            elif node in {'colour', 'background', 'effects'}: self.local_editor_tabs.setCurrentIndex(2)

        def simple_action(self, action):
            if action == 'native_review': self.request_native_detail([.5,.5]); return
            if action == 'check_plan': self.check_creative_task_plan(); return
            if action == 'start_plan': self.start_planned_creative_task(); return
            if action == 'professional': self.set_professional_mode(True); return
            if action == 'generation_settings':
                self.set_professional_mode(True); self.tabs.setCurrentIndex(1)
                self.provider_panel_toggle.setChecked(True); self.provider.setFocus(); return
            if action == 'select_candidate':
                self.set_professional_mode(True)
                self.tabs.setCurrentIndex(1)  # 候选页：先选设计，再开始后续图层流程
                self.operation_state.setText('请比较角色风格候选，采用一张后才进入仿色、融合和后期。')
                return
            if action == 'continue_post':
                if not self.source_run_id or not self.source_run_has_reference or not self.design_reconstruction_plan:
                    self.operation_state.setText('请先选择候选并完成仿色融合准备。'); return
                if self.render_pending: return
                self._checkpoint_session()
                if self._session_checkpoint_failed:
                    self.operation_state.setText(self.session_draft_state.text()); return
                intent = self.simple_creation or {'source': str(Path(self.source_run_source).resolve()), 'mode': 'create_cos'}
                intent['stage'] = 'post_save_wait' if self._session_dirty else 'post'
                self.simple_creation = intent
                if self._session_dirty:
                    self.operation_state.setText('正在保存当前仿色与融合版本；保存完成后自动进入特效与完整后期。')
                    self.save_source_lights()
                else:
                    self.complete_current_post()
                self.refresh_cos_workspace()
                return
            if action == 'compare':
                self.simple_comparing = not self.simple_comparing; self.refresh_cos_workspace(); return
            if action == 'cancel':
                self.cancel_startup_service(suppress=True)
                # Stop the automatic chain immediately, including a success/cancel race.
                intent = self.simple_creation or {}
                if self.creative_task_start_pending and self.creative_task_execution_label:
                    self.simple_cancel_submissions.add(self.creative_task_execution_label)
                if self.portrait_beauty_request == 'submitting': self.simple_cancel_submissions.add('portrait-beauty-job')
                if self.design_reconstruction_request == 'submitting': self.simple_cancel_submissions.add('design-reconstruct-job')
                self.simple_creation = None; self._reset_quick_progress()
                self.invalidate_creative_task_plan()
                if self.render_pending and self.active_job_id and self.active_job_id != intent.get('initial_job'):
                    self.task('POST', f'/api/jobs/{self.active_job_id}/cancel', {}, 'job-cancel')
                self.refresh_cos_workspace()
                return
            if action == 'save': self.save_source_lights(); return
            if action == 'export': self.export_source_run(); return
            if action not in {'beauty', 'create_cos', 'existing_reference'} or self.render_pending or self.simple_creation: return
            source = self.source_edit.text().strip()
            if not source: self.choose_source(); return
            if self.embedded_backend and self.embedded_server is None: self.start_backend()
            self.simple_creation = {'source': str(Path(source).resolve()), 'mode': action, 'stage': 'base', 'initial_job': self.active_job_id}
            self._reset_quick_progress()
            if action == 'beauty' and self.source_run_id and Path(source).resolve() == Path(self.source_run_source).resolve():
                # Reusing an open revision must pass the same capability plan;
                # startup's degraded/ready flag is never execution authority.
                self.request_creative_task_plan('local_portrait', source=source,
                    parent_run_id=self.source_run_id, prepare_existing_beauty=True)
                if self.creative_task_plan_request is None: self.simple_creation = None
                else: self.simple_creation['plan_label'] = self.creative_task_plan_request['label']
            else:
                if action == 'existing_reference':
                    reference, _filter = QFileDialog.getOpenFileName(self, '选择已有参考图', str(Path(source).parent), 'Images (*.png *.jpg *.jpeg *.tif *.tiff)')
                    if not reference:
                        self.simple_creation = None; self.refresh_cos_workspace(); return
                    mode = 'existing_reference'
                else:
                    reference = None; mode = 'local_portrait' if action == 'beauty' else 'create_reference'
                parent = (self.source_run_id if self.source_run_id and self.source_run_source
                    and Path(source).resolve() == Path(self.source_run_source).resolve()
                    and not self._session_dirty and self.source_status in {'needs_visual_review','accepted','delivered'} else None)
                self.request_creative_task_plan(mode, source=source, reference=reference, parent_run_id=parent)
                if self.creative_task_plan_request is None: self.simple_creation = None
                elif self.simple_creation is not None:
                    self.simple_creation['plan_label'] = self.creative_task_plan_request['label']
            self.refresh_cos_workspace()

        def _advance_simple_creation(self, stage):
            intent = self.simple_creation
            if not intent: return
            if (self.source_generated_appearance and stage == 'base'
                    and intent.get('stage') in {'base', 'awaiting_variant', 'selecting_variant'}
                    and not self._active_production_job()
                    and intent.get('source') == str(Path(self.source_run_source).resolve())):
                self.simple_creation = None
                self.operation_state.setText('所选生成画面已保存，可审片或导出。')
                self.refresh_cos_workspace()
                return
            # Explicit multi-candidate requests pause for selection. Once the
            # user adopts one, continue its local editing chain. Single-image
            # production is completed by the backend and has no second chain.
            if stage == 'base' and intent.get('stage') in {'awaiting_variant', 'selecting_variant'} and self.source_run_has_reference:
                intent['stage'] = 'beauty'
                self.prepare_portrait_beauty()
                if self.portrait_beauty_request is None: self.simple_creation = None
                self.refresh_cos_workspace()
                return
            if intent['stage'] != stage: return
            if (not self.source_run_id or intent['source'] != str(Path(self.source_run_source).resolve())
                    or self.source_status in {'failed', 'blocked', 'cancelled', 'render_failed', 'needs_rerender'}):
                self.simple_creation = None; return
            if stage == 'base':
                if intent['mode'] == 'create_cos' and self.source_status == 'awaiting_reference':
                    if self.product_run.get('variants'):
                        intent['stage'] = 'awaiting_variant'
                        self.operation_state.setText('候选设计已生成；请选择一张后继续 Cos 后期。')
                        self.refresh_cos_workspace()
                    else:
                        self.operation_state.setText('候选生成未返回图片；请检查任务详情后重试。')
                        self.simple_creation = None
                    return
                from cos_workflow_graph import preset
                self.workflow_template_changed(preset('portrait' if intent['mode'] == 'beauty' else 'cos_scene'))
                intent['stage'] = 'beauty'; self.prepare_portrait_beauty()
                if self.portrait_beauty_request is None: self.simple_creation = None
            elif stage == 'beauty' and intent['mode'] in {'create_cos', 'existing_reference'} and self.source_run_has_reference:
                intent['stage'] = 'colour'; self.reconstruct_design()
                if self.design_reconstruction_request is None: self.simple_creation = None
            else:
                self.simple_creation = None
            self.refresh_cos_workspace()
        def _apply_workflow_intent(self):
            if not self.workflow_graph: return
            from cos_workflow_graph import revision_for_template
            patch = revision_for_template(self._cos_draft_run(), self.workflow_graph)
            if 'portrait_beauty' in patch: self.load_portrait_beauty(patch['portrait_beauty'], self.portrait_beauty_faces)
            if 'design_reconstruction_plan' in patch: self.load_design_reconstruction(patch['design_reconstruction_plan'])

        def accordion(self, title, content, expanded=False):
            panel = QWidget(); layout = QVBoxLayout(panel); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(5)
            toggle = QToolButton(); toggle.setObjectName("accordion"); toggle.setText(title); toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon); toggle.setCheckable(True); toggle.setChecked(expanded)
            toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow); layout.addWidget(toggle); layout.addWidget(content); content.setVisible(expanded)
            toggle.toggled.connect(lambda checked: (content.setVisible(checked), toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)))
            return panel, toggle
        def scrollable_page(self, content):
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(content)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded); scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            return scroll
        def project_page(self):
            from generation_prompt import DEFAULT_PROMPT, compose_generation_prompt
            from preset_store import list_presets
            page = QWidget(); layout = QHBoxLayout(page); layout.setContentsMargins(14, 14, 14, 10)
            split = QSplitter(Qt.Horizontal); layout.addWidget(split)
            canvas = QWidget(); canvas.setObjectName("canvas"); canvas_layout = QVBoxLayout(canvas); canvas_layout.setContentsMargins(14, 12, 14, 10)
            canvas_bar = QHBoxLayout(); self.canvas_title = QLabel("你的下一张作品"); self.canvas_title.setObjectName("sectionTitle"); canvas_bar.addWidget(self.canvas_title); canvas_bar.addStretch()
            self.project_view_selector = QComboBox(); self.project_view_selector.addItem("原片", "source"); self.project_view_selector.addItem("GPT 候选", "reference"); self.project_view_selector.addItem("高清草稿", "final"); canvas_bar.addWidget(self.project_view_selector)
            self.choose_source_button = QPushButton("导入原图"); self.choose_source_button.setObjectName("secondaryButton"); self.choose_source_button.clicked.connect(self.choose_source); canvas_bar.addWidget(self.choose_source_button); canvas_layout.addLayout(canvas_bar)
            self.project_preview = Preview("原片"); self.project_preview.path_label.hide(); self.project_preview.scroll.setObjectName("canvas"); self.project_preview.image.setObjectName("canvas"); self.project_preview.image.setText("导入 JPG / RAW 原图\n在这里预览你的作品"); self.project_preview.image.adjustSize(); canvas_layout.addWidget(self.project_preview, 1)
            self.project_image_info = QLabel("导入原图，选择成片效果"); self.project_image_info.setObjectName("caption"); canvas_layout.addWidget(self.project_image_info)
            self.project_view_selector.currentIndexChanged.connect(self.update_project_canvas); split.addWidget(canvas)
            inspector = QWidget(); inspector.setObjectName("inspector"); form = QVBoxLayout(inspector); form.setContentsMargins(18, 18, 18, 18); form.setSpacing(12)
            heading = QLabel("创作设置"); heading.setObjectName("sectionTitle"); form.addWidget(heading)
            self.source_edit = QLineEdit(); self.source_edit.setPlaceholderText("原图路径"); self.batch_sources = []
            self.creative_task_mode = QComboBox()
            for mode, label in CREATIVE_TASK_MODES.items(): self.creative_task_mode.addItem(label, mode)
            form.addWidget(QLabel('本次输入方式')); form.addWidget(self.creative_task_mode)
            form.addWidget(self.build_editing_scope_panel())
            self.creative_task_brief = QPlainTextEdit(); self.creative_task_brief.setPlaceholderText('简明要求：角色、氛围、重点，以及需要保留的妆造/纹样/道具。批次会共享这些要求。')
            self.creative_task_brief.setMinimumHeight(65); self.creative_task_brief.setMaximumHeight(95); form.addWidget(self.creative_task_brief)
            self.creative_task_reference = QLineEdit(); self.creative_task_reference.setPlaceholderText('已有参考路径（PNG / JPG / TIFF）'); self.creative_task_reference.hide(); form.addWidget(self.creative_task_reference)
            self.creative_background_mode = QComboBox(); self.creative_background_mode.setAccessibleName('专业场景调整范围')
            for label, value in (('原景增强', 'enhance_source'), ('严格保留原背景', 'preserve'), ('允许换景', 'replace')):
                self.creative_background_mode.addItem(label, value)
            form.addWidget(QLabel('场景调整范围')); form.addWidget(self.creative_background_mode)
            from simple_cos_workspace import same_photo_finish_checkbox
            self.creative_same_photo_finish = same_photo_finish_checkbox()
            self.creative_same_photo_finish.toggled.connect(self.set_same_photo_finish)
            form.addWidget(self.creative_same_photo_finish)
            constraints = QHBoxLayout(); self.creative_preserve_background = QCheckBox('保留原背景'); self.creative_no_effects = QCheckBox('不加特效')
            self.creative_preserve_background.hide()
            self.creative_preserve_background.setChecked(True); self.creative_no_effects.setChecked(True)
            constraints.addWidget(self.creative_preserve_background); constraints.addWidget(self.creative_no_effects); form.addLayout(constraints)
            self.creative_text_vision = QCheckBox('观察原片并设计本次编辑（开始时发送原片与创作要求）'); self.creative_text_vision.setEnabled(False); form.addWidget(self.creative_text_vision)
            self.creative_visual_review = QCheckBox('独立审片（额外耗时与费用，会发送成片）'); self.creative_visual_review.setEnabled(False); self.creative_visual_review.hide(); form.addWidget(self.creative_visual_review)
            plan_actions = QHBoxLayout(); self.creative_plan_button = QPushButton('检查制作计划'); self.creative_plan_button.clicked.connect(self.check_creative_task_plan)
            self.creative_start_button = QPushButton('开始制作'); self.creative_start_button.setObjectName('primaryButton'); self.creative_start_button.setEnabled(False); self.creative_start_button.clicked.connect(self.start_planned_creative_task)
            plan_actions.addWidget(self.creative_plan_button); plan_actions.addWidget(self.creative_start_button); form.addLayout(plan_actions)
            self.creative_plan_summary = QPlainTextEdit('先检查计划：这里只读取配置与文件，不启动模型、上传图片或创建制作任务。'); self.creative_plan_summary.setReadOnly(True)
            self.creative_plan_summary.setMinimumHeight(150); self.creative_plan_summary.setMaximumHeight(260)
            plan_panel, self.creative_plan_toggle = self.accordion('本次执行计划与范围', self.creative_plan_summary); form.addWidget(plan_panel)
            self.creative_task_mode.currentIndexChanged.connect(self.creative_task_mode_changed)
            self.creative_background_mode.currentIndexChanged.connect(self.background_mode_changed)
            self.creative_task_brief.textChanged.connect(self.invalidate_creative_task_plan)
            self.creative_task_reference.textChanged.connect(self.invalidate_creative_task_plan)
            for control in (self.creative_preserve_background, self.creative_no_effects, self.creative_text_vision, self.creative_visual_review): control.toggled.connect(self.creative_task_policy_changed)
            self.existing_design_button = QPushButton("选择已有参考并检查后期计划"); self.existing_design_button.setObjectName("secondaryButton")
            self.existing_design_button.setToolTip("复用参考并检查完整后期；默认零生图。主动授权局部精修时，以本次检查计划预算为准。")
            self.existing_design_button.clicked.connect(self.use_existing_design_reference); form.addWidget(self.existing_design_button)
            self.existing_design_state = QLabel("已有参考与原片成对规划；缺失或不支持的步骤会列入计划。"); self.existing_design_state.setWordWrap(True); self.existing_design_state.setObjectName("caption"); form.addWidget(self.existing_design_state)
            self.local_source_button = QPushButton("检查自然原片后期计划"); self.local_source_button.setObjectName("secondaryButton"); self.local_source_button.setToolTip("检查原片本地处理；不调用外部文本、视觉或生图服务。"); self.local_source_button.clicked.connect(self.run_local_source); form.addWidget(self.local_source_button)
            project_files=QHBoxLayout();self.import_project_button=QPushButton('导入工程');self.export_project_button=QPushButton('导出已保存工程')
            self.import_project_button.clicked.connect(self.import_portable_project);self.export_project_button.clicked.connect(self.export_portable_project)
            self.export_project_button.setToolTip('打包当前已保存版本、原片与设计图。请先保存微调；未保存的编辑不会混入工程包。')
            self.export_project_button.setEnabled(False);project_files.addWidget(self.import_project_button);project_files.addWidget(self.export_project_button);form.addLayout(project_files)
            self.portable_project_state=QLabel('工程包保存原片、设计、高清图层与编辑参数；微调请先保存版本。');self.portable_project_state.setWordWrap(True);self.portable_project_state.setObjectName('caption');form.addWidget(self.portable_project_state)
            form.addWidget(self.product_result_panel())
            self.pipeline_summary = QPlainTextEdit("打开已保存工程后显示全流程进度与下一步。"); self.pipeline_summary.setReadOnly(True); self.pipeline_summary.setMinimumHeight(150); self.pipeline_summary.setMaximumHeight(210); self.pipeline_summary.setToolTip("以当前已保存版本的资产和处理记录为准；未保存编辑需先保存。")
            pipeline_panel, _toggle = self.accordion('流程明细与来源', self.pipeline_summary); form.addWidget(pipeline_panel)
            self.preparation_version_workspace = QLabel('打开已保存版本后显示处理版本。'); self.preparation_version_workspace.setWordWrap(True); form.addWidget(self.preparation_version_workspace)
            self.complete_post_button = QPushButton('按当前算法重新制作')
            self.complete_post_button.setToolTip('复用原片、参考和美化，按当前算法重新准备完整后期并保存新版本；不新增生图。')
            self.complete_post_button.setEnabled(False)
            self.complete_post_button.clicked.connect(self.complete_current_post)
            form.addWidget(self.complete_post_button)
            self.character_name = QLineEdit(); self.character_name.setPlaceholderText("角色名 · 由你确认，可留空")
            self.character_name.setAccessibleName('专业角色名')
            self.character_notes = QLineEdit(); self.character_notes.setPlaceholderText("作品、版本或你希望的氛围（可选）")
            self.character_notes.setAccessibleName('专业角色作品版本说明')
            form.addWidget(self.character_name); form.addWidget(self.character_notes)
            from simple_cos_workspace import RegionalRefinementControls
            self.regional_controls = RegionalRefinementControls(); form.addWidget(self.regional_controls)
            self.regional_controls.setAccessibleName('专业模式局部精修设置')
            self.regional_controls.enabledChanged.connect(self.set_regional_enabled)
            self.regional_controls.boxChanged.connect(self.set_regional_box)
            self.character_recognize = QPushButton("识别角色并推荐风格"); self.character_recognize.setObjectName("secondaryButton"); self.character_recognize.clicked.connect(self.recognize_character)
            self.character_result = QLabel("默认按你填写的角色与要求制作；额外识别与研究须启用服务。"); self.character_result.setWordWrap(True)
            self.recommended_styles = QComboBox(); self.recommended_styles.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon); self.recommended_styles.setMinimumContentsLength(12); self.recommended_styles.addItem("识别后显示推荐", None)
            self.character_apply_style = QPushButton("使用推荐风格"); self.character_apply_style.clicked.connect(self.apply_recommended_style)
            character_details = QWidget(); character_layout = QVBoxLayout(character_details); character_layout.setContentsMargins(0, 0, 0, 0); character_layout.addWidget(self.character_recognize); character_layout.addWidget(self.character_result); character_layout.addWidget(self.recommended_styles); character_layout.addWidget(self.character_apply_style)
            character_panel, self.character_details_toggle = self.accordion("角色识别与初步建议", character_details); form.addWidget(character_panel)
            self.preset = QComboBox(); self.preset.addItem("按角色与本图自动设计", ""); self.preset_rows = {row["id"]: row for row in list_presets()}
            for identifier, row in self.preset_rows.items(): self.preset.addItem(row["name"], identifier)
            # Keep the first-run route generic. The source-refinement profile
            # remains available in the selector, but it must be chosen by the
            # user instead of becoming an invisible task constraint.
            self.preset.setCurrentIndex(0)
            self.preset.currentIndexChanged.connect(self.select_preset)
            self.preset.activated.connect(self.preset_activated)
            style_label = QLabel("艺术风格"); style_label.setObjectName("caption"); form.addWidget(style_label); form.addWidget(self.preset)
            self.generation_style = QLabel(self.preset_rows.get(self.preset.currentData(), {}).get('description') or '依据角色、服装和场景设计合适的灯光与氛围。'); self.generation_style.setWordWrap(True); self.generation_style.setObjectName("caption"); form.addWidget(self.generation_style)
            quantity = QHBoxLayout(); quantity.addWidget(QLabel("候选数量")); self.variant_count = QSpinBox(); self.variant_count.setRange(1, 8); self.variant_count.setValue(1); quantity.addWidget(self.variant_count); quantity.addStretch()
            self.batch_button = QPushButton("检查批量创作计划"); self.batch_button.setObjectName("secondaryButton"); self.batch_button.clicked.connect(self.run_batch); quantity.addWidget(self.batch_button); form.addLayout(quantity)
            self.run_source_button = QPushButton("检查创作参考与后期计划"); self.run_source_button.setObjectName("secondaryButton"); self.run_source_button.clicked.connect(self.run_source_only); form.addWidget(self.run_source_button)
            auto_note = QLabel("先检查本次阶段、模型缺口、服务发送与预算，再开始制作；保存、自动审片与人工批准分别记录。"); auto_note.setObjectName("caption"); auto_note.setWordWrap(True); form.addWidget(auto_note)
            self.design_style = QLabel("本图采用风格：生成时自动设计"); self.design_style.setObjectName("caption"); self.design_style.setWordWrap(True); form.addWidget(self.design_style)
            self.final_prompt = QTextEdit(); self.final_prompt.setReadOnly(True); self.final_prompt.setPlaceholderText("制作完成后显示实际采用的编辑提示词，只读展示。"); self.final_prompt.setMinimumHeight(130); self.final_prompt.setMaximumHeight(185)
            final_panel, self.final_prompt_toggle = self.accordion("实际提示词（只读）", self.final_prompt); form.addWidget(final_panel)
            self.final_prompt_toggle.toggled.connect(lambda opened: self.final_prompt.verticalScrollBar().setValue(0) if opened else None)
            self.prompt = QTextEdit(); self.prompt.setPlainText(compose_generation_prompt(DEFAULT_PROMPT,
                self.preset_rows.get(self.preset.currentData(), {}).get('prompt'))); self.prompt.setMinimumHeight(150); self.prompt.setMaximumHeight(220)
            self.prompt_tabs = QTabWidget(); input_tab = QWidget(); input_layout = QVBoxLayout(input_tab); input_layout.addWidget(QLabel("摄影要求与自由艺术方向")); input_layout.addWidget(self.prompt)
            prompt_row = QHBoxLayout(); self.restore_prompt = QPushButton("恢复默认英文提示词"); self.apply_style_prompt = QPushButton("应用所选风格提示词"); self.restore_prompt.clicked.connect(self.restore_generation_prompt); self.apply_style_prompt.clicked.connect(self.apply_selected_style_prompt)
            prompt_row.addWidget(self.restore_prompt); prompt_row.addWidget(self.apply_style_prompt); input_layout.addLayout(prompt_row); self.prompt_tabs.addTab(input_tab, "自由方向")
            research_tab = QWidget(); research_layout = QVBoxLayout(research_tab); self.research_summary = QLabel("尚无保存设计；开始制作后记录实际观察。"); self.research_summary.setWordWrap(True); research_layout.addWidget(self.research_summary)
            self.research_sources = QLabel("来源：实际原图观察或保存的参考资料"); self.research_sources.setWordWrap(True); self.research_sources.setTextFormat(Qt.RichText); self.research_sources.setOpenExternalLinks(True); research_layout.addWidget(self.research_sources)
            self.design_summary = QPlainTextEdit(); self.design_summary.setReadOnly(True); self.design_summary.setMinimumHeight(110); self.design_summary.setMaximumHeight(170); research_layout.addWidget(self.design_summary); self.prompt_tabs.addTab(research_tab, "研究与分析")
            design_panel, self.design_details_toggle = self.accordion("角色资料与设计详情", self.prompt_tabs); form.addWidget(design_panel)
            native_note = QLabel("原图与参考图先仿色，再与背景融合；人物结构和细节来自原片。审片页可调整整体强度、背景与人工精修，历史版本可查看。")
            native_note.setWordWrap(True); native_note.setObjectName("caption"); form.addWidget(native_note)
            history = QWidget(); history_layout = QVBoxLayout(history); history_layout.setContentsMargins(0, 0, 0, 0)
            self.recovery_label = QLabel("暂无恢复记录"); self.recovery_label.setWordWrap(True); history_layout.addWidget(self.recovery_label)
            self.recent_projects = QComboBox(); self.recent_projects.setMinimumContentsLength(10); history_layout.addWidget(self.recent_projects)
            open_recent = QPushButton("打开最近运行"); open_recent.clicked.connect(self.open_recent_project); history_layout.addWidget(open_recent)
            self.recent_projects_label = QLabel("最近项目：读取中…"); self.recent_projects_label.setWordWrap(True); history_layout.addWidget(self.recent_projects_label)
            self.template_selector = QComboBox(); self.template_selector.addItem("选择命名模板", None); history_layout.addWidget(self.template_selector)
            self.template_name = QLineEdit(); self.template_name.setPlaceholderText("模板名称"); history_layout.addWidget(self.template_name)
            save_template = QPushButton("保存当前参数"); load_template = QPushButton("套用模板"); save_template.clicked.connect(self.save_workbench_template); load_template.clicked.connect(self.load_workbench_template); history_layout.addWidget(save_template); history_layout.addWidget(load_template)
            history_panel, _toggle = self.accordion("最近项目与模板", history); form.addWidget(history_panel)
            files = QWidget(); files_layout = QVBoxLayout(files); files_layout.setContentsMargins(0, 0, 0, 0); files_layout.addWidget(self.source_edit)
            choose_batch = QPushButton("选择多张原图"); choose_batch.clicked.connect(self.choose_batch_sources); files_layout.addWidget(choose_batch)
            self.batch_state = QLabel("尚未启动批量任务"); self.batch_state.setWordWrap(True); files_layout.addWidget(self.batch_state)
            local_batch = QHBoxLayout()
            self.local_batch_pair_button = QPushButton('配对已有参考'); self.local_batch_pair_button.clicked.connect(self.choose_local_batch_references)
            self.local_batch_start_button = QPushButton('检查已有参考批量后期计划'); self.local_batch_start_button.clicked.connect(self.run_local_reference_batch)
            local_batch.addWidget(self.local_batch_pair_button); local_batch.addWidget(self.local_batch_start_button); files_layout.addLayout(local_batch)
            self.local_batch_pairs = []; self.local_batch_id = None
            self.local_batch_tree = QTreeWidget(); self.local_batch_tree.setColumnCount(3); self.local_batch_tree.setHeaderLabels(['原图', '参考来源', '制作状态']); self.local_batch_tree.setMaximumHeight(180); files_layout.addWidget(self.local_batch_tree)
            self.local_batch_retry_button = QPushButton('重试选中失败项'); self.local_batch_retry_button.clicked.connect(self.retry_local_batch_item); self.local_batch_retry_button.setEnabled(False)
            self.local_batch_retry_button.setToolTip('按本项保存的策略重试，保留其他成功项；生成结果未知时不会自动再次生成。'); files_layout.addWidget(self.local_batch_retry_button)
            self.local_batch_restore_button = QPushButton('恢复最近制作批次'); self.local_batch_restore_button.clicked.connect(lambda: self.task('GET', '/api/batch', None, 'local-batch-restore')); files_layout.addWidget(self.local_batch_restore_button)
            cleanup_preview = QPushButton("查看可清理历史"); cleanup_apply = QPushButton("清理过时失败记录"); cleanup_preview.clicked.connect(lambda: self.task("GET", "/api/cleanup", None, "cleanup-preview")); cleanup_apply.clicked.connect(self.apply_cleanup); files_layout.addWidget(cleanup_preview); files_layout.addWidget(cleanup_apply)
            files_panel, _toggle = self.accordion("文件与存储", files); form.addWidget(files_panel); form.addStretch()
            self.inspector_scroll = QScrollArea(); self.inspector_scroll.setWidgetResizable(True); self.inspector_scroll.setWidget(inspector); self.inspector_scroll.setMinimumWidth(350); self.inspector_scroll.setMaximumWidth(480); self.inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            split.addWidget(self.inspector_scroll); split.setStretchFactor(0, 1); split.setStretchFactor(1, 0); split.setSizes([900, 400]); return page
        def build_editing_improve_button(self):
            button = QPushButton('按独立审片改进'); button.setAccessibleName('按独立审片改进')
            button.clicked.connect(self.plan_editing_improvement)
            button.setVisible(False); button.setEnabled(False)
            self.editing_improve_buttons.append(button)
            return button

        def editing_improvement_state(self):
            """A UI invitation only; the checked backend verifies review provenance."""
            run = self.product_run
            scope = (run.get('task_plan') or {}).get('editing_scope') or {}
            session = run.get('photo_edit_session') or {}
            review = session.get('independent_evaluation') or {}
            provenance = review.get('provenance') or {}
            observed = {row.get('role'): row.get('sha256') for row in review.get('observed_images', []) if isinstance(row, dict)}
            axes = [(review.get(key) or {}).get('status') for key in ('credibility', 'creative_benefit')]
            expected_source=(run.get('source') or {}).get('sha256')
            if (run.get('options') or {}).get('photo_appearance'):
                from scoped_workflow import _native_asset_fact
                receipt=_native_asset_fact(self.root_path,run.get('initial_source_appearance') or session.get('source_appearance'),run,scope)
                expected_source=((receipt or {}).get('files',{}).get('preview') or {}).get('sha256')
            source = (run.get('source') or {}).get('path')
            draft = (run.get('artifacts') or {}).get('auto_draft')
            current = bool(source and self.source_edit.text().strip()
                and Path(source).resolve() == Path(self.source_edit.text().strip()).resolve()
                and run.get('run_id') == self.source_run_id)
            visible = bool(self.source_generated_appearance and current
                and scope.get('work_type') in {'retouch', 'partial_composite', 'full_composite'}
                and all(value in {'fail', 'unknown', 'pass'} for value in axes)
                and provenance.get('network_used') is True and provenance.get('generation_used') is False
                and provenance.get('model') and expected_source and observed.get('source') == expected_source
                and isinstance(observed.get('candidate'), str) and len(observed['candidate']) == 64)
            if self.source_generated_appearance and current and ((run.get('options') or {}).get('shaping_state') or {}).get('enabled'):
                return {'visible':True, 'enabled':False, 'shaping_active':True,
                    'reason':'当前形体结果支持改力度、恢复和导出；普通反馈继续编辑尚未支持。'}
            if not visible: return {'visible': False, 'enabled': False, 'reason': ''}
            publication = run.get('result_publication') or {}
            reason = ''
            if session.get('status') == 'needs_attention' or (run.get('editing_iteration_result') or {}).get('failure'):
                reason = '已有改进请求结果需要核查；请先查看原任务，不会重复提交模型请求。'
            elif publication.get('state') != 'committed' or publication.get('run_id') != run.get('run_id'):
                reason = '当前版本尚未完成保存提交。'
            elif self._session_dirty or self._session_checkpoint_failed:
                reason = '请先保留当前未保存的修改。'
            elif self._production_mutation_busy() or self.simple_creation or self._session_loading:
                reason = '当前任务仍在处理，请等待完成或取消。'
            elif self.creative_task_plan_request is not None:
                reason = '正在检查制作计划，请等待结果。'
            elif not source or not draft or not Path(source).is_file() or not Path(draft).is_file():
                reason = '当前原片或保存成片不可用。'
            elif not session.get('path') or not Path(session['path']).is_file():
                reason = '当前独立审片记录不可用。'
            if reason: return {'visible': True, 'enabled': False, 'reason': reason}
            stat = Path(draft).stat()
            parent = {'run_id': run['run_id'], 'revision': run.get('revision'),
                'source_sha256': run['source']['sha256'], 'scope_id': scope.get('scope_id'),
                'draft': {'path': str(Path(draft).resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns},
                'review_sha256': hashlib.sha256(json.dumps(review, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
            return {'visible': True, 'enabled': True, 'reason': '', 'parent': parent,
                'review_passed':bool(session.get('automated_target_met') and all(value=='pass' for value in axes))}

        def update_editing_improvement_actions(self):
            if not hasattr(self, 'source_edit'): return
            state = self.editing_improvement_state()
            disclosure = '先检查范围与预算。开始后会向已配置服务发送原片、当前成片、审片及填写的反馈，新增生图与视觉调用，可能产生费用；原版本保留。'
            for button in self.editing_improve_buttons:
                button.setVisible(state['visible']); button.setEnabled(state['enabled'])
                button.setText('普通反馈编辑（形体结果暂不支持）' if state.get('shaping_active') else
                    '按我的反馈改进' if state.get('review_passed') else '按独立审片改进')
                button.setAccessibleName(button.text())
                button.setToolTip(state['reason'] if state.get('shaping_active') else (state.get('reason') or '') + disclosure)

        def plan_editing_improvement(self):
            self._checkpoint_session()
            state = self.editing_improvement_state()
            if not state['enabled']:
                self.operation_state.setText(state.get('reason') or '当前保存版本没有可用于改进的独立审片。')
                return
            feedback,confirmed=QInputDialog.getMultiLineText(self,'改进当前保存版本',
                '请说明希望改善什么；原制作范围保持不变，先检查预算再开始。\n'
                + ('当前自动审片已通过，仍可提出你的要求；必须填写反馈。' if state['review_passed'] else
                   '可以填写你的反馈；留空则依据当前独立审片改进。'))
            if not confirmed:return
            feedback=feedback.strip()
            if len(feedback)>8000 or (state['review_passed'] and not feedback):
                self.operation_state.setText('请填写不超过 8000 字符的具体用户反馈；未开始制作。')
                return
            current=self.editing_improvement_state()
            if not current['enabled'] or current.get('parent')!=state['parent']:
                self.operation_state.setText('当前保存版本或审片已变化；未开始制作，请重新提交反馈。')
                return
            run = copy.deepcopy(self.product_run)
            options = run.get('options') or {}
            original = run.get('task_intent') or options.get('task_intent') or {}
            constraints = original.get('constraints') or {}
            settings = self._template_payload()
            keys = ('provider', 'model', 'vision_model', 'api_base', 'api_url', 'api_key_env', 'model_registry',
                    'algorithm_profile', 'material_finish_profile', 'character_name', 'character_notes', 'preset')
            request = {key: copy.deepcopy(options.get(key, settings.get(key))) for key in keys
                       if options.get(key, settings.get(key)) is not None}
            request.update(source=run['source']['path'], reference=run['artifacts']['auto_draft'],
                parent_run_id=run['run_id'], sample=run.get('sample') or self.source_sample, variant_count=1,
                editing_request={key: (run['task_plan']['editing_scope'])[key]
                    for key in ('work_type', 'style', 'target_setting', 'scene_target', 'language')},
                editing_iteration={'maximum_improvements': 1, 'reason': '根据独立审片纠正主要缺陷'},
                creative_task={'schema_version': 'cosplay_task_intent.v1', 'mode': 'existing_reference',
                    'brief': original.get('brief') or '',
                    'constraints': {key: copy.deepcopy(constraints[key]) for key in
                        ('preserve_background', 'no_effects', 'background_mode') if key in constraints},
                    'services': {'text_vision': True, 'visual_review': True}})
            if feedback:
                request['editing_iteration'].update(user_feedback=feedback,reason='根据用户反馈改进当前保存版本')
            self.invalidate_creative_task_plan()
            self.submit_creative_task_plan(request, mode='existing_reference', improvement_parent=state['parent'])

        def product_result_panel(self):
            panel = QWidget(); layout = QVBoxLayout(panel); layout.setContentsMargins(0, 4, 0, 4); layout.setSpacing(5)
            title = QLabel('导入原片后开始制作'); title.setObjectName('sectionTitle'); title.setWordWrap(True)
            findings = QLabel('保存版本、证据与人工批准分别显示。'); findings.setWordWrap(True)
            note = QLabel(''); note.setWordWrap(True); note.setObjectName('caption')
            actions = QHBoxLayout(); selector = QComboBox(); selector.setMinimumContentsLength(12)
            selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            selector.addItem('打开作品后显示下一步', None)
            button = QPushButton('执行下一步'); button.setEnabled(False)
            button.clicked.connect(lambda: self.run_product_action(selector.currentData()))
            selector.currentIndexChanged.connect(lambda _index: self.update_product_action_buttons())
            actions.addWidget(selector, 1); actions.addWidget(button)
            layout.addWidget(title); layout.addWidget(findings); layout.addWidget(note); layout.addLayout(actions)
            layout.addWidget(self.build_editing_improve_button())
            layout.addWidget(self.build_shaping_panel())
            self.product_result_panels.append({'title':title,'findings':findings,'note':note,'selector':selector,'button':button})
            return panel

        def build_shaping_panel(self):
            panel = QGroupBox('形体调整（可选）'); layout = QVBoxLayout(panel)
            parts = {}
            for key, title in (('face', '脸部轮廓'), ('waist', '腰身'), ('arms', '手臂'), ('legs', '腿部轮廓')):
                row = QHBoxLayout(); enabled = QCheckBox(title); enabled.setAccessibleName('启用'+title)
                amount = QDoubleSpinBox(); amount.setRange(.05, 1.); amount.setSingleStep(.05)
                amount.setDecimals(2); amount.setValue(.3); amount.setAccessibleName(title+'请求强度')
                amount.setToolTip('请求幅度，不表示精确的人体比例。未勾选的部位不授权改变。')
                row.addWidget(enabled); row.addWidget(amount); layout.addLayout(row)
                parts[key] = (enabled, amount)
                enabled.toggled.connect(lambda checked, key=key, amount=amount:
                    self.change_shaping_part(key, amount.value() if checked else 0.))
                amount.valueChanged.connect(lambda value, key=key, enabled=enabled:
                    self.change_shaping_part(key, value) if enabled.isChecked() else None)
            note = QLabel('默认关闭。检查不调用模型；预览最多一次生图、三次视觉调用，可能产生费用。'); note.setWordWrap(True)
            check = QPushButton('检查形体预览'); check.clicked.connect(lambda: self.plan_shaping('apply'))
            start = QPushButton('开始已检查的形体任务'); start.clicked.connect(self.start_planned_creative_task)
            before = QCheckBox('查看形体调整前版本'); before.toggled.connect(self.compare_shaping)
            adopt = QPushButton('采用此候选版本'); adopt.clicked.connect(self.adopt_shaping_candidate)
            cancel = QPushButton('取消预览，保留当前版本'); cancel.clicked.connect(self.cancel_shaping_preview)
            restore = QPushButton('关闭并恢复调整前版本'); restore.clicked.connect(lambda: self.plan_shaping('restore'))
            for widget in (note, check, start, before, adopt, cancel, restore): layout.addWidget(widget)
            self.shaping_panels.append({'panel':panel, 'parts':parts, 'note':note, 'check':check,
                'start':start, 'before':before, 'adopt':adopt, 'cancel':cancel, 'restore':restore})
            return panel

        def shaping_parent_state(self, *, ignore_busy=False):
            run = self.product_run; options = run.get('options') or {}
            if not isinstance(options, dict): return {'enabled':False, 'reason':'当前保存版本格式无效。'}
            current = options.get('photo_appearance'); state = options.get('shaping_state') or {}
            if not isinstance(state, dict): return {'enabled':False, 'reason':'当前形体状态格式无效。'}
            base = state.get('base_appearance') or current
            reason = ''
            if not current or not base or run.get('run_id') != self.source_run_id:
                reason = '请先打开当前格式的已保存成片。'
            elif self.source_edit.text().strip() != str((run.get('source') or {}).get('path') or ''):
                reason = '当前原片选择已变化，请重新打开保存版本。'
            elif (run.get('result_publication') or {}).get('state') != 'committed':
                reason = '当前版本尚未完整保存。'
            elif self._session_dirty or self._session_checkpoint_failed:
                reason = '请先保存当前修改。'
            elif (run.get('photo_edit_session') or {}).get('status') == 'needs_attention' or (run.get('editing_iteration_result') or {}).get('failure'):
                reason = '已有请求结果需要核查；不会重复发送。'
            elif not ignore_busy and (self._production_mutation_busy() or self._session_loading or self.creative_task_plan_request is not None):
                reason = '当前任务仍在处理或检查，请等待完成。'
            elif not ignore_busy and self.shaping_adoption is not None:
                reason = '正在后台核验采用条件；可取消，当前版本保留。'
            if reason: return {'enabled':False, 'reason':reason}
            from shaping_inspection import observe_run
            fact = self.shaping_parent_fact
            if fact and self._shaping_fact_current(fact, run):
                return copy.deepcopy(fact['state']) if fact.get('ok') else {
                    'enabled':False, 'reason':'当前冻结基线不可用：'+fact.get('reason', '检查失败')}
            observed = observe_run(self.root_path, run)
            pending = self.shaping_inspections.get('parent')
            if not self.shaping_inspection_closed and (not pending or pending['observed'] != observed):
                self.shaping_parent_fact = None
                self._start_shaping_inspection('parent', run, observed=observed)
            return {'enabled':False, 'pending':True, 'reason':'正在后台核验冻结基线；可调整部位，当前版本保留。'}

        def _shaping_fact_current(self, fact, run):
            from shaping_inspection import observe_run
            return bool(fact and fact.get('snapshot') == observe_run(self.root_path, run,
                watch_paths=fact.get('watch_paths') or ()))

        def _start_shaping_inspection(self, kind, run, **context):
            previous = self.shaping_inspections.pop(kind, None)
            if previous: previous['task'].cancel.set()
            label = 'shaping-inspection:'+uuid.uuid4().hex
            task = ShapingInspectionTask(self.root_path, kind, copy.deepcopy(run), label,
                parent_run=copy.deepcopy(context.get('parent_run')),
                parent_identity=copy.deepcopy(context.get('parent')), request=copy.deepcopy(context.get('request')))
            record = {**context, 'label':label, 'task':task, 'run':copy.deepcopy(run),
                'revision':self.shaping_revision}
            self.shaping_inspections[kind] = record
            relay = TaskRelay(self, label, self.job_observation)
            self.pending_tasks[relay] = task
            task.signals.result.connect(relay.result); task.signals.error.connect(relay.error); task.signals.finished.connect(relay.finished)
            self.pool.start(task)
            return record

        def _discard_shaping_inspection(self, kind):
            pending = self.shaping_inspections.pop(kind, None)
            if pending: pending['task'].cancel.set()

        def _shaping_candidate_current(self, candidate, state):
            return bool(candidate and candidate.get('parent') == state.get('parent')
                and candidate.get('revision') == self.shaping_revision
                and self._shaping_fact_current(candidate.get('inspection'), candidate['run']))

        def _shaping_adoption_current(self, context, fact):
            candidate = self.shaping_candidate
            return bool(not self.shaping_inspection_closed and self.shaping_adoption is context
                and not context['task'].cancel.is_set() and not self._session_dirty
                and not self._session_checkpoint_failed and not self._session_loading
                and not self._production_mutation_busy() and self.creative_task_plan_request is None
                and self.shaping_execution is None and context['revision'] == self.shaping_revision
                and candidate and candidate.get('parent') == context['parent']
                and candidate['run'].get('run_id') == context['run'].get('run_id')
                and self.product_run.get('run_id') == self.source_run_id == context['parent']['run_id']
                and self.source_edit.text().strip() == str((self.product_run.get('source') or {}).get('path') or '')
                and self._shaping_fact_current(fact.get('parent_fact'), self.product_run)
                and self._shaping_fact_current(fact, candidate['run']))

        def handle_shaping_inspection(self, label, value):
            if not str(label).startswith('shaping-inspection:'): return False
            found = next(((kind, context) for kind, context in self.shaping_inspections.items()
                if context['label'] == label), None)
            if not found or self.shaping_inspection_closed: return True
            kind, context = found; self.shaping_inspections.pop(kind, None)
            if kind == 'parent':
                if self._shaping_fact_current(value, self.product_run): self.shaping_parent_fact = copy.deepcopy(value)
                self.update_shaping_actions(); return True
            parent_fact = value.get('parent_fact')
            if parent_fact and self._shaping_fact_current(parent_fact, self.product_run):
                self.shaping_parent_fact = copy.deepcopy(parent_fact)
            if kind == 'adopt':
                if value.get('ok') and self._shaping_adoption_current(context, value):
                    try:
                        from workbench_store import select_saved_result
                        def commit_selection():
                            select_saved_result(self.root_path, context['run']['run_id'], explicit=True, lock_timeout=0,
                                expected_manifest_sha256=value['manifest_sha256'], return_state=False,
                                guard=lambda:self._shaping_adoption_current(context, value))
                        loaded = self.load_source_run(context['run'], context.get('stage_report'),
                            selection_explicit=True, record_selection=False, before_load=commit_selection)
                    except (OSError, ValueError, TimeoutError) as error:
                        self.operation_state.setText('采用未完成，当前版本与候选保留；可重试：'+str(error))
                    else:
                        self.operation_state.setText('已采用保存候选；人工接受仍需另行确认。' if loaded else
                            '采用未完成；当前版本与候选保留，请先保存当前修改。')
                else:
                    self.operation_state.setText('候选未采用，当前版本保留：'+(value.get('reason') or '版本或选择已变化'))
                if self.shaping_adoption is context: self.shaping_adoption = None
                self.update_shaping_actions(); return True
            state = self.shaping_parent_state()
            if (context['revision'] != self.shaping_revision or context['parent'] != state.get('parent')
                    or not state.get('enabled') or not self._shaping_fact_current(value, context['run'])):
                self.operation_state.setText('候选检查已过期；当前版本保留，未切换。')
            elif not value.get('ok'):
                self.operation_state.setText('形体候选未通过冻结资产检查：'+value.get('reason', '检查失败'))
            else:
                self.shaping_candidate = {'run':context['run'], 'stage_report':context.get('stage_report'),
                    'parent':context['parent'], 'revision':context['revision'], 'inspection':copy.deepcopy(value)}
                self.compare_shaping(False)
                self.operation_state.setText('候选已保存并预览；点击采用才更改当前保存版本。')
            self.update_shaping_actions(); return True

        def change_shaping_part(self, key, amount):
            if self.shaping_candidate or 'candidate' in self.shaping_inspections or self.shaping_adoption:
                self.cancel_shaping_preview()
            if amount: self.shaping_parts[key] = round(float(amount), 2)
            else: self.shaping_parts.pop(key, None)
            self.shaping_revision += 1
            self.invalidate_creative_task_plan()
            self.update_shaping_actions()

        def update_shaping_actions(self):
            if not self.shaping_panels or not hasattr(self, 'source_edit'): return
            state = self.shaping_parent_state(); execution = self.shaping_execution
            candidate = self.shaping_candidate
            current_candidate = self._shaping_candidate_current(candidate, state)
            planned = self.creative_task_plan or {}
            checked = bool(planned.get('shaping_parent') and planned.get('plan', {}).get('ready') is True
                and planned['shaping_parent'] == state.get('parent'))
            editable = state['enabled'] and execution is None and self.shaping_adoption is None
            configurable = (state['enabled'] or state.get('pending')) and execution is None and self.shaping_adoption is None
            for controls in self.shaping_panels:
                controls['panel'].setVisible(bool((self.product_run.get('options') or {}).get('photo_appearance')))
                for key, (enabled, amount) in controls['parts'].items():
                    enabled.blockSignals(True); enabled.setChecked(key in self.shaping_parts); enabled.blockSignals(False)
                    amount.blockSignals(True)
                    if key in self.shaping_parts: amount.setValue(self.shaping_parts[key])
                    amount.blockSignals(False)
                    enabled.setEnabled(bool(configurable)); amount.setEnabled(bool(configurable and key in self.shaping_parts))
                controls['check'].setEnabled(editable and bool(self.shaping_parts))
                controls['start'].setEnabled(editable and checked)
                controls['restore'].setEnabled(editable and state.get('restore_available', False))
                controls['before'].setEnabled(current_candidate or bool(state.get('restore_available')))
                controls['adopt'].setEnabled(editable and current_candidate)
                controls['cancel'].setEnabled(bool(candidate or execution or 'candidate' in self.shaping_inspections or self.shaping_adoption))
                controls['note'].setText(state['reason'] or
                    ('候选已完整保存；比较不会更改当前版本，点击采用后才切换。自动审片不等于人工接受。' if current_candidate else
                     '默认关闭。改力度从同一母版重算。脸部使用零生图轮廓形变，身体至多一次生图；均至多三次视觉调用。恢复零模型。'))

        def plan_shaping(self, action):
            state = self.shaping_parent_state()
            if not state['enabled'] or self.shaping_execution or self.shaping_adoption:
                self.operation_state.setText(state.get('reason') or '已有形体请求仍待核查。'); return
            if action == 'apply' and not self.shaping_parts: return
            if action == 'restore' and not state.get('restore_available'): return
            run = self.product_run; options = run.get('options') or {}; settings = self._template_payload()
            keys = ('provider', 'model', 'vision_model', 'api_base', 'api_url', 'api_key_env', 'model_registry',
                    'algorithm_profile', 'material_finish_profile', 'character_name', 'character_notes', 'preset')
            request = {key:copy.deepcopy(options.get(key, settings.get(key))) for key in keys
                if options.get(key, settings.get(key)) is not None}
            intent = copy.deepcopy(run.get('task_intent') or options.get('task_intent') or {})
            intent.update(mode='existing_reference', services={'text_vision':action=='apply', 'visual_review':action=='apply'})
            intent.pop('regional_refinement', None)
            request.update(source=run['source']['path'], reference=state['preview'], parent_run_id=run['run_id'],
                sample=run['sample'], variant_count=1, preserve_project_selection=True, creative_task=intent,
                editing_request={key:run['task_plan']['editing_scope'][key]
                    for key in ('work_type', 'style', 'target_setting', 'scene_target', 'language')},
                shaping_request={'action':'apply', 'parts':copy.deepcopy(self.shaping_parts)} if action=='apply' else {'action':'restore'})
            self.cancel_shaping_preview()
            self.invalidate_creative_task_plan()
            self.submit_creative_task_plan(request, mode='existing_reference', shaping_parent=state['parent'])

        def compare_shaping(self, before):
            state = self.shaping_parent_state(ignore_busy=True); candidate = self.shaping_candidate
            if not state.get('enabled'): return
            if candidate and not self._shaping_candidate_current(candidate, state): return
            path = state['preview'] if before else (
                candidate['run']['artifacts']['auto_draft'] if candidate else self.product_run['artifacts']['auto_draft'])
            self.previews['final'].replace_preview_path(Path(path))
            for controls in self.shaping_panels:
                controls['before'].blockSignals(True); controls['before'].setChecked(before); controls['before'].blockSignals(False)

        def cancel_shaping_preview(self):
            execution = self.shaping_execution
            if execution:
                execution['cancelled'] = True
                if execution.get('job_id'):
                    self.task('POST', '/api/jobs/'+execution['job_id']+'/cancel', {}, 'shaping-cancel:'+execution['token'])
            self._discard_shaping_inspection('candidate'); self._discard_shaping_inspection('adopt')
            self.shaping_adoption = None; self.shaping_candidate = None
            path = (self.product_run.get('artifacts') or {}).get('auto_draft')
            if path and Path(path).is_file(): self.previews['final'].replace_preview_path(Path(path))
            for controls in self.shaping_panels:
                controls['before'].blockSignals(True); controls['before'].setChecked(False); controls['before'].blockSignals(False)
            self.update_shaping_actions()

        def adopt_shaping_candidate(self):
            candidate = self.shaping_candidate; state = self.shaping_parent_state()
            if (not self._shaping_candidate_current(candidate, state) or not state.get('enabled')
                    or self.shaping_execution or self.shaping_adoption):
                self.operation_state.setText('当前版本已变化，不能采用此候选。'); return
            self.shaping_adoption = self._start_shaping_inspection('adopt', candidate['run'],
                parent_run=self.product_run, parent=state['parent'], stage_report=candidate.get('stage_report'),
                request=(candidate['run'].get('options') or {}).get('shaping_state', {}).get('request'))
            self.operation_state.setText('正在后台重新核验候选与父版本字节；当前版本保留。')
            self.update_shaping_actions()

        def validate_shaping_candidate(self, run):
            """UI browsing fact only. Adoption always starts another worker."""
            candidate = self.shaping_candidate
            return bool(candidate and self._shaping_fact_current(candidate.get('inspection'), run))

        def handle_shaping_result(self, label, value):
            execution = self.shaping_execution
            if str(label).startswith('shaping-cancel:'): return True
            if not execution: return str(label).startswith(('shaping-start:', 'shaping-candidate:'))
            token = execution['token']
            if label == 'shaping-start:'+token:
                self.creative_task_start_pending = False; self.creative_task_execution_label = None
                if value.get('async') and value.get('job_id'):
                    execution['job_id'] = str(value['job_id']); self.render_pending = True
                    self.observe_job(execution['job_id']); self.refresh_jobs()
                    if execution.get('cancelled'):
                        self.task('POST', '/api/jobs/'+execution['job_id']+'/cancel', {}, 'shaping-cancel:'+token)
                elif isinstance(value.get('run'), dict):
                    return self.handle_shaping_result('shaping-candidate:'+token, value)
                else:
                    execution['unknown'] = True
                    self.operation_state.setText('形体开始请求结果待核查；当前版本保留，不会自动重发。')
                self.update_creative_task_actions(); return True
            if label in {'job-status', 'job-stream'} and execution.get('job_id'):
                job = value.get('job' if label=='job-status' else 'data') or {}
                identifier = str(job.get('job_id') or self.active_job_id)
                if identifier != execution['job_id']: return False
                self.job_rows[identifier] = {**self.job_rows.get(identifier, {}), **job}
                status = job.get('status'); self.operation_state.setText('形体任务：'+str(status or '处理中')+'；当前保存版本保留。')
                if status in {'succeeded', 'failed', 'blocked', 'cancelled'}:
                    self.job_stream_active = False; self.job_poll_timer.stop()
                    self.job_poll_terminal_observation = self.job_observation
                    self.render_pending = False; self.active_checked_task = None
                    run_id = job.get('run_id') or (job.get('data') or {}).get('run_id')
                    if status=='succeeded' and run_id and not execution.get('cancelled'):
                        execution['run_id'] = run_id
                        self.task('GET', f'/api/source-runs/{run_id}', None, 'shaping-candidate:'+token)
                    else:
                        self.shaping_execution = None
                        self.operation_state.setText('形体任务已停止；当前保存版本保留。')
                    self.update_creative_task_actions()
                elif label=='job-status': self._schedule_job_poll(750)
                return True
            if label == 'shaping-candidate:'+token:
                self.render_pending = False; self.active_checked_task = None
                state = self.shaping_parent_state(ignore_busy=True); run = value.get('run') or {}
                self.shaping_execution = None
                if (not isinstance(run, dict) or execution.get('cancelled') or state.get('parent') != execution['parent']
                        or execution['revision'] != self.shaping_revision
                        or (execution.get('run_id') and run.get('run_id') != execution['run_id'])
                        or run.get('parent_run_id') != execution['parent']['run_id']
                        or (run.get('result_publication') or {}).get('state') != 'committed'):
                    self.operation_state.setText('候选已保留在任务历史；当前版本或选择已变化，未切换。')
                    self.update_shaping_actions(); return True
                saved = (run.get('options') or {}).get('shaping_state') or {}
                if not isinstance(saved, dict):
                    self.operation_state.setText('候选形体记录格式无效；当前版本保留，未采用。')
                    self.update_shaping_actions(); return True
                base = saved.get('base_appearance') or {}
                typed = (run.get('options') or {}).get('photo_appearance') or {}
                if (base.get('asset_id') != execution['parent']['base_id']
                        or base.get('receipt_sha256') != execution['parent']['base_receipt']
                        or saved.get('selected_asset_id') != typed.get('asset_id')
                        or saved.get('request') != execution['request']):
                    self.operation_state.setText('候选的调整前基线或母版绑定不符；未采用。')
                    self.update_shaping_actions(); return True
                self._start_shaping_inspection('candidate', run, parent_run=self.product_run,
                    parent=execution['parent'], request=execution['request'], stage_report=value.get('stage_report'))
                self.operation_state.setText('候选已保存，正在后台核验；当前版本保留。')
                self.update_shaping_actions()
                return True
            return str(label).startswith(('shaping-start:', 'shaping-candidate:'))

        def show_product_result(self, report, run):
            self.product_stage_report = copy.deepcopy(report); self.product_run = copy.deepcopy(run)
            self.refresh_product_result_actions()
            self.refresh_cos_workspace()

        def refresh_product_result_actions(self):
            work = self.product_stage_report.get('whole_work') or {}
            diagnostic = self.product_stage_report.get('cosplay_post_diagnosis') or {}
            rows = work.get('open_findings') or []
            states = {'fail':'需改善','unknown':'证据不足','unavailable':'证据不可用'}
            finding_text = '；'.join(str(row.get('label') or row['criterion'])+'：'+states.get(row['status'],'待核对') for row in rows[:3])
            if len(rows)>3: finding_text += f'；另有 {len(rows)-3} 项见流程明细。'
            if not finding_text: finding_text = '请对照原片与最终版本复核后批准。' if work.get('result_state')!='accepted' else '当前保存版本已记录人工批准。'
            note = '处理执行结束后，作品仍按缺项、证据和人工批准判断。'
            if diagnostic.get('binding_state')=='inherited_same_pixels':
                note = '当前画面沿用旧版本审片；这里只展示已知问题，修复会先重新检查当前成片。'
            elif diagnostic.get('binding_state')=='stale':
                note = '旧审片与当前保存版本不匹配；请重新核对当前证据。'
            actions = list(work.get('recovery_actions') or [])
            if self._session_dirty:
                actions.insert(0, {'action_id':'save_native_revision','label':'保存高清新版本','supported':True,
                    'reason':'当前新增修改尚未保存。','generation_required':False,'QA_required':False})
                note = '当前编辑尚未保存；以下证据和诊断属于已保存版本。'
            for panel in self.product_result_panels:
                panel['title'].setText('当前修改尚未保存' if self._session_dirty else str(work.get('result_label') or '尚未打开保存作品'))
                panel['findings'].setText(finding_text); panel['note'].setText(note); panel['base_note']=note
                panel['note'].setToolTip('审片生产版本：'+str(diagnostic.get('producer_run_id') or '未记录'))
                selector=panel['selector']; previous=(selector.currentData() or {}).get('action_id')
                selector.blockSignals(True); selector.clear()
                for action in actions: selector.addItem(action['label'],action)
                if not actions: selector.addItem('暂无可执行下一步',None)
                index=next((i for i,action in enumerate(actions) if action['action_id']==previous),0)
                selector.setCurrentIndex(index); selector.blockSignals(False)
            self.update_product_action_buttons()
            self.update_editing_improvement_actions()

        def _active_production_job(self):
            kinds = {'one_click', 'native_design_reconstruction', 'prepare_portrait_beauty', 'select_reference_variant'}
            return next((job for job in self.job_rows.values()
                         if job.get('kind') in kinds and job.get('status') in {'queued', 'running', 'cancelling'}), None)

        def _production_mutation_busy(self):
            return bool(self.render_pending or self.creative_task_start_pending or self._active_production_job())

        def update_product_action_buttons(self):
            for panel in self.product_result_panels:
                action=panel['selector'].currentData() or {}; action_id=action.get('action_id')
                read_only=action_id in {'inspect_native_evidence','inspect_scene_producer','review_and_accept'}
                ready=bool(action.get('supported') and self.source_run_id and (read_only or not self._production_mutation_busy()))
                if self.source_generated_appearance and not read_only: ready = False
                if action_id=='save_native_revision': ready=ready and bool(getattr(self,'source_light_save',None) and self.source_light_save.isEnabled())
                if self._session_dirty and action_id in {'retry_local_processing','prepare_reconstruction','repair_final_locally','review_and_accept'}: ready=False
                panel['button'].setEnabled(ready); panel['button'].setToolTip(
                    '此生成版本不支持旧原片图层处理；可从原片创建新的设计任务。'
                    if self.source_generated_appearance and not read_only else str(action.get('reason') or ''))
                panel['note'].setText(str(panel.get('base_note') or '')+
                    ('\n此操作会新增文本审片请求，不新增生图。' if action.get('QA_required') else ''))

        def run_product_action(self, action):
            if not isinstance(action,dict) or not action.get('supported') or not self.source_run_id: return
            if self.source_generated_appearance and action.get('action_id') not in {'inspect_native_evidence', 'inspect_scene_producer', 'review_and_accept'}:
                self.operation_state.setText('此生成版本不支持旧原片图层处理；可从原片创建新的设计任务。')
                return
            action_id=action.get('action_id')
            if action_id in {'inspect_native_evidence','inspect_scene_producer'}:
                self.open_product_evidence(native=action_id=='inspect_native_evidence'); return
            if self._production_mutation_busy():
                self.operation_state.setText('当前任务仍在处理；先等待或取消，再重试或准备。'); return
            if self._session_dirty and action_id!='save_native_revision':
                self.operation_state.setText('请先保存当前修改，再继续本地恢复或批准。'); return
            if action_id=='retry_local_processing':
                recovery={'generate':False,'async':True}
                if action.get('regional_candidate_reuse') and self.regional_core_box:
                    recovery['regional_reuse_core_box']=list(self.regional_core_box)
                self.task('POST',f'/api/source-runs/{self.source_run_id}/retry',
                    recovery,'local-retry')
            elif action_id=='prepare_reconstruction': self.reconstruct_design()
            elif action_id=='save_native_revision': self.save_source_lights()
            elif action_id=='repair_final_locally':
                self.task('POST',f'/api/source-runs/{self.source_run_id}/repair-final',{},'repair-final')
                self.operation_state.setText('正在重新审片并本地修复；会新增文本审片请求，不新增生图。')
            elif action_id=='review_and_accept':
                self.tabs.setCurrentIndex(2); self.local_editor_tabs.setCurrentIndex(self.local_editor_tabs.count()-1)
                self.visual_checks['identity'].setFocus(); self.operation_state.setText('查看全图与原尺寸证据后逐项复核；批准会单独记录。')

        def product_evidence_paths(self, *, native=False):
            run=self.product_run; review=run.get('cosplay_post_review') or {}; result=[]; seen=set()
            def add(label,value):
                value=value.get('path') if isinstance(value,dict) else value
                if not isinstance(value,str) or not value: return
                path=Path(value); path=path if path.is_absolute() else self.root_path/path
                if path.is_file() and str(path.resolve()) not in seen:
                    seen.add(str(path.resolve()));result.append((label,path))
            review_dir=self.root_path/'outputs/runs'/str(run.get('run_id') or '')/'cosplay-post-review'
            if native:
                for path in sorted(review_dir.glob('native-*.png')):
                    if path.stem[len('native-'):].isdigit(): add('原尺寸对照 '+path.stem,str(path))
                for row in review.get('native_crops') or []:
                    add('原尺寸 '+str(row.get('pair_id') or '')+' · '+str(row.get('role') or ''),row.get('asset'))
                if not result: return []
            add('全图 原片 / 参考 / 成片对照',str(review_dir/'comparison.png'))
            add('审片原始全图对照',review.get('comparison'))
            if not native:
                add('当前已保存成片',(run.get('artifacts') or {}).get('auto_draft'))
                add('当前设计参考',(run.get('reference') or {}).get('path'))
            return result

        def open_product_evidence(self, *, native=False):
            rows=self.product_evidence_paths(native=native)
            if not rows:
                self.tabs.setCurrentIndex(2); self.layer_tabs.setCurrentWidget(self.previews['final'])
                if native:
                    regions=((self.product_run.get('cosplay_post_review') or {}).get('native_selection') or {}).get('regions') or []
                    face=next((row for row in regions if row.get('pair_id')=='face_upper'),{})
                    box=face.get('native_box');size=(self.product_run.get('source') or {}).get('oriented_size')
                    center=[(box[0]+box[2])/(2*size[0]),(box[1]+box[3])/(2*size[1])] if box and size else None
                    self.request_native_detail(center)
                self.operation_state.setText('没有已存审片板；请核对当前已保存画面与原尺寸细节。'); return
            dialog=QDialog(self);dialog.setWindowTitle('眼口与原尺寸证据' if native else '主效与空间对照');dialog.resize(1000,860)
            layout=QVBoxLayout(dialog);diagnostic=self.product_stage_report.get('cosplay_post_diagnosis') or {}
            scope=QLabel('已保存证据；不包含未保存修改。'+('继承审片来自 '+str(diagnostic.get('producer_run_id'))+'，不能直接授予修复资格。' if diagnostic.get('binding_state')=='inherited_same_pixels' else '证据只能证明实际显示的区域。'))
            scope.setWordWrap(True);layout.addWidget(scope)
            selector=QComboBox()
            for label,path in rows: selector.addItem(label,str(path))
            layout.addWidget(selector);image=QLabel();image.setAlignment(Qt.AlignCenter)
            scroll=QScrollArea();scroll.setWidget(image);scroll.setAlignment(Qt.AlignCenter);layout.addWidget(scroll,1)
            def show(index):
                image.setPixmap(QPixmap(str(rows[index][1])));image.adjustSize()
            selector.currentIndexChanged.connect(show);show(0)
            self.product_evidence_dialog=dialog;dialog.show()

        def request_source_preview(self, source):
            path = Path(source).expanduser()
            if not path.is_absolute(): path = self.root_path / path
            path = path.resolve(); self.project_images["source"] = path
            self.project_proxies.pop("source", None); self.project_dimensions.pop("source", None)
            if self.project_view_selector.currentData() == "source":
                self.project_preview.set_path(None); self.project_preview.image.setText("正在载入原片预览…"); self.project_preview.image.adjustSize()
                self.canvas_title.setText(path.name); self.canvas_title.setToolTip(str(path)); self.project_image_info.setText("正在载入原片预览…")
            if not path.is_file(): return
            task = SourcePreviewTask(self.root_path, path); relay = TaskRelay(self, "source-preview", self.job_observation); self.pending_tasks[relay] = task
            task.signals.result.connect(relay.result); task.signals.error.connect(relay.error); task.signals.finished.connect(relay.finished); self.pool.start(task)
        def update_project_canvas(self, _index=None):
            name = self.project_view_selector.currentData(); path = self.project_images.get(name); proxy = self.project_proxies.get(name)
            self.project_preview.set_path(path, proxy)
            if path:
                self.canvas_title.setText(path.name); self.canvas_title.setToolTip(str(path)); QTimer.singleShot(0, self.project_preview.fit_view)
                size = self.project_dimensions.get(name)
                if not size:
                    reader = QImageReader(str(path)); reader.setAutoTransform(True); image_size = reader.size()
                    if image_size.isValid(): size = [image_size.width(), image_size.height()]; self.project_dimensions[name] = size
                dimensions_text = f" · {size[0]} × {size[1]}" if size else ""
                label = {"source": "原片", "reference": "GPT 候选", "final": "高清草稿"}.get(name, "")
                self.project_image_info.setText(path.name + dimensions_text + " · " + label + (" · 代理预览" if proxy else ""))
            else:
                self.project_preview.image.setText("导入原图开始创作" if name == "source" else "生成后可以在这里查看" )
                self.project_preview.image.adjustSize()
                self.project_image_info.setText("导入原图，选择成片效果" if name == "source" else "生成后显示图像信息")
        def _appearance_scope_options(self):
            """Serialize the regional controls without inventing image logic."""
            extra = self.appearance_extra_controls[0] if self.appearance_extra_controls else {}
            if not extra:
                return {"scope": "global"}
            rect = []
            try:
                rect = [max(0.0, min(1.0, float(v.strip()))) for v in extra["local_rect"].text().split(",")]
            except (TypeError, ValueError):
                rect = [0.0, 0.0, 1.0, 1.0]
            if len(rect) != 4:
                rect = [0.0, 0.0, 1.0, 1.0]
            return {"scope": extra["region"].currentData(), "local_rect": rect,
                    "local_scope": extra["local_scope"].currentData(),
                    "local_feather": extra["local_feather"].value() / 100.0}

        def select_preset(self, _index):
            row = self.preset_rows.get(self.preset.currentData(), {})
            self.refinement_profile_choice = ({'id': 'source_refinement', 'version': 1}
                if self.preset.currentData() == 'source_refinement' else None)
            self.apply_selected_style_prompt()
            if row.get("appearance"):
                self.appearance.setPlainText(json.dumps({"params": row.get("appearance", {})}, ensure_ascii=False, indent=2))
                for key, slider in self.appearance_sliders.items():
                    slider.blockSignals(True); slider.setValue(round(float(row.get("appearance", {}).get(key, 0)) * 100)); slider.blockSignals(False)
                    self.appearance_value_labels[key].setText(str(slider.value()))
            if row.get("manual_layers"):
                self.manual_layers.setPlainText(json.dumps({"schema_version": "manual_layers.v1", "layers": row.get("manual_layers", {})}, ensure_ascii=False, indent=2)); self.sync_manual_controls()
        def preset_activated(self, _index):
            # activated also fires when the user chooses the already selected
            # empty item; currentIndexChanged cannot express that opt-out.
            self.refinement_profile_choice = ({'id': 'source_refinement', 'version': 1}
                if self.preset.currentData() == 'source_refinement' else None)
            self.invalidate_creative_task_plan()
        def restore_generation_prompt(self):
            from generation_prompt import DEFAULT_PROMPT
            self.prompt.setPlainText(DEFAULT_PROMPT)
        def apply_selected_style_prompt(self):
            from generation_prompt import compose_generation_prompt
            row = self.preset_rows.get(self.preset.currentData(), {})
            self.prompt.setPlainText(compose_generation_prompt(self.prompt.toPlainText(), row.get("prompt") or ""))
            self.update_generation_style_description()
        def update_generation_style_description(self):
            row = self.preset_rows.get(self.preset.currentData(), {})
            self.generation_style.setText(f"{row.get('name', '按角色自动推荐')}：{row.get('description', '根据角色、服装、道具和场景选择适合的灯光与氛围。')} 提示词可自由编辑。")
        def recognize_character(self):
            if not self.allow_creative_text_vision(): return
            source = self.source_edit.text().strip() or self.source_run_source
            if not source: QMessageBox.information(self, "缺少原图", "请选择一张 JPG/RAW 原图。"); return
            checked=self.creative_task_plan or {}
            plan=checked.get('plan') or {}
            if checked.get('revision')!=self.creative_task_revision or plan.get('ready') is not True:
                self.operation_state.setText('角色识别需要当前已检查的任务与外发许可；请先检查制作计划。')
                return
            request=copy.deepcopy(checked['request'])
            request.update(task_intent=copy.deepcopy(plan['intent']),task_plan=copy.deepcopy(plan))
            self.character_result.setText("正在识别角色并分析适合的风格…")
            self.character_details_toggle.setChecked(True)
            self.task('POST','/api/character/recognize',request,'character-recognized')
        def show_character_profile(self, profile):
            if not profile:
                self.character_result.setText("角色：可选填写或显式识别；未确定时按可见服装、道具与场景设计。")
                self.recommended_styles.clear(); self.recommended_styles.addItem("识别后显示角色推荐风格", None); return
            name = profile.get("character_name") or "未知角色"
            series = str(profile.get("series") or "")
            identification = str(profile.get("identification") or "")
            identification_text = {"user_supplied": "用户提供角色名", "user_provided": "用户提供角色名", "recognized": "根据原图识别", "confirmed": "根据原图识别", "inferred": "根据造型推测", "tentative": "根据造型推测", "unknown": "未确定具体角色", "visual_only": "依据可见造型"}.get(identification, "依据原图分析" if identification else "")
            cues = profile.get("visual_cues") or []
            cue_text = "、".join(str(cue) for cue in cues) if isinstance(cues, list) else str(cues)
            confidence = profile.get("confidence")
            confidence_text = {"high": "高", "medium": "中等", "low": "低", "unknown": "未确定", "certain": "明确"}.get(str(confidence).lower(), "未确定" if confidence is not None else "")
            if isinstance(confidence, (int, float)): confidence_text = f"{confidence * 100:.0f}%" if 0 <= confidence <= 1 else str(confidence)
            self.character_result.setText(f"角色：{name}" + (f" / {series}" if series else "") +
                (f" / 识别依据：{identification_text}" if identification_text else "") + (f" / 识别可信度：{confidence_text}" if confidence is not None else "") +
                (f"\n可见线索：{cue_text}" if cue_text else "") + ("\n未确定具体角色，将依据可见造型设计，不需额外确认。" if not profile.get("character_name") or identification == "unknown" else ""))
            self.recommended_styles.clear()
            for row in profile.get("recommended_styles") or []:
                preset = self.preset_rows.get(row.get("id"), {})
                self.recommended_styles.addItem(f"{preset.get('name', row.get('id', '推荐风格'))}：{row.get('reason', '')}", row)
            if not self.recommended_styles.count(): self.recommended_styles.addItem("暂无明确风格推荐，可使用自动设计或手动风格", None)
        def apply_recommended_style(self):
            from generation_prompt import compose_generation_prompt
            row = self.recommended_styles.currentData()
            if not isinstance(row, dict): return
            index = self.preset.findData(row.get("id"))
            if index >= 0: self.preset.setCurrentIndex(index)
            self.prompt.setPlainText(compose_generation_prompt(self.prompt.toPlainText(), row.get("english_direction") or self.preset_rows.get(row.get("id"), {}).get("prompt") or ""))
        def generation_input_prompt(self):
            from generation_prompt import compose_generation_prompt
            row = self.preset_rows.get(self.preset.currentData(), {})
            return self.prompt.toPlainText().strip() or compose_generation_prompt(style_prompt=row.get("prompt"))
        def clear_generation_design_view(self, *_args):
            self.final_prompt.clear(); self.design_summary.clear()
            self.research_sources.setText('来源：实际制作或已有参考的保存记录')
            self.research_summary.setText('开始制作后显示实际原图观察与设计；这里不单独调用模型。')
            self.design_style.setText('保存设计（只读）')

        def show_generation_design(self, design, profile=None):
            """Display saved observed/imported facts without injecting a new task."""
            if profile: self.show_character_profile(profile)
            self.design_style.setText('保存设计（只读）')
            observed=design.get('observed_design') or {}
            if observed:
                lines=['本图设计：'+str(observed.get('summary') or '')]
                observation=observed.get('observation') or {}
                if observation.get('description'):lines.append('原图观察：'+str(observation['description']))
                for change in observed.get('changes') or []:
                    lines.append('编辑：'+str(change.get('target') or '')+' · '+str(change.get('instruction') or '')+
                        ('\n目的：'+str(change['purpose']) if change.get('purpose') else ''))
                for key,label in (('lighting','统一用光'),('keep','保留事实'),('uncertain','未确定项')):
                    value=observed.get(key)
                    if value:lines.append(label+'：'+('；'.join(str(row) for row in value) if isinstance(value,list) else str(value)))
                self.research_summary.setText('已保存的原图观察与编辑设计；显示不新增视觉、研究或生图请求。')
                self.research_sources.setText('来源：本次原图与已保存观察记录；不冒称联网角色研究。')
            else:
                brief=design.get('design_brief') or {}
                lines=[label+'：'+str(brief[key]) for key,label in (('summary','保存设计'),('scene_analysis','原图分析'),
                    ('reference_application','参考应用'),('lighting_plan','用光'),('tradeoffs','取舍')) if isinstance(brief,dict) and brief.get(key)]
                self.research_summary.setText('已有参考或历史保存设计，仅供只读查看；不会复用为新任务的外部规划身份。')
                research=design.get('character_research') or {};links=[]
                for source in research.get('sources') or research.get('references') or []:
                    url=str(source.get('url') or '');parsed=urlparse(url)
                    if parsed.scheme in {'http','https'} and parsed.netloc:
                        links.append(f'<a href="{html.escape(url,quote=True)}">{html.escape(str(source.get("title") or url))}</a>')
                self.research_sources.setText('保存资料链接：'+(' · '.join(links) if links else '未记录可核实链接'))
            if design.get('status') in {'unavailable','failed'}:lines.append('保存的设计未完成：'+str(design.get('error') or '尚缺证据'))
            self.design_summary.setPlainText('\n\n'.join(lines))
            self.final_prompt.setPlainText(str(design.get('effective_prompt') or ''));self.prompt_tabs.setCurrentIndex(1)

        def generation_request_options(self):
            settings=self._template_payload()
            settings['prompt']=self.generation_input_prompt();settings['input_prompt']=settings['prompt']
            settings['effective_prompt']=settings['prompt'];settings['generate']=True
            return settings

        def candidate_page(self):
            page = QWidget(); lay = QVBoxLayout(page); form = QFormLayout()
            self.provider = QComboBox(); self.provider.addItems(["qwen_image_2_1", "ccswitch_sub2api", "openai-compatible", "qwen_image_2_1_uncensored", "qwen_image_2_1_fast", "qwen_image_2_1_uncensored_fast", "manual_import", "comfyui", "local"])
            self.model = QLineEdit("gpt-image-2.5-sunburst")
            self.provider.currentTextChanged.connect(self.select_provider)
            self.provider_state = QLabel("Provider 配置尚未读取")
            form.addRow("Provider", self.provider); form.addRow("Model / ComfyUI workflow JSON", self.model); form.addRow("配置状态", self.provider_state)
            self.analysis_models = QComboBox()
            self.analysis_models.addItem('当前 DINOv3 + SAM3.1', 'current')
            self.analysis_models.setToolTip('当前唯一模型配置：DINOv3 ViT-L/16、SAM3.1 静态实例分割、Sapiens2 与 ViTMatte。')
            form.addRow('本地分析方案', self.analysis_models)
            provider_content = QWidget(); provider_content.setLayout(form); provider_panel, self.provider_panel_toggle = self.accordion("生成服务与模型", provider_content); lay.addWidget(provider_panel)
            prompt_note = QLabel("英文提示词与生成风格在项目页编辑；下方重新生成使用当前单图的原图和编辑参数。"); prompt_note.setWordWrap(True); lay.addWidget(prompt_note)
            variants_box = QGroupBox("多版候选浏览与选择"); variants_lay = QVBoxLayout(variants_box)
            self.variant_list = QListWidget(); self.variant_list.setViewMode(QListWidget.ViewMode.IconMode); self.variant_list.setResizeMode(QListWidget.ResizeMode.Adjust); self.variant_list.setIconSize(QSize(100, 140)); self.variant_list.setGridSize(QSize(170, 200)); self.variant_list.setWordWrap(True)
            self.variant_list.setMinimumHeight(230); self.variant_list.currentItemChanged.connect(self.preview_variant)
            self.variant_preview = Preview("候选预览"); self.variant_preview.path_label.setWordWrap(True); self.variant_preview.setMinimumHeight(280)
            variants_split = QSplitter(Qt.Horizontal); variants_split.addWidget(self.variant_list); variants_split.addWidget(self.variant_preview); variants_split.setStretchFactor(1, 1); variants_lay.addWidget(variants_split)
            variant_row = QHBoxLayout(); self.variant_state = QLabel("生成后可比较多个候选，选择一版制作新的高清草稿。"); self.variant_state.setWordWrap(True)
            self.variant_use = QPushButton("采用所选设计"); self.variant_use.setEnabled(False); self.variant_use.clicked.connect(self.select_variant)
            self.variant_use.setObjectName("primaryButton")
            variant_row.addWidget(self.variant_state, 1); variant_row.addWidget(self.variant_use); variants_lay.addLayout(variant_row); lay.addWidget(variants_box, 1)
            settings = QWidget(); settings_form = QFormLayout(settings)
            self.provider_endpoint = QLineEdit(); self.provider_endpoint.setPlaceholderText("API base URL，例如 http://host/v1")
            self.provider_secret_env = QLineEdit(); self.provider_secret_env.setPlaceholderText("环境变量名，例如 OPENAI_API_KEY；不要填写密钥")
            self.provider_credential = QComboBox(); self.provider_credential.addItems(["auto", "environment", "ccswitch", "none"])
            settings_form.addRow("Endpoint", self.provider_endpoint); settings_form.addRow("凭据环境变量名", self.provider_secret_env); settings_form.addRow("凭据来源", self.provider_credential)
            setting_buttons = QHBoxLayout(); self.provider_save = QPushButton("保存连接设置"); self.provider_discover = QPushButton("读取 CCSwitch 当前通道")
            self.provider_save.clicked.connect(self.save_provider_settings); self.provider_discover.clicked.connect(lambda: self.task("GET", "/api/provider-settings/discover", None, "provider-discovery"))
            setting_buttons.addWidget(self.provider_save); setting_buttons.addWidget(self.provider_discover); settings_form.addRow(setting_buttons)
            settings_form.addRow(QLabel("CCSwitch 凭据只在后端内存使用，不展示或写入本项目；读取后点击保存即可采用此通道。")); settings_panel, _toggle = self.accordion("高级连接配置", settings); lay.addWidget(settings_panel)
            self.gen = QPushButton("GPT 重新生成当前原图"); lay.addWidget(self.gen)
            self.candidate_info = QPlainTextEdit(); self.candidate_info.setReadOnly(True); self.candidate_info.setMaximumHeight(160)
            info_panel, _toggle = self.accordion("运行详情", self.candidate_info); lay.addWidget(info_panel)
            self.gen.clicked.connect(self.generate); return page
        def populate_variants(self, run):
            self.variant_list.blockSignals(True); self.variant_list.clear()
            selected_id = str(run.get("selected_variant_id") or "")
            selected_item = None
            for row in run.get("variants") or []:
                variant_id = str(row.get("variant_id") or "")
                label = f"版本 {row.get('index', self.variant_list.count() + 1)} · {row.get('style_name') or row.get('style_id') or '角色推荐'}"
                if variant_id == selected_id: label += "\n当前草稿"
                item = QListWidgetItem(label); item.setData(Qt.UserRole, row); item.setToolTip(str(row.get("candidate_path") or ""))
                path = Path(str(row.get("candidate_path") or ""))
                if not path.is_absolute(): path = self.root_path / path
                if path.is_file():
                    reader = QImageReader(str(path)); size = reader.size()
                    if size.isValid(): reader.setScaledSize(size.scaled(self.variant_list.iconSize(), Qt.KeepAspectRatio))
                    thumbnail = reader.read()
                    if not thumbnail.isNull(): item.setIcon(QIcon(QPixmap.fromImage(thumbnail)))
                self.variant_list.addItem(item)
                if variant_id == selected_id: selected_item = item
            self.variant_list.blockSignals(False)
            if self.variant_list.count():
                self.variant_list.setCurrentItem(selected_item or self.variant_list.item(0))
            else:
                self.variant_preview.set_path(None); self.variant_state.setText("此运行暂无多版候选；在项目页选择生成数量后重新生成。"); self.variant_use.setEnabled(False)
        def preview_variant(self, item, _previous=None):
            row = item.data(Qt.UserRole) if item else None
            if not isinstance(row, dict): return
            path = Path(str(row.get("candidate_path") or ""))
            if not path.is_absolute(): path = self.root_path / path
            self.variant_preview.set_path(path)
            self.variant_state.setText(f"候选：{row.get('variant_id', '')} / {row.get('style_name') or row.get('style_id') or '角色推荐'}；选中只预览，采用后进入设计重建。")
            self.variant_use.setEnabled(bool(self.source_run_id and row.get("variant_id")) and not self.render_pending)
        def select_variant(self):
            item = self.variant_list.currentItem()
            row = item.data(Qt.UserRole) if item else None
            if not isinstance(row, dict) or not self.source_run_id or self.render_pending: return
            if self.simple_creation and self.simple_creation.get('stage') == 'awaiting_variant':
                self.simple_creation['stage'] = 'selecting_variant'
            self.task("POST", f"/api/source-runs/{self.source_run_id}/select-variant", {"variant_id": row["variant_id"]}, "source-select-variant")
        def review_page(self):
            page = QWidget(); lay = QVBoxLayout(page); self.review_title = QLabel("请选择样例"); self.review_state = QLabel("review 状态：未知"); lay.addWidget(self.review_title); lay.addWidget(self.review_state)
            self.preparation_version_review = QLabel(''); self.preparation_version_review.setWordWrap(True); lay.addWidget(self.preparation_version_review)
            self.reprepare_current_button = QPushButton('按当前算法重新制作'); self.reprepare_current_button.setEnabled(False)
            self.reprepare_current_button.clicked.connect(self.complete_current_post); lay.addWidget(self.reprepare_current_button)
            lay.addWidget(self.product_result_panel())
            review_action = QPushButton("打开审片证据"); lay.addWidget(review_action)
            review_action.clicked.connect(self.open_product_evidence)
            checks_box = QWidget(); checks_form = QFormLayout(checks_box)
            self.visual_checks = {}
            for key in VISUAL_CHECK_KEYS:
                combo = QComboBox(); combo.addItems(["not_checked", "pass", "rework", "hold"])
                combo.currentTextChanged.connect(lambda _value: self.update_accept_enabled())
                self.visual_checks[key] = combo
                checks_form.addRow(VISUAL_CHECK_LABELS[key], combo)
            checks_panel, _toggle = self.accordion("视觉检查与交付确认", checks_box); lay.addWidget(checks_panel)
            self.layer_tabs = QTabWidget()
            for name, label in (("source", "原片"), ("reference", "GPT 候选"), ("final", "高清草稿"), ("alpha", "遮罩"), ("silhouette", "人物轮廓"), ("risk", "风险导航")):
                self.previews[name] = Preview(label); self.layer_tabs.addTab(self.previews[name], label)
            self.native_detail_token = ""
            self.native_live_detail_context = None
            self.native_live_detail_pending_center = None
            self.native_detail_button = QPushButton("查看已保存版本的原尺寸细节")
            self.native_detail_button.setEnabled(False)
            self.native_detail_button.setToolTip("双击成片任意位置查看已保存版本的原生像素。")
            self.native_detail_button.clicked.connect(lambda: self.request_native_detail())
            self.previews['final'].nativeDetailRequested.connect(self.request_native_detail)
            detail_row = QHBoxLayout();detail_row.addWidget(self.native_detail_button)
            self.native_live_detail_button = QPushButton("查看当前调整的原尺寸细节")
            self.native_live_detail_button.setEnabled(False)
            self.native_live_detail_button.setToolTip("包含当前未保存的边缘、光色与精修调整；首次需原尺寸计算，请稍候。")
            self.native_live_detail_button.clicked.connect(lambda:self.request_live_native_detail())
            detail_row.addWidget(self.native_live_detail_button);lay.addLayout(detail_row)
            edit_split = QSplitter(Qt.Horizontal); edit_split.addWidget(self.layer_tabs)
            edit_split.addWidget(self.source_light_panel()); edit_split.setStretchFactor(0, 1); edit_split.setStretchFactor(1, 0); edit_split.setSizes([900, 340])
            lay.addWidget(edit_split, 3)
            risk_row = QHBoxLayout(); risk_row.addWidget(QLabel("风险区域")); self.risk_crop = QComboBox(); self.risk_crop.addItem("结构导航图", "")
            self.risk_crop.currentIndexChanged.connect(self.select_risk_crop); risk_row.addWidget(self.risk_crop, 1); lay.addLayout(risk_row)
            alpha_editor = QWidget(); alpha_editor_layout = QVBoxLayout(alpha_editor); alpha_editor_layout.setContentsMargins(0, 0, 0, 0)
            alpha_tools = QHBoxLayout(); alpha_tools.addWidget(QLabel("Source matte（只读）")); self.alpha_tool = QComboBox(); self.alpha_tool.addItems(["paint", "erase"])
            self.alpha_overlay = QComboBox(); self.alpha_overlay.addItem("Alpha 置信度", "trust"); self.alpha_overlay.addItem("边界风险图", "structure"); self.alpha_overlay.addItem("置信度 + 风险", "trust+structure"); self.alpha_overlay.setToolTip("只用于查看 source matte，不会改变 source 几何")
            self.alpha_radius = QSlider(Qt.Horizontal); self.alpha_radius.setRange(5, 120); self.alpha_radius.setValue(25); self.alpha_radius.setToolTip("画笔半径")
            self.alpha_opacity = QSlider(Qt.Horizontal); self.alpha_opacity.setRange(5, 100); self.alpha_opacity.setValue(100); self.alpha_opacity.setToolTip("不透明度")
            self.alpha_undo = QPushButton("撤销"); self.alpha_redo = QPushButton("重做"); self.alpha_reset = QPushButton("重置")
            alpha_tools.addWidget(self.alpha_tool); alpha_tools.addWidget(QLabel("叠加")); alpha_tools.addWidget(self.alpha_overlay); alpha_tools.addWidget(QLabel("半径")); alpha_tools.addWidget(self.alpha_radius, 1); alpha_tools.addWidget(QLabel("不透明度")); alpha_tools.addWidget(self.alpha_opacity, 1); alpha_tools.addWidget(self.alpha_undo); alpha_tools.addWidget(self.alpha_redo); alpha_tools.addWidget(self.alpha_reset); alpha_editor_layout.addLayout(alpha_tools)
            self.alpha_canvas = AlphaCanvas(); alpha_editor_layout.addWidget(self.alpha_canvas); self.alpha_canvas.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            alpha_panel, _toggle = self.accordion("Source matte 证据（只读）", alpha_editor); lay.addWidget(alpha_panel)
            self.alpha_tool.currentTextChanged.connect(self.alpha_canvas.set_tool); self.alpha_overlay.currentIndexChanged.connect(lambda index: self.alpha_canvas.set_overlay_mode(self.alpha_overlay.itemData(index))); self.alpha_radius.valueChanged.connect(self.alpha_canvas.set_radius); self.alpha_opacity.valueChanged.connect(self.alpha_canvas.set_opacity); self.alpha_undo.clicked.connect(self.alpha_canvas.undo); self.alpha_redo.clicked.connect(self.alpha_canvas.redo); self.alpha_reset.clicked.connect(self.alpha_canvas.reset); self.alpha_canvas.editsChanged.connect(self.alpha_edits_changed)
            split = QSplitter(Qt.Horizontal); self.alpha = QPlainTextEdit(); self.alpha.setReadOnly(True); self.appearance = QPlainTextEdit()
            self.appearance.setPlainText(json.dumps({"schema_version":"appearance_revision_draft.v1","revision":1,"params":{"exposure":0,"contrast":0,"highlights":0,"shadows":0,"saturation":0,"temperature":0,"tint":0,"sharpness":0}}, ensure_ascii=False, indent=2))
            a = QWidget(); al = QVBoxLayout(a); al.addWidget(QLabel("Source matte（由原生准备阶段生成，当前仅供查看）")); alpha_json, _toggle = self.accordion("Source matte JSON（只读）", self.alpha); al.addWidget(alpha_json); ab = QPushButton("Source matte 不支持手动 revision"); self.alpha_save = ab; ab.setEnabled(False); al.addWidget(ab)
            b = QWidget(); bl = QVBoxLayout(b); bl.addWidget(QLabel("Native appearance revision 参数（作用于 source-bound 结果）")); self.appearance_sliders = {}; self.appearance_value_labels = {}
            for key, label in (("exposure", "曝光"), ("contrast", "对比度"), ("highlights", "高光"), ("shadows", "阴影"), ("saturation", "饱和度"), ("temperature", "色温"), ("tint", "色调"), ("sharpness", "锐度")):
                row = QHBoxLayout(); row.addWidget(QLabel(label)); slider = QSlider(Qt.Horizontal); slider.setRange(-100, 100); slider.setValue(0); slider.setToolTip(label); row.addWidget(slider, 1); value_label = QLabel("0"); row.addWidget(value_label); slider.valueChanged.connect(lambda value, k=key, out=value_label: self.update_appearance_parameter(k, value, out)); self.appearance_sliders[key] = slider; self.appearance_value_labels[key] = value_label; bl.addLayout(row)
            appearance_json, _toggle = self.accordion("高级外观参数 JSON", self.appearance); bl.addWidget(appearance_json); bb = QPushButton("保存 appearance revision 草稿"); bl.addWidget(bb)
            self.manual_layers = QPlainTextEdit(); self.manual_layers.setPlainText(json.dumps(self.default_manual_layers(), ensure_ascii=False, indent=2))
            c = QWidget(); cl = QVBoxLayout(c); cl.addWidget(QLabel("局部手动层（不改 source 几何）"))
            self.manual_layer_controls = {}
            cl.addWidget(self.manual_layers_panel())
            manual_json, _toggle = self.accordion("高级图层 JSON", self.manual_layers); cl.addWidget(manual_json)
            cl.addWidget(QLabel("四层均为外观叠加：人物 source 几何、脸、手、头发和服装主体不会被重绘。"))
            split.addWidget(a); split.addWidget(b); split.addWidget(c)
            retouch_panel, _toggle = self.accordion("光色调整、手动图层与保存", split); lay.addWidget(retouch_panel)
            # Keep infrequent review evidence and editing tools available without
            # consuming the photo height on a 1366x768 display.
            advanced = QWidget(); advanced_layout = QVBoxLayout(advanced); advanced_layout.setContentsMargins(0, 6, 0, 0); advanced_layout.setSpacing(7)
            for widget in (self.review_state, review_action, checks_panel, alpha_panel, retouch_panel):
                lay.removeWidget(widget); advanced_layout.addWidget(widget)
            self.review_state.setWordWrap(True); advanced_layout.addStretch()
            self.local_editor_tabs.addTab(self.scrollable_page(advanced), "审片与高级")
            bb.clicked.connect(lambda: self.save_revision("appearance")); return page

        def source_light_panel(self):
            panel = QWidget(); panel.setMinimumWidth(320); panel.setMaximumWidth(370)
            outer = QVBoxLayout(panel); outer.setContentsMargins(10, 0, 0, 0); outer.setSpacing(8)
            title = QLabel("仿色融合与精修"); title.setObjectName("sectionTitle"); outer.addWidget(title)
            self.source_light_state = QLabel("打开高清草稿后，可规划或添加原片补光。"); self.source_light_state.setWordWrap(True); self.source_light_state.setMaximumHeight(45); outer.addWidget(self.source_light_state)
            self.session_draft_state = QLabel(""); self.session_draft_state.setWordWrap(True); outer.addWidget(self.session_draft_state)
            self.local_editor_tabs = QTabWidget(); self.local_editor_tabs.setDocumentMode(True)
            light_tab = QWidget(); light_layout = QVBoxLayout(light_tab); light_layout.setContentsMargins(0, 6, 0, 0); light_layout.setSpacing(7)
            toolbar = QHBoxLayout(); self.source_light_enabled = QCheckBox("启用")
            toolbar.addStretch(); toolbar.addWidget(self.source_light_enabled); light_layout.addLayout(toolbar)
            self.source_light_list = QListWidget(); self.source_light_list.setFixedHeight(80); light_layout.addWidget(self.source_light_list)
            buttons = QHBoxLayout(); self.source_light_add = QPushButton("添加柔光"); self.source_light_remove = QPushButton("删除")
            self.source_light_overlay = QCheckBox("显示范围"); buttons.addWidget(self.source_light_add); buttons.addWidget(self.source_light_remove); buttons.addWidget(self.source_light_overlay); light_layout.addLayout(buttons)
            self.source_light_form = QWidget(); form = QFormLayout(self.source_light_form); form.setContentsMargins(0, 0, 0, 0); form.setSpacing(7)
            self.source_light_label = QLineEdit()
            self.source_light_field_enabled = QCheckBox("此柔光生效")
            self.source_light_scope = QComboBox()
            for label, value in (("整幅图像", "all"), ("仅人物", "subject"), ("仅背景", "background")): self.source_light_scope.addItem(label, value)
            self.source_light_spins = {}
            for key, label, low, high, step, digits in (
                ("center_x", "中心 X", 0, 1, .01, 3), ("center_y", "中心 Y", 0, 1, .01, 3),
                ("radius_x", "水平半径", .001, 1, .01, 3), ("radius_y", "垂直半径", .001, 1, .01, 3),
                ("rotation_deg", "旋转 °", -360, 360, 5, 1), ("exposure_ev", "曝光 EV", -2, 2, .05, 2),
                ("temperature", "色温", -1, 1, .05, 2), ("tint", "色调", -1, 1, .05, 2),
                ("saturation", "饱和度", -1, 1, .05, 2), ("opacity", "强度", 0, 1, .05, 2)):
                spin = QDoubleSpinBox(); spin.setRange(low, high); spin.setSingleStep(step); spin.setDecimals(digits); spin.setKeyboardTracking(False)
                spin.setToolTip("位置与半径按原片宽高归一化" if key.startswith(("center", "radius")) else label)
                self.source_light_spins[key] = spin; spin.valueChanged.connect(self.edit_source_light)
            for label, keys in (("曝光 / 强度", ("exposure_ev", "opacity")), ("色温 / 色调", ("temperature", "tint")),
                                ("中心 X / Y", ("center_x", "center_y")), ("半径 X / Y", ("radius_x", "radius_y")), ("饱和 / 旋转°", ("saturation", "rotation_deg"))):
                row = QWidget(); pair = QHBoxLayout(row); pair.setContentsMargins(0, 0, 0, 0); pair.setSpacing(5)
                for key in keys: pair.addWidget(self.source_light_spins[key], 1)
                form.addRow(label, row)
            form.addRow("作用范围", self.source_light_scope); form.addRow("名称", self.source_light_label); form.addRow(self.source_light_field_enabled)
            light_layout.addWidget(self.source_light_form, 1)
            for content, label in ((light_tab, "柔光"), (self.source_color_panel(), "局部纠偏"), (self.design_reconstruction_panel(), "设计图层"), (self.clone_panel(), "仿制修复"), (self.tone_curve_panel(), "曲线"), (self.hsl_panel(), "颜色 HSL"), (self.local_colour_brush_panel(), "颜色笔刷"), (self.native_matte_panel(), "人物边缘")):
                scroll = self.scrollable_page(content); scroll.setMinimumHeight(150); self.local_editor_tabs.addTab(scroll, label)
            self.portrait_beauty_page = self.scrollable_page(self.portrait_beauty_panel())
            self.local_editor_tabs.addTab(self.portrait_beauty_page, '人像美化')
            outer.addWidget(self.local_editor_tabs, 1)
            self.local_editor_tabs.currentChanged.connect(self.local_editor_changed)
            actions = QHBoxLayout(); self.source_light_preview = QPushButton("预览"); self.source_light_save = QPushButton("保存高清版本"); self.source_light_save.setObjectName("primaryButton")
            actions.addWidget(self.source_light_preview); actions.addWidget(self.source_light_save, 1); outer.addLayout(actions)
            self.source_light_compare = QCheckBox("查看保存前草稿"); outer.addWidget(self.source_light_compare)
            note = QLabel("在原片或草稿上编辑；保存会创建新的高清版本。"); note.setWordWrap(True); note.setObjectName("mutedLabel"); outer.addWidget(note)
            self.source_light_add.clicked.connect(self.add_source_light); self.source_light_remove.clicked.connect(self.remove_source_light)
            self.source_light_list.currentRowChanged.connect(self.select_source_light); self.source_light_enabled.toggled.connect(self.toggle_source_lights)
            self.source_light_label.textEdited.connect(self.edit_source_light); self.source_light_field_enabled.toggled.connect(self.edit_source_light); self.source_light_scope.currentIndexChanged.connect(self.edit_source_light)
            self.source_light_overlay.toggled.connect(self.update_source_light_overlay); self.source_light_compare.toggled.connect(self.compare_source_lights)
            self.portrait_beauty_compare.toggled.connect(self.source_light_compare.setChecked)
            self.source_light_compare.toggled.connect(self.portrait_beauty_compare.setChecked)
            self.source_light_preview.clicked.connect(self.preview_source_lights); self.source_light_save.clicked.connect(self.save_source_lights)
            self.source_light_form.setEnabled(False); self.update_source_light_actions(); return panel

        def update_source_light_actions(self):
            available = bool(self.source_run_id) and not self.render_pending
            self.source_light_add.setEnabled(available); self.source_light_enabled.setEnabled(available and self.source_light_plan is not None)
            has_selection = self.source_light_list.currentRow() >= 0
            self.source_light_remove.setEnabled(available and has_selection); self.source_light_form.setEnabled(available and has_selection)
            has_plan = self.portrait_beauty is not None or self.source_light_plan is not None or self.source_color_plan is not None or self.design_reconstruction_plan is not None or (hasattr(self,'manual_layers') and bool(self._manual_data()['layers'].get('clone_retouch',{}).get('strokes') or self._manual_data()['layers'].get('tone_curves') or self._manual_data()['layers'].get('hsl_adjust') or self._manual_data()['layers'].get('local_colour_brush',{}).get('strokes')))
            current = not self._paired_subject_state()['reconstruction_required']
            self.source_light_preview.setEnabled(available and has_plan and current); self.source_light_save.setEnabled(available and has_plan and current)
            self.update_source_color_actions()
            self.update_design_reconstruction_actions()
            self.update_clone_overlay()
            self.update_tone_curve_actions()
            self.update_hsl_actions()
            self.update_local_brush_overlay()
            self.update_native_matte_overlay()
            self.update_live_native_detail_actions()
            self.update_portrait_beauty_actions()
            self.update_generated_appearance_actions()

        def update_generated_appearance_actions(self):
            """Keep unsupported source-space editors out of a generated canvas."""
            if not hasattr(self, 'local_editor_tabs'): return
            readonly = self.source_generated_appearance
            self.local_editor_tabs.setEnabled(not readonly)
            if hasattr(self, 'appearance'):
                self.appearance.setReadOnly(readonly); self.manual_layers.setReadOnly(readonly)
            for slider in getattr(self, 'appearance_sliders', {}).values(): slider.setEnabled(not readonly)
            if hasattr(self, 'source_adjust'):
                self.source_adjust.setEnabled(bool(self.source_run_id) and not readonly and not self.render_pending)
            if hasattr(self, 'simple_workspace'):
                self.simple_workspace.strength_group.setEnabled(not readonly)
            if not readonly: return
            self.source_light_timer.stop(); self.native_matte_timer.stop()
            note = '冻结外观版本已保存。可对比、查看原尺寸、审片和导出；原片图层工具不适用于此版本。进一步修改请检查新的改进任务。'
            self.source_light_state.setText(note)
            for name in ('source_light_preview', 'source_light_save', 'source_adjust', 'complete_post_button',
                         'reprepare_current_button', 'portrait_beauty_prepare', 'native_live_detail_button', 'variant_use'):
                control = getattr(self, name, None)
                if control is not None: control.setEnabled(False); control.setToolTip(note)
            if hasattr(self, 'simple_workspace'):
                quick = self.simple_workspace
                same = bool(self.source_edit.text().strip() and self.source_run_source
                    and Path(self.source_edit.text().strip()).resolve() == Path(self.source_run_source).resolve())
                if same:
                    quick.save_button.setEnabled(False); quick.save_button.setToolTip(note)
                    quick.beauty_button.setEnabled(False); quick.beauty_button.setToolTip(note)
                    quick.strength_group.setEnabled(False)
                    quick.post_button.setVisible(False)
                    quick.set_preparation_state(note)

        def _paired_subject_state(self, plan=None):
            from pipeline_stage_report import paired_subject_observation_state
            from design_reconstruction import SUPPORTED_NATIVE_LAYER_KINDS
            plan = (self.design_reconstruction_plan if plan is None else plan) or {}
            source_sha = getattr(self, 'source_run_source_sha256', None) or plan.get('source_sha256')
            candidate_sha = getattr(self, 'source_run_candidate_sha256', None) or plan.get('candidate_sha256')
            state = paired_subject_observation_state(plan, source_sha256=source_sha,
                candidate_sha256=candidate_sha,
                source_only_plan=(None if getattr(self, 'source_run_has_reference', False)
                    else getattr(self, 'source_run_only_plan', None)))
            from design_scene_assets import validate_contact_receiver_contract
            contacts = [layer for layer in plan.get('layers', []) if layer.get('kind') == 'contact_shadow']
            errors = []
            alpha_sha = ((plan.get('mask_roles') or {}).get('foreground_alpha') or {}).get('sha256')
            for layer in contacts:
                try:
                    validate_contact_receiver_contract(layer, source_sha256=source_sha,
                        candidate_sha256=candidate_sha, source_alpha_sha256=alpha_sha)
                except (ValueError, TypeError, KeyError) as error:
                    errors.append(str(error))
            state['contact_contract_state'] = ('not_requested' if not contacts else
                'reconstruction_required' if errors else 'current')
            state['contact_contract_errors'] = errors
            unsupported = sorted({str(layer.get('kind')) for layer in plan.get('layers', [])
                                  if layer.get('kind') not in SUPPORTED_NATIVE_LAYER_KINDS})
            state['unsupported_design_layers'] = unsupported
            state['reconstruction_required'] = state['reconstruction_required'] or bool(plan.get('enabled') and (errors or unsupported))
            return state

        def _allow_current_render(self):
            if self.source_generated_appearance:
                self.update_generated_appearance_actions()
                return False
            if not self._paired_subject_state()['reconstruction_required']: return True
            self.source_light_state.setText('已保存画面可查看；请先点击准备仿色融合，再预览或保存新效果。不重新生图。')
            self.update_source_light_actions(); return False

        def update_live_native_detail_actions(self):
            if not hasattr(self,'native_live_detail_button'):return
            plan=self.design_reconstruction_plan or {}
            ready=bool(self.source_run_id) and (plan.get('execution') or {}).get('source_reference')=='paired_native_baseline' and not self._paired_subject_state()['reconstruction_required']
            self.native_live_detail_button.setEnabled(ready and not self.render_pending and self.native_live_detail_context is None and self.native_matte_request is None)

        def load_source_lights(self, plan):
            self.source_light_loading = True
            self.source_light_plan = json.loads(json.dumps(plan)) if isinstance(plan, dict) else None
            self.source_light_enabled.setChecked(bool((self.source_light_plan or {}).get("enabled", True)) if self.source_light_plan else False)
            self.source_light_list.clear()
            for field in (self.source_light_plan or {}).get("light_fields", []):
                self.source_light_list.addItem(str(field.get("label") or field.get("id") or "柔光"))
            self.source_light_list.setCurrentRow(0 if self.source_light_list.count() else -1)
            self.source_light_loading = False; self.select_source_light(self.source_light_list.currentRow()); self.update_source_light_actions()
            summary = (self.source_light_plan or {}).get("summary")
            self.source_light_state.setText("局部光色方案已载入；调整自动预览，保存高清版本。" if summary else ("调整参数自动预览；保存生成高清版本。" if self.source_light_plan else "此版本没有局部补光方案。可根据原片规划或手动添加。"))
            self.source_light_state.setToolTip(str(summary or ""))
            self.update_source_light_overlay()

        def select_source_light(self, index):
            fields = (self.source_light_plan or {}).get("light_fields", [])
            self.source_light_loading = True
            field = fields[index] if 0 <= index < len(fields) else None
            if field:
                self.source_light_label.setText(str(field.get("label") or "柔光")); self.source_light_field_enabled.setChecked(field.get("enabled", True))
                self.source_light_scope.setCurrentIndex(max(0, self.source_light_scope.findData(field.get("scope", "all"))))
                center = field.get("center", [.5, .5]); radius = field.get("radius", [.2, .2])
                values = {"center_x": center[0], "center_y": center[1], "radius_x": radius[0], "radius_y": radius[1], **field}
                for key, spin in self.source_light_spins.items(): spin.setValue(float(values.get(key, 1 if key == "opacity" else 0)))
            self.source_light_loading = False; self.update_source_light_actions(); self.update_source_light_overlay()

        def edit_source_light(self, _value=None):
            if self.source_light_loading: return
            index = self.source_light_list.currentRow(); fields = (self.source_light_plan or {}).get("light_fields", [])
            if not 0 <= index < len(fields): return
            field = fields[index]; field.update({"label": self.source_light_label.text().strip() or "柔光", "enabled": self.source_light_field_enabled.isChecked(), "scope": self.source_light_scope.currentData(),
                "center": [self.source_light_spins[key].value() for key in ("center_x", "center_y")], "radius": [self.source_light_spins[key].value() for key in ("radius_x", "radius_y")]})
            for key in ("rotation_deg", "exposure_ev", "temperature", "tint", "saturation", "opacity"): field[key] = self.source_light_spins[key].value()
            self.source_light_list.item(index).setText(field["label"]); self.source_lights_changed()

        def toggle_source_lights(self, checked):
            if self.source_light_loading or self.source_light_plan is None: return
            self.source_light_plan["enabled"] = checked; self.source_lights_changed()

        def add_source_light(self):
            if not self.source_run_id or self.render_pending: return
            plan = self.source_light_plan or {"schema_version": "source_render_plan.v1", "enabled": True, "summary": "手动局部光色", "light_fields": []}
            fields = plan.setdefault("light_fields", []); self.source_light_sequence += 1
            fields.append({"id": "manual-" + uuid.uuid4().hex[:10], "label": f"柔光 {len(fields) + 1}", "center": [.5, .35], "radius": [.2, .2], "rotation_deg": 0, "exposure_ev": .25, "temperature": 0, "tint": 0, "saturation": 0, "opacity": 1, "scope": "subject", "enabled": True})
            self.load_source_lights(plan); self.source_light_list.setCurrentRow(len(fields) - 1); self.source_lights_changed()

        def remove_source_light(self):
            index = self.source_light_list.currentRow()
            if self.source_light_plan is None or index < 0: return
            plan = self.source_light_plan; del plan["light_fields"][index]; self.load_source_lights(plan)
            self.source_light_list.setCurrentRow(min(index, self.source_light_list.count() - 1)); self.source_lights_changed()

        def source_lights_changed(self):
            if self.source_light_origin in {"model", "model_with_user_edits"}: self.source_light_origin = "model_with_user_edits"
            self.source_light_version += 1; self.source_light_state.setText("局部光色已修改，正在准备预览…")
            self.source_light_compare.setChecked(False); self.update_source_light_overlay(); self.source_light_timer.start()

        def update_source_light_overlay(self, _checked=None):
            view = self.previews.get("final")
            if view is None: return
            view.light_fields = (self.source_light_plan or {}).get("light_fields", []) if (self.source_light_plan or {}).get("enabled", True) else []
            index = self.source_light_list.currentRow(); view.selected_light_id = view.light_fields[index].get("id", "") if 0 <= index < len(view.light_fields) else ""
            view.show_light_fields = self.source_light_overlay.isChecked() and self.local_editor_tabs.currentIndex() == 0; view.render_overlays()

        def preview_source_lights(self):
            self.source_light_timer.stop()
            if not self.source_run_id or self.render_pending: return
            if not self._allow_current_render(): return
            if self.native_matte_request is not None:return
            if self.native_matte_strokes and self.native_matte_preview_version!=self.native_matte_version:
                self.preview_native_matte();return
            self.source_light_sequence += 1; token = str(self.source_light_sequence)
            self.source_light_requests[token] = (self.source_run_id, self.source_light_version)
            self.receiver_shadow_preview_requests[token] = {'snapshot': self.receiver_edit_snapshot(), 'matte_version': self.native_matte_version,
                'matte_strokes': json.loads(json.dumps(self.native_matte_strokes))}
            self.source_light_state.setText("正在更新局部光色预览…")
            body = {"request_id": token, **self._source_edit_payload()}
            self.task("POST", f"/api/source-runs/{self.source_run_id}/preview", body, "source-light-preview:" + token)

        def compare_source_lights(self, before):
            view = self.previews["final"]
            path = self.source_light_draft_path if before else getattr(self, "source_light_preview_path", None) or self.source_light_draft_path
            if path: view.replace_preview_path(path); self.update_source_light_overlay(); self.update_source_color_overlay()

        def save_source_lights(self):
            if not self.source_run_id or self.render_pending: return
            if not self._allow_current_render(): return
            self.source_light_timer.stop(); self.source_light_version += 1
            payload = self._source_edit_payload()
            if self.native_matte_strokes:
                payload.update(strokes=json.loads(json.dumps(self.native_matte_strokes)),cache_key=self.native_matte_cache_key)
                self.source_light_state.setText('正在保存人物边缘与高清版本…')
                self.task('POST',f'/api/source-runs/{self.source_run_id}/matte-adjust',payload,'source-adjust');return
            self.source_light_state.setText("正在制作原尺寸高清版本…"); self.operation_state.setText("正在高清处理，完成后打开新版本")
            self.task("POST", f"/api/source-runs/{self.source_run_id}/adjust", payload, "source-adjust")

        def _source_edit_payload(self):
            """Preview and save submit the same current component snapshots."""
            appearance = json.loads(self.appearance.toPlainText())
            manual = json.loads(self.manual_layers.toPlainText())
            return {'appearance_revision':self._appearance_revision_payload(appearance),
                    'manual_layers':manual.get('layers',manual)}

        def portrait_beauty_panel(self):
            panel = QWidget(); layout = QVBoxLayout(panel); layout.setContentsMargins(0, 6, 0, 0); layout.setSpacing(8)
            self.portrait_beauty_prepare = QPushButton('一键准备人像美化 · BeautyGRPO')
            self.portrait_beauty_prepare.setObjectName('primaryButton')
            self.portrait_beauty_prepare.setToolTip('本地准备模型原始美化效果，无需参考图。任务中心可查看进度或取消；完成后预览并保存高清版本。')
            self.portrait_beauty_prepare.clicked.connect(self.prepare_portrait_beauty); layout.addWidget(self.portrait_beauty_prepare)
            self.portrait_beauty_state = QLabel('打开原片版本后可准备人像美化，无需参考图。')
            self.portrait_beauty_state.setWordWrap(True); layout.addWidget(self.portrait_beauty_state)
            row = QHBoxLayout(); row.addWidget(QLabel('美化强度'))
            self.portrait_beauty_strength = QSlider(Qt.Horizontal); self.portrait_beauty_strength.setRange(0, 100)
            self.portrait_beauty_strength.setValue(100); row.addWidget(self.portrait_beauty_strength, 1)
            self.portrait_beauty_strength_value = QLabel('100%'); row.addWidget(self.portrait_beauty_strength_value)
            self.portrait_beauty_strength.valueChanged.connect(self.edit_portrait_beauty_strength); layout.addLayout(row)
            self.portrait_beauty_compare = QCheckBox('查看调整前的保存画面')
            self.portrait_beauty_compare.setToolTip('沿用同一预览的缩放和位置；取消勾选查看当前美化预览。')
            layout.addWidget(self.portrait_beauty_compare)
            restoration = QGroupBox('局部恢复原片 · 按人脸'); restore_layout = QVBoxLayout(restoration)
            self.portrait_beauty_face = QComboBox(); self.portrait_beauty_face.currentIndexChanged.connect(self.update_portrait_beauty_actions)
            restore_layout.addWidget(self.portrait_beauty_face)
            self.portrait_beauty_restore_face = QCheckBox('恢复所选人脸的原片外观')
            self.portrait_beauty_restore_face.setToolTip('关闭这张人脸的美化，恢复该人脸范围；其它人脸保留美化。此入口按人脸操作，不是画笔。')
            self.portrait_beauty_restore_face.toggled.connect(self.toggle_portrait_beauty_face); restore_layout.addWidget(self.portrait_beauty_restore_face)
            self.portrait_beauty_scope = QLabel('准备完成后选择人脸。恢复范围为该人脸的美化区域。')
            self.portrait_beauty_scope.setWordWrap(True); restore_layout.addWidget(self.portrait_beauty_scope); layout.addWidget(restoration)
            note = QLabel('默认 100% 保留模型生成的肤质、清晰度和色调；按接缝及遮挡融合。可降低强度或按人脸恢复原片，调整后预览并保存。')
            note.setWordWrap(True); note.setObjectName('mutedLabel'); layout.addWidget(note); layout.addStretch()
            return panel

        def update_portrait_beauty_actions(self, *_args):
            if not hasattr(self, 'portrait_beauty_prepare'): return
            available = bool(self.source_run_id) and not self.render_pending
            ready = self.portrait_beauty is not None
            self.portrait_beauty_prepare.setEnabled(available and self.portrait_beauty_request is None)
            self.portrait_beauty_prepare.setText('重新准备人像美化 · BeautyGRPO' if ready else '一键准备人像美化 · BeautyGRPO')
            self.portrait_beauty_strength.setEnabled(available and ready)
            self.portrait_beauty_compare.setEnabled(available and ready)
            self.portrait_beauty_face.setEnabled(available and ready and bool(self.portrait_beauty_faces))
            identifier = self.portrait_beauty_face.currentData()
            face = next((row for row in self.portrait_beauty_faces if row['id'] == identifier), None)
            self.portrait_beauty_restore_face.setEnabled(available and ready and face is not None)
            self.portrait_beauty_restore_face.blockSignals(True)
            self.portrait_beauty_restore_face.setChecked(bool(ready and identifier in self.portrait_beauty.get('disabled_faces', [])))
            self.portrait_beauty_restore_face.blockSignals(False)
            box = face.get('native_box') if face else None
            self.portrait_beauty_scope.setText(('所选人脸美化区域：原片坐标 ' + ', '.join(str(value) for value in box) + '；整张人脸恢复，不是画笔。')
                if isinstance(box, list) and len(box) == 4 else '准备完成后选择人脸。恢复范围为该人脸的美化区域。')

        def load_portrait_beauty(self, config, faces=None):
            self.portrait_beauty_loading = True
            self.portrait_beauty = copy.deepcopy(config) if isinstance(config, dict) and config.get('result') else None
            self.portrait_beauty_faces = []
            if self.portrait_beauty is not None:
                if faces is None:
                    try:
                        document = json.loads(Path(self.portrait_beauty['result']).read_text(encoding='utf-8'))
                        faces = document.get('faces', [])
                    except (OSError, TypeError, ValueError):
                        faces = []
                self.portrait_beauty_faces = [copy.deepcopy(row) for row in faces or []
                    if isinstance(row, dict) and isinstance(row.get('id'), str)]
            self.portrait_beauty_face.blockSignals(True); self.portrait_beauty_face.clear()
            for index, face in enumerate(self.portrait_beauty_faces):
                self.portrait_beauty_face.addItem('人脸 ' + str(index + 1) + ' · ' + face['id'], face['id'])
            self.portrait_beauty_face.blockSignals(False)
            strength = float(self.portrait_beauty.get('strength', 1)) if self.portrait_beauty else 1.
            self.portrait_beauty_strength.blockSignals(True); self.portrait_beauty_strength.setValue(round(strength * 100)); self.portrait_beauty_strength.blockSignals(False)
            self.portrait_beauty_strength_value.setText(str(self.portrait_beauty_strength.value()) + '%')
            self.portrait_beauty_compare.blockSignals(True); self.portrait_beauty_compare.setChecked(False); self.portrait_beauty_compare.blockSignals(False)
            self.portrait_beauty_state.setText('BeautyGRPO 原始生成外观已载入；可调强度或按人脸恢复原片。' if self.portrait_beauty else '打开原片版本后可准备人像美化，无需参考图。')
            self.portrait_beauty_loading = False; self.update_portrait_beauty_actions()

        def portrait_beauty_changed(self):
            if self.workflow_graph:
                self.workflow_graph['nodes']['beauty'] = {'enabled': True, 'strength': float(self.portrait_beauty.get('strength', 1.))}
            self.simple_comparing = False
            self.source_light_version += 1
            self.source_light_compare.setChecked(False)
            self._checkpoint_session(); self.update_source_light_actions(); self.source_light_timer.start()

        def edit_portrait_beauty_strength(self, value):
            self.portrait_beauty_strength_value.setText(str(value) + '%')
            if self.portrait_beauty_loading or self.portrait_beauty is None: return
            self.portrait_beauty['strength'] = value / 100.
            self.portrait_beauty_changed()

        def toggle_portrait_beauty_face(self, restored):
            if self.portrait_beauty_loading or self.portrait_beauty is None: return
            identifier = self.portrait_beauty_face.currentData()
            if identifier is None: return
            disabled = set(self.portrait_beauty.get('disabled_faces', []))
            if restored: disabled.add(identifier)
            else: disabled.discard(identifier)
            self.portrait_beauty['disabled_faces'] = sorted(disabled); self.portrait_beauty_changed()

        def prepare_portrait_beauty(self, _checked=False):
            if not self.source_run_id or self.render_pending or self.portrait_beauty_request is not None: return
            self._checkpoint_session()
            if self._session_checkpoint_failed: return
            try: payload = self._source_edit_payload()
            except (TypeError, ValueError) as error:
                self.portrait_beauty_state.setText('当前参数无法读取，尚未提交美化准备：' + str(error)); return
            self.source_light_timer.stop(); self.native_matte_timer.stop()
            self.portrait_beauty_request = 'submitting'
            self.portrait_beauty_state.setText('正在本地准备 BeautyGRPO 原始效果；任务中心可等待或取消。')
            self.local_editor_tabs.setCurrentWidget(self.portrait_beauty_page); self.layer_tabs.setCurrentWidget(self.previews['final'])
            self.task('POST', f'/api/source-runs/{self.source_run_id}/prepare-portrait-beauty', payload, 'portrait-beauty-job')

        def handle_portrait_beauty_job(self, value):
            from session_draft import digest
            job = value.get('job') or {}
            if job.get('kind') != 'prepare_portrait_beauty': return False
            if job.get('job_id') != self.active_job_id: return True
            job_id = job['job_id']; status = job.get('status'); self.portrait_beauty_job_id = job_id
            self.job_details.setPlainText(json.dumps(job, ensure_ascii=False, indent=2))
            if hasattr(self, 'simple_workspace'):
                self._show_quick_job_progress(job)
            if status in {'queued', 'running', 'cancelling'}:
                self.render_pending = True; self.portrait_beauty_request = job_id
                self.operation_state.setText(str(job.get('message') or '正在准备人像美化；任务中心可取消。'))
                self.portrait_beauty_state.setText(self.operation_state.text()); self.update_accept_enabled()
                self._schedule_job_poll(750)
                return True
            self.render_pending = False; self.portrait_beauty_request = None; self.update_accept_enabled()
            if self.portrait_beauty_result_seen == job_id: return True
            if status != 'succeeded':
                self.simple_creation = None
                self.operation_state.setText('人像美化准备已取消；当前编辑保留。' if status == 'cancelled' else '人像美化准备未完成；当前编辑保留，可重试。')
                self.portrait_beauty_state.setText(self.operation_state.text())
                if job.get('error'): self.log_text('[portrait beauty] ' + str(job['error']))
                return True
            receipt = value.get('portrait_beauty_result') or {}
            if receipt.get('run_id') != self.source_run_id or receipt.get('source_sha256') != self.source_run_source_sha256:
                self.simple_creation = None
                self.operation_state.setText('美化结果已保留；请打开任务对应原片版本后再载入。'); return True
            try: current_sha = digest(self._source_edit_payload())
            except (TypeError, ValueError): current_sha = None
            if receipt.get('input_sha256') != current_sha:
                self.simple_creation = None
                self.operation_state.setText('美化结果已保留；当前精修已有变化，请按当前参数重试准备。'); return True
            if receipt.get('no_change') is True:
                self.simple_creation = None
                self.portrait_beauty_result_seen = job_id
                self.operation_state.setText('未找到可美化的可见人脸；当前编辑与原片保留。')
                self.portrait_beauty_state.setText(self.operation_state.text()); return True
            config = (receipt.get('appearance_revision') or {}).get('portrait_beauty')
            if (receipt.get('preparation_contract') != 'portrait_beauty_layer.v1' or not isinstance(config, dict)
                    or not config.get('result') or not Path(config['result']).is_absolute()):
                self.simple_creation = None
                self.operation_state.setText('美化任务缺少有效冻结结果；当前编辑保留，请重试。'); return True
            self.portrait_beauty_result_seen = job_id
            self.load_portrait_beauty(config, receipt.get('faces'))
            self._apply_workflow_intent()
            self.local_editor_tabs.setCurrentWidget(self.portrait_beauty_page); self.layer_tabs.setCurrentWidget(self.previews['final'])
            self.source_light_version += 1; self.source_light_compare.setChecked(False)
            self._checkpoint_session(); self.update_source_light_actions()
            self.operation_state.setText('BeautyGRPO 原始美化效果已准备；预览后保存高清新版本。')
            self.portrait_beauty_state.setText(self.operation_state.text()); self.preview_source_lights()
            self._advance_simple_creation('beauty')
            return True

        def design_reconstruction_panel(self):
            panel = QWidget(); layout = QVBoxLayout(panel); layout.setContentsMargins(0, 6, 0, 0); layout.setSpacing(8)
            self.design_reconstruct_button = QPushButton("准备仿色融合 / 重试"); self.design_reconstruct_button.setObjectName("primaryButton")
            self.design_reconstruct_button.setToolTip("复用原片、已选参考和当前精修，不新增生图。任务中心显示各阶段进度，可取消；准备完成后仍需审片并保存高清新版本。")
            self.design_reconstruct_button.clicked.connect(self.reconstruct_design); layout.addWidget(self.design_reconstruct_button)
            self.boundary_light_coordination_button = QPushButton('同时更新边界受光')
            self.boundary_light_coordination_button.setToolTip('依次更新有依据的发梢与肩袖受光，完成后一起应用；保留开关和强度，可取消，不新增生图。仍需预览原尺寸细节并保存新版本。')
            self.boundary_light_coordination_button.clicked.connect(self.prepare_boundary_light_coordination); layout.addWidget(self.boundary_light_coordination_button)
            self.hair_boundary_light_button = QPushButton('补充参考发梢受光')
            self.hair_boundary_light_button.setToolTip('基于已准备的人物外观补充有依据的发梢受光；可取消，完成后预览并保存新版本。')
            self.hair_boundary_light_button.clicked.connect(self.prepare_hair_boundary_light); layout.addWidget(self.hair_boundary_light_button)
            self.cloth_boundary_light_button = QPushButton('补充参考肩袖受光')
            self.cloth_boundary_light_button.setToolTip('基于当前人物外观和发梢受光，补充有依据的肩袖受光；完成后预览并保存新版本。')
            self.cloth_boundary_light_button.clicked.connect(self.prepare_cloth_boundary_light); layout.addWidget(self.cloth_boundary_light_button)
            self.design_stack_enabled = QCheckBox("启用设计图层"); self.design_stack_enabled.toggled.connect(self.toggle_design_reconstruction); layout.addWidget(self.design_stack_enabled)
            self.design_stack_summary = QLabel("选择原图与参考后，准备人物仿色和背景融合。"); self.design_stack_summary.setWordWrap(True); self.design_stack_summary.setObjectName("caption"); layout.addWidget(self.design_stack_summary)
            from scene_receiver_intent import FOOT_STANCES, GROUND_TYPES
            receiver_form = QFormLayout(); self.receiver_stance = QComboBox(); self.receiver_ground = QComboBox()
            for value, label in FOOT_STANCES.items(): self.receiver_stance.addItem(label, value)
            for value, label in GROUND_TYPES.items(): self.receiver_ground.addItem(label, value)
            self.receiver_stance.setToolTip("左右指人物本人的左右脚；未指定时保留未知。")
            self.receiver_ground.setToolTip("描述希望人物站在哪里；不是地面几何或反射测量。")
            receiver_form.addRow("落脚关系", self.receiver_stance); receiver_form.addRow("地面类型", self.receiver_ground); layout.addLayout(receiver_form)
            self.receiver_tool = QComboBox()
            for label, value in (('平移照片', 'pan'), ('画接地范围', 'region'), ('点选四角平面', 'plane'), ('人物左脚落点', 'left'), ('人物右脚落点', 'right'), ('移动已有点', 'move')):
                self.receiver_tool.addItem(label, value)
            self.receiver_tool.setToolTip('在成片上点选；区域至少三点，平面沿边依次四点。左右指人物本人的左右。')
            receiver_form.addRow('画布工具', self.receiver_tool)
            receiver_actions = QHBoxLayout(); self.receiver_finish = QPushButton('确认点选'); self.receiver_cancel = QPushButton('取消点选'); self.receiver_clear = QPushButton('清除此项')
            receiver_actions.addWidget(self.receiver_finish); receiver_actions.addWidget(self.receiver_cancel); receiver_actions.addWidget(self.receiver_clear); layout.addLayout(receiver_actions)
            receiver_history = QHBoxLayout(); self.receiver_undo = QPushButton('撤销接地编辑'); self.receiver_redo = QPushButton('重做')
            receiver_history.addWidget(self.receiver_undo); receiver_history.addWidget(self.receiver_redo); layout.addLayout(receiver_history)
            self.receiver_intent_state = QLabel('在成片上画接地范围，再点选平面与脚落点。所画平面是艺术投影；不会自动开启投影或反射。')
            self.receiver_intent_state.setWordWrap(True); self.receiver_intent_state.setObjectName("caption"); layout.addWidget(self.receiver_intent_state)
            self.receiver_stance.currentIndexChanged.connect(self.edit_receiver_intent); self.receiver_ground.currentIndexChanged.connect(self.edit_receiver_intent)
            self.receiver_tool.currentIndexChanged.connect(self.update_receiver_overlay)
            view = self.previews['final']; view.receiverGeometryEdited.connect(self.edit_receiver_geometry); view.receiverDraftChanged.connect(self.receiver_draft_changed)
            self.receiver_finish.clicked.connect(view.commit_receiver_draft); self.receiver_cancel.clicked.connect(view.cancel_receiver_draft); self.receiver_clear.clicked.connect(self.clear_receiver_geometry)
            self.receiver_undo.clicked.connect(lambda: self.restore_receiver_history(-1)); self.receiver_redo.clicked.connect(lambda: self.restore_receiver_history(1))
            shadow_group = QGroupBox('地面投影'); shadow_layout = QVBoxLayout(shadow_group)
            self.receiver_shadow_enabled = QCheckBox('地面艺术投影（原片轮廓）'); shadow_layout.addWidget(self.receiver_shadow_enabled)
            shadow_form = QFormLayout(); self.receiver_shadow_direction = QComboBox()
            for label, direction in (('主光方向未指定', None), ('左上方主光', [-1., 1.]), ('右上方主光', [1., 1.]), ('上方主光', [0., 1.])):
                self.receiver_shadow_direction.addItem(label, direction)
            shadow_form.addRow('主光方向', self.receiver_shadow_direction)
            self.receiver_shadow_spins = {}
            for key, label, low, high, step in (('length', '投影长度（相对平面）', .01, 4., .05), ('opacity', '投影强度', 0., 1., .05), ('softness', '柔边（相对平面）', 0., .25, .005)):
                spin = QDoubleSpinBox(); spin.setRange(low, high); spin.setDecimals(3); spin.setSingleStep(step); spin.setKeyboardTracking(False)
                self.receiver_shadow_spins[key] = spin; shadow_form.addRow(label, spin)
                spin.valueChanged.connect(lambda value, name=key: self.edit_receiver_shadow_plan(name, value))
            shadow_layout.addLayout(shadow_form)
            self.receiver_shadow_state = QLabel(); self.receiver_shadow_state.setWordWrap(True); self.receiver_shadow_state.setObjectName('caption'); shadow_layout.addWidget(self.receiver_shadow_state)
            layout.addWidget(shadow_group)
            self.receiver_shadow_enabled.toggled.connect(lambda value: self.edit_receiver_shadow_plan('enabled', value))
            self.receiver_shadow_direction.currentIndexChanged.connect(lambda _index: self.edit_receiver_shadow_plan('direction_to_light_xy', self.receiver_shadow_direction.currentData()))
            self.load_receiver_shadow_plan()
            self.design_stack_list = QListWidget(); self.design_stack_list.setMinimumHeight(150); self.design_stack_list.setMaximumHeight(220); self.design_stack_list.currentRowChanged.connect(self.select_design_layer); layout.addWidget(self.design_stack_list)
            self.design_layer_enabled = QCheckBox("此图层生效"); self.design_layer_enabled.toggled.connect(self.edit_design_layer); layout.addWidget(self.design_layer_enabled)
            row = QHBoxLayout(); self.design_layer_strength_label = QLabel("图层强度"); row.addWidget(self.design_layer_strength_label); self.design_layer_strength = QDoubleSpinBox(); self.design_layer_strength.setRange(0, 1); self.design_layer_strength.setDecimals(2); self.design_layer_strength.setSingleStep(.05); self.design_layer_strength.setKeyboardTracking(False); self.design_layer_strength.valueChanged.connect(self.edit_design_layer); row.addWidget(self.design_layer_strength, 1); layout.addLayout(row)
            self.design_layer_help = QLabel(); self.design_layer_help.setWordWrap(True); self.design_layer_help.setObjectName("caption"); layout.addWidget(self.design_layer_help)
            independent = QFormLayout(); self.reference_control_spins = {}
            for key, label in (('colour_strength', '颜色强度'), ('lightness_strength', '明暗强度')):
                spin = QDoubleSpinBox(); spin.setRange(0, 1); spin.setDecimals(2); spin.setSingleStep(.05); spin.setKeyboardTracking(False); spin.setValue(1)
                spin.valueChanged.connect(lambda value, name=key: self.edit_reference_control(name, value)); self.reference_control_spins[key] = spin; independent.addRow(label, spin)
            layout.addLayout(independent)
            self.reference_support_text = QPlainTextEdit(); self.reference_support_text.setReadOnly(True); self.reference_support_text.setMaximumHeight(150); self.reference_support_text.setPlaceholderText('逐区域参考兼容性与降级'); layout.addWidget(self.reference_support_text)
            note = QLabel("开关或调整单个图层，预览后保存高清成片。"); note.setWordWrap(True); note.setObjectName("caption"); layout.addWidget(note); layout.addStretch()
            return panel

        def load_receiver_intent(self, value=None, *, reset_history=True):
            from scene_receiver_intent import normalize_receiver_intent
            declaration = normalize_receiver_intent(value, source_canvas_size=self.receiver_source_canvas_size)
            self.receiver_intent_loading = True
            try:
                self.receiver_intent = declaration
                if reset_history:
                    self.receiver_history = [self.receiver_edit_snapshot()]; self.receiver_history_index = 0
                for combo, key in ((self.receiver_stance, "foot_stance"), (self.receiver_ground, "ground_type")):
                    previous = combo.blockSignals(True)
                    try: combo.setCurrentIndex(combo.findData(declaration[key]))
                    finally: combo.blockSignals(previous)
            finally:
                self.receiver_intent_loading = False
            self.update_receiver_overlay()
            self.receiver_intent_state.setText('接地范围 ' + str(len(declaration['receiver_region'] or [])) + ' 点 · 平面 ' + str(len(declaration['receiver_plane'] or [])) + ' 角 · 落点 ' + str(len(declaration['foot_anchors'])) + ' 个。所画平面是艺术投影；不会自动开启投影或反射。')

        def edit_receiver_intent(self, _index=None):
            if self.receiver_intent_loading or self._session_loading or not self.source_run_id or self.render_pending: return
            declaration = json.loads(json.dumps(self.receiver_intent))
            declaration.update(foot_stance=self.receiver_stance.currentData(),
                ground_type=self.receiver_ground.currentData(), origin="user_authored")
            self.commit_receiver_intent(declaration)

        def commit_receiver_intent(self, declaration):
            from scene_receiver_intent import normalize_receiver_intent
            try: declaration = normalize_receiver_intent(declaration, source_canvas_size=self.receiver_source_canvas_size)
            except (TypeError, ValueError) as error:
                self.load_receiver_intent(self.receiver_intent, reset_history=False)
                self.receiver_intent_state.setText('接地编辑与已有声明冲突，保留原状态：' + str(error)); return False
            if declaration == self.receiver_intent: return True
            snapshot = {'receiver_intent': declaration, 'receiver_shadow_plan': self.receiver_shadow_plan}
            self.receiver_history = self.receiver_history[:self.receiver_history_index + 1] + [json.loads(json.dumps(snapshot))]
            self.receiver_history_index = len(self.receiver_history) - 1
            self.load_receiver_intent(declaration, reset_history=False)
            self.receiver_intent_state.setText('接地编辑已修改，保存高清版本时一并保存。所画平面是艺术投影；不会自动开启投影或反射。')
            # An intent change checkpoints only small editable data; it never
            # schedules a preview, POST, model call or new image generation.
            self._checkpoint_session()
            return True

        def edit_receiver_geometry(self, geometry):
            if self.receiver_intent_loading or self._session_loading or self.render_pending or not self.previews['final'].receiver_active: return
            declaration = json.loads(json.dumps(self.receiver_intent)); declaration.update(geometry, origin='user_authored')
            self.commit_receiver_intent(declaration)

        def clear_receiver_geometry(self):
            if not self.receiver_tool.isEnabled(): return
            declaration = json.loads(json.dumps(self.receiver_intent)); tool = self.receiver_tool.currentData()
            if tool in {'region', 'plane'}: declaration['receiver_' + tool] = None
            elif tool in {'left', 'right'}: declaration['foot_anchors'].pop(tool, None)
            else: return
            declaration['origin'] = 'user_authored'; self.previews['final'].cancel_receiver_draft(); self.commit_receiver_intent(declaration)

        def restore_receiver_history(self, direction):
            if self.render_pending or not self.source_run_id: return
            index = max(0, min(len(self.receiver_history) - 1, self.receiver_history_index + direction))
            if index == self.receiver_history_index or index < 0: return
            self.receiver_history_index = index; self.previews['final'].cancel_receiver_draft()
            snapshot = self.receiver_history[index]
            self.load_receiver_intent(snapshot['receiver_intent'], reset_history=False)
            self.load_receiver_shadow_plan(snapshot['receiver_shadow_plan'], reset_history=False); self._checkpoint_session()

        def receiver_edit_snapshot(self):
            return json.loads(json.dumps({'receiver_intent': self.receiver_intent, 'receiver_shadow_plan': self.receiver_shadow_plan}))

        def receiver_draft_changed(self, points):
            self.update_receiver_geometry_actions()
            if points:
                label = '区域至少三点' if self.receiver_tool.currentData() == 'region' else '沿边依次四角'
                self.receiver_intent_state.setText(label + '，已点选 ' + str(len(points)) + ' 点；确认后保存，Esc 取消。')

        def update_receiver_overlay(self, _index=None):
            if not hasattr(self, 'receiver_tool'): return
            from design_reconstruction import has_native_foreground
            from scene_receiver_intent import GROUNDED_FEET
            view = self.previews['final']; mode = self.receiver_tool.currentData()
            ready = bool(self.source_run_id and self.receiver_source_canvas_size and self.design_reconstruction_plan) and has_native_foreground(self.design_reconstruction_plan) and not self.render_pending
            visible = self.local_editor_tabs.currentIndex() == 2
            feet = GROUNDED_FEET[self.receiver_intent['foot_stance']]
            active = ready and visible and mode != 'pan' and (mode not in {'left', 'right'} or mode in feet)
            geometry = {key: json.loads(json.dumps(self.receiver_intent[key])) for key in ('receiver_region', 'receiver_plane', 'foot_anchors')}
            if (view.receiver_geometry != geometry or view.receiver_canvas_size != self.receiver_source_canvas_size or
                    view.receiver_mode != mode or view.receiver_active != active or view.receiver_visible != visible):
                view._receiver_draft = []; view._receiver_drag = None
            view.receiver_geometry = geometry; view.receiver_canvas_size = self.receiver_source_canvas_size
            view.receiver_mode = mode; view.receiver_active = active; view.receiver_visible = visible
            if active:
                view.local_brush_active = False; view.clone_active = False; view.color_brush_mode = 'pan'
                self.layer_tabs.setCurrentWidget(view)
            if visible: view.image.setCursor(Qt.CrossCursor if active else Qt.OpenHandCursor)
            self.receiver_tool.setEnabled(ready)
            for side in ('left', 'right'):
                self.receiver_tool.model().item(self.receiver_tool.findData(side)).setEnabled(ready and side in feet)
            view.render_overlays(); self.update_receiver_geometry_actions()

        def update_receiver_geometry_actions(self):
            view = self.previews['final']; mode = self.receiver_tool.currentData(); count = len(view._receiver_draft)
            self.receiver_finish.setEnabled(view.receiver_active and ((mode == 'region' and count >= 3) or (mode == 'plane' and count == 4)))
            self.receiver_cancel.setEnabled(view.receiver_active and bool(count or view._receiver_drag))
            has_geometry = bool(self.receiver_intent.get('receiver_' + mode)) if mode in {'region', 'plane'} else mode in self.receiver_intent['foot_anchors']
            self.receiver_clear.setEnabled(view.receiver_active and has_geometry)
            ready = bool(self.source_run_id) and not self.render_pending
            self.receiver_undo.setEnabled(ready and self.receiver_history_index > 0)
            self.receiver_redo.setEnabled(ready and self.receiver_history_index + 1 < len(self.receiver_history))
            self.update_receiver_shadow_actions()

        def load_receiver_shadow_plan(self, value=None, *, reset_history=True):
            from authored_receiver_shadow import normalize_receiver_shadow_plan
            plan = normalize_receiver_shadow_plan(value); self.receiver_shadow_loading = True
            try:
                self.receiver_shadow_plan = plan; self.receiver_shadow_enabled.setChecked(plan['enabled'])
                while self.receiver_shadow_direction.count() > 4: self.receiver_shadow_direction.removeItem(4)
                index = self.receiver_shadow_direction.findData(plan['direction_to_light_xy'])
                if index < 0:
                    self.receiver_shadow_direction.addItem('已保存的主光方向', plan['direction_to_light_xy']); index = 4
                self.receiver_shadow_direction.setCurrentIndex(index)
                for key, spin in self.receiver_shadow_spins.items(): spin.setValue(plan[key])
            finally: self.receiver_shadow_loading = False
            if reset_history:
                self.receiver_history = [self.receiver_edit_snapshot()]; self.receiver_history_index = 0
            self.update_receiver_geometry_actions()

        def edit_receiver_shadow_plan(self, key, value):
            if self.receiver_shadow_loading or self._session_loading or not self.source_run_id or self.render_pending: return
            from authored_receiver_shadow import normalize_receiver_shadow_plan
            plan = normalize_receiver_shadow_plan({**self.receiver_shadow_plan, key: value})
            if plan == self.receiver_shadow_plan: return
            self.receiver_shadow_plan = plan
            self.receiver_history = self.receiver_history[:self.receiver_history_index + 1] + [self.receiver_edit_snapshot()]
            self.receiver_history_index = len(self.receiver_history) - 1
            self.update_receiver_geometry_actions(); self._checkpoint_session()

        def update_receiver_shadow_actions(self):
            if not hasattr(self, 'receiver_shadow_enabled'): return
            from design_reconstruction import has_native_foreground
            from scene_receiver_intent import GROUNDED_FEET
            ready = bool(self.source_run_id and self.design_reconstruction_plan) and has_native_foreground(self.design_reconstruction_plan) and not self.render_pending
            for widget in (self.receiver_shadow_enabled, self.receiver_shadow_direction, *self.receiver_shadow_spins.values()): widget.setEnabled(ready)
            missing = []
            if not self.receiver_intent['receiver_region']: missing.append('接地范围')
            if not self.receiver_intent['receiver_plane']: missing.append('四角平面')
            if self.receiver_intent['ground_type'] == 'unknown': missing.append('地面类型')
            feet = GROUNDED_FEET[self.receiver_intent['foot_stance']]
            if not feet: missing.append('着地关系与脚落点')
            elif any(side not in self.receiver_intent['foot_anchors'] for side in feet): missing.append('对应脚落点')
            if self.receiver_shadow_plan['direction_to_light_xy'] is None: missing.append('明确的主光方向')
            state = '未开启。' if not self.receiver_shadow_plan['enabled'] else '已选择；将在预览/保存时检查资产并生成。'
            result = self.receiver_shadow_result
            current = (result and result['snapshot'] == self.receiver_edit_snapshot() and
                result['matte_version'] == self.native_matte_version and result['matte_strokes'] == self.native_matte_strokes)
            if current:
                execution = result['execution']; status = execution.get('status')
                if status == 'ready': state = '已选择。' + result['origin'] + '已生成艺术投影。'
                elif status == 'partial':
                    reasons = {'artist_light_direction_unspecified': '主光方向未指定',
                        'grounded_receiver_plane_and_foot_anchors_incomplete': '地面、平面或对应落脚信息未补齐',
                        'receiver_shadow_inputs_unqualified': '原片、环境或脚部观察未满足本次生成条件'}
                    state = '已选择。' + result['origin'] + '未生成：' + reasons.get(execution.get('reason'), '本次处理尚未完成') + '。'
                elif status == 'disabled': state = '未开启，' + result['origin'] + '未生成艺术投影。'
            if missing: state += '尚需' + '、'.join(missing) + '；缺失时跳过此投影，仍可普通编辑和保存。'
            else: state += '可继续普通编辑和保存。'
            self.receiver_shadow_state.setText(state + '平面与长度是艺术选择，不是地面测量。')

        def load_receiver_shadow_result(self, execution, *, snapshot, matte_version, origin, matte_strokes=None):
            self.receiver_shadow_result = ({'execution': json.loads(json.dumps(execution)),
                'snapshot': json.loads(json.dumps(snapshot)), 'matte_version': matte_version,
                'matte_strokes': json.loads(json.dumps(self.native_matte_strokes if matte_strokes is None else matte_strokes)), 'origin': origin}
                if isinstance(execution, dict) else None)
            self.update_receiver_shadow_actions()

        def design_layer(self):
            layers = (self.design_reconstruction_plan or {}).get("layers", []); index = self.design_stack_list.currentRow()
            return layers[index] if 0 <= index < len(layers) else None

        @staticmethod
        def _unsupported_design_layer(layer):
            from design_reconstruction import SUPPORTED_NATIVE_LAYER_KINDS
            return bool(layer) and layer.get('kind') not in SUPPORTED_NATIVE_LAYER_KINDS

        @staticmethod
        def _automatic_contact_available(layer):
            if (layer or {}).get('kind') != 'contact_shadow':
                return False
            from design_scene_assets import validate_contact_receiver_contract
            try:
                result = validate_contact_receiver_contract(layer)
            except (ValueError, TypeError, KeyError):
                return False
            return (result.get('valid') is True
                    and result.get('qualified_for_automatic_contact') is True)

        def update_design_reconstruction_actions(self):
            available = bool(self.source_run_id) and not self.render_pending
            self.receiver_stance.setEnabled(available); self.receiver_ground.setEnabled(available)
            candidate = bool(self.previews.get("reference") and self.previews["reference"].path)
            self.design_reconstruct_button.setEnabled(available and candidate and self.design_reconstruction_request is None)
            self.boundary_light_coordination_button.setEnabled(available and candidate and self.design_reconstruction_request is None
                and self._boundary_light_coordination_available())
            kinds = [row.get('kind') for row in (self.design_reconstruction_plan or {}).get('layers', [])]
            self.hair_boundary_light_button.setText('重新准备参考发梢受光' if 'source_boundary_light' in kinds else '补充参考发梢受光')
            self.hair_boundary_light_button.setToolTip('当前已有肩袖受光，建议使用“同时更新边界受光”；单独更新发梢后需重新协调肩袖，再预览并保存。'
                if 'source_cloth_boundary_light' in kinds else '基于已准备的人物外观补充有依据的发梢受光；可取消，完成后预览并保存新版本。')
            self.cloth_boundary_light_button.setText('重新协调肩袖暖光' if 'source_cloth_boundary_light' in kinds else '补充参考肩袖受光')
            self.hair_boundary_light_button.setEnabled(available and candidate and self.design_reconstruction_request is None
                and self._hair_boundary_preparation_available())
            self.cloth_boundary_light_button.setEnabled(available and candidate and self.design_reconstruction_request is None
                and self._cloth_boundary_preparation_available())
            self.design_stack_enabled.setEnabled(available and self.design_reconstruction_plan is not None)
            selected = available and self.design_layer() is not None
            retired = self._unsupported_design_layer(self.design_layer())
            generated_shadow = (self.design_layer() or {}).get('kind') == 'authored_receiver_shadow'
            contact_unavailable = ((self.design_layer() or {}).get('kind') == 'contact_shadow'
                                   and not self._automatic_contact_available(self.design_layer()))
            self.design_layer_enabled.setEnabled(selected and not retired and not generated_shadow and not contact_unavailable); self.design_layer_strength.setEnabled(selected and not retired and not generated_shadow and not contact_unavailable)
            controlled = self.design_layer() or {}
            v2 = (controlled.get('kind') == 'reference_appearance_stack'
                  and (controlled.get('params') or {}).get('schema') == 'reference_appearance_stack.v2'
                  and all(controlled.get('assets', {}).get(key) for key in ('colour_delta', 'lightness_delta', 'control_receipt')))
            for spin in self.reference_control_spins.values():
                spin.setEnabled(selected and v2 and not retired)
                spin.setToolTip('按已保存的原生颜色/明暗分解调整；整体强度仍控制总效果。' if v2 else '该版本契约不受支持；请使用当前完整仿色融合流程。')
            self.update_receiver_overlay()

        def load_design_reconstruction(self, plan):
            self.design_reconstruction_loading = True
            self.design_reconstruction_plan = json.loads(json.dumps(plan)) if isinstance(plan, dict) else None
            self.design_stack_enabled.setChecked(bool((self.design_reconstruction_plan or {}).get("enabled", True)) if self.design_reconstruction_plan else False)
            self.design_stack_list.clear()
            for layer in (self.design_reconstruction_plan or {}).get("layers", []): self.design_stack_list.addItem(str(layer.get("label") or layer.get("id") or "设计图层"))
            self.design_stack_list.setCurrentRow(0 if self.design_stack_list.count() else -1)
            summary = str((self.design_reconstruction_plan or {}).get("summary") or "")
            required = self._paired_subject_state()['reconstruction_required']
            self.design_stack_summary.setText('已保存画面可查看；本地图层需更新，请点击准备仿色融合（不生图）。' if required else "设计图层已载入，按需调整强度与开关；整体效果仍需审片。" if self.design_reconstruction_plan else "选择原图与参考后，准备人物仿色和背景融合。")
            self.design_stack_summary.setToolTip(summary)
            self.design_reconstruction_loading = False; self.select_design_layer(self.design_stack_list.currentRow()); self.update_design_reconstruction_actions()

        def select_design_layer(self, _index):
            self.design_reconstruction_loading = True; layer = self.design_layer()
            self.design_layer_enabled.setChecked(bool((layer or {}).get("enabled", True)) if layer else False)
            self.design_layer_strength.setValue(float((layer or {}).get("strength", 1)))
            reference_stack = (layer or {}).get('kind') == 'reference_appearance_stack'
            self.design_layer_strength_label.setText('整体仿色融合强度' if reference_stack else '图层强度')
            self.design_layer_strength.setToolTip('0 为当前源人物底色，1 为本次完整目标；不撤销人工笔刷。' if reference_stack else '调整当前图层强度。')
            for key, spin in self.reference_control_spins.items(): spin.setValue(float((layer or {}).get(key, 1)))
            if reference_stack:
                from source_appearance_responsibility import reference_stack_state
                state = reference_stack_state({'source': {'sha256': getattr(self, 'source_run_source_sha256', None)}, 'reference': {'sha256': getattr(self, 'source_run_candidate_sha256', None)}}, self.design_reconstruction_plan)
                support = state.get('reference_support') or {}
                lines = []; labels = {'geometry': '结构', 'material': '材质', 'statistics': '统计', 'spatial_colour': '空间颜色', 'lightness': '明暗', 'donor': '边缘供体'}
                region_labels = {'face':'面部','left_arm':'左臂','right_arm':'右臂','left_hand':'左手','right_hand':'右手',
                    'legs':'腿部','hair':'头发','clothing':'服装','shoes':'鞋','bag':'包','lips':'嘴唇'}
                for region, row in (support.get('regions') or {}).items():
                    detail = []
                    for key, title in labels.items():
                        value = row.get(key) or {}; status = value.get('status', 'unknown')
                        detail.append(title + '：' + {'supported': '支持', 'partial': '部分支持', 'unsupported': '降级', 'unknown': '未核实'}.get(status, status) + ('（' + str(value['reason']) + '）' if value.get('reason') else ''))
                    lines.append(region_labels.get(region, str(region)) + '\n' + '；'.join(detail))
                missing_message = ('逐区域报告未通过来源与完整性校验；支持程度待核实，需重新准备并审片。'
                    if (layer.get('params') or {}).get('reference_support') else
                    '此保存版本没有逐区域兼容性报告；支持程度待核实，仍需整体与原尺寸审片。')
                self.reference_support_text.setPlainText('\n\n'.join(lines) or missing_message)
            else: self.reference_support_text.clear()
            explanations = {
                'reference_appearance_stack': '整体仿色融合：0 为当前源人物底色，1 为本次准备的完整目标。原片人物结构与纹理保留；调到 0 不撤销人工笔刷。此强度统一调整本次结果，不独立修改仿色、材质融合或场景协调。调整后预览审片，再保存高清新版本。',
                'subject_illumination': '历史人物光色图层：准备仿色融合后再编辑新的整体结果；已保存画面保留。',
                'skin_appearance': '肤色协调：调整皮肤外观，保留当前光照的明暗。',
                'subject_appearance': '材质颜色协调：调整服装、头发与鞋的颜色，保留当前明暗。',
                'trusted_source_reflection': '历史镜面人物片段：已停止执行，仅保留保存图像与来源记录，不能启用。',
                'captured_floor_reflection': '历史原片湿地倒影：已停止执行，仅保留保存图像与来源记录，不能启用。',
                'contact_shadow': '鞋底接触增强：仅在原片鞋底与目标地面共同支持的范围自动生效，可关闭或调整强度。此层提供局部接触衰减；投影阴影和湿地反射仍待处理。',
                'authored_receiver_shadow': '地面艺术投影：通过地面艺术投影控件调整，预览/保存自动重建。此生成层的开关和强度由专用控件负责。',
                'rim_light': '额外轮廓光：手动增加亮边；与自动发丝受光分开。',
                'source_boundary_light': '参考发梢受光：仅在有依据的原片头发范围补充参考受光，可关闭或调整强度。调整后预览，并查看原尺寸发丝细节再保存。',
                'source_cloth_boundary_light': '参考肩袖受光：仅在有依据的原片衣料范围补充参考受光，可关闭或调整强度。发梢受光或前序外观调整后需重新准备；查看原尺寸衣料细节再保存。',
                'bloom': '额外柔光：为当前成片亮区增加光学散射。'}
            if self._unsupported_design_layer(layer):
                self.design_layer_help.setText('此图层已退出当前处理路径，仅可查看保存画面与来源记录。准备仿色融合后再预览或保存新效果。')
            else:
                help_text=explanations.get((layer or {}).get('kind'), '调整此层后点击预览；保存会创建新的高清版本。')
                if reference_stack and (layer.get('params') or {}).get('schema') == 'reference_appearance_stack.v2':
                    help_text += ' 颜色与明暗控件分别调整已保存目标的分量，整体强度继续控制总效果；不重新运行仿色。'
                if layer and layer.get('assets',{}).get('field',{}).get('contract_schema')=='source_coordinate_associated_response.v1':
                    help_text='历史人物颜色方案：可查看已保存画面；准备仿色融合后再调整。头发、鞋和透明材料仍需原尺寸审片。'
                if layer and layer.get('kind')=='contact_shadow':
                    help_text += ' 若已开启的艺术投影本次成功生成，将由它接管接收面衰减；关闭艺术投影恢复本层处理。'
                    qualification=((layer.get('params') or {}).get('receiver_qualification') or {})
                    if self._automatic_contact_available(layer):
                        help_text+=' 当前有 %s 个接收面支持像素；实际接地效果请查看鞋底原尺寸对照。' % qualification.get('joint_support_pixels_near_contact',0)
                    else:
                        help_text+=' 当前缺少有效的原片归属或地面支持，不能启用。'
                self.design_layer_help.setText(help_text)
            self.design_reconstruction_loading = False; self.update_design_reconstruction_actions()

        def design_reconstruction_changed(self):
            if self.workflow_graph:
                from cos_workflow_graph import build_graph
                draft = self._cos_draft_run(); draft['options']['appearance_revision'].pop('workflow_graph', None)
                for node in build_graph(draft)['nodes']:
                    if node['prepared'] and node['id'] != 'beauty' and node['editable']:
                        self.workflow_graph['nodes'][node['id']] = node['parameters']
            self.source_light_version += 1; self.source_light_compare.setChecked(False)
            self.source_light_state.setText("设计图层已修改，正在准备成片预览…"); self.update_source_light_actions(); self.source_light_timer.start()

        def edit_reference_control(self, key, value):
            if self.design_reconstruction_loading: return
            layer = self.design_layer()
            if not layer or layer.get('kind') != 'reference_appearance_stack' or not self.reference_control_spins[key].isEnabled(): return
            layer[key] = float(value); self.design_reconstruction_changed()

        def toggle_design_reconstruction(self, checked):
            if self.design_reconstruction_loading or self.design_reconstruction_plan is None: return
            requested = json.loads(json.dumps(self.design_reconstruction_plan)); requested['enabled'] = checked
            if self._paired_subject_state(requested)['reconstruction_required']:
                self.design_reconstruction_loading = True
                self.design_stack_enabled.setChecked(bool(self.design_reconstruction_plan.get('enabled')))
                self.design_reconstruction_loading = False
                self.source_light_state.setText('启用人物仿色融合前需点击准备仿色融合；已保存画面保留。'); return
            self.design_reconstruction_plan["enabled"] = checked; self.design_reconstruction_changed()

        def edit_design_layer(self, _value=None):
            if self.design_reconstruction_loading: return
            layer = self.design_layer()
            if layer and layer.get('kind') == 'authored_receiver_shadow': return
            if layer is None or self._unsupported_design_layer(layer):
                if layer is not None and self._unsupported_design_layer(layer):
                    self.design_layer_enabled.setChecked(False)
                return
            if layer.get('kind') == 'subject_illumination':
                requested = json.loads(json.dumps(self.design_reconstruction_plan))
                proposed = next(item for item in requested['layers'] if item['id'] == layer['id'])
                proposed.update(enabled=self.design_layer_enabled.isChecked(), strength=self.design_layer_strength.value())
                if self._paired_subject_state(requested)['reconstruction_required']:
                    self.select_design_layer(self.design_stack_list.currentRow())
                    self.source_light_state.setText('启用人物仿色融合前需点击准备仿色融合；已有画面与参数保留。'); return
            if layer.get('kind')=='contact_shadow':
                if not self._automatic_contact_available(layer):
                    layer['enabled']=False
                    self.design_layer_enabled.setChecked(False)
                    return
            layer["enabled"] = self.design_layer_enabled.isChecked(); layer["strength"] = self.design_layer_strength.value(); self.design_reconstruction_changed()

        def _hair_boundary_preparation_available(self):
            return self._boundary_preparation_available('source_boundary_light')

        def _cloth_boundary_preparation_available(self):
            return self._boundary_preparation_available('source_cloth_boundary_light')

        def _boundary_light_coordination_available(self):
            return self._hair_boundary_preparation_available() and self._cloth_boundary_preparation_available()

        def _boundary_preparation_available(self, kind):
            from design_reconstruction import validate_current_native_plan
            plan = self.design_reconstruction_plan or {}
            try: validate_current_native_plan(plan)
            except (KeyError, TypeError, ValueError, StopIteration): return False
            appearance = [row for row in plan.get('layers', []) if row.get('kind') == 'reference_appearance_stack']
            return bool(len(appearance) == 1 and (appearance[0].get('params') or {}).get('schema') == 'reference_appearance_stack.v2'
                and sum(row.get('kind') == kind for row in plan.get('layers', [])) <= 1)

        def _reconstruction_name(self):
            return {'hair_boundary_light': '发梢受光', 'cloth_boundary_light': '肩袖受光',
                'boundary_light_coordination': '发梢与肩袖受光'}.get(self.native_reconstruction_preparation_kind, '仿色融合')

        def prepare_boundary_light_coordination(self, _checked=False):
            self.reconstruct_design(preparation_kind='boundary_light_coordination')

        def prepare_hair_boundary_light(self, _checked=False):
            self.reconstruct_design(preparation_kind='hair_boundary_light')

        def prepare_cloth_boundary_light(self, _checked=False):
            self.reconstruct_design(preparation_kind='cloth_boundary_light')

        def reconstruct_design(self, _checked=False, *, preparation_kind='reference_appearance'):
            if not self.source_run_id or self.render_pending: return
            if preparation_kind == 'hair_boundary_light' and not self._hair_boundary_preparation_available():
                self.source_light_state.setText('请先准备当前人物仿色融合；发梢受光图层需保持唯一。'); return
            if preparation_kind == 'cloth_boundary_light' and not self._cloth_boundary_preparation_available():
                self.source_light_state.setText('请先准备当前人物仿色融合；肩袖受光图层需保持唯一。'); return
            if preparation_kind == 'boundary_light_coordination' and not self._boundary_light_coordination_available():
                self.source_light_state.setText('请先准备当前人物仿色融合；发梢和肩袖受光图层各需保持唯一。'); return
            self.native_reconstruction_preparation_kind = preparation_kind
            name = self._reconstruction_name()
            if self.native_matte_request is not None or (self.native_matte_strokes
                    and self.native_matte_preview_version != self.native_matte_version):
                self.design_reconstruction_after_matte = (self.source_run_id, self.native_matte_version)
                self.design_reconstruction_after_matte_kind = preparation_kind
                self.source_light_state.setText(f'正在准备当前人物边缘，完成后继续准备{name}。')
                self.preview_native_matte(); return
            self._checkpoint_session()
            if self._session_checkpoint_failed: return
            try: payload = self._source_edit_payload()
            except (TypeError, ValueError) as error:
                self.source_light_state.setText(f'精修参数无法读取，尚未提交{name}准备：' + str(error)); return
            if preparation_kind in ('hair_boundary_light', 'cloth_boundary_light', 'boundary_light_coordination'): payload['preparation_kind'] = preparation_kind
            self.source_light_timer.stop(); self.native_matte_timer.stop()
            self.design_reconstruction_request = 'submitting'
            self.source_light_state.setText(f'正在提交{name}准备；可在任务中心等待或取消，不重新生图。')
            self.local_editor_tabs.setCurrentIndex(2); self.layer_tabs.setCurrentWidget(self.previews['final'])
            self.task('POST', f'/api/source-runs/{self.source_run_id}/reconstruct-design', payload, 'design-reconstruct-job')

        def handle_reconstruction_job(self, value):
            """Restore a prepared native plan only into its exact edit snapshot."""
            from session_draft import digest
            job = value.get('job') or {}
            if job.get('kind') != 'native_design_reconstruction': return False
            if job.get('job_id') != self.active_job_id: return True
            job_id = job['job_id']; status = job.get('status')
            self.native_reconstruction_preparation_kind = job.get('preparation_kind', 'reference_appearance')
            name = self._reconstruction_name()
            self.native_reconstruction_job_id = job_id
            self.job_details.setPlainText(json.dumps(job, ensure_ascii=False, indent=2))
            if hasattr(self, 'simple_workspace'):
                self._show_quick_job_progress(job)
            if status in {'queued', 'running', 'cancelling'}:
                self.render_pending = True; self.design_reconstruction_request = job_id
                self.operation_state.setText(str(job.get('message') or f'正在准备{name}；任务中心可取消。'))
                self.update_accept_enabled()
                self._schedule_job_poll(750)
                return True
            self.render_pending = False; self.design_reconstruction_request = None
            self.update_accept_enabled()
            if self.native_reconstruction_result_seen == job_id: return True
            if status != 'succeeded':
                self.simple_creation = None
                message = (f'{name}准备已取消；当前编辑与已保存画面保留。' if status == 'cancelled'
                    else f'{name}准备未完成；当前编辑保留，可重新提交本地准备。')
                self.operation_state.setText(message); self.source_light_state.setText(message)
                if job.get('error'): self.log_text('[native reconstruction] ' + str(job['error']))
                return True
            receipt = value.get('reconstruction_result') or {}
            if (receipt.get('run_id') != self.source_run_id
                    or receipt.get('source_sha256') != self.source_run_source_sha256
                    or receipt.get('candidate_sha256') != self.source_run_candidate_sha256):
                self.simple_creation = None
                self.operation_state.setText('本地准备结果已保留；先打开任务对应版本，再在任务中心选择此任务。')
                return True
            try: current_sha = digest(self._source_edit_payload())
            except (TypeError, ValueError): current_sha = None
            if receipt.get('input_sha256') != current_sha:
                self.simple_creation = None
                self.operation_state.setText(f'准备结果已保留；当前精修已有变化，请按当前参数重试{name}准备。')
                return True
            plan = receipt.get('design_reconstruction_plan')
            if not isinstance(plan, dict):
                self.simple_creation = None
                self.operation_state.setText(f'本地任务缺少准备结果，请重试{name}准备。'); return True
            self.native_reconstruction_result_seen = job_id
            if job.get('preparation_kind') in ('hair_boundary_light', 'cloth_boundary_light', 'boundary_light_coordination') and receipt.get('no_change') is True:
                message = (f'{name}没有新的可用改动；当前编辑与已保存原图保留。'
                    if job.get('preparation_kind') == 'boundary_light_coordination' else
                    f'{name}依据不足，未改变画面；当前编辑与已保存原图保留。')
                self.operation_state.setText(message); self.source_light_state.setText(message)
                # In the default Cos route, insufficient evidence for one
                # boundary segment is a local finding, not a reason to drop
                # the already prepared Beauty/fusion result. Save that result
                # and continue to the post stage with the missing segment
                # recorded by the job receipt.
                if (self.simple_creation and self.simple_creation.get('stage') == 'boundary'
                        and job.get('preparation_kind') == 'boundary_light_coordination'):
                    self.simple_creation['stage'] = 'post_save_wait'
                    self.operation_state.setText(message + '正在保存当前融合版本并继续后期。')
                    QTimer.singleShot(0, self.save_source_lights)
                return True
            self.load_design_reconstruction(plan); self.design_reconstruction_prepared = True
            self._apply_workflow_intent()
            self.local_editor_tabs.setCurrentIndex(2); self.layer_tabs.setCurrentWidget(self.previews['final'])
            self.source_light_version += 1; self._checkpoint_session()
            self.operation_state.setText(f'{name}已准备，待审片且尚未保存；预览后保存高清新版本。')
            automatic_handoff = bool(self.simple_creation and (
                (job.get('preparation_kind') == 'reference_appearance'
                 and self.simple_creation.get('stage') == 'colour') or
                (job.get('preparation_kind') == 'boundary_light_coordination'
                 and self.simple_creation.get('stage') == 'boundary')))
            self.update_source_light_actions()
            # The automatic route immediately submits the next local worker;
            # avoid starting a competing preview render for the same snapshot.
            # The final high-resolution save is the first preview of this
            # combined edit. Manual professional edits retain live preview.
            if not automatic_handoff:
                self.preview_source_lights()
            # The quick Cos route owns the next hand-off.  A selected
            # reference first prepares ordinary appearance fusion, then the
            # same edit snapshot is sent through coordinated boundary light,
            # and only then is the high-resolution revision saved. Professional
            # manual reconstruction keeps its previous behaviour.
            if (self.simple_creation and job.get('preparation_kind') == 'reference_appearance'
                    and self.simple_creation.get('stage') == 'colour'):
                self.simple_creation['stage'] = 'boundary'
                QTimer.singleShot(0, lambda: self.reconstruct_design(preparation_kind='boundary_light_coordination'))
            elif (self.simple_creation and job.get('preparation_kind') == 'boundary_light_coordination'
                    and self.simple_creation.get('stage') == 'boundary'):
                self.simple_creation['stage'] = 'post_save_wait'
                QTimer.singleShot(0, self.save_source_lights)
            else:
                self._advance_simple_creation('colour')
            return True

        def source_color_panel(self):
            panel = QWidget(); layout = QVBoxLayout(panel); layout.setContentsMargins(0, 6, 0, 0); layout.setSpacing(7)
            self.source_color_auto_button = QPushButton("本地自动纠偏"); self.source_color_auto_button.setObjectName("primaryButton"); layout.addWidget(self.source_color_auto_button)
            header = QHBoxLayout(); self.source_color_enabled = QCheckBox("启用")
            header.addStretch(); header.addWidget(self.source_color_enabled); layout.addLayout(header)
            self.source_color_list = QListWidget(); self.source_color_list.setFixedHeight(64); layout.addWidget(self.source_color_list)
            row = QHBoxLayout(); self.source_color_add = QPushButton("新建选区"); self.source_color_remove = QPushButton("删除"); self.source_color_overlay = QCheckBox("显示选区")
            row.addWidget(self.source_color_add); row.addWidget(self.source_color_remove); row.addWidget(self.source_color_overlay); layout.addLayout(row)
            brushes = QHBoxLayout(); self.source_color_tool = QComboBox(); self.source_color_tool.addItem("平移照片", "pan"); self.source_color_tool.addItem("添加选区", "add"); self.source_color_tool.addItem("擦除选区", "erase")
            self.source_color_undo = QPushButton("撤销"); self.source_color_redo = QPushButton("重做"); brushes.addWidget(self.source_color_tool, 1); brushes.addWidget(self.source_color_undo); brushes.addWidget(self.source_color_redo); layout.addLayout(brushes)
            brush_size = QHBoxLayout(); self.source_color_radius = QDoubleSpinBox(); self.source_color_radius.setRange(.002, .2); self.source_color_radius.setDecimals(3); self.source_color_radius.setSingleStep(.005); self.source_color_radius.setValue(.02)
            self.source_color_feather = QDoubleSpinBox(); self.source_color_feather.setRange(0, .25); self.source_color_feather.setDecimals(3); self.source_color_feather.setSingleStep(.005); self.source_color_feather.setKeyboardTracking(False)
            brush_size.addWidget(QLabel("画笔")); brush_size.addWidget(self.source_color_radius, 1); brush_size.addWidget(QLabel("羽化")); brush_size.addWidget(self.source_color_feather, 1); layout.addLayout(brush_size)
            self.source_color_form = QWidget(); form = QFormLayout(self.source_color_form); form.setContentsMargins(0, 0, 0, 0); form.setSpacing(7)
            self.source_color_target = QPushButton("目标色"); self.source_color_reference = QPushButton("原片参照色")
            form.addRow("纠偏目标", self.source_color_target)
            self.source_color_strength = QDoubleSpinBox(); self.source_color_strength.setRange(0, 1); self.source_color_strength.setDecimals(2); self.source_color_strength.setSingleStep(.05); self.source_color_strength.setKeyboardTracking(False)
            self.source_color_exposure = QDoubleSpinBox(); self.source_color_exposure.setRange(-2, 2); self.source_color_exposure.setDecimals(2); self.source_color_exposure.setSingleStep(.05); self.source_color_exposure.setKeyboardTracking(False)
            pair = QWidget(); pairs = QHBoxLayout(pair); pairs.setContentsMargins(0, 0, 0, 0); pairs.addWidget(self.source_color_strength, 1); pairs.addWidget(self.source_color_exposure, 1); form.addRow("强度 / EV", pair)
            self.source_color_guard = QCheckBox("保护原片其他颜色"); form.addRow(self.source_color_guard)
            form.addRow("原片参照", self.source_color_reference)
            self.source_color_tolerance = QDoubleSpinBox(); self.source_color_tolerance.setRange(.01, 1); self.source_color_tolerance.setDecimals(2); self.source_color_tolerance.setSingleStep(.05); self.source_color_tolerance.setKeyboardTracking(False); form.addRow("颜色容差", self.source_color_tolerance)
            self.source_color_base = QComboBox(); self.source_color_base.addItem("画笔选区", "empty"); self.source_color_base.addItem("椭圆 + 画笔", "ellipse"); self.source_color_base.addItem("自动语义选区", "mask"); form.addRow("基础选区", self.source_color_base)
            self.source_color_shape = {}
            for key, low, high, step in (("center_x", 0, 1, .01), ("center_y", 0, 1, .01), ("radius_x", .001, 1, .01), ("radius_y", .001, 1, .01), ("rotation_deg", -360, 360, 5)):
                spin = QDoubleSpinBox(); spin.setRange(low, high); spin.setDecimals(3 if key != "rotation_deg" else 1); spin.setSingleStep(step); spin.setKeyboardTracking(False); self.source_color_shape[key] = spin; spin.valueChanged.connect(self.edit_source_color)
            for title, keys in (("中心 X / Y", ("center_x", "center_y")), ("半径 X / Y", ("radius_x", "radius_y"))):
                pair = QWidget(); pairs = QHBoxLayout(pair); pairs.setContentsMargins(0, 0, 0, 0)
                for key in keys: pairs.addWidget(self.source_color_shape[key], 1)
                form.addRow(title, pair)
            form.addRow("旋转 °", self.source_color_shape["rotation_deg"])
            self.source_color_label = QLineEdit(); self.source_color_region_enabled = QCheckBox("此选区生效"); form.addRow("选区名称", self.source_color_label); form.addRow(self.source_color_region_enabled)
            layout.addWidget(self.source_color_form, 1)
            self.source_color_add.clicked.connect(self.add_source_color); self.source_color_remove.clicked.connect(self.remove_source_color)
            self.source_color_auto_button.clicked.connect(self.auto_source_colors)
            self.source_color_list.currentRowChanged.connect(self.select_source_color); self.source_color_enabled.toggled.connect(self.toggle_source_colors)
            self.source_color_undo.clicked.connect(lambda: self.restore_source_color_history(-1)); self.source_color_redo.clicked.connect(lambda: self.restore_source_color_history(1))
            self.source_color_overlay.toggled.connect(self.update_source_color_overlay); self.source_color_tool.currentIndexChanged.connect(self.update_source_color_overlay); self.source_color_radius.valueChanged.connect(self.update_source_color_overlay)
            self.source_color_label.textEdited.connect(self.edit_source_color); self.source_color_region_enabled.toggled.connect(self.edit_source_color); self.source_color_guard.toggled.connect(self.edit_source_color)
            self.source_color_base.currentIndexChanged.connect(self.edit_source_color)
            for spin in (self.source_color_feather, self.source_color_strength, self.source_color_exposure, self.source_color_tolerance): spin.valueChanged.connect(self.edit_source_color)
            self.source_color_target.clicked.connect(lambda: self.choose_source_color("target_rgb")); self.source_color_reference.clicked.connect(lambda: self.choose_source_color("reference_rgb"))
            for name in ("source", "final"): self.previews[name].colorStrokeFinished.connect(self.source_color_stroke)
            return panel

        def update_source_color_actions(self):
            ready = bool(self.source_run_id) and not self.render_pending
            selected = self.source_color_region() is not None
            self.source_color_add.setEnabled(ready)
            self.source_color_auto_button.setEnabled(ready and self.source_color_auto_request is None)
            self.source_color_enabled.setEnabled(ready and self.source_color_plan is not None)
            for widget in (self.source_color_remove, self.source_color_form, self.source_color_tool, self.source_color_radius, self.source_color_feather): widget.setEnabled(ready and selected)
            for widget in self.source_color_shape.values(): widget.setEnabled(ready and selected and self.source_color_base.currentData() == "ellipse")
            self.source_color_undo.setEnabled(ready and self.source_color_history_index > 0); self.source_color_redo.setEnabled(ready and self.source_color_history_index + 1 < len(self.source_color_history))
            self.update_source_color_overlay()

        def source_color_region(self):
            regions = (self.source_color_plan or {}).get("regions", []); index = self.source_color_list.currentRow()
            return regions[index] if 0 <= index < len(regions) else None

        def load_source_colors(self, plan, reset_history=True):
            self.source_color_loading = True; self.source_color_plan = json.loads(json.dumps(plan)) if isinstance(plan, dict) else None
            self.source_color_enabled.setChecked(bool((self.source_color_plan or {}).get("enabled", True)) if self.source_color_plan else False)
            self.source_color_list.clear()
            for region in (self.source_color_plan or {}).get("regions", []): self.source_color_list.addItem(str(region.get("label") or region.get("id") or "局部纠偏"))
            self.source_color_list.setCurrentRow(0 if self.source_color_list.count() else -1); self.source_color_loading = False
            if reset_history:
                self.source_color_history = [self.source_color_snapshot()]; self.source_color_history_index = 0
            self.select_source_color(self.source_color_list.currentRow()); self.update_source_color_actions()

        def set_source_color_swatch(self, button, rgb, label):
            color = QColor.fromRgbF(*rgb); button.setText(f"{label} {color.name().upper()}"); button.setStyleSheet(f"border-left: 12px solid {color.name()};")

        def select_source_color(self, _index):
            region = self.source_color_region(); self.source_color_loading = True
            if region:
                selection = region.get("selection") or {}; correction = region.get("correction") or {}; guard = region.get("color_guard") or {}
                self.source_color_label.setText(str(region.get("label") or "局部纠偏")); self.source_color_region_enabled.setChecked(region.get("enabled", True)); self.source_color_guard.setChecked(guard.get("enabled", True))
                self.source_color_strength.setValue(float(correction.get("strength", .5))); self.source_color_exposure.setValue(float(correction.get("exposure_ev", 0))); self.source_color_tolerance.setValue(float(guard.get("tolerance", .25)))
                self.source_color_feather.setValue(float(selection.get("feather", .01))); self.source_color_base.setCurrentIndex(max(0, self.source_color_base.findData(selection.get("base", "ellipse"))))
                self.source_color_base.model().item(self.source_color_base.findData("mask")).setEnabled(bool(selection.get("mask")))
                center = selection.get("center", [.5, .35]); radius = selection.get("radius", [.1, .1]); geometry = {"center_x": center[0], "center_y": center[1], "radius_x": radius[0], "radius_y": radius[1], "rotation_deg": selection.get("rotation_deg", 0)}
                for key, spin in self.source_color_shape.items(): spin.setValue(float(geometry[key]))
                self.set_source_color_swatch(self.source_color_target, correction.get("target_rgb", [.65, .45, .38]), "目标")
                self.set_source_color_swatch(self.source_color_reference, guard.get("reference_rgb", [.5, .35, .25]), "原片")
            self.source_color_loading = False; self.update_source_color_actions(); self.update_source_color_overlay()

        def edit_source_color(self, _value=None):
            if self.source_color_loading: return
            region = self.source_color_region()
            if region is None: return
            region["label"] = self.source_color_label.text().strip() or "局部纠偏"; region["enabled"] = self.source_color_region_enabled.isChecked()
            selection = region.setdefault("selection", {}); selection.update(base=self.source_color_base.currentData(), feather=self.source_color_feather.value(),
                center=[self.source_color_shape[key].value() for key in ("center_x", "center_y")], radius=[self.source_color_shape[key].value() for key in ("radius_x", "radius_y")], rotation_deg=self.source_color_shape["rotation_deg"].value())
            region.setdefault("correction", {}).update(strength=self.source_color_strength.value(), exposure_ev=self.source_color_exposure.value())
            region.setdefault("color_guard", {}).update(enabled=self.source_color_guard.isChecked(), tolerance=self.source_color_tolerance.value())
            self.source_color_list.currentItem().setText(region["label"]); self.source_colors_changed()

        def choose_source_color(self, key):
            region = self.source_color_region()
            if region is None: return
            node = region["correction"] if key == "target_rgb" else region["color_guard"]
            color = QColorDialog.getColor(QColor.fromRgbF(*node[key]), self, "选择自然纠偏目标色" if key == "target_rgb" else "选择原片颜色保护参照")
            if color.isValid():
                node[key] = [round(value, 5) for value in (color.redF(), color.greenF(), color.blueF())]
                self.select_source_color(self.source_color_list.currentRow()); self.source_colors_changed()

        def toggle_source_colors(self, checked):
            if self.source_color_loading or self.source_color_plan is None: return
            self.source_color_plan["enabled"] = checked; self.source_colors_changed()

        def add_source_color(self):
            if not self.source_run_id or self.render_pending: return
            plan = self.source_color_plan or {"schema_version": "source_color_plan.v1", "enabled": True, "summary": "手动局部肤色纠偏", "regions": []}
            regions = plan.setdefault("regions", []); regions.append({"id": "manual-" + uuid.uuid4().hex[:10], "label": f"纠偏选区 {len(regions) + 1}", "enabled": True,
                "selection": {"base": "empty", "center": [.5, .35], "radius": [.1, .1], "rotation_deg": 0, "feather": .01, "strokes": []},
                "correction": {"target_rgb": [.65, .45, .38], "strength": .5, "exposure_ev": 0}, "color_guard": {"enabled": True, "reference_rgb": [.5, .35, .25], "tolerance": .25}})
            self.load_source_colors(plan, reset_history=False); self.source_color_list.setCurrentRow(len(regions) - 1)
            self.local_editor_tabs.setCurrentIndex(1); self.source_color_tool.setCurrentIndex(1); self.source_color_overlay.setChecked(True)
            self.layer_tabs.setCurrentWidget(self.previews["final"]); self.source_colors_changed()

        def remove_source_color(self):
            index = self.source_color_list.currentRow()
            if self.source_color_plan is None or index < 0: return
            plan = self.source_color_plan; del plan["regions"][index]; self.load_source_colors(plan, reset_history=False)
            self.source_color_list.setCurrentRow(min(index, self.source_color_list.count() - 1)); self.source_colors_changed()

        def source_color_stroke(self, stroke):
            region = self.source_color_region()
            if region is None or self.render_pending: return
            region.setdefault("selection", {}).setdefault("strokes", []).append(stroke); self.source_colors_changed()

        def source_colors_changed(self, record_history=True):
            if self.source_color_origin in {"model", "model_with_user_edits"}: self.source_color_origin = "model_with_user_edits"
            elif self.source_color_origin in {"local_automatic", "local_automatic_with_user_edits"}: self.source_color_origin = "local_automatic_with_user_edits"
            if record_history:
                self.source_color_history = self.source_color_history[:self.source_color_history_index + 1]
                self.source_color_history.append(self.source_color_snapshot()); self.source_color_history_index += 1
            self.source_light_version += 1; self.source_light_state.setText("局部纠偏已修改，正在准备预览…")
            self.source_light_compare.setChecked(False); self.update_source_color_actions(); self.update_source_color_overlay(); self.source_light_timer.start()

        def restore_source_color_history(self, direction):
            target = self.source_color_history_index + direction
            if not 0 <= target < len(self.source_color_history): return
            self.source_color_history_index = target; index = self.source_color_list.currentRow()
            snapshot = self.source_color_history[target]
            self.load_source_colors(snapshot["source_color_plan"], reset_history=False); self.source_color_list.setCurrentRow(min(index, self.source_color_list.count() - 1))
            self.source_colors_changed(record_history=False)

        def source_color_snapshot(self):
            return json.loads(json.dumps({"source_color_plan": self.source_color_plan}))

        def local_editor_changed(self, _index):
            self.update_source_light_overlay(); self.update_source_color_overlay();self.update_clone_overlay();self.update_local_brush_overlay();self.update_native_matte_overlay()
            self.update_receiver_overlay()
            if _index in (4,5):self.layer_tabs.setCurrentWidget(self.previews['final'])

        def tone_curve_panel(self):
            panel=QWidget();layout=QVBoxLayout(panel);layout.setContentsMargins(0,6,0,0)
            self.curve_enabled=QCheckBox('启用 RGB 曲线');self.curve_enabled.setChecked(True);layout.addWidget(self.curve_enabled)
            form=QFormLayout();self.curve_scope=QComboBox()
            for label,value in [('人物','foreground'),('背景','background'),('成片','composite')]:self.curve_scope.addItem(label,value)
            self.curve_channel=QComboBox();self.curve_channel.addItems(['RGB','R','G','B']);form.addRow('作用范围',self.curve_scope);form.addRow('通道',self.curve_channel);layout.addLayout(form)
            self.curve_scope_enabled=QCheckBox('此范围生效');self.curve_scope_enabled.setChecked(True);layout.addWidget(self.curve_scope_enabled)
            self.curve_canvas=ToneCurveCanvas();layout.addWidget(self.curve_canvas)
            note=QLabel('单击曲线添加控制点，拖动调整；右键删除控制点。RGB 调整明暗，R / G / B 调整通道颜色。人物边缘和高光余量保留。');note.setWordWrap(True);layout.addWidget(note)
            resets=QHBoxLayout();self.curve_reset=QPushButton('复位当前通道');self.curve_reset_all=QPushButton('复位全部');resets.addWidget(self.curve_reset);resets.addWidget(self.curve_reset_all);layout.addLayout(resets)
            history=QHBoxLayout();self.curve_undo=QPushButton('撤销');self.curve_redo=QPushButton('重做');history.addWidget(self.curve_undo);history.addWidget(self.curve_redo);layout.addLayout(history);layout.addStretch()
            self.curve_canvas.curveEdited.connect(self.edit_tone_curve);self.curve_scope.currentIndexChanged.connect(self.select_tone_curve);self.curve_channel.currentTextChanged.connect(self.select_tone_curve)
            self.curve_enabled.toggled.connect(lambda value:self.toggle_tone_curves(value,False));self.curve_scope_enabled.toggled.connect(lambda value:self.toggle_tone_curves(value,True))
            self.curve_reset.clicked.connect(lambda:self.reset_tone_curves(False));self.curve_reset_all.clicked.connect(lambda:self.reset_tone_curves(True));self.curve_undo.clicked.connect(lambda:self.restore_tone_curve_history(-1));self.curve_redo.clicked.connect(lambda:self.restore_tone_curve_history(1))
            return panel

        def tone_curve_data(self):
            from tone_curves import empty_tone_curves
            return self._manual_data()['layers'].get('tone_curves',empty_tone_curves())

        def write_tone_curves(self,layer,history=True):
            data=self._manual_data();data['layers']['tone_curves']=layer;self._write_manual_data(data)
            if history:
                snapshot=json.loads(json.dumps(layer))
                if self.curve_history_index>=0 and self.curve_history[self.curve_history_index]==snapshot:return
                self.curve_history=self.curve_history[:self.curve_history_index+1]+[snapshot];self.curve_history_index=len(self.curve_history)-1
            self.source_light_version+=1;self.source_light_compare.setChecked(False);self.update_source_light_actions();self.source_light_timer.start()

        def load_tone_curves(self):
            layer=self.tone_curve_data();self.curve_history=[json.loads(json.dumps(layer))];self.curve_history_index=0
            self.curve_loading=True;self.curve_enabled.setChecked(layer.get('enabled',True));self.curve_loading=False;self.select_tone_curve()

        def select_tone_curve(self,_value=None):
            if not hasattr(self,'manual_layers'):return
            layer=self.tone_curve_data();scope=layer['scopes'][self.curve_scope.currentData()]
            self.curve_canvas.set_curve(scope['channels'][self.curve_channel.currentText()],self.curve_channel.currentText())
            self.curve_loading=True;self.curve_scope_enabled.setChecked(scope.get('enabled',True));self.curve_loading=False;self.update_tone_curve_actions()

        def edit_tone_curve(self,points):
            if self.curve_loading or self.render_pending:return
            layer=self.tone_curve_data();layer['scopes'][self.curve_scope.currentData()]['channels'][self.curve_channel.currentText()]=points;self.write_tone_curves(layer)

        def toggle_tone_curves(self,enabled,scoped):
            if self.curve_loading:return
            layer=self.tone_curve_data()
            if scoped:layer['scopes'][self.curve_scope.currentData()]['enabled']=bool(enabled)
            else:layer['enabled']=bool(enabled)
            self.write_tone_curves(layer)

        def reset_tone_curves(self,all_channels):
            from tone_curves import empty_tone_curves
            if all_channels:layer=empty_tone_curves()
            else:
                layer=self.tone_curve_data();layer['scopes'][self.curve_scope.currentData()]['channels'][self.curve_channel.currentText()]=[[0.,0.],[1.,1.]]
            self.write_tone_curves(layer);self.curve_loading=True;self.curve_enabled.setChecked(layer['enabled']);self.curve_loading=False;self.select_tone_curve()

        def restore_tone_curve_history(self,direction):
            index=max(0,min(len(self.curve_history)-1,self.curve_history_index+direction))
            if index==self.curve_history_index or self.render_pending:return
            self.curve_history_index=index;layer=json.loads(json.dumps(self.curve_history[index]));self.write_tone_curves(layer,False)
            self.curve_loading=True;self.curve_enabled.setChecked(layer['enabled']);self.curve_loading=False;self.select_tone_curve()

        def update_tone_curve_actions(self):
            if not hasattr(self,'curve_canvas'):return
            ready=bool(self.source_run_id) and not self.render_pending
            for widget in (self.curve_canvas,self.curve_reset,self.curve_reset_all,self.curve_enabled,self.curve_scope_enabled):widget.setEnabled(ready)
            self.curve_undo.setEnabled(ready and self.curve_history_index>0);self.curve_redo.setEnabled(ready and self.curve_history_index+1<len(self.curve_history))

        def hsl_panel(self):
            panel=QWidget();layout=QVBoxLayout(panel);layout.setContentsMargins(0,6,0,0)
            self.hsl_enabled=QCheckBox('启用颜色 HSL');self.hsl_enabled.setChecked(True);layout.addWidget(self.hsl_enabled)
            form=QFormLayout();self.hsl_scope=QComboBox();self.hsl_family=QComboBox()
            for label,value in [('人物','foreground'),('背景','background'),('成片','composite')]:self.hsl_scope.addItem(label,value)
            for label,value in [('全部颜色','all'),('红色','red'),('橙色','orange'),('黄色','yellow'),('绿色','green'),('青色','aqua'),('蓝色','blue'),('紫色','purple'),('洋红','magenta')]:self.hsl_family.addItem(label,value)
            form.addRow('作用范围',self.hsl_scope);form.addRow('颜色分区',self.hsl_family)
            self.hsl_scope_enabled=QCheckBox('此范围生效');self.hsl_scope_enabled.setChecked(True);form.addRow(self.hsl_scope_enabled)
            self.hsl_sliders={};self.hsl_spins={}
            for key,label,extent,suffix in [('hue','色相',180,'°'),('saturation','饱和度',100,'%'),('lightness','明度',100,'%')]:
                row=QWidget();pair=QHBoxLayout(row);pair.setContentsMargins(0,0,0,0)
                slider=QSlider(Qt.Horizontal);slider.setRange(-extent,extent)
                spin=QSpinBox();spin.setRange(-extent,extent);spin.setSuffix(suffix);spin.setKeyboardTracking(False)
                pair.addWidget(slider,1);pair.addWidget(spin);form.addRow(label,row)
                self.hsl_sliders[key]=slider;self.hsl_spins[key]=spin
                slider.valueChanged.connect(lambda value,k=key:self.hsl_slider_value(k,value))
                slider.sliderReleased.connect(self.edit_hsl)
                spin.valueChanged.connect(lambda value,k=key:self.hsl_spin_value(k,value))
            layout.addLayout(form)
            note=QLabel('分别调整人物、背景或整张成片。颜色分区柔和衔接；近灰色区域由「全部颜色」控制。调整不改变人物轮廓或像素位置。');note.setWordWrap(True);layout.addWidget(note)
            resets=QHBoxLayout();self.hsl_reset=QPushButton('复位当前颜色');self.hsl_reset_all=QPushButton('复位全部');resets.addWidget(self.hsl_reset);resets.addWidget(self.hsl_reset_all);layout.addLayout(resets)
            history=QHBoxLayout();self.hsl_undo=QPushButton('撤销');self.hsl_redo=QPushButton('重做');history.addWidget(self.hsl_undo);history.addWidget(self.hsl_redo);layout.addLayout(history);layout.addStretch()
            self.hsl_scope.currentIndexChanged.connect(self.select_hsl);self.hsl_family.currentIndexChanged.connect(self.select_hsl)
            self.hsl_enabled.toggled.connect(lambda value:self.toggle_hsl(value,False));self.hsl_scope_enabled.toggled.connect(lambda value:self.toggle_hsl(value,True))
            self.hsl_reset.clicked.connect(lambda:self.reset_hsl(False));self.hsl_reset_all.clicked.connect(lambda:self.reset_hsl(True));self.hsl_undo.clicked.connect(lambda:self.restore_hsl_history(-1));self.hsl_redo.clicked.connect(lambda:self.restore_hsl_history(1))
            return panel

        def hsl_data(self):
            from hsl_adjust import empty_hsl_adjust
            return self._manual_data()['layers'].get('hsl_adjust',empty_hsl_adjust())

        def write_hsl(self,layer,history=True):
            data=self._manual_data();data['layers']['hsl_adjust']=layer;self._write_manual_data(data)
            if history:
                snapshot=json.loads(json.dumps(layer))
                if self.hsl_history_index>=0 and self.hsl_history[self.hsl_history_index]==snapshot:return
                self.hsl_history=self.hsl_history[:self.hsl_history_index+1]+[snapshot];self.hsl_history_index=len(self.hsl_history)-1
            self.source_light_version+=1;self.source_light_compare.setChecked(False);self.update_source_light_actions();self.source_light_timer.start()

        def load_hsl(self):
            layer=self.hsl_data();self.hsl_history=[json.loads(json.dumps(layer))];self.hsl_history_index=0
            self.hsl_loading=True;self.hsl_enabled.setChecked(layer.get('enabled',True));self.hsl_loading=False;self.select_hsl()

        def select_hsl(self,_value=None):
            if not hasattr(self,'manual_layers'):return
            scope=self.hsl_data()['scopes'][self.hsl_scope.currentData()];settings=scope['colours'][self.hsl_family.currentData()]
            self.hsl_loading=True;self.hsl_scope_enabled.setChecked(scope.get('enabled',True))
            for key in self.hsl_spins:self.hsl_spins[key].setValue(round(settings[key]));self.hsl_sliders[key].setValue(round(settings[key]))
            self.hsl_loading=False;self.update_hsl_actions()

        def hsl_slider_value(self,key,value):
            self.hsl_spins[key].blockSignals(True);self.hsl_spins[key].setValue(value);self.hsl_spins[key].blockSignals(False)
            if not self.hsl_loading and not self.hsl_sliders[key].isSliderDown():self.edit_hsl()

        def hsl_spin_value(self,key,value):
            self.hsl_sliders[key].blockSignals(True);self.hsl_sliders[key].setValue(value);self.hsl_sliders[key].blockSignals(False)
            self.edit_hsl()

        def edit_hsl(self):
            if self.hsl_loading or self.render_pending:return
            layer=self.hsl_data();layer['scopes'][self.hsl_scope.currentData()]['colours'][self.hsl_family.currentData()]={key:float(spin.value()) for key,spin in self.hsl_spins.items()};self.write_hsl(layer)

        def toggle_hsl(self,enabled,scoped):
            if self.hsl_loading or self.render_pending:return
            layer=self.hsl_data()
            if scoped:layer['scopes'][self.hsl_scope.currentData()]['enabled']=bool(enabled)
            else:layer['enabled']=bool(enabled)
            self.write_hsl(layer)

        def reset_hsl(self,all_colours):
            from hsl_adjust import empty_hsl_adjust
            if all_colours:layer=empty_hsl_adjust()
            else:
                layer=self.hsl_data();layer['scopes'][self.hsl_scope.currentData()]['colours'][self.hsl_family.currentData()]={'hue':0.,'saturation':0.,'lightness':0.}
            self.write_hsl(layer);self.hsl_loading=True;self.hsl_enabled.setChecked(layer['enabled']);self.hsl_loading=False;self.select_hsl()

        def restore_hsl_history(self,direction):
            index=max(0,min(len(self.hsl_history)-1,self.hsl_history_index+direction))
            if index==self.hsl_history_index or self.render_pending:return
            self.hsl_history_index=index;layer=json.loads(json.dumps(self.hsl_history[index]));self.write_hsl(layer,False)
            self.hsl_loading=True;self.hsl_enabled.setChecked(layer['enabled']);self.hsl_loading=False;self.select_hsl()

        def update_hsl_actions(self):
            if not hasattr(self,'hsl_spins'):return
            ready=bool(self.source_run_id) and not self.render_pending
            for widget in (*self.hsl_spins.values(),*self.hsl_sliders.values(),self.hsl_scope,self.hsl_family,self.hsl_reset,self.hsl_reset_all,self.hsl_enabled,self.hsl_scope_enabled):widget.setEnabled(ready)
            self.hsl_undo.setEnabled(ready and self.hsl_history_index>0);self.hsl_redo.setEnabled(ready and self.hsl_history_index+1<len(self.hsl_history))

        def native_matte_panel(self):
            panel=QWidget();layout=QVBoxLayout(panel);layout.setContentsMargins(0,6,0,0)
            note=QLabel('沿人物边缘画少量笔迹：绿色保留人物，红色去除背景。结果自动更新；保存会创建高清子版本。');note.setWordWrap(True);layout.addWidget(note)
            form=QFormLayout();self.native_matte_tool=QComboBox()
            for name,mode in [('保留人物','foreground'),('去除背景','background'),('平移 / 查看细节','pan')]:self.native_matte_tool.addItem(name,mode)
            form.addRow('画布工具',self.native_matte_tool)
            self.native_matte_radius=QDoubleSpinBox();self.native_matte_radius.setRange(1,120);self.native_matte_radius.setValue(12);self.native_matte_radius.setSuffix(' px');self.native_matte_radius.setKeyboardTracking(False);form.addRow('原生半径',self.native_matte_radius);layout.addLayout(form)
            row=QHBoxLayout();self.native_matte_undo=QPushButton('撤销新笔迹');self.native_matte_redo=QPushButton('重做');row.addWidget(self.native_matte_undo);row.addWidget(self.native_matte_redo);layout.addLayout(row)
            self.native_matte_reset=QPushButton('清除未保存笔迹');layout.addWidget(self.native_matte_reset)
            self.native_matte_detail=QPushButton('查看修正区域原尺寸');self.native_matte_detail.clicked.connect(self.show_native_matte_detail);layout.addWidget(self.native_matte_detail)
            self.native_matte_state=QLabel('选择高清原生工程后修正人物边缘。');self.native_matte_state.setWordWrap(True);layout.addWidget(self.native_matte_state);layout.addStretch()
            self.native_matte_tool.currentIndexChanged.connect(self.update_native_matte_overlay);self.native_matte_radius.valueChanged.connect(self.update_native_matte_overlay)
            self.native_matte_undo.clicked.connect(lambda:self.restore_native_matte_history(-1));self.native_matte_redo.clicked.connect(lambda:self.restore_native_matte_history(1));self.native_matte_reset.clicked.connect(self.reset_native_matte)
            self.previews['final'].localColourStrokeFinished.connect(self.add_native_matte_stroke)
            return panel

        def load_native_matte(self):
            self.native_matte_timer.stop();self.native_matte_request=None;self.native_matte_version+=1
            self.native_matte_base_plan=json.loads(json.dumps(self.design_reconstruction_plan))
            self.native_matte_strokes=[];self.native_matte_history=[[]];self.native_matte_history_index=0;self.native_matte_cache_key=None;self.native_matte_preview_version=-1;self.native_matte_result=None
            self.native_matte_state.setText('已保存的人物边缘作为当前基线；新笔迹可以撤销或保存到子版本。')
            self.update_native_matte_overlay()

        def update_native_matte_overlay(self,_value=None):
            if not hasattr(self,'native_matte_tool'):return
            from design_reconstruction import has_native_foreground
            ready=bool(self.source_run_id and self.design_reconstruction_plan) and has_native_foreground(self.design_reconstruction_plan) and not self.render_pending
            active=ready and self.local_editor_tabs.currentIndex()==7 and self.native_matte_tool.currentData()!='pan'
            view=self.previews['final'];view.matte_strokes=json.loads(json.dumps(self.native_matte_strokes)) if self.local_editor_tabs.currentIndex()==7 else []
            if self.local_editor_tabs.currentIndex()==7:
                view.local_brush_active=active;view.local_brush_mode=self.native_matte_tool.currentData();view.local_brush_radius=self.native_matte_radius.value()/max(1,min(self.clone_native_size))
                view.image.setCursor(Qt.CrossCursor if active else Qt.OpenHandCursor);self.layer_tabs.setCurrentWidget(view)
            view.render_overlays()
            self.native_matte_tool.setEnabled(ready);self.native_matte_radius.setEnabled(ready);self.native_matte_reset.setEnabled(ready and bool(self.native_matte_strokes))
            self.native_matte_undo.setEnabled(ready and self.native_matte_history_index>0);self.native_matte_redo.setEnabled(ready and self.native_matte_history_index+1<len(self.native_matte_history))
            self.native_matte_detail.setEnabled(ready and bool(getattr(self,'native_matte_result',None)))
            if self.native_matte_request is not None:self.source_light_save.setEnabled(False)

        def add_native_matte_stroke(self,stroke):
            if self.local_editor_tabs.currentIndex()!=7 or not self.source_run_id or self.render_pending:return
            stroke.update(id=uuid.uuid4().hex[:12],mode=self.native_matte_tool.currentData(),radius_px=float(self.native_matte_radius.value()),native_canvas_size=list(self.clone_native_size))
            self.native_matte_strokes.append(stroke)
            self.native_matte_history=self.native_matte_history[:self.native_matte_history_index+1]+[json.loads(json.dumps(self.native_matte_strokes))];self.native_matte_history_index=len(self.native_matte_history)-1
            self.native_matte_changed()

        def restore_native_matte_pair(self):
            base=self.native_matte_base_plan;plan=self.design_reconstruction_plan
            if not base or not plan:return
            for name in ('foreground_alpha','edge_applicability','edited_edge_light_applicability','matte_weak_added_material'):
                if name in base.get('mask_roles',{}):plan['mask_roles'][name]=json.loads(json.dumps(base['mask_roles'][name]))
                else:plan['mask_roles'].pop(name,None)
            original=next(l for l in base['layers'] if l['kind']=='background_design')
            layer=next(l for l in plan['layers'] if l['kind']=='background_design')
            for name in ('foreground_premultiplied','old_environment'):layer['assets'][name]=json.loads(json.dumps(original['assets'][name]))
            for name in ('source_baseline','native_matte_revision'):
                if name in base:plan[name]=json.loads(json.dumps(base[name]))
                else:plan.pop(name,None)

        def native_matte_changed(self):
            self.native_matte_version+=1;self.native_matte_cache_key=None;self.native_matte_result=None;self.native_matte_preview_version=-1
            if not self.native_matte_strokes:self.restore_native_matte_pair()
            self.source_light_version+=1;self.source_light_compare.setChecked(False)
            self.native_matte_state.setText(str(len(self.native_matte_strokes))+' 笔 · 正在准备人物边缘预览…');self.update_native_matte_overlay();self.native_matte_timer.start()

        def restore_native_matte_history(self,direction):
            if self.render_pending:return
            index=max(0,min(len(self.native_matte_history)-1,self.native_matte_history_index+direction))
            if index==self.native_matte_history_index:return
            self.native_matte_history_index=index;self.native_matte_strokes=json.loads(json.dumps(self.native_matte_history[index]));self.native_matte_changed()

        def reset_native_matte(self):
            if not self.native_matte_strokes or self.render_pending:return
            self.native_matte_strokes=[];self.native_matte_history=self.native_matte_history[:self.native_matte_history_index+1]+[[]];self.native_matte_history_index=len(self.native_matte_history)-1;self.native_matte_changed()

        def preview_native_matte(self):
            self.native_matte_timer.stop()
            if self.native_matte_request is not None or self.render_pending:return
            if not self._allow_current_render(): return
            token=uuid.uuid4().hex;self.native_matte_request=(token,self.source_run_id,self.native_matte_version)
            body={'request_id':token,'strokes':json.loads(json.dumps(self.native_matte_strokes)),**self._source_edit_payload()}
            self.native_matte_state.setText('正在修正人物边缘…');self.update_source_light_actions()
            self.task('POST',f'/api/source-runs/{self.source_run_id}/matte-preview',body,'native-matte:'+token)

        def show_native_matte_detail(self):
            value=getattr(self,'native_matte_result',None)
            if not value:return
            dialog=QDialog(self);dialog.setWindowTitle('人物边缘 · 当前修正的原尺寸细节');dialog.resize(840,840);layout=QVBoxLayout(dialog)
            label=QLabel('人物边缘的原尺寸对比；光色精修以主画布预览为准。新增边缘可能缺少可靠原片信息，请放大查看。');label.setWordWrap(True);layout.addWidget(label)
            image=QLabel();image.setPixmap(QPixmap(value['native_preview_path']));image.adjustSize();scroll=QScrollArea();scroll.setWidget(image);scroll.setAlignment(Qt.AlignCenter);layout.addWidget(scroll,1)
            self.native_matte_detail_dialog=dialog;dialog.show()

        def local_colour_brush_panel(self):
            panel=QWidget();layout=QVBoxLayout(panel);layout.setContentsMargins(0,6,0,0)
            note=QLabel('在成片上拖动，局部调整明暗、冷暖或绿洋红。每笔记录当前设置；擦除只移除本工具调整。半径按原生像素保存，人物透明边缘和高光余量保留。');note.setWordWrap(True);layout.addWidget(note)
            self.local_brush_enabled=QCheckBox('启用颜色笔刷');self.local_brush_enabled.setChecked(True);layout.addWidget(self.local_brush_enabled)
            layout.addWidget(QLabel('下一笔设置'))
            form=QFormLayout();self.local_brush_scope=QComboBox()
            for label,scope in [('人物','foreground'),('背景','background'),('成片','composite')]:self.local_brush_scope.addItem(label,scope)
            form.addRow('作用范围',self.local_brush_scope)
            self.local_brush_tool=QComboBox()
            for label,mode in [('绘制调整','paint'),('擦除调整','erase'),('平移 / 查看细节','pan')]:self.local_brush_tool.addItem(label,mode)
            form.addRow('画布工具',self.local_brush_tool)
            self.local_brush_radius=QDoubleSpinBox();self.local_brush_radius.setRange(1,1200);self.local_brush_radius.setValue(120);self.local_brush_radius.setSuffix(' px');self.local_brush_radius.setKeyboardTracking(False);form.addRow('原生半径',self.local_brush_radius)
            self.local_brush_hardness=QDoubleSpinBox();self.local_brush_hardness.setRange(0,1);self.local_brush_hardness.setSingleStep(.1);self.local_brush_hardness.setValue(.3);form.addRow('硬度',self.local_brush_hardness)
            self.local_brush_opacity=QDoubleSpinBox();self.local_brush_opacity.setRange(.05,1);self.local_brush_opacity.setSingleStep(.05);self.local_brush_opacity.setValue(.5);form.addRow('不透明度',self.local_brush_opacity)
            self.local_brush_spins={}
            for key,label,minimum,maximum,step,initial in [('exposure','曝光 EV',-3,3,.1,.3),('temperature','冷暖',-100,100,5,0),('tint','绿 / 洋红',-100,100,5,0)]:
                spin=QDoubleSpinBox();spin.setRange(minimum,maximum);spin.setSingleStep(step);spin.setValue(initial);spin.setKeyboardTracking(False);self.local_brush_spins[key]=spin;form.addRow(label,spin)
            layout.addLayout(form)
            saved=QWidget();saved_layout=QVBoxLayout(saved);saved_layout.setContentsMargins(0,4,0,0)
            self.local_brush_strokes=QListWidget();self.local_brush_strokes.setFixedHeight(100);saved_layout.addWidget(self.local_brush_strokes)
            self.local_brush_selected_state=QLabel('选择一笔后可修改，画布虚线标出其范围。');self.local_brush_selected_state.setWordWrap(True);saved_layout.addWidget(self.local_brush_selected_state)
            selected_form=QFormLayout();self.local_brush_selected_spins={}
            for key,label,low,high,step,initial in (
                ('radius_px','原生半径',1,1200,10,120),('hardness','硬度',0,1,.1,.3),('opacity','不透明度',.05,1,.05,.5),
                ('exposure','曝光 EV',-3,3,.1,.3),('temperature','冷暖',-100,100,5,0),('tint','绿 / 洋红',-100,100,5,0)):
                spin=QDoubleSpinBox();spin.setRange(low,high);spin.setSingleStep(step);spin.setValue(initial);spin.setKeyboardTracking(False)
                if key=='radius_px':spin.setSuffix(' px')
                self.local_brush_selected_spins[key]=spin;selected_form.addRow(label,spin);spin.valueChanged.connect(lambda value,k=key:self.edit_selected_local_brush(k,value))
            saved_layout.addLayout(selected_form)
            self.local_brush_delete=QPushButton('删除选中笔画');self.local_brush_delete.clicked.connect(self.delete_selected_local_brush);saved_layout.addWidget(self.local_brush_delete)
            saved_panel,self.local_brush_strokes_toggle=self.accordion('编辑已有笔画',saved,False);layout.insertWidget(2,saved_panel)
            self.local_brush_strokes.currentRowChanged.connect(self.select_local_brush_stroke)
            self.local_brush_strokes_toggle.toggled.connect(self.update_local_brush_overlay)
            row=QHBoxLayout();self.local_brush_undo=QPushButton('撤销');self.local_brush_redo=QPushButton('重做');row.addWidget(self.local_brush_undo);row.addWidget(self.local_brush_redo);layout.addLayout(row)
            self.local_brush_reset=QPushButton('清除当前范围');layout.addWidget(self.local_brush_reset)
            self.local_brush_state=QLabel('尚未绘制');self.local_brush_state.setWordWrap(True);layout.addWidget(self.local_brush_state);layout.addStretch()
            self.local_brush_enabled.toggled.connect(self.toggle_local_brush)
            for control in (self.local_brush_scope,self.local_brush_tool):control.currentIndexChanged.connect(self.update_local_brush_overlay)
            self.local_brush_radius.valueChanged.connect(self.update_local_brush_overlay)
            self.local_brush_undo.clicked.connect(lambda:self.restore_local_brush_history(-1));self.local_brush_redo.clicked.connect(lambda:self.restore_local_brush_history(1));self.local_brush_reset.clicked.connect(self.reset_local_brush)
            self.previews['final'].localColourStrokeFinished.connect(self.add_local_brush_stroke)
            return panel

        def local_brush_data(self):
            from local_colour_brush import empty_local_colour_brush
            return self._manual_data()['layers'].get('local_colour_brush',empty_local_colour_brush())

        def write_local_brush(self,layer,history=True):
            data=self._manual_data();data['layers']['local_colour_brush']=layer;self._write_manual_data(data)
            if history:
                snapshot=json.loads(json.dumps(layer))
                if self.local_brush_history_index>=0 and self.local_brush_history[self.local_brush_history_index]==snapshot:return
                self.local_brush_history=self.local_brush_history[:self.local_brush_history_index+1]+[snapshot];self.local_brush_history_index=len(self.local_brush_history)-1
            self.refresh_local_brush_strokes()
            self.source_light_state.setText('颜色笔画已修改，正在准备预览…')
            self.source_light_version+=1;self.source_light_compare.setChecked(False);self.update_source_light_actions();self.source_light_timer.start()

        def add_local_brush_stroke(self,stroke):
            if not self.source_run_id or self.render_pending or self.local_editor_tabs.currentIndex()!=6:return
            params={key:float(spin.value())/(1 if key=='exposure' else 100) for key,spin in self.local_brush_spins.items()}
            stroke.update({'id':uuid.uuid4().hex[:12],'scope':self.local_brush_scope.currentData(),'native_canvas_size':list(self.clone_native_size),
                'radius_px':self.local_brush_radius.value(),'hardness':self.local_brush_hardness.value(),'opacity':self.local_brush_opacity.value(),'params':params})
            layer=self.local_brush_data();layer['strokes'].append(stroke);self.write_local_brush(layer)

        def toggle_local_brush(self,enabled):
            if self.local_brush_loading:return
            layer=self.local_brush_data();layer['enabled']=bool(enabled);self.write_local_brush(layer)

        def refresh_local_brush_strokes(self):
            strokes=self.local_brush_data().get('strokes',[])
            self.local_brush_strokes_toggle.setText(f'编辑已有笔画 · {len(strokes)} 笔')
            selected=next((i for i,stroke in enumerate(strokes) if self.local_brush_selected_id and stroke.get('id')==self.local_brush_selected_id),
                          min(self.local_brush_selected_index,len(strokes)-1))
            self.local_brush_strokes.blockSignals(True);self.local_brush_strokes.clear()
            scope_names={'foreground':'人物','background':'背景','composite':'成片'}
            for index,stroke in enumerate(strokes):
                mode='擦除' if stroke.get('mode')=='erase' else '调整'
                exposure=float(stroke.get('params',{}).get('exposure',0))
                detail='' if stroke.get('mode')=='erase' else f' · {exposure:+.2f} EV'
                self.local_brush_strokes.addItem(f"{index+1} · {scope_names.get(stroke.get('scope'),'人物')} · {mode} · {stroke['radius_px']:g} px{detail}")
            self.local_brush_strokes.setCurrentRow(selected);self.local_brush_strokes.blockSignals(False)
            self.select_local_brush_stroke(selected)

        def selected_local_brush_stroke(self):
            strokes=self.local_brush_data().get('strokes',[]);index=self.local_brush_strokes.currentRow()
            return strokes[index] if 0<=index<len(strokes) else None

        def select_local_brush_stroke(self,index):
            self.local_brush_selected_index=index;stroke=self.selected_local_brush_stroke()
            self.local_brush_selected_id=stroke.get('id') if stroke else None
            self.local_brush_selection_loading=True
            if stroke:
                for key,spin in self.local_brush_selected_spins.items():
                    value=stroke.get(key,0) if key in ('radius_px','hardness','opacity') else stroke.get('params',{}).get(key,0)*(1 if key=='exposure' else 100)
                    spin.setValue(float(value))
                self.local_brush_selected_state.setText(f"第 {index+1} 笔 · 已有调整；修改仅作用于这一笔。")
            else:self.local_brush_selected_state.setText('选择一笔后可修改，画布虚线标出其范围。')
            self.local_brush_selection_loading=False;self.update_local_brush_overlay()

        def edit_selected_local_brush(self,key,value):
            if self.local_brush_selection_loading or self.render_pending:return
            index=self.local_brush_strokes.currentRow();layer=self.local_brush_data()
            if not 0<=index<len(layer['strokes']):return
            stroke=layer['strokes'][index]
            if key in ('radius_px','hardness','opacity'):stroke[key]=float(value)
            elif stroke.get('mode')!='erase':stroke.setdefault('params',{})[key]=float(value)/(1 if key=='exposure' else 100)
            self.write_local_brush(layer)

        def delete_selected_local_brush(self):
            if self.render_pending:return
            index=self.local_brush_strokes.currentRow();layer=self.local_brush_data()
            if not 0<=index<len(layer['strokes']):return
            del layer['strokes'][index];self.write_local_brush(layer)

        def restore_local_brush_history(self,direction):
            if self.render_pending:return
            index=max(0,min(len(self.local_brush_history)-1,self.local_brush_history_index+direction))
            if index==self.local_brush_history_index:return
            self.local_brush_history_index=index;layer=json.loads(json.dumps(self.local_brush_history[index]));self.write_local_brush(layer,False)
            self.local_brush_enabled.blockSignals(True);self.local_brush_enabled.setChecked(layer.get('enabled',True));self.local_brush_enabled.blockSignals(False);self.update_local_brush_overlay()

        def reset_local_brush(self):
            if self.render_pending:return
            layer=self.local_brush_data();scope=self.local_brush_scope.currentData();layer['strokes']=[s for s in layer['strokes'] if s.get('scope')!=scope];self.write_local_brush(layer)

        def load_local_brush(self):
            layer=self.local_brush_data();self.local_brush_history=[json.loads(json.dumps(layer))];self.local_brush_history_index=0
            self.local_brush_selected_id=None;self.local_brush_selected_index=-1;self.refresh_local_brush_strokes()
            self.local_brush_enabled.blockSignals(True);self.local_brush_enabled.setChecked(layer.get('enabled',True));self.local_brush_enabled.blockSignals(False)
            self.previews['final']._local_brush_stroke=None;self.update_local_brush_overlay()

        def update_local_brush_overlay(self,_value=None):
            if not hasattr(self,'local_brush_tool'):return
            ready=bool(self.source_run_id) and not self.render_pending
            active=ready and self.local_editor_tabs.currentIndex()==6 and self.local_brush_enabled.isChecked() and self.local_brush_tool.currentData()!='pan'
            view=self.previews['final'];view.local_brush_active=active;view.local_brush_mode=self.local_brush_tool.currentData();view.local_brush_radius=self.local_brush_radius.value()/max(1,min(self.clone_native_size))
            selected=self.selected_local_brush_stroke() if ready else None
            view.selected_local_brush=selected if ready and self.local_editor_tabs.currentIndex()==6 and self.local_brush_strokes_toggle.isChecked() else None
            if active:self.layer_tabs.setCurrentWidget(view);view.image.setCursor(Qt.CrossCursor)
            elif self.local_editor_tabs.currentIndex()==6:view.image.setCursor(Qt.OpenHandCursor)
            view.render_overlays()
            for widget in (self.local_brush_enabled,self.local_brush_scope,self.local_brush_tool,self.local_brush_radius,self.local_brush_hardness,self.local_brush_opacity,self.local_brush_reset,*self.local_brush_spins.values()):widget.setEnabled(ready)
            self.local_brush_undo.setEnabled(ready and self.local_brush_history_index>0);self.local_brush_redo.setEnabled(ready and self.local_brush_history_index+1<len(self.local_brush_history))
            self.local_brush_strokes.setEnabled(ready);self.local_brush_strokes_toggle.setEnabled(ready)
            self.local_brush_delete.setEnabled(ready and selected is not None)
            for key,spin in self.local_brush_selected_spins.items():spin.setEnabled(ready and selected is not None and (key in ('radius_px','hardness','opacity') or selected.get('mode')!='erase'))
            count=len(self.local_brush_data().get('strokes',[])) if ready else 0;self.local_brush_state.setText(str(count)+' 笔 · '+str(self.clone_native_size[0])+' × '+str(self.clone_native_size[1])+' 原生像素')

        def clone_panel(self):
            panel=QWidget();layout=QVBoxLayout(panel);layout.setContentsMargins(0,6,0,0)
            note=QLabel('在高清草稿上按 Alt + 单击同材质区域取样，再拖动画笔。自然修复融合目标光色，仿制图章直接复制。半径按原生像素保存；每笔可撤销，保存会创建子版本。');note.setWordWrap(True);layout.addWidget(note)
            self.clone_enabled=QCheckBox('启用仿制修复');self.clone_enabled.setChecked(True);layout.addWidget(self.clone_enabled)
            form=QFormLayout();self.clone_scope=QComboBox();self.clone_scope.addItem('人物（保护原生 alpha）','foreground');self.clone_scope.addItem('背景（仅背景取样）','background');form.addRow('修复对象',self.clone_scope)
            self.clone_tool=QComboBox();self.clone_tool.addItem('仿制图章','paint');self.clone_tool.addItem('平移 / 查看细节','pan');form.addRow('画布工具',self.clone_tool)
            self.clone_mode=QComboBox();self.clone_mode.addItem('仿制图章','clone');self.clone_mode.addItem('自然修复（保留目标光色）','heal');form.addRow('修复模式',self.clone_mode)
            self.clone_radius=QDoubleSpinBox();self.clone_radius.setRange(1,500);self.clone_radius.setValue(35);self.clone_radius.setSuffix(' px');self.clone_radius.setKeyboardTracking(False);form.addRow('原生半径',self.clone_radius)
            self.clone_hardness=QDoubleSpinBox();self.clone_hardness.setRange(0,1);self.clone_hardness.setSingleStep(.1);self.clone_hardness.setValue(.5);form.addRow('硬度',self.clone_hardness)
            self.clone_opacity=QDoubleSpinBox();self.clone_opacity.setRange(.05,1);self.clone_opacity.setSingleStep(.05);self.clone_opacity.setValue(1);form.addRow('不透明度',self.clone_opacity);layout.addLayout(form)
            row=QHBoxLayout();self.clone_undo=QPushButton('撤销一笔');self.clone_redo=QPushButton('重做');row.addWidget(self.clone_undo);row.addWidget(self.clone_redo);layout.addLayout(row)
            self.clone_state=QLabel('尚未取样');self.clone_state.setWordWrap(True);layout.addWidget(self.clone_state);layout.addStretch()
            self.clone_undo.clicked.connect(lambda:self.restore_clone_history(-1));self.clone_redo.clicked.connect(lambda:self.restore_clone_history(1))
            self.clone_enabled.toggled.connect(self.toggle_clone);self.clone_tool.currentIndexChanged.connect(self.update_clone_overlay);self.clone_scope.currentIndexChanged.connect(self.update_clone_overlay);self.clone_radius.valueChanged.connect(self.update_clone_overlay)
            self.previews['final'].cloneSourcePicked.connect(self.pick_clone_source);self.previews['final'].cloneStrokeFinished.connect(self.add_clone_stroke)
            return panel

        def clone_data(self):
            return self._manual_data()['layers'].get('clone_retouch',{'schema_version':'clone_retouch.v1','enabled':True,'strokes':[]})

        def write_clone_data(self,layer):
            data=self._manual_data();data['layers']['clone_retouch']=layer;self._write_manual_data(data)

        def pick_clone_source(self,point):
            self.clone_source=list(point);self.update_clone_overlay()

        def add_clone_stroke(self,stroke):
            if not self.source_run_id or self.render_pending:return
            stroke.update({'id':uuid.uuid4().hex[:12],'scope':self.clone_scope.currentData(),'mode':self.clone_mode.currentData(),
                'native_canvas_size':list(self.clone_native_size),'radius_px':self.clone_radius.value(),
                'hardness':self.clone_hardness.value(),'opacity':self.clone_opacity.value()})
            self.clone_history=self.clone_history[:self.clone_history_index]+[stroke];self.clone_history_index=len(self.clone_history)
            layer=self.clone_data();layer['strokes']=json.loads(json.dumps(self.clone_history));self.write_clone_data(layer);self.clone_changed()

        def restore_clone_history(self,direction):
            if self.render_pending:return
            index=max(0,min(len(self.clone_history),self.clone_history_index+direction))
            if index==self.clone_history_index:return
            self.clone_history_index=index;layer=self.clone_data();layer['strokes']=json.loads(json.dumps(self.clone_history[:index]));self.write_clone_data(layer);self.clone_changed()

        def toggle_clone(self,enabled):
            layer=self.clone_data();layer['enabled']=bool(enabled);self.write_clone_data(layer)
            if self.source_run_id:self.clone_changed()

        def clone_changed(self):
            self.source_light_version+=1;self.source_light_compare.setChecked(False);self.update_clone_overlay();self.update_source_light_actions();self.source_light_timer.start()

        def load_clone(self):
            layer=self.clone_data();self.clone_history=json.loads(json.dumps(layer.get('strokes',[])));self.clone_history_index=len(self.clone_history);self.clone_source=None
            self.clone_enabled.blockSignals(True);self.clone_enabled.setChecked(layer.get('enabled',True));self.clone_enabled.blockSignals(False)
            mode=self.clone_history[-1].get('mode','clone') if self.clone_history else 'clone'
            self.clone_mode.setCurrentIndex(self.clone_mode.findData(mode))
            self.previews['final']._clone_stroke=None;self.update_clone_overlay()

        def update_clone_overlay(self,_value=None):
            if not hasattr(self,'clone_tool'):return
            active=self.local_editor_tabs.currentIndex()==3 and bool(self.source_run_id) and not self.render_pending and self.clone_enabled.isChecked() and self.clone_tool.currentData()=='paint'
            view=self.previews['final'];view.clone_active=active;view.clone_source=self.clone_source;view.clone_radius=self.clone_radius.value()/max(1,min(self.clone_native_size))
            if active:self.layer_tabs.setCurrentWidget(view);view.image.setCursor(Qt.CrossCursor)
            elif self.local_editor_tabs.currentIndex()==3:view.image.setCursor(Qt.OpenHandCursor)
            view.render_overlays();self.clone_undo.setEnabled(not self.render_pending and self.clone_history_index>0);self.clone_redo.setEnabled(not self.render_pending and self.clone_history_index<len(self.clone_history))
            anchor='' if self.clone_source is None else '取样 '+str(tuple(round(v*(d-1)) for v,d in zip(self.clone_source,self.clone_native_size)))+' · '
            self.clone_state.setText(anchor+str(self.clone_history_index)+' 笔 · '+str(self.clone_native_size[0])+' × '+str(self.clone_native_size[1])+' 原生像素')

        def update_source_color_overlay(self, _value=None):
            if self.source_color_loading: return
            active = hasattr(self, "local_editor_tabs") and self.local_editor_tabs.currentIndex() == 1
            region = self.source_color_region(); selection = region.get("selection") if region else None
            if active and region and self.source_color_tool.currentData() != "pan" and self.layer_tabs.currentWidget() not in {self.previews["source"], self.previews["final"]}:
                self.layer_tabs.setCurrentWidget(self.previews["final"])
            for name in ("source", "final"):
                view = self.previews[name]; view.color_selection = selection if active else None
                view.show_color_selection = active and self.source_color_overlay.isChecked()
                view.color_brush_mode = self.source_color_tool.currentData() if active and not self.render_pending else "pan"
                view.color_brush_radius = self.source_color_radius.value(); view.render_overlays()
                view.image.setCursor(Qt.CrossCursor if active and view.color_brush_mode != "pan" else Qt.OpenHandCursor)

        def auto_source_colors(self):
            if not self.source_run_id or self.render_pending: return
            self.source_light_sequence += 1; token = str(self.source_light_sequence)
            self.source_light_requests[token] = (self.source_run_id, self.source_light_version); self.source_color_auto_request = token
            self.source_light_state.setText("正在准备本地语义选区…"); self.update_source_color_actions()
            self.task("POST", f"/api/source-runs/{self.source_run_id}/auto-color", {}, "source-color-auto:" + token)

        def add_local_plans_to_revision(self, revision):
            if self.workflow_graph is not None: revision['workflow_graph'] = copy.deepcopy(self.workflow_graph)
            else: revision.pop('workflow_graph', None)
            if self.portrait_beauty is not None: revision['portrait_beauty'] = copy.deepcopy(self.portrait_beauty)
            else: revision.pop('portrait_beauty', None)
            for prefix, plan, origin, design in (("source_render_plan", self.source_light_plan, self.source_light_origin, self.source_light_design), ("source_color_plan", self.source_color_plan, self.source_color_origin, self.source_color_design)):
                if plan is not None:
                    revision[prefix] = json.loads(json.dumps(plan)); revision[prefix + "_origin"] = origin
                    if design: revision[prefix + "_design"] = design
                    elif prefix == "source_color_plan": revision.pop(prefix + "_design", None)
            if self.source_color_provenance: revision["source_color_plan_provenance"] = self.source_color_provenance
            else: revision.pop("source_color_plan_provenance", None)
            if self.design_reconstruction_plan is not None: revision["design_reconstruction_plan"] = json.loads(json.dumps(self.design_reconstruction_plan))
            return revision

        @staticmethod
        def default_manual_layers():
            return {"schema_version": "manual_layers.v1", "revision": 1, "layers": {
                "background": {"enabled": False, "color": "#182b4d", "opacity": 0.18, "region": [0, 0, 1, 1]},
                "particles": {"enabled": False, "color": "#ffe3ad", "opacity": 0.45, "count": 80, "seed": 7, "radius": [1, 4]},
                "outline_light": {"enabled": False, "color": "#8fd7ff", "opacity": 0.35, "width": 5},
                "garment_texture": {"enabled": False, "color": "#fff1c7", "opacity": 0.10, "region": [0.15, 0.30, 0.85, 0.95], "spacing": 48},
            }}

        def _manual_data(self):
            try:
                value = json.loads(self.manual_layers.toPlainText())
                if isinstance(value, dict):
                    value.setdefault("schema_version", "manual_layers.v1")
                    value.setdefault("revision", 1)
                    value.setdefault("layers", {})
                    return value
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            return self.default_manual_layers()

        def _write_manual_data(self, data):
            self.manual_layers.blockSignals(True)
            self.manual_layers.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))
            self.manual_layers.blockSignals(False)

        def _set_manual_value(self, layer, key, value):
            data = self._manual_data(); layers = data.setdefault("layers", {})
            node = layers.setdefault(layer, {})
            node[key] = value
            self._write_manual_data(data)

        def manual_layers_panel(self):
            panel = QWidget(); outer = QVBoxLayout(panel); outer.setContentsMargins(0, 0, 0, 0)
            specs = {
                "background": ("背景色块", "opacity", 0, 100, 18, "强度"),
                "particles": ("粒子", "count", 0, 2000, 80, "数量"),
                "outline_light": ("外轮廓光", "width", 1, 31, 5, "宽度"),
                "garment_texture": ("服装纹理", "spacing", 8, 256, 48, "间距"),
            }
            for layer, (title, value_key, low, high, initial, value_label_text) in specs.items():
                group = QGroupBox(title); row = QHBoxLayout(group)
                enabled = QCheckBox("启用"); enabled.setToolTip("开启后通过 source-runs adjust 重渲染")
                enabled.toggled.connect(lambda value, name=layer: self._set_manual_value(name, "enabled", bool(value)))
                row.addWidget(enabled)
                slider = QSlider(Qt.Horizontal); slider.setRange(low, high); slider.setValue(initial); slider.setToolTip(value_label_text)
                value_label = QLabel(str(initial)); value_label.setMinimumWidth(42)
                slider.valueChanged.connect(lambda value, name=layer, key=value_key, out=value_label: (out.setText(str(value)), self._set_manual_value(name, key, value / 100.0 if key == "opacity" else int(value))))
                row.addWidget(QLabel(value_label_text)); row.addWidget(slider, 1); row.addWidget(value_label)
                opacity_slider = None; opacity_label = None
                if value_key != "opacity":
                    row.addWidget(QLabel("强度")); opacity_slider = QSlider(Qt.Horizontal); opacity_slider.setRange(0, 100); opacity_slider.setValue(45); opacity_label = QLabel("45%"); opacity_label.setMinimumWidth(42)
                    opacity_slider.valueChanged.connect(lambda value, name=layer, out=opacity_label: (out.setText(f"{value}%"), self._set_manual_value(name, "opacity", round(value / 100.0, 3))))
                    row.addWidget(opacity_slider, 1); row.addWidget(opacity_label)
                color_button = QPushButton("颜色"); color_button.setToolTip("选择该层颜色")
                color_button.clicked.connect(lambda _checked=False, name=layer, button=color_button: self.choose_manual_color(name, button))
                row.addWidget(color_button)
                self.manual_layer_controls[layer] = {"enabled": enabled, "slider": slider, "value_label": value_label, "opacity_slider": opacity_slider, "opacity_label": opacity_label, "color": color_button, "value_key": value_key}
                outer.addWidget(group)
            sync = QPushButton("从高级 JSON 同步控件")
            sync.clicked.connect(self.sync_manual_controls)
            outer.addWidget(sync)
            return panel

        def choose_manual_color(self, layer, button):
            data = self._manual_data(); current = str(data.get("layers", {}).get(layer, {}).get("color", "#ffffff"))
            color = QColorDialog.getColor(QColor(current), self, "选择" + layer + "颜色")
            if not color.isValid():
                return
            value = color.name().lower(); button.setStyleSheet("background-color: %s" % value); button.setText(value)
            self._set_manual_value(layer, "color", value)

        def sync_manual_controls(self):
            data = self._manual_data(); layers = data.get("layers", {})
            for layer, controls in self.manual_layer_controls.items():
                node = layers.get(layer, {}) if isinstance(layers.get(layer), dict) else {}
                controls["enabled"].blockSignals(True); controls["enabled"].setChecked(bool(node.get("enabled", False))); controls["enabled"].blockSignals(False)
                key = controls["value_key"]; raw = node.get(key, controls["slider"].value())
                try: value = round(float(raw) * 100) if key == "opacity" else int(raw)
                except (TypeError, ValueError): value = controls["slider"].value()
                value = max(controls["slider"].minimum(), min(controls["slider"].maximum(), value))
                controls["slider"].blockSignals(True); controls["slider"].setValue(value); controls["slider"].blockSignals(False); controls["value_label"].setText(str(value))
                if controls.get("opacity_slider") is not None:
                    try: opacity_value = round(float(node.get("opacity", 0.45)) * 100)
                    except (TypeError, ValueError): opacity_value = 45
                    opacity_value = max(0, min(100, opacity_value)); controls["opacity_slider"].blockSignals(True); controls["opacity_slider"].setValue(opacity_value); controls["opacity_slider"].blockSignals(False); controls["opacity_label"].setText(f"{opacity_value}%")
                color = str(node.get("color", "#ffffff")); controls["color"].setText(color); controls["color"].setStyleSheet("background-color: %s" % color)
        def delivery_page(self):
            page = QWidget(); lay = QVBoxLayout(page); self.delivery = QPlainTextEdit(); self.delivery.setReadOnly(True); lay.addWidget(self.delivery)
            export_box = QGroupBox("导出当前单图版本的副本"); export_layout = QVBoxLayout(export_box)
            export_row = QHBoxLayout(); self.export_directory = QLineEdit(); self.export_directory.setPlaceholderText("选择导出目录")
            choose_export = QPushButton("选择目录"); choose_export.clicked.connect(self.choose_export_directory)
            self.export_format = QComboBox(); self.export_format.addItems(["JPEG", "PNG", "TIFF", "TIFF16", "TIFF32_LINEAR"]); self.export_format.setToolTip("TIFF16 为当前版本的16位sRGB成片；TIFF32_LINEAR 为保留高光的32位线性工作主图，色彩空间与显示变换记录在sidecar。")
            self.export_artifact = QComboBox(); self.export_artifact.addItem("审片草稿", "draft"); self.export_artifact.addItem("已批准交付", "delivery")
            self.export_button = QPushButton("导出副本"); self.export_button.setEnabled(False); self.export_button.clicked.connect(self.export_source_run)
            export_row.addWidget(self.export_directory, 1); export_row.addWidget(choose_export); export_row.addWidget(self.export_format); export_row.addWidget(self.export_artifact); export_row.addWidget(self.export_button); export_layout.addLayout(export_row)
            self.export_state = QLabel("导出使用当前已保存母版；回执显示实际尺寸、位深与 ICC 信息。已批准交付保持不变。")
            self.export_state.setWordWrap(True); export_layout.addWidget(self.export_state); lay.addWidget(export_box)
            history_row = QHBoxLayout(); self.delivery_history_refresh = QPushButton("刷新单图交付历史"); history_row.addWidget(self.delivery_history_refresh); lay.addLayout(history_row)
            self.delivery_history = QPlainTextEdit(); self.delivery_history.setReadOnly(True); self.delivery_history.setPlaceholderText("单图 active delivery 与历史版本"); lay.addWidget(self.delivery_history, 1)
            lay.addWidget(QLabel("Run / revision 历史树（选择任意历史版本载入审片预览）"))
            self.source_history_tree = QTreeWidget(); self.source_history_tree.setColumnCount(4)
            self.source_history_tree.setHeaderLabels(["Run / revision", "状态", "交付", "更新时间"])
            self.source_history_tree.setMinimumHeight(150); self.source_history_tree.setAlternatingRowColors(True)
            self.source_history_tree.currentItemChanged.connect(self.select_source_history)
            lay.addWidget(self.source_history_tree, 1)
            self.saved_export_versions = QTreeWidget(); self.saved_export_versions.setColumnCount(3); self.saved_export_versions.setHeaderLabels(['勾选已保存版本', '状态', '样例']); self.saved_export_versions.setMaximumHeight(180); lay.addWidget(self.saved_export_versions)
            self.batch_export_button = QPushButton('批量导出勾选的已保存版本'); self.batch_export_button.clicked.connect(self.export_selected_versions); lay.addWidget(self.batch_export_button)
            self.source_copy = QPushButton("复制当前版本为新 revision")
            self.source_copy.clicked.connect(self.copy_source_run); lay.addWidget(self.source_copy)
            self.source_reviewer = QLineEdit(); self.source_reviewer.setPlaceholderText("视觉批准人（单图交付必填）"); lay.addWidget(self.source_reviewer)
            row = QHBoxLayout(); self.source_accept = QPushButton("批准单图草稿并交付"); self.source_accept.setEnabled(False); self.source_retry = QPushButton("重新生图"); self.source_retry.setToolTip("点击后按当前候选数量调用模型生成新候选；失败或超时不会自动重复生图。"); self.source_adjust = QPushButton("应用 revision 并重渲染"); row.addWidget(self.source_accept); row.addWidget(self.source_retry); row.addWidget(self.source_adjust); lay.addLayout(row)
            rollback_row = QHBoxLayout(); self.rollback_target = QLineEdit(); self.rollback_target.setPlaceholderText("回滚目标 run id（已批准历史版本）"); self.source_rollback = QPushButton("切换 active delivery"); rollback_row.addWidget(self.rollback_target, 1); rollback_row.addWidget(self.source_rollback); lay.addLayout(rollback_row)
            self.source_accept.clicked.connect(self.accept_source_run); self.source_retry.clicked.connect(self.retry_source_run); self.source_adjust.clicked.connect(self.adjust_source_run); self.source_rollback.clicked.connect(self.rollback_source_delivery); self.delivery_history_refresh.clicked.connect(self.load_delivery_history); return page
        def jobs_page(self):
            page = QWidget(); lay = QVBoxLayout(page)
            lay.addWidget(QLabel("任务保存在项目目录中；选择任务恢复进度或载入对应草稿。"))
            self.job_timing = QLabel('阶段耗时将在任务开始后显示；样本不足时显示估算中。')
            self.job_timing.setWordWrap(True); lay.addWidget(self.job_timing)
            row = QHBoxLayout(); refresh = QPushButton("刷新任务列表"); cancel = QPushButton("取消选中任务")
            refresh.clicked.connect(self.refresh_jobs); cancel.clicked.connect(self.cancel_selected_job)
            row.addWidget(refresh); row.addWidget(cancel); row.addStretch(); lay.addLayout(row)
            self.jobs_tree = QTreeWidget(); self.jobs_tree.setColumnCount(6)
            self.jobs_tree.setHeaderLabels(["任务", "状态", "阶段", "进度", "Run", "更新时间"])
            self.jobs_tree.setAlternatingRowColors(True); self.jobs_tree.currentItemChanged.connect(self.select_job)
            lay.addWidget(self.jobs_tree, 1)
            self.job_details = QPlainTextEdit(); self.job_details.setReadOnly(True); lay.addWidget(self.job_details, 1)
            return page
        def refresh_jobs(self):
            self.task("GET", "/api/jobs", None, "jobs-list")
        def populate_jobs(self, value):
            jobs = value.get("jobs", [])
            self.job_rows = {str(job["job_id"]): job for job in jobs if job.get("job_id")}
            self.jobs_tree.blockSignals(True); self.jobs_tree.clear()
            for job_id, job in self.job_rows.items():
                progress = job.get("progress")
                scope=job.get('editing_request') or (job.get('task_plan') or {}).get('editing_scope') or {}
                if job.get('status') in {'queued','running','cancelling'} and (scope.get('work_type') in {'partial_composite','full_composite'}
                        or job.get('progress_basis')=='indeterminate_scoped_workflow'):
                    progress=None
                item = QTreeWidgetItem(self.jobs_tree, [job_id, str(job.get("status") or ""), str(job.get("stage") or ""), f"{float(progress):.0%}" if progress is not None else "—", str(job.get("run_id") or ""), str(job.get("updated_at") or "")])
                item.setData(0, Qt.UserRole, job_id)
                if job_id == self.active_job_id: self.jobs_tree.setCurrentItem(item)
            self.jobs_tree.blockSignals(False)
        def observe_job(self, job_id):
            self.job_poll_timer.stop(); self.job_poll_terminal_observation = None
            self.job_observation += 1
            self.active_job_id = str(job_id); self.job_stream_active = False
            self.start_job_stream(); self.poll_active_job()
        def select_job(self, item, _previous=None):
            if item is None: return
            job_id = str(item.data(0, Qt.UserRole) or "")
            job = self.job_rows.get(job_id, {})
            self.job_details.setPlainText(json.dumps(job, ensure_ascii=False, indent=2))
            if self.render_pending and self.active_job_id and job_id != self.active_job_id:
                self.operation_state.setText('当前任务仍在处理；先等待或取消当前任务。'); return
            if job_id != self.active_job_id:
                if job.get("status") in {"queued", "running", "cancelling"}: self.observe_job(job_id)
                else:
                    self.job_poll_timer.stop()
                    self.job_observation += 1; self.active_job_id = job_id; self.job_stream_active = False
                    self.job_poll_terminal_observation = self.job_observation
            if job.get('kind') == 'native_design_reconstruction':
                self.native_reconstruction_job_id = job_id
                self.task('GET', f'/api/jobs/{job_id}', None, 'job-status')
            elif job.get("run_id"):
                self.task("GET", f"/api/source-runs/{job['run_id']}", None, "job-run-loaded")
        def cancel_selected_job(self):
            item = self.jobs_tree.currentItem()
            if item is not None:
                self.task("POST", f"/api/jobs/{item.data(0, Qt.UserRole)}/cancel", {}, "job-cancel")
        def _select_preferred_local_provider(self):
            """Choose the qualified local default only when it is ready.

            Persisted settings and an explicit selector action take precedence;
            this helper is used solely when no default has ever been saved.
            """
            if self.generation_model_user_selected or self.provider_default:
                return False
            row = self.provider_rows.get(PREFERRED_LOCAL_GENERATION_PROVIDER) or {}
            if not row.get('ready'):
                return False
            if self.provider.currentText() == PREFERRED_LOCAL_GENERATION_PROVIDER:
                return True
            self.provider.blockSignals(True)
            self.provider.setCurrentText(PREFERRED_LOCAL_GENERATION_PROVIDER)
            self.provider.blockSignals(False)
            self.select_provider(PREFERRED_LOCAL_GENERATION_PROVIDER)
            return True

        def select_provider(self, provider):
            row = self.provider_rows.get(provider, {})
            configured = self.provider_settings.get(provider, {})
            model = configured.get("model") or row.get("model") or ("" if provider in {"comfyui", "local", "manual_import", "qwen_image_2_1", "qwen_image_2_1_uncensored", "qwen_image_2_1_fast", "qwen_image_2_1_uncensored_fast"} else "gpt-image-2.5-sunburst")
            self.model.setText(str(model))
            display = row.get('display') or provider
            status = row.get('status', '未配置')
            readiness = generation_readiness_text({**row, 'id': provider})
            self.provider_state.setText(f"{display} · {readiness} / {status} · {row.get('endpoint') or '未配置地址'}")
            self.model.setToolTip('当前生图模型或 ComfyUI 工作流：' + (str(model) if model else '请填写模型/工作流路径'))
            self.provider_endpoint.setText(str(configured.get("endpoint") or row.get("endpoint") or ""))
            self.provider_secret_env.setText(str(configured.get("secret_env") or ""))
            self.provider_credential.setCurrentText(str(configured.get("credential_source") or "auto"))
            self.refresh_generation_models()

        def refresh_generation_models(self, *_args, locked=None):
            if not hasattr(self, 'simple_workspace'): return
            selected = self.provider.currentText()
            choices = []
            for provider, row in self.provider_rows.items():
                if not row.get('generation_supported'): continue
                model = self.model.text().strip() if provider == selected else str(row.get('model') or '')
                display = str(row.get('display') or provider)
                if row.get('kind') == 'comfyui':
                    label = display if str(row.get('model_env') or '').startswith('COSPLAY_QWEN_') else display + (' · ' + Path(model).stem if model else '')
                    route = 'ComfyUI 生成'
                else:
                    label = (model + ' · ' if model else '') + display
                    route = 'API 生成'
                state = generation_readiness_text({**row, 'id': provider})
                if provider == selected and self.generation_model_save_error: state += ' / 默认未保存'
                choices.append({'id':provider, 'label':label, 'detail':route + ' · ' + state})
            if locked is None:
                locked = bool(self.render_pending or self.creative_task_start_pending or self.simple_creation or self._session_loading)
            self.simple_workspace.set_generation_models(choices, selected, locked=locked)
            if (self.generation_model_save_pending and not self.generation_model_save_active
                    and not self.generation_model_save_timer.isActive()
                    and not self._session_loading and (not self.embedded_backend or self.embedded_server is not None)):
                self.generation_model_save_timer.start()
            self.schedule_selected_service()

        def select_generation_model(self, provider):
            if not self.simple_workspace.generation_model.isEnabled() or provider not in self.provider_rows: return
            self.generation_model_user_selected = True
            self._startup_service_suppressed=False
            self.generation_model_save_error = False
            self.provider.setCurrentText(provider)
            self.generation_model_save_pending = {**(self.generation_model_save_pending or {}), 'default_provider':provider}
            self.generation_model_save_timer.start()

        def persist_generation_model(self):
            if self.generation_model_save_active or not self.generation_model_save_pending: return
            if self._session_loading or (self.embedded_backend and self.embedded_server is None): return
            self.generation_model_save_active = self.generation_model_save_pending
            self.generation_model_save_pending = None
            payload = self.generation_model_save_active
            label = 'provider-settings-saved' if 'providers' in payload else 'generation-model-selected'
            self.task('POST', '/api/provider-settings', payload, label)

        def save_provider_settings(self):
            provider = self.provider.currentText()
            settings = {"endpoint": self.provider_endpoint.text().strip(), "model": self.model.text().strip(), "secret_env": self.provider_secret_env.text().strip(), "credential_source": self.provider_credential.currentText()}
            pending = self.generation_model_save_pending or {}
            self.generation_model_save_pending = {'default_provider':provider,
                'providers':{**pending.get('providers', {}), provider:settings}}
            self.generation_model_save_timer.stop(); self.persist_generation_model()
        def choose_export_directory(self):
            directory = QFileDialog.getExistingDirectory(self, "选择导出目录", self.export_directory.text().strip() or str(self.root_path))
            if directory: self.export_directory.setText(directory)
        def export_source_run(self):
            if not self.source_run_id or self.render_pending: return
            if self._unsaved_session_blocks_export(): return
            directory = self.export_directory.text().strip()
            if not directory:
                self.choose_export_directory(); directory = self.export_directory.text().strip()
            if not directory: return
            artifact = self.export_artifact.currentData()
            if artifact == "delivery" and self.source_status not in {"accepted", "delivered"}:
                QMessageBox.information(self, "此版本尚未批准", "选择草稿导出，或完成审片批准后导出交付副本。"); return
            self.task("POST", f"/api/source-runs/{self.source_run_id}/export", {"directory": directory, "format": self.export_format.currentText(), "artifact": artifact}, "source-export")

        def export_selected_versions(self):
            if self.render_pending or self._unsaved_session_blocks_export(): return
            directory = self.export_directory.text().strip()
            if not directory: self.choose_export_directory(); directory = self.export_directory.text().strip()
            if not directory: return
            items = []
            def collect(node):
                if node.checkState(0) == Qt.Checked:
                    items.append({'run_id': node.data(0, Qt.UserRole), 'artifact': self.export_artifact.currentData()})
                for index in range(node.childCount()): collect(node.child(index))
            for index in range(self.saved_export_versions.topLevelItemCount()): collect(self.saved_export_versions.topLevelItem(index))
            if not items:
                for index in range(self.source_history_tree.topLevelItemCount()): collect(self.source_history_tree.topLevelItem(index))
            if not items: self.export_state.setText('请先勾选要导出的已保存版本。'); return
            self.task('POST', '/api/source-runs/export-batch', {'items': items, 'directory': directory, 'format': self.export_format.currentText()}, 'saved-export-batch')
        @Slot(str)
        def log_text(self, text): self.log.appendPlainText(str(text).rstrip())
        def sample(self): return self.source_sample
        def task(self, method, path, body, label):
            if self.source_generated_appearance and method != 'GET' and path.startswith(f'/api/source-runs/{self.source_run_id}/'):
                allowed = {'native-detail', 'export', 'export-project', 'copy', 'accept', 'review'}
                if path.rsplit('/', 1)[-1] not in allowed:
                    self.operation_state.setText('此生成版本可查看、审片和导出；旧原片图层编辑不会提交。修改设计请从原片创建新任务。')
                    return
            if label in {'job-status', 'job-events'} and any(
                    relay.label == label and relay.current() for relay in self.pending_tasks):
                return
            if self.embedded_backend and self.embedded_server is None:
                self.operation_state.setText('请先启动当前工程的后台。'); return
            if label == 'model-setup' and self.model_setup_submission is not None: return
            if label in {'model-setup', 'model-setup-saved'}: self.model_setup_observation += 1
            if self._session_loading and method != "GET": return
            if label in {"history-run-selected", "project-imported"}:
                if self.render_pending:
                    self.operation_state.setText('当前任务仍在处理；先等待或取消，再切换版本。'); return
                self._checkpoint_session()
                if self._session_checkpoint_failed: return
            if label in {"source-adjust", "source-retry", "local-retry", "repair-final", "source-copy", "one-click", "complete-post", "local-only", "existing-design", "source-select-variant", "design-reconstruct-job", "portrait-beauty-job"}:
                if self.render_pending or self._active_production_job():
                    self.log_text("[render] 请求进行中，请等待草稿完成"); return
                self._checkpoint_session()
                if self._session_checkpoint_failed:
                    if label == 'design-reconstruct-job':
                        self.design_reconstruction_request = None; self.update_source_light_actions()
                    if label == 'portrait-beauty-job':
                        self.portrait_beauty_request = None; self.update_source_light_actions()
                    return
                if label == "source-adjust" and self._session_binding:
                    from session_draft import SessionDraftStore
                    try:
                        self._session_submission = SessionDraftStore(self.root_path).mark_submitted(self._session_binding, self._session_snapshot())
                        body = json.loads(json.dumps(body))
                        body.setdefault("appearance_revision", {})["native_session_submission"] = self._session_submission
                    except (OSError, TypeError, ValueError) as error:
                        self.session_draft_state.setText("会话保留失败，高清保存未提交：" + str(error)); return
                self.render_pending = True; self.update_accept_enabled()
                self.operation_state.setText(f'正在提交{self._reconstruction_name()}准备任务；准备阶段不更新已保存版本。' if label == 'design-reconstruct-job' else "正在高清处理，完成后打开新版本" if label == "source-adjust" else "正在按已检查计划本地处理原片。" if label == "local-only" else "正在复用参考执行已检查计划；新增图像模型调用以本次冻结预算为准。" if label == "existing-design" else f"操作状态：{label} pending / run={self.source_run_id} / r{self.source_revision}")
            self.log_text(f"[{label}] {method} {path}"); task = ApiTask(self.backend_url(), method, path, body, label)
            relay = TaskRelay(self, label, self.job_observation)
            self.pending_tasks[relay] = task
            task.signals.result.connect(relay.result); task.signals.error.connect(relay.error); task.signals.finished.connect(relay.finished)
            self.pool.start(task)
        def request_native_detail(self, center=None):
            if not self.source_run_id or not self.native_detail_button.isEnabled(): return
            if center is None:
                view = self.previews['final']
                point = view.image.mapFrom(view.scroll.viewport(),view.scroll.viewport().rect().center())
                center = [min(1.,max(0.,point.x()/max(1,view.image.width()))),
                          min(1.,max(0.,point.y()/max(1,view.image.height())))]
            self.native_detail_token = self.source_run_id+":"+uuid.uuid4().hex[:8]
            self.task("POST",f"/api/source-runs/{self.source_run_id}/native-detail",
                      {'center':center,'edge':768,'include_source':True,'request_id':self.native_detail_token},
                      "native-detail:"+self.native_detail_token)

        def request_live_native_detail(self,center=None):
            if not self.source_run_id or self.render_pending or self.native_live_detail_context is not None:return
            if not self._allow_current_render(): return
            if center is None:
                view=self.previews['final'];point=view.image.mapFrom(view.scroll.viewport(),view.scroll.viewport().rect().center())
                center=[min(1.,max(0.,point.x()/max(1,view.image.width()))),min(1.,max(0.,point.y()/max(1,view.image.height())))]
            if self.native_matte_strokes and self.native_matte_preview_version!=self.native_matte_version:
                self.native_live_detail_pending_center=center
                self.operation_state.setText('人物边缘更新后将显示当前调整的原尺寸细节…')
                self.preview_source_lights();return
            token=uuid.uuid4().hex
            self.native_live_detail_context=(token,self.source_run_id,self.source_light_version,self.native_matte_version)
            self.native_live_detail_pending_center=None
            self.update_live_native_detail_actions()
            self.operation_state.setText('正在计算当前调整的原尺寸细节，完成后打开…')
            self.task('POST',f'/api/source-runs/{self.source_run_id}/native-detail-preview',
                {'center':center,'edge':768,'request_id':token,**self._source_edit_payload()},'native-live-detail:'+token)

        def show_live_native_detail(self,value):
            dialog=QDialog(self);dialog.setWindowTitle(f"r{value['revision']} · 当前未保存调整 · 原尺寸细节");dialog.resize(840,860)
            layout=QVBoxLayout(dialog);box=value['crop_box']
            layout.addWidget(QLabel(f"当前未保存调整 · {value['native_size'][0]} × {value['native_size'][1]}\n"
                f"源像素区域 {box[0]}, {box[1]} → {box[2]}, {box[3]}；未缩放。此预览没有保存新版本。"))
            image=QLabel();image.setPixmap(QPixmap(value['preview_path']));image.adjustSize()
            scroll=QScrollArea();scroll.setWidget(image);scroll.setAlignment(Qt.AlignCenter);layout.addWidget(scroll,1)
            layout.addWidget(QLabel('显示请求时的参数；继续修改后请重新查看。'))
            self.native_live_detail_dialog=dialog;dialog.show()

        def show_native_detail(self, value):
            dialog = QDialog(self); dialog.setWindowTitle(f"r{value['revision']} · 原尺寸细节")
            dialog.resize(840,860)
            layout = QVBoxLayout(dialog)
            box = value['crop_box']
            layout.addWidget(QLabel(f"已保存版本 · {value['native_size'][0]} × {value['native_size'][1]}\n"
                f"原生像素区域 {box[0]}, {box[1]} → {box[2]}, {box[3]}；未缩放。尚未保存的修改不在此视图中。"))
            pair = QHBoxLayout(); scrolls = []
            source = value.get('source_preview_path')
            if not source:
                layout.addWidget(QLabel('原片基线对照不可用：' + str((value.get('source_comparison') or {}).get('unavailable_reason') or '未提供同坐标原片基线；未使用代理图代替。')))
            for title, path in ([('原片基线（美化前）', source)] if source else []) + [('已保存成片', value['preview_path'])]:
                column = QVBoxLayout(); column.addWidget(QLabel(title))
                image = QLabel(); image.setPixmap(QPixmap(path)); image.adjustSize()
                scroll = QScrollArea(); scroll.setWidget(image); scroll.setAlignment(Qt.AlignCenter); column.addWidget(scroll,1)
                pair.addLayout(column); scrolls.append(scroll)
            if len(scrolls) == 2:
                for first, second in ((scrolls[0],scrolls[1]), (scrolls[1],scrolls[0])):
                    first.horizontalScrollBar().valueChanged.connect(second.horizontalScrollBar().setValue)
                    first.verticalScrollBar().valueChanged.connect(second.verticalScrollBar().setValue)
                dialog.resize(1460,900)
            layout.addLayout(pair,1)
            self.native_detail_dialog = dialog
            dialog.show()

        def result(self, label, value):
            if self.handle_shaping_inspection(label, value): return
            if self.handle_shaping_result(label, value): return
            if str(label).startswith('startup-service:'):
                self.receive_startup_service(label,value);return
            active = self.active_checked_task
            if active and isinstance(value, dict):
                if label == active['callback']:
                    if value.get('async') and value.get('job_id'):
                        active['job_id'] = str(value['job_id'])
                    else:
                        self.active_checked_task = None
                elif label in {'job-status', 'job-stream'}:
                    observed = value.get('job' if label == 'job-status' else 'data') or {}
                    identifier = str(observed.get('job_id') or self.active_job_id)
                    if (identifier == active.get('job_id') and observed.get('status') in
                            {'succeeded', 'failed', 'blocked', 'cancelled'}):
                        self.active_checked_task = None
            # Polling/SSE carry fresher progress than the occasional jobs list.
            # Keep that observation available to ordinary UI refreshes too.
            if label in {'job-status', 'job-stream'} and isinstance(value, dict):
                observed = value.get('job' if label == 'job-status' else 'data') or {}
                if isinstance(observed, dict) and self.active_job_id:
                    identifier = str(observed.get('job_id') or self.active_job_id)
                    if identifier == self.active_job_id:
                        self.job_rows[identifier] = {**self.job_rows.get(identifier, {}), **observed}
                        timing = observed.get('timing') or (observed.get('data') or {}).get('timing')
                        if isinstance(timing, dict):
                            from workflow_timing import progress_text
                            self.job_timing.setText(progress_text(timing))
            if label.startswith('creative-task-plan:'):
                self.receive_creative_task_plan(label, value); return
            if self.creative_task_start_pending and label == self.creative_task_execution_label:
                self.creative_task_start_pending = False; self.creative_task_execution_label = None
                self.creative_task_retry_batch_id = None
                self.update_creative_task_actions()
            if label in self.simple_cancel_submissions:
                self.simple_cancel_submissions.discard(label)
                if value.get('async') and value.get('job_id'):
                    identifier = value['job_id']
                    if label == 'portrait-beauty-job':
                        self.portrait_beauty_job_id = identifier; self.portrait_beauty_request = identifier
                    elif label == 'design-reconstruct-job':
                        self.native_reconstruction_job_id = identifier; self.design_reconstruction_request = identifier
                    self.observe_job(identifier)
                    self.task('POST', f'/api/jobs/{identifier}/cancel', {}, 'job-cancel')
                    self.refresh_jobs(); return
            if label in {'model-setup', 'model-setup-saved'}:
                if label == 'model-setup-saved' and self.model_setup_dialog is not None:
                    for key, edit in self.model_path_edits.items():
                        if edit.text() == (self.model_setup_submission or {}).get(key): edit.setModified(False)
                if label == 'model-setup-saved': self.model_setup_submission = None
                self.show_model_setup_status(value.get('setup') or {})
                return
            if label == 'design-reconstruct-job':
                if value.get('async') and value.get('job_id'):
                    self.native_reconstruction_preparation_kind = (value.get('job') or {}).get('preparation_kind', self.native_reconstruction_preparation_kind)
                    self.native_reconstruction_job_id = value['job_id']
                    self.design_reconstruction_request = value['job_id']
                    self.observe_job(value['job_id']); self.refresh_jobs()
                    self.operation_state.setText(f'{self._reconstruction_name()}准备任务已提交；任务中心可等待或取消，未保存新版本。')
                else: self.error('本地准备任务未创建，当前编辑保留。', label)
                return
            if label == 'portrait-beauty-job':
                if value.get('async') and value.get('job_id'):
                    self.portrait_beauty_job_id = value['job_id']; self.portrait_beauty_request = value['job_id']
                    self.observe_job(value['job_id']); self.refresh_jobs()
                    self.operation_state.setText('人像美化任务已提交；任务中心可等待或取消，完成后预览并保存。')
                    self.portrait_beauty_state.setText(self.operation_state.text())
                else: self.error('人像美化任务未创建，当前编辑保留。', label)
                return
            if label.startswith('native-live-detail:'):
                context=self.native_live_detail_context
                if not context or label!='native-live-detail:'+context[0]:return
                self.native_live_detail_context=None;self.update_live_native_detail_actions()
                if context[1:]!=(self.source_run_id,self.source_light_version,self.native_matte_version):
                    self.operation_state.setText('调整已更新，请重新查看当前原尺寸细节。');return
                if value.get('request_id')==context[0] and value.get('run_id')==self.source_run_id:
                    self.show_live_native_detail(value);self.operation_state.setText('当前调整的原尺寸细节已打开；未保存新版本。')
                return
            if label.startswith('native-matte:'):
                context=self.native_matte_request
                if not context or label!='native-matte:'+context[0]:return
                self.native_matte_request=None
                if context[1]!=self.source_run_id or context[2]!=self.native_matte_version:
                    self.update_source_light_actions();self.native_matte_timer.start();return
                self.native_matte_cache_key=value['cache_key'];self.native_matte_preview_version=self.native_matte_version;self.native_matte_result=value
                self.load_design_reconstruction(value['patched_plan'])
                self.previews['final'].replace_preview_path(Path(value['preview_path']))
                self.native_matte_state.setText(str(len(self.native_matte_strokes))+' 笔 · 已更新人物边缘，可查看原尺寸或保存。')
                self.source_light_version+=1;self.update_source_light_actions();self.source_light_timer.start()
                pending = getattr(self, 'design_reconstruction_after_matte', None)
                self.design_reconstruction_after_matte = None
                if pending == (self.source_run_id, self.native_matte_version):
                    self.reconstruct_design(preparation_kind=getattr(self, 'design_reconstruction_after_matte_kind', 'reference_appearance'))
                if self.native_live_detail_pending_center is not None:self.request_live_native_detail(self.native_live_detail_pending_center)
                return
            if label.startswith("native-detail:"):
                if value.get('request_id') == self.native_detail_token and value.get('run_id') == self.source_run_id:
                    self.show_native_detail(value)
                return
            if label.startswith(("source-light-", "source-color-auto:")):
                token = label.split(":", 1)[1]; context = self.source_light_requests.pop(token, None)
                receiver_receipt = self.receiver_shadow_preview_requests.pop(token, None)
                if label.startswith("source-color-auto:") and self.source_color_auto_request == token:
                    self.source_color_auto_request = None; self.update_source_color_actions()
                if context != (self.source_run_id, self.source_light_version): return
                if value.get("run_id") != self.source_run_id: return
                if label.startswith("source-color-auto:"):
                    self.source_color_provenance = value.get("provenance")
                    self.source_color_design = None; self.source_color_origin = "local_automatic"
                    self.load_source_colors(value.get("source_color_plan")); self.local_editor_tabs.setCurrentIndex(1); self.source_light_version += 1
                    self.source_color_overlay.setChecked(True); self.update_source_light_actions(); self.preview_source_lights()
                elif str(value.get("request_id")) == token:
                    receipt = receiver_receipt
                    if (not receipt or receipt['snapshot'] != self.receiver_edit_snapshot() or
                            receipt['matte_version'] != self.native_matte_version or receipt['matte_strokes'] != self.native_matte_strokes): return
                    path = Path(str(value.get("preview_path") or ""))
                    if not path.is_file(): return
                    self.source_light_preview_path = path
                    if not self.source_light_compare.isChecked():
                        self.previews["final"].replace_preview_path(path); self.update_source_light_overlay(); self.update_source_color_overlay()
                    if self.local_editor_tabs.currentIndex() in {0, 2}: self.layer_tabs.setCurrentWidget(self.previews["final"])
                    size = value.get("preview_size") or []
                    size_text = f"{size[0]} × {size[1]}" if len(size) == 2 else "代理尺寸"
                    self.source_light_state.setText(f"预览已更新（{size_text}），尚未保存。")
                    self.load_receiver_shadow_result(value.get('authored_receiver_shadow_execution'),
                        snapshot=receipt['snapshot'], matte_version=receipt['matte_version'], origin='本次预览', matte_strokes=receipt['matte_strokes'])
                    self.refresh_cos_workspace()
                return
            if label == "source-preview":
                source = Path(str(value.get("source") or ""))
                if source == self.project_images.get("source"):
                    self.project_proxies["source"] = Path(value["proxy_path"])
                    self.project_dimensions["source"] = [value["width"], value["height"]]
                    if self.project_view_selector.currentData() == "source": self.update_project_canvas()
                    if self.previews.get("source") and self.previews["source"].path == source: self.previews["source"].set_path(source, self.project_proxies["source"])
                    self.refresh_cos_workspace()
                return
            if label in {"source-adjust", "source-retry", "local-retry", "repair-final", "source-copy", "one-click", "complete-post", "local-only", "existing-design", "source-select-variant"} and not value.get("async"):
                self.render_pending = False; self.update_accept_enabled()
            if label == 'job-status':
                job = value.get('job') or {}
                self.log_text(json.dumps({key: job.get(key) for key in
                    ('job_id', 'status', 'stage', 'progress', 'message')}, ensure_ascii=False))
            elif label != 'jobs-list':
                self.log_text(json.dumps(value, ensure_ascii=False, indent=2))
            if isinstance(value, dict) and label not in {"source-delivery-history", "source-run-history", "job-events", "jobs-list", "job-status", "job-stream", "job-run-loaded"}:
                status = value.get("status") or value.get("review_status")
                if value.get("needs_rerender"):
                    self.operation_state.setText(f"操作状态：{label} 完成，需重新渲染")
                elif status:
                    self.operation_state.setText(f"操作状态：{label} → {status}")
                else:
                    self.operation_state.setText(f"操作状态：{label} 完成")
            if label in {"one-click", "complete-post", "local-only", "existing-design", "source-retry", "local-retry", "repair-final", "source-select-variant"}:
                run = (value.get("run") or value.get("child_run")) if isinstance(value, dict) else None
                if isinstance(run, dict):
                    was_post = bool(label in {'one-click', 'complete-post', 'existing-design'} and self.simple_creation
                                    and self.simple_creation.get('stage') in {'post', 'production'})
                    self.load_source_run(run)
                    if was_post and self.simple_creation:
                        self.simple_creation = None; self._reset_quick_progress()
                        self.operation_state.setText(post_completion_message(run))
                        self.refresh_cos_workspace()
                elif isinstance(value, dict) and value.get("async") and value.get("job_id"):
                    self.observe_job(value["job_id"])
                    self.operation_state.setText(f"操作状态：后台任务 {self.active_job_id} 已启动")
                    self.refresh_jobs()
            elif label in {"source-accept", "source-adjust", "source-copy"}:
                run = value.get("run") if isinstance(value, dict) else None
                if isinstance(run, dict):
                    self.load_source_run(run, saved_submission=label == "source-adjust")
                    if label == "source-adjust" and (run.get("render") or {}).get("run_id") == run.get("run_id") and run.get("status") not in {"failed", "blocked", "cancelled", "render_failed", "needs_rerender"}:
                        if self.source_run_id == run.get("run_id"):
                            self.operation_state.setText("高清版本已保存")
                            if self.source_light_plan is not None or self.source_color_plan is not None or self.design_reconstruction_plan is not None: self.source_light_state.setText("高清版本已保存；可继续调整设计图层与局部光色。")
                            if label == "source-adjust" and self.simple_creation and self.simple_creation.get('stage') == 'post_save_wait':
                                self.simple_creation['stage'] = 'post'
                                self.operation_state.setText('仿色与融合版本已保存；正在进入特效与完整后期。')
                                QTimer.singleShot(0, self.complete_current_post)
                        else:
                            self.operation_state.setText("高清子版本已保存；当前新增修改仍未保存")
                elif label == "source-adjust" and value.get("async") and value.get("job_id"):
                    self.observe_job(value["job_id"])
            elif label == 'project-exported':
                self.portable_project_state.setText('已导出保存版本 '+str(value.get('run_id',''))+'：'+str(value.get('archive','')))
                self.operation_state.setText('已保存工程导出完成')
            elif label == 'project-imported':
                self.load_source_run(value['run'])
                self.portable_project_state.setText('工程已导入：'+str(value.get('run_id',self.source_run_id)))
                self.operation_state.setText('工程已导入，可继续编辑和导出')
                self.task('GET','/api/recent',None,'recent')
            elif label == "jobs-list": self.populate_jobs(value)
            elif label == "job-cancel": self.refresh_jobs()
            elif label == "character-recognized": self.show_character_profile(value.get("profile") or {})
            elif label == 'generation-model-selected':
                self.generation_model_save_active = None
                self.generation_model_save_error = False
                self.provider_settings = value.get('providers') or {}
                self.refresh_generation_models()
            elif label in {"provider-settings", "provider-settings-saved"}:
                self.provider_settings = value.get("providers") or {}
                self.provider_default = value.get("default_provider")
                if label == "provider-settings-saved":
                    self.generation_model_save_active = None
                    self.generation_model_save_error = False
                    self.select_provider(self.provider.currentText()); self.task("GET", "/api/providers", None, "providers")
                elif not self.provider_settings_loaded:
                    default = value.get("default_provider")
                    if not self.generation_model_user_selected:
                        if default and self.provider.findText(default) >= 0: self.provider.setCurrentText(default)
                        self.select_provider(self.provider.currentText())
                        self._select_preferred_local_provider()
                self.provider_settings_loaded = True
                if label == "provider-settings" and not self.generation_model_user_selected:
                    self._select_preferred_local_provider()
                self.refresh_generation_models()
            elif label == "provider-discovery":
                discovery = value.get("discovery") or value.get("provider") or {}
                if discovery.get("available"):
                    self.provider.setCurrentText("ccswitch_sub2api")
                    self.provider_endpoint.setText(str(discovery.get("endpoint") or ""))
                    if discovery.get("model"): self.model.setText(str(discovery["model"]))
                    self.provider_credential.setCurrentText("ccswitch"); self.provider_secret_env.clear()
                    self.provider_state.setText(f"已读取 {discovery.get('provider_name', 'CCSwitch')} / {discovery.get('status', 'available')}；保存后采用")
                else:
                    self.provider_state.setText(f"CCSwitch 通道不可用：{discovery.get('status', '未发现')}")
            elif label in {"templates", "template-saved"}:
                rows = value.get("templates", []) if isinstance(value, dict) else []
                if hasattr(self, "template_selector"):
                    current = self.template_selector.currentData()
                    current_name = current.get("name") if isinstance(current, dict) else None
                    self.template_selector.blockSignals(True); self.template_selector.clear(); self.template_selector.addItem("选择命名模板", None)
                    selected_index = 0
                    for row in rows:
                        if not isinstance(row, dict): continue
                        self.template_selector.addItem(str(row.get("name", "")), row)
                        if row.get("name") == current_name: selected_index = self.template_selector.count() - 1
                    self.template_selector.setCurrentIndex(selected_index); self.template_selector.blockSignals(False)
                if label == "template-saved":
                    self.operation_state.setText("操作状态：参数模板已保存")
            elif label == "recent":
                self._startup_recent_loaded=True
                rows = value.get("recent", []) if isinstance(value, dict) else []
                current = self.recent_projects.currentData()
                selected_id = (current or {}).get('run_id') if isinstance(current, dict) else None
                selection = value.get('selection') or {}
                if selected_id is None:
                    selected_id = selection.get('selected_saved_run_id') if selection.get('selection_explicit') else selection.get('latest_saved_run_id')
                self.recent_projects.blockSignals(True)
                self.recent_projects.clear()
                for row in rows:
                    if isinstance(row, dict): self.recent_projects.addItem(f"{row.get('sample', '?')} / {row.get('run_id', '?')}", row)
                for index in range(self.recent_projects.count()):
                    if (self.recent_projects.itemData(index) or {}).get('run_id') == selected_id:
                        self.recent_projects.setCurrentIndex(index); break
                self.recent_projects.blockSignals(False)
                if hasattr(self, "recent_projects_label"):
                    entries = [f"{row.get('sample', '?')} / {row.get('run_id', '?')}" for row in rows[:8] if isinstance(row, dict)]
                    self.recent_projects_label.setText("最近项目：" + ("；".join(entries) if entries else "暂无记录"))
                if (not self._startup_result_requested and not self.source_run_id and not self.render_pending
                        and not self._session_dirty and not self.simple_creation):
                    self._startup_result_requested = True
                    restore_id = selection.get('selected_saved_run_id') if selection.get('selection_explicit') else selection.get('latest_saved_run_id')
                    if restore_id:
                        self._startup_result_context = {'source_input': self.source_edit.text(),
                            'explicit': bool(selection.get('selection_explicit')), 'run_id': restore_id}
                        self.task('GET', f'/api/source-runs/{restore_id}', None, 'startup-result-loaded')
                self.schedule_selected_service()
            elif label == "cleanup-preview":
                self.operation_state.setText(str(value.get('reason')) if value.get('blocked') else f"可清理历史：{value.get('run_count', 0)} 个运行 / {value.get('job_count', 0)} 个任务 / {value.get('bytes', 0)} bytes" if value.get('ok') else '清理检查未通过：'+str(value.get('errors') or '请查看日志'))
            elif label == "cleanup-apply":
                self.operation_state.setText(str(value.get('reason')) if value.get('blocked') else f"已直接删除过时记录：{value.get('removed_runs', 0)} 个运行 / {value.get('removed_jobs', 0)} 个任务 / {value.get('deleted_bytes', 0)} bytes" if value.get('ok') else '清理未完成：'+str(value.get('errors') or '请查看日志'))
                if value.get('ok'): self.refresh()
            elif label == 'local-batch-restore':
                batches = [batch for batch in (value.get('batches') or []) if self.is_creative_task_batch(batch)]
                if batches:
                    batch = max(batches, key=lambda row: row.get('created_at', ''))
                    self.local_batch_id = batch['batch_id']; self.local_batch_pairs = []
                    self.show_local_batch(batch); self.poll_batch(batch['batch_id'])
                else: self.batch_state.setText('尚无已保存的当前制作批次。')
            elif label in {'batch-started', 'local-batch-started'}:
                batch = value.get("batch", {}) if isinstance(value, dict) else {}
                self.batch_state.setText(f"批量任务：{batch.get('batch_id', '?')} / {len(batch.get('items', []))} 张已排队")
                if self.is_creative_task_batch(batch): self.local_batch_id = batch.get('batch_id'); self.show_local_batch(batch)
                self.poll_batch(batch.get("batch_id"))
            elif label == "batch-status":
                batch = value.get("batch", {}) if isinstance(value, dict) else {}
                status = batch.get("status", "?")
                message = {"queued": "排队中", "running": "处理中", "cancelling": "正在取消",
                           "awaiting_reference": "候选已生成时请选择一张；没有候选时再提供已有参考或重新生成",
                           "cancelled": "已取消；取消的任务未完成出片", "failed": "存在失败或丢失任务，请查看任务详情",
                           "succeeded": "批量执行结束，作品仍需逐图检查缺项与批准"}.get(status, status)
                self.batch_state.setText(f"批量任务：{batch.get('batch_id', '?')} / {message} / {batch.get('summary', {})}")
                if self.is_creative_task_batch(batch) and batch.get('batch_id') == self.local_batch_id: self.show_local_batch(batch)
                if status in {"queued", "running", "cancelling"}:
                    QTimer.singleShot(1500, lambda: self.poll_batch(batch.get("batch_id")))
            elif label == 'saved-export-batch':
                batch = value.get('batch') or {}; failed = [str(row.get('run_id')) + '：' + str(row.get('error')) for row in batch.get('items', []) if row.get('status') == 'failed']
                self.export_state.setText('批量导出：' + str(batch.get('summary')) + '\n回执：' + str(value.get('receipt', '')) + ('\n' + '\n'.join(failed) if failed else ''))
            elif label == "source-export":
                metadata = value.get("metadata") or {}
                icc = "已嵌入 ICC" if metadata.get("icc_profile_embedded") else "未嵌入 ICC"
                precision = metadata.get("precision_origin", "")
                self.export_state.setText(f"已导出：{value.get('path', '')}\n{metadata.get('format', self.export_format.currentText())} / {metadata.get('width', '?')} × {metadata.get('height', '?')} / {metadata.get('actual_bit_depth', '?')}-bit / {metadata.get('color_space', '未声明色彩空间')} / {icc}\n精度来源：{precision or '未声明'}；版本状态：{metadata.get('status', self.source_status)}；导出副本不修改当前交付。")
            elif label == "providers":
                selected = self.provider.currentText()
                current_model = self.model.text()
                current_connection = (self.provider_endpoint.text(), self.provider_secret_env.text(), self.provider_credential.currentText())
                already_loaded = bool(self.provider_rows)
                self.provider_rows = {entry["id"]: entry for entry in value.get("providers", [])}
                self.provider.blockSignals(True); self.provider.clear(); self.provider.addItems(list(self.provider_rows))
                for index, row in enumerate(self.provider_rows.values()):
                    self.provider.setItemData(index, row.get("display") or row["id"], Qt.ToolTipRole)
                if selected in self.provider_rows: self.provider.setCurrentText(selected)
                self.provider.blockSignals(False); self.select_provider(self.provider.currentText())
                self._select_preferred_local_provider()
                if already_loaded and selected == self.provider.currentText():
                    self.model.setText(current_model); self.provider_endpoint.setText(current_connection[0]); self.provider_secret_env.setText(current_connection[1]); self.provider_credential.setCurrentText(current_connection[2])
                self.refresh_generation_models()
            elif label == "job-status":
                observed_job = value.get('job') or {}
                if observed_job.get('status') in {'succeeded', 'failed', 'blocked', 'cancelled'}:
                    self.job_poll_timer.stop()
                    self.job_poll_terminal_observation = self.job_observation
                if self.handle_portrait_beauty_job(value): return
                if self.handle_reconstruction_job(value): return
                job = value.get("job") if isinstance(value, dict) else {}
                if isinstance(job, dict):
                    status = str(job.get("status") or "unknown")
                    self.operation_state.setText(str(job.get('message') or f"任务 {status} · {job.get('stage') or '处理中'}"))
                    if status=='succeeded': self.operation_state.setText('本次处理执行结束；正在读取作品缺项、证据与批准状态。')
                    if status in {"failed", "blocked", "cancelled"}:
                        detail = (job.get('error') or {}).get('message') if isinstance(job.get('error'), dict) else job.get('error')
                        self.operation_state.setText(str(detail or self.operation_state.text()) + '；任务已停止，可查看任务详情。')
                    self.job_details.setPlainText(json.dumps(job, ensure_ascii=False, indent=2))
                    if hasattr(self, 'simple_workspace'):
                        self._show_quick_job_progress(job)
                    if status in {"queued", "running", "cancelling"} and self.active_job_id:
                        self._schedule_job_poll(750)
                    elif status in {"succeeded", "failed", "blocked", "cancelled"}:
                        if status != 'succeeded': self.simple_creation = None
                        run_id = job.get("run_id")
                        self._session_commit_child_id = run_id if status == "succeeded" and self._session_submission else None
                        if status != 'succeeded' or not run_id:
                            self.render_pending = False; self.update_accept_enabled()
                        self.refresh_jobs()
                        if status == 'succeeded' and run_id:
                            self.task("GET", f"/api/source-runs/{run_id}", None, "job-run-loaded")
            elif label == "job-events":
                events = value.get("events") if isinstance(value, dict) else []
                if events: self.log_text("[job-events] " + json.dumps(events[-5:], ensure_ascii=False))
            elif label == "job-stream":
                event = value if isinstance(value, dict) else {}
                data = event.get("data") if isinstance(event.get("data"), dict) else {}
                if self.active_job_id == self.portrait_beauty_job_id:
                    self.operation_state.setText(str(data.get('message') or '正在观察人像美化准备任务。'))
                    self.portrait_beauty_state.setText(self.operation_state.text())
                    if data.get('status') in {'succeeded', 'failed', 'blocked', 'cancelled'}:
                        self.job_stream_active = False
                        self.task('GET', f'/api/jobs/{self.active_job_id}', None, 'job-status'); self.refresh_jobs()
                    return
                if hasattr(self, 'simple_workspace') and data:
                    self._show_quick_job_progress(data)
                if self.active_job_id == self.native_reconstruction_job_id:
                    self.operation_state.setText(str(data.get('message') or f'正在观察{self._reconstruction_name()}准备任务。'))
                    if data.get('status') in {'succeeded', 'failed', 'blocked', 'cancelled'}:
                        self.job_stream_active = False
                        self.task('GET', f'/api/jobs/{self.active_job_id}', None, 'job-status'); self.refresh_jobs()
                    return
                if data:
                    status = str(data.get("status") or "unknown")
                    self.operation_state.setText(str(data.get('message') or f"任务 {status} · {data.get('stage') or '处理中'}"))
                    if status=='succeeded': self.operation_state.setText('本次处理执行结束；正在读取作品缺项、证据与批准状态。')
                    if status in {"failed", "blocked", "cancelled"}: self.operation_state.setText(self.operation_state.text() + "；任务已停止，不会自动重试或重新生图。")
                    self.log_text("[job-stream] " + json.dumps(data, ensure_ascii=False))
                    if status in {"succeeded", "failed", "blocked", "cancelled"}:
                        if status != 'succeeded': self.simple_creation = None
                        self.job_stream_active = False
                        self.refresh_jobs()
                        run_id = data.get("run_id") or (data.get("data") or {}).get("run_id")
                        self._session_commit_child_id = run_id if status == "succeeded" and self._session_submission else None
                        if status == 'succeeded' and run_id: self.task("GET", f"/api/source-runs/{run_id}", None, "job-run-loaded")
                        else:
                            self.render_pending = False; self.update_accept_enabled()
            elif label == 'startup-result-loaded':
                context = self._startup_result_context
                self._startup_result_context = None
                run = value.get('run') if isinstance(value, dict) else None
                if (context and isinstance(run, dict) and run.get('run_id') == context['run_id']
                        and not self.source_run_id and not self.render_pending and not self._session_dirty
                        and not self.simple_creation and self.source_edit.text() == context['source_input']):
                    self.load_source_run(run, value.get('stage_report'), selection_explicit=context['explicit'])
                self.schedule_selected_service()
            elif label in {"source-run-loaded", "job-run-loaded"}:
                run = value.get("run") if isinstance(value, dict) else None
                if isinstance(run, dict):
                    if label == 'job-run-loaded' and (run.get('task_plan') or {}).get('shaping_request'):
                        self.operation_state.setText('形体候选已保留；请在形体预览中明确采用，或从历史中明确选择保存版本。')
                        return
                    terminal_saved_post = label == 'job-run-loaded' and not self.simple_creation
                    was_async_post = bool(label == 'job-run-loaded' and self.simple_creation
                                          and self.simple_creation.get('stage') in {'post', 'production'})
                    self.load_source_run(run, value.get("stage_report"), saved_submission=label == "job-run-loaded" and run.get("run_id") == self._session_commit_child_id)
                    if terminal_saved_post and self.source_run_id == run.get('run_id') and not self._session_dirty:
                        self.operation_state.setText(post_completion_message(run))
                    # Complete production and explicit post completion share
                    # this saved asynchronous callback. End the UI intent
                    # without starting another preparation chain.
                    if was_async_post and self.simple_creation and not self._active_production_job():
                        self.simple_creation = None; self._reset_quick_progress()
                        self.operation_state.setText(post_completion_message(run))
                        self.refresh_cos_workspace()
            elif label == "history-run-selected":
                run = value.get("run") if isinstance(value, dict) else None
                if isinstance(run, dict): self.load_source_run(run, value.get("stage_report"), selection_explicit=True)
            elif label == "status": self.load(value)
            elif label == "source-delivery-history":
                self.delivery_history.setPlainText(json.dumps(value, ensure_ascii=False, indent=2))
            elif label == "source-run-history":
                self.populate_source_history(value)
            elif label == "recovery":
                checked = {self.saved_export_versions.topLevelItem(i).data(0, Qt.UserRole) for i in range(self.saved_export_versions.topLevelItemCount()) if self.saved_export_versions.topLevelItem(i).checkState(0) == Qt.Checked}
                self.saved_export_versions.clear()
                for row in value.get('source_runs', []):
                    if not row.get('auto_draft'): continue
                    item = QTreeWidgetItem(self.saved_export_versions, [str(row['run_id']), str(row.get('status')), str(row.get('sample'))]); item.setData(0, Qt.UserRole, row['run_id']); item.setFlags(item.flags() | Qt.ItemIsUserCheckable); item.setCheckState(0, Qt.Checked if row['run_id'] in checked else Qt.Unchecked)
                source_current = value.get("source_current") if isinstance(value, dict) else None
                if source_current:
                    self.recovery_label.setText(f"恢复状态：可恢复单图 {source_current.get('sample')} / {source_current.get('status')} / revision={source_current.get('revision', 1)}")
                else:
                    self.recovery_label.setText("恢复状态：没有已保存原片版本；任务中心保留已提交任务。")
            elif label == "source-rollback":
                self.load_delivery_history()
            elif label in {"generate-reference","one-click","existing-design","source-accept","source-adjust","source-retry","source-rollback"}: QTimer.singleShot(500, self.refresh)
        def error(self, error, label=None):
            if str(label or '').startswith(('shaping-start:', 'shaping-candidate:')):
                execution = self.shaping_execution
                if execution and str(label).endswith(execution['token']):
                    execution['unknown'] = True
                    self.creative_task_start_pending = False; self.creative_task_execution_label = None
                    self.operation_state.setText('形体请求结果需要核查；当前保存版本保留，不会自动重发。')
                    self.update_creative_task_actions()
                return
            if str(label or '').startswith('startup-service:'):
                self.receive_startup_service(label,error=error);return
            if label=='startup-result-loaded':
                self._startup_result_context=None
                self.schedule_selected_service()
            if self.active_checked_task and label == self.active_checked_task['callback']:
                self.active_checked_task = None
            if label in {'generation-model-selected', 'provider-settings-saved'}:
                self.generation_model_save_active = None
                self.generation_model_save_error = True
                self.operation_state.setText('当前生成模型已选择，但默认选择未保存；可在配置模型中重试保存。')
                self.log_text('[generation model settings] ' + str(error)); self.refresh_generation_models(); return
            self.simple_cancel_submissions.discard(label)
            if (str(label or '').startswith('creative-task-plan:') or label in {'local-only', 'one-click', 'complete-post', 'portrait-beauty-job', 'design-reconstruct-job'}):
                self.simple_creation = None
            if str(label or '').startswith('creative-task-plan:'):
                context = self.creative_task_plan_request
                if context and context['label'] == label:
                    self.creative_task_plan_request = None; self.creative_task_plan = None
                    self.creative_plan_summary.setPlainText('计划检查未完成：' + str(error) + '\n未开始制作，可修正输入或配置后重新检查。')
                    self.operation_state.setText('制作计划未取得；未创建处理任务。'); self.update_creative_task_actions()
                return
            if self.creative_task_start_pending and label == self.creative_task_execution_label:
                retry_batch_id = self.creative_task_retry_batch_id; self.creative_task_retry_batch_id = None
                self.creative_task_start_pending = False; self.creative_task_execution_label = None
                self.creative_plan_summary.appendPlainText('\n本项重试结果待核实；正在读取批次状态，不会自动重发。' if retry_batch_id else '\n开始请求未完成；需要重新检查计划，系统不会自动重发。')
                self.update_creative_task_actions()
                if retry_batch_id: self.poll_batch(retry_batch_id)
            if label in {'model-setup', 'model-setup-saved'}:
                if label == 'model-setup-saved': self.model_setup_submission = None
                self.model_setup_summary.setText('模型配置未更新：' + str(error))
                if self.model_setup_dialog is not None: self.model_dialog_status.setText(self.model_setup_summary.text())
                self.log_text('[model setup] ' + str(error)); return
            if label == 'job-status' and self.active_job_id == self.native_reconstruction_job_id:
                self.operation_state.setText('本地任务状态连接中断；当前编辑保留，正在重新查询进度。')
                self._schedule_job_poll(1500)
                self.log_text('[native reconstruction status] ' + str(error)); return
            if label == 'job-status' and self.active_job_id == self.portrait_beauty_job_id:
                self.operation_state.setText('美化任务连接中断；当前编辑保留，正在重新查询进度。')
                self._schedule_job_poll(1500)
                self.log_text('[portrait beauty status] ' + str(error)); return
            if label == 'job-status':
                self._schedule_job_poll(1500)
                self.operation_state.setText('任务状态连接中断；正在重新查询进度。')
                self.log_text('[job status] ' + str(error)); return
            if label == 'portrait-beauty-job':
                self.render_pending = False; self.portrait_beauty_request = None; self.update_accept_enabled()
                self.operation_state.setText('人像美化任务未提交；当前编辑保留，可重试本地准备。')
                self.portrait_beauty_state.setText(self.operation_state.text()); self.log_text('[portrait beauty] ' + str(error)); return
            if label == 'design-reconstruct-job':
                self.render_pending = False; self.design_reconstruction_request = None
                self.update_accept_enabled()
                self.operation_state.setText(f'{self._reconstruction_name()}准备任务未提交；当前编辑保留，可重新提交本地准备。')
                self.source_light_state.setText(self.operation_state.text()); self.log_text('[native reconstruction] ' + str(error))
                return
            if str(label or '').startswith('native-live-detail:'):
                context=self.native_live_detail_context
                if context and label=='native-live-detail:'+context[0]:
                    self.native_live_detail_context=None;self.update_live_native_detail_actions();self.operation_state.setText('原尺寸预览未完成：'+str(error));self.log_text('[native detail] '+str(error))
                return
            if str(label or '').startswith('native-matte:'):
                context=self.native_matte_request
                if context and label=='native-matte:'+context[0]:
                    self.native_matte_request=None;self.native_matte_state.setText('人物边缘未更新：'+str(error));self.update_source_light_actions()
            if label and label.startswith(("source-light-", "source-color-auto:")):
                token = label.split(":", 1)[1]; context = self.source_light_requests.pop(token, None)
                self.update_source_light_actions()
                if self.source_color_auto_request == token:
                    self.source_color_auto_request = None; self.update_source_color_actions()
                if context == (self.source_run_id, self.source_light_version):
                    self.source_light_state.setText("局部光色请求未完成；可查看缺失原因后重试。未触发生图。")
                    self.log_text("[local-light] " + error)
                return
            if label == "source-preview":
                if self.project_view_selector.currentData() == "source": self.project_image_info.setText("原片预览未载入：" + error)
                return
            if label == "character-recognized": self.character_result.setText("角色分析失败：" + error + "；可重新识别或填写角色名后生成。")
            if label in {"source-adjust", "source-retry", "local-retry", "repair-final", "source-copy", "one-click", "complete-post", "local-only", "existing-design", "job-run-loaded", "source-select-variant"}:
                self.render_pending = False; self.update_accept_enabled()
                if label == 'source-adjust' and self.simple_creation and self.simple_creation.get('stage') == 'post_save_wait':
                    self.simple_creation = None
            if label in {'repair-final','local-retry'}:
                self.operation_state.setText('本地恢复未完成，已保留当前作品；不会自动重试或新增生图。')
                self.log_text('[ERROR] '+error); return
            if label == 'complete-post':
                self.operation_state.setText('本地重新制作未完成：' + str(error) + '；已保存版本保留。')
                self.log_text('[ERROR] ' + str(error)); self.refresh_cos_workspace(); return
            self.operation_state.setText("已有参考的后期请求未完成，请查看任务中心或日志；不会自动生图。" if label == "existing-design" else "本地处理请求未完成，请查看任务中心或日志；未触发生图。" if label == "local-only" else "生成请求未完成，请查看任务中心；不会自动重试生成。点击重新生图才会新增候选。" if label in {"one-click", "source-retry", "generate-reference"} else "操作状态：失败，请查看日志"); self.log_text("[ERROR] " + error)
        def job_stream_error(self, error):
            # A proxy, older backend, or a closed desktop connection can make
            # SSE unavailable.  The existing status/events poller remains the
            # compatibility path and will continue from the durable journal.
            self.job_stream_active = False
            self.log_text("[job-stream] unavailable; fallback to polling: " + error)
            self._schedule_job_poll(100)
        def job_stream_finished(self):
            # A bounded stream can end without an exception (for example
            # after its 60s server timeout).  Release the flag so the durable
            # polling fallback keeps observing the job.
            if self.job_stream_active:
                self.job_stream_active = False
                self._schedule_job_poll(100)
        def _schedule_job_poll(self, delay):
            if not self.active_job_id or self.job_poll_terminal_observation == self.job_observation: return
            if self.job_poll_timer.isActive() and self.job_poll_timer.remainingTime() <= delay: return
            self.job_poll_timer.start(delay)
        def start_job_stream(self):
            if not self.active_job_id or self.job_stream_active: return
            self.job_stream_active = True
            task = JobStreamTask(self.backend_url(), f"/api/jobs/{self.active_job_id}/events/stream")
            relay = TaskRelay(self, "job-stream", self.job_observation, stream=True)
            self.pending_tasks[relay] = task
            task.signals.result.connect(relay.result); task.signals.error.connect(relay.error); task.signals.finished.connect(relay.finished)
            self.pool.start(task)
        def poll_active_job(self):
            if not self.active_job_id or self.job_poll_terminal_observation == self.job_observation: return
            self.job_poll_timer.stop()
            self.task("GET", f"/api/jobs/{self.active_job_id}", None, "job-status")
            if not self.job_stream_active:
                self.task("GET", f"/api/jobs/{self.active_job_id}/events", None, "job-events")
        def refresh(self):
            if self.embedded_backend and self.embedded_server is None: return
            self.task("GET", "/api/status", None, "status")
            self.task("GET", "/api/recovery", None, "recovery")
            self.refresh_jobs()
            self.task("GET", "/api/providers", None, "providers")
            self.task("GET", "/api/provider-settings", None, "provider-settings")
            self.task("GET", "/api/templates", None, "templates")
            self.task("GET", "/api/recent", None, "recent")
            self.task('GET', '/api/model-setup', None, 'model-setup')

        def prewarm_startup(self):
            """Initialize only the selected service after settings are restored.

            Local Qwen exposes service startup, not load-only model warming.
            Beauty is loaded by an explicitly requested portrait operation.
            """
            if self.embedded_backend and self.embedded_server is None:
                self.set_startup_readiness('degraded')
                self.set_startup_status('本地后台尚未启动；请重试启动后检查制作计划。')
                return
            self._startup_service_enabled=True
            self.set_startup_readiness('ready')
            self.set_startup_status('后台可用；正在检查本次选择的服务，模型加载状态另行显示。')
            self.refresh();self.schedule_selected_service()

        def selected_startup_service(self):
            if (not self._startup_service_enabled or self._startup_service_closed or self._startup_service_suppressed
                    or not self.provider_settings_loaded or not self.provider_rows
                    or not self._startup_recent_loaded or self._startup_result_context is not None
                    or self.creative_task_mode.currentData()!='create_reference'
                    or self._production_mutation_busy() or self.simple_creation or self._session_loading):return None
            scope=self.editing_request_payload('create_reference')
            if not scope or scope['work_type'] not in {'partial_composite','full_composite'}:return None
            provider=self.provider.currentText();row=self.provider_rows.get(provider) or {}
            if provider not in {'qwen_image_2_1','qwen_image_2_1_uncensored','qwen_image_2_1_fast','qwen_image_2_1_uncensored_fast'} or row.get('ready') is not True:return None
            endpoint=self.provider_endpoint.text().strip() or row.get('endpoint')
            from start_qwen_image import ENDPOINT
            # External endpoints are operator owned; startup must not start,
            # stop, probe or implicitly use a remote provider.
            if not endpoint or endpoint.rstrip('/')!=ENDPOINT:return None
            return {'root':str(self.root_path),'backend_generation':self.backend_generation,
                'mode':'create_reference','work_type':scope['work_type'],'provider':provider,
                'model':self.model.text().strip(),'endpoint':endpoint.rstrip('/')}

        def schedule_selected_service(self,*_args):
            if not getattr(self,'_startup_service_enabled',False):return
            desired=self.selected_startup_service();current=self._startup_service_context
            if desired!=self._startup_service_last_target:
                if desired:self._startup_service_records.pop(json.dumps(desired,sort_keys=True),None)
                self._startup_service_last_target=desired
            if current and current['target']!=desired:
                current['cancel'].set();self._startup_service_records.pop(current['key'],None)
                self._startup_service_context=None
            self.startup_service_timer.stop()
            if desired and json.dumps(desired,sort_keys=True) not in self._startup_service_records:
                self.startup_service_timer.start()

        def start_selected_service(self):
            target=self.selected_startup_service()
            if not target:return
            key=json.dumps(target,sort_keys=True)
            if key in self._startup_service_records:return
            cancelled=threading.Event();context={'target':target,'key':key,'cancel':cancelled}
            self._startup_service_context=context
            self._startup_service_records[key]={'state':'starting','model_load_verified':False,
                'model_warming_performed':False,'image_submissions':0}
            label='startup-service:'+uuid.uuid4().hex
            task=StartupServiceTask(target['provider'],target['endpoint'],cancelled)
            relay=TaskRelay(self,label,self.job_observation)
            context['label']=label
            self.pending_tasks[relay]=task
            task.signals.result.connect(relay.result);task.signals.error.connect(relay.error);task.signals.finished.connect(relay.finished)
            self.set_startup_status('后台可用；正在初始化所选 Qwen 服务，模型加载状态尚未验证。')
            self.pool.start(task)

        def receive_startup_service(self,label,value=None,error=None):
            context=self._startup_service_context
            if (not context or context.get('label')!=label or context['cancel'].is_set()
                    or context['target']!=self.selected_startup_service()):return
            record=(value if error is None else {'state':'service_unavailable','model_load_verified':False,
                'model_warming_performed':False,'error':str(error),'image_submissions':0})
            self._startup_service_records[context['key']]=record
            self._startup_service_context=None
            provider=context['target']['provider'];row=self.provider_rows.get(provider)
            if row is not None:
                row.update(network_checked=error is None,
                    service_ready=True if error is None and record.get('state')=='service_ready' else None)
            self.set_startup_status('Qwen 服务可连接；模型加载未验证，首次生成可能仍需加载。' if error is None else
                'Qwen 服务初始化未完成；可检查本次计划，实际开始时重新核验服务。')
            if error is not None:self.log_text('[startup service] '+str(error))
            self.refresh_generation_models()

        def cancel_startup_service(self,*,suppress=False,closing=False):
            self.startup_service_timer.stop()
            if suppress:self._startup_service_suppressed=True
            if closing:self._startup_service_closed=True
            current=self._startup_service_context
            if current:
                current['cancel'].set();self._startup_service_records.pop(current['key'],None)
                self._startup_service_context=None

        def set_startup_readiness(self, state):
            self.startup_readiness = str(state)
            self.refresh_cos_workspace()

        def set_startup_status(self, message):
            # A late prewarm receipt must not overwrite an active task's status.
            self.models_button.setToolTip(str(message))
            if not (self.render_pending or self.creative_task_start_pending or self.simple_creation):
                self.operation_state.setText(str(message))

        def backend_url(self):
            return self.url if self.embedded_backend else self.url_edit.text().strip()

        def show_model_setup_status(self, setup):
            identity = lambda value: (value.get('config'), value.get('registry_path'), value.get('beauty'))
            if identity(setup) != identity(self.model_setup_status): self.invalidate_creative_task_plan()
            self.model_setup_status = setup
            missing = [row['label'] for row in setup.get('paths', []) if not row.get('available')]
            if setup.get('paths_available'):
                message = '分析模型路径已就绪'
                if 'beauty' in setup:
                    message += '；人像美化已就绪。' if setup['beauty']['available'] else '；人像美化待配置。'
            elif missing:
                message = '待配置或缺失：' + '、'.join(missing)
            else:
                message = '模型配置无法读取：' + str(setup.get('error') or '请选择配置')
            self.model_setup_summary.setText(message)
            self.model_setup_summary.setToolTip(str(setup.get('registry_path') or '') + '\n' + str(setup.get('error') or ''))
            if self.model_setup_dialog is not None:
                self.model_dialog_status.setText(message)
                if setup.get('model_directory'):
                    self.model_directory_label.setText('模型目录：'+setup['model_directory'])
                for index, group in enumerate(setup.get('groups') or []):
                    if index < len(self.model_group_states):
                        self.model_group_states[index].setText('已找到' if group['available'] else '待配置：'+'、'.join(group['missing']))
                beauty = setup.get('beauty') or {}
                self.beauty_model_state.setText('已找到 · BeautyGRPO + FLUX + SegFace' if beauty.get('available') else '待配置 · 点击自动查找或选择安装配置')
                self.beauty_model_state.setToolTip(str(beauty.get('config_path') or '')+'\n'+str(beauty.get('error') or ''))
                for row in setup.get('paths', []):
                    edit = self.model_path_edits[row['key']]
                    if not edit.isModified(): edit.setText(row.get('path') or '')
                    edit.setToolTip('路径可用' if row.get('available') else '待配置或文件缺失')
                    self.model_path_states[row['key']].setText('未保存' if edit.isModified() else '可用' if row.get('available') else '缺失')
                for row in (setup.get('discovery') or {}).get('paths', []):
                    edit = self.model_path_edits[row['key']]
                    if row['available'] and not edit.isModified():
                        edit.setText(row['path']); edit.setModified(True)
                        self.model_path_states[row['key']].setText('已找到 · 待保存')
            if ('beauty' in setup and not self.model_auto_discovery_attempted
                    and not (setup.get('paths_available') and setup['beauty'].get('available'))
                    and not self.render_pending and not self.simple_creation):
                self.model_auto_discovery_attempted = True
                self.discover_models()

        def open_model_setup(self):
            if self.model_setup_dialog is not None:
                self.model_setup_dialog.show(); self.model_setup_dialog.raise_(); return
            from current_model_registry import PATH_KEYS, PATH_LABELS
            dialog = QDialog(self); dialog.setWindowTitle('模型管理'); dialog.resize(890, 740)
            layout = QVBoxLayout(dialog)
            intro = QLabel('自动查找已有安装；如果模型放在其他位置，选择一次模型文件夹即可。配置会保存，重启和更新软件后继续使用。')
            intro.setWordWrap(True); layout.addWidget(intro)
            from current_model_registry import model_directory
            self.model_directory_label = QLabel('模型目录：'+str(model_directory()))
            self.model_directory_label.setWordWrap(True); self.model_directory_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(self.model_directory_label)
            self.model_dialog_status = QLabel(); self.model_dialog_status.setWordWrap(True); layout.addWidget(self.model_dialog_status)
            discovery = QHBoxLayout(); auto = QPushButton('自动查找已有模型'); folder = QPushButton('选择模型文件夹')
            discovery.addWidget(auto); discovery.addWidget(folder); discovery.addStretch(); layout.addLayout(discovery)
            auto.clicked.connect(lambda: self.discover_models()); folder.clicked.connect(self.choose_model_directory)
            body = QWidget(); body_layout = QVBoxLayout(body)
            from model_setup import GROUPS
            self.model_group_states = []
            for name, purpose, _keys in GROUPS:
                card = QGroupBox(name); card_layout = QHBoxLayout(card)
                description = QLabel(purpose); card_layout.addWidget(description)
                state = QLabel('尚未检查'); state.setWordWrap(True); card_layout.addWidget(state, 1)
                self.model_group_states.append(state); body_layout.addWidget(card)
            beauty_card = QGroupBox('人像美化'); beauty_layout = QVBoxLayout(beauty_card)
            self.beauty_model_state = QLabel('尚未检查'); self.beauty_model_state.setWordWrap(True)
            beauty_select = QPushButton('选择 BeautyGRPO 安装配置'); beauty_select.clicked.connect(self.select_beauty_config)
            beauty_layout.addWidget(self.beauty_model_state); beauty_layout.addWidget(beauty_select); body_layout.insertWidget(0, beauty_card)
            advanced = QWidget(); advanced_layout = QVBoxLayout(advanced)
            select_existing = QPushButton('导入分析模型配置 JSON'); advanced_layout.addWidget(select_existing)
            fields = QWidget(); form = QFormLayout(fields); self.model_path_edits = {}; self.model_path_states = {}
            for key in PATH_KEYS:
                field = QWidget(); row = QHBoxLayout(field); row.setContentsMargins(0, 0, 0, 0)
                edit = QLineEdit(); edit.setPlaceholderText('选择本机已安装路径'); self.model_path_edits[key] = edit
                state = QLabel('缺失'); self.model_path_states[key] = state
                browse = QPushButton('选择'); row.addWidget(edit, 1); row.addWidget(state); row.addWidget(browse)
                browse.clicked.connect(lambda _checked=False, key=key: self.choose_model_path(key))
                form.addRow(PATH_LABELS[key], field)
            advanced_layout.addWidget(fields)
            advanced_panel, _toggle = self.accordion('高级：逐项路径与配置 JSON', advanced, expanded=False)
            body_layout.addWidget(advanced_panel)
            layout.addWidget(self.scrollable_page(body), 1)
            controls = QHBoxLayout(); save = QPushButton('保存并使用'); check = QPushButton('重新检查'); close = QPushButton('关闭')
            controls.addWidget(save); controls.addWidget(check); controls.addStretch(); controls.addWidget(close); layout.addLayout(controls)
            save.clicked.connect(self.save_model_paths); check.clicked.connect(lambda: self.task('GET', '/api/model-setup', None, 'model-setup'))
            select_existing.clicked.connect(self.select_model_registry); close.clicked.connect(dialog.close)
            self.model_setup_dialog = dialog
            self.show_model_setup_status(self.model_setup_status)
            dialog.finished.connect(lambda _result: setattr(self, 'model_setup_dialog', None))
            dialog.show(); self.task('GET', '/api/model-setup', None, 'model-setup')

        def discover_models(self, directory=None):
            if self.model_setup_submission is not None or self.render_pending or self.simple_creation: return
            if self.embedded_backend and self.embedded_server is None: return
            self.model_auto_discovery_attempted = True
            self.model_setup_submission = {key: edit.text() for key, edit in self.model_path_edits.items()} if self.model_setup_dialog else {}
            body = {'action': 'discover'}
            if directory: body['directory'] = directory
            self.model_setup_summary.setText('正在查找本机已有模型…')
            if self.model_setup_dialog: self.model_dialog_status.setText('正在查找本机已有模型…')
            self.task('POST', '/api/model-setup', body, 'model-setup-saved')

        def choose_model_directory(self):
            from current_model_registry import model_directory
            directory = QFileDialog.getExistingDirectory(self.model_setup_dialog, '选择模型总目录或原源码项目目录', str(model_directory()))
            if directory: self.discover_models(directory)

        def select_beauty_config(self):
            if self.model_setup_submission is not None: return
            path, _filter = QFileDialog.getOpenFileName(self.model_setup_dialog, '选择 BeautyGRPO installation.json', str(Path.home()), 'JSON (*.json)')
            if path:
                self.model_setup_submission = {key: edit.text() for key, edit in self.model_path_edits.items()}
                self.task('POST', '/api/model-setup', {'beauty_config_path': path}, 'model-setup-saved')

        def choose_model_path(self, key):
            from current_model_registry import DIRECTORY_KEYS, PATH_LABELS
            current = self.model_path_edits[key].text().strip() or str(self.root_path)
            if key in DIRECTORY_KEYS:
                selected = QFileDialog.getExistingDirectory(self.model_setup_dialog, PATH_LABELS[key], current)
            else:
                selected, _filter = QFileDialog.getOpenFileName(self.model_setup_dialog, PATH_LABELS[key], current)
            if selected:
                self.model_path_edits[key].setText(str(Path(selected).resolve())); self.model_path_edits[key].setModified(True)

        def select_model_registry(self):
            if self.model_setup_submission is not None: return
            path, _filter = QFileDialog.getOpenFileName(self.model_setup_dialog, '选择模型配置', str(self.root_path), 'JSON (*.json)')
            if path:
                if self.embedded_backend and self.embedded_server is None: return
                self.model_setup_submission = {key: edit.text() for key, edit in self.model_path_edits.items()}
                self.task('POST', '/api/model-setup', {'registry_path': path}, 'model-setup-saved')

        def save_model_paths(self):
            if self.model_setup_submission is not None or self.embedded_backend and self.embedded_server is None: return
            from current_model_registry import SCHEMA, PROFILES, PATH_KEYS
            values = {key: self.model_path_edits[key].text().strip() for key in PATH_KEYS}
            self.model_setup_submission = {key: edit.text() for key, edit in self.model_path_edits.items()}
            self.task('POST', '/api/model-setup', {'config': {'schema': SCHEMA, 'ready': True, **PROFILES, **values}}, 'model-setup-saved')

        def current_model_options(self):
            # The backend owns configuration, including in source external mode.
            if self.model_setup_status.get('configured') and self.model_setup_status.get('registry_path'):
                return {'algorithm_profile': 'dinov3_vitl16',
                        'material_finish_profile': 'owner_adaptive_multiscale_skin_v2',
                        'model_registry': self.model_setup_status['registry_path']}
            return None

        def build_editing_scope_panel(self):
            from photo_editing import WORK_TYPES, STYLES
            panel = QWidget(); form = QVBoxLayout(panel); form.setContentsMargins(0, 0, 0, 0)
            work = QComboBox(); work.setAccessibleName('制作范围')
            for value, label in WORK_TYPES.items(): work.addItem(label, value)
            style = QComboBox(); style.setAccessibleName('风格表达')
            for value, label in STYLES.items(): style.addItem(label, value)
            setting = QComboBox(); setting.setAccessibleName('明确的室内外目标')
            for label, value in (('保持原片室内外关系', 'same'), ('明确改为室内', 'indoor'), ('明确改为室外', 'outdoor')):
                setting.addItem(label, value)
            scene = QLineEdit(); scene.setAccessibleName('大合成目标场景')
            scene.setPlaceholderText('明确的新场景；留空时不授权改变室内外')
            target = QWidget(); target_form = QVBoxLayout(target); target_form.setContentsMargins(0, 0, 0, 0)
            target_form.addWidget(QLabel('大合成目标')); target_form.addWidget(setting); target_form.addWidget(scene)
            note = QLabel(); note.setWordWrap(True); note.setObjectName('caption')
            review = QCheckBox('独立审片（额外发送成片）'); review.setAccessibleName('独立审片')
            review.setToolTip('使用已配置的文本/视觉服务比较原片与成片，增加调用、耗时与可能的费用；可关闭。自动评价不等于人工批准。')
            autofinish = QCheckBox('自动完成：未通过初审时最多再改两轮')
            autofinish.setAccessibleName('默认自动纠偏')
            autofinish.setToolTip('首次任务内按需要生成编辑或做 CPU 摄影融合，每轮独立复审。默认最多额外两次GPU生图、四次视觉调用，可能增加费用与等待；CPU工具不新增生图，初审通过不追加。可关闭。')
            form.addWidget(QLabel('制作范围')); form.addWidget(work)
            form.addWidget(QLabel('风格表达')); form.addWidget(style); form.addWidget(target); form.addWidget(review); form.addWidget(autofinish); form.addWidget(note)
            controls = {'panel': panel, 'work_type': work, 'style': style, 'target_setting': setting,
                        'scene_target': scene, 'target': target, 'review': review, 'autofinish':autofinish, 'note': note}
            self.editing_scope_panels.append(controls)
            for key in ('work_type', 'style', 'target_setting'):
                controls[key].currentIndexChanged.connect(lambda _index, key=key, controls=controls:
                    self.editing_scope_changed(key, controls[key].currentData()))
            scene.textChanged.connect(lambda value: self.editing_scope_changed('scene_target', value))
            review.toggled.connect(lambda enabled: self.creative_visual_review.setChecked(enabled))
            autofinish.toggled.connect(self.set_editing_autofinish)
            return panel

        def editing_autofinish_payload(self, mode=None):
            mode=mode or self.creative_task_mode.currentData()
            scope=self.editing_request_payload(mode)
            if mode!='create_reference' or not scope:return None
            enabled=self.editing_autofinish_enabled and self.creative_visual_review.isChecked()
            return {'maximum_improvements':self.editing_autofinish_maximum if enabled else 0,
                'maximum_zero_image_finishes':self.editing_autofinish_zero_finishes if enabled else 0}

        def set_editing_autofinish(self, enabled):
            self.editing_autofinish_enabled=bool(enabled)
            self.invalidate_creative_task_plan()
            self.sync_editing_scope_controls()

        def apply_editing_autofinish_options(self, value):
            if not isinstance(value,dict):return
            count=value.get('maximum_improvements')
            if type(count) is not int or not 0<=count<=8:return
            zero_finishes=value.get('maximum_zero_image_finishes',0)
            if type(zero_finishes) is not int or not 0<=zero_finishes<=1 or (zero_finishes and not count):return
            self.editing_autofinish_enabled=bool(count)
            self.editing_autofinish_zero_finishes=zero_finishes
            if count:self.editing_autofinish_maximum=count
            self.invalidate_creative_task_plan();self.sync_editing_scope_controls()

        def editing_request_payload(self, mode=None):
            mode = mode or self.creative_task_mode.currentData()
            if mode == 'existing_reference': return None
            request = copy.deepcopy(self.editing_requests[mode])
            if mode == 'local_portrait': request['work_type'] = 'retouch'
            if request['work_type'] != 'full_composite':
                request.update(target_setting='same', scene_target='')
            return request

        def editing_scope_changed(self, key, value):
            mode = self.creative_task_mode.currentData()
            if mode == 'existing_reference': return
            request = self.editing_requests[mode]
            if key in {'scene_target', 'target_setting'} and request['work_type'] != 'full_composite': return
            request[key] = value
            if key == 'work_type':
                # A previous scene change does not silently authorize a new one.
                request.update(target_setting='same', scene_target='')
                work = request['work_type']
                target_mode = 'create_reference' if mode == 'local_portrait' and work != 'retouch' else mode
                if mode != target_mode:
                    self.editing_requests[target_mode] = copy.deepcopy(request)
                    self.creative_task_mode.setCurrentIndex(self.creative_task_mode.findData(target_mode))
                self.set_background_mode({'retouch': 'preserve', 'partial_composite': 'enhance_source', 'full_composite': 'replace'}[work])
                self.creative_no_effects.setChecked(work == 'retouch')
            self.clear_generation_design_view(); self.invalidate_creative_task_plan()
            self.update_creative_service_actions()

        def apply_editing_request_options(self, value):
            mode = self.creative_task_mode.currentData()
            if not isinstance(value, dict) or mode == 'existing_reference': return
            from photo_editing import editing_scope
            scope = editing_scope(value, local=mode == 'local_portrait')
            self.editing_requests[mode] = {key: scope[key] for key in ('work_type', 'style', 'target_setting', 'scene_target', 'language')}
            self.set_background_mode({'retouch': 'preserve', 'partial_composite': 'enhance_source', 'full_composite': 'replace'}[scope['work_type']])
            self.clear_generation_design_view(); self.invalidate_creative_task_plan()
            self.update_creative_service_actions()

        def sync_editing_scope_controls(self, *, busy=None):
            if not hasattr(self, 'creative_task_mode'): return
            mode = self.creative_task_mode.currentData()
            request = self.editing_request_payload(mode)
            saved_scope = ((self.product_run.get('task_plan') or {}).get('editing_scope') or {})
            saved_improvement = bool(mode == 'existing_reference' and self.source_generated_appearance and saved_scope)
            display = request or (saved_scope if saved_improvement else self.editing_requests[mode])
            if busy is None: busy = bool(self.render_pending or self.creative_task_start_pending or self._active_production_job())
            for controls in self.editing_scope_panels:
                for key in ('work_type', 'style', 'target_setting', 'scene_target'):
                    widget = controls[key]; widget.blockSignals(True)
                    value = display[key]
                    if key == 'scene_target': widget.setText(value)
                    else: widget.setCurrentIndex(widget.findData(value))
                    widget.blockSignals(False)
                    widget.setEnabled(not busy and request is not None)
                controls['target'].setVisible(bool((request or saved_improvement) and display['work_type'] == 'full_composite'))
                if hasattr(self, 'creative_visual_review'):
                    controls['review'].blockSignals(True)
                    controls['review'].setChecked(self.creative_visual_review.isChecked())
                    controls['review'].blockSignals(False)
                    controls['review'].setEnabled(not busy and not saved_improvement and mode != 'local_portrait'
                        and self.creative_visual_review.isEnabled())
                    policy=self.editing_autofinish_payload(mode)
                    controls['autofinish'].blockSignals(True)
                    controls['autofinish'].setChecked(bool(policy and policy['maximum_improvements']))
                    finishes=self.editing_autofinish_zero_finishes
                    controls['autofinish'].setText(f'自动完成：首次编辑后最多再改 {self.editing_autofinish_maximum} 轮，另有 {finishes} 轮零生图收尾')
                    controls['autofinish'].setToolTip(f'先由模型选择首次编辑，其后最多额外 {self.editing_autofinish_maximum} 次图像额度和 {finishes} 轮零生图收尾，共最多 {2*(self.editing_autofinish_maximum+finishes)} 次额外视觉调用。GPU或外部模型增加费用与等待；最后收尾仅提供零生图工具，达到目标即停止，可关闭额外纠偏。')
                    controls['autofinish'].blockSignals(False)
                    controls['autofinish'].setVisible(policy is not None)
                    controls['autofinish'].setEnabled(not busy and policy is not None and self.creative_visual_review.isChecked())
                controls['note'].setText('这里显示该保存版本的制作范围；按独立审片改进会沿用此范围并重新审片。'
                    if saved_improvement else
                    '已有参考沿用原有输入路线；观察式制作范围暂不支持导入参考。' if request is None else
                    '保留本人、妆造、姿态、体型与重要道具。精修使用本地处理，不发送外部视觉服务。' if mode == 'local_portrait' else
                    '保留本人、妆造、姿态、体型与重要道具。制作时向视觉服务发送原片与要求，可能产生费用；独立审片额外发送成片。')
            if hasattr(self, 'creative_no_effects'):
                self.creative_no_effects.setEnabled(not busy and not (request and request['work_type'] == 'retouch'))
            if hasattr(self, 'simple_workspace'):
                self.simple_workspace.no_effects.setEnabled(not busy and not (request and request['work_type'] == 'retouch'))
                self.simple_workspace.regional.setVisible(mode == 'existing_reference')
                self.simple_workspace.select_region.setVisible(mode == 'existing_reference')
            if hasattr(self, 'regional_controls'):
                self.regional_controls.setVisible(mode == 'existing_reference')
            # Scope owns scene permission for source-based work. Imported references
            # keep their existing scene and matching-reference controls.
            for control in (getattr(self, 'creative_background_mode', None),
                            getattr(getattr(self, 'simple_workspace', None), 'background_mode', None)):
                if control is not None:
                    control.setVisible(mode == 'existing_reference')
                    form = control.parentWidget().layout(); index = form.indexOf(control)
                    if index > 0 and isinstance(form.itemAt(index-1).widget(), QLabel):
                        form.itemAt(index-1).widget().setVisible(mode == 'existing_reference')

        def creative_task_payload(self, mode=None):
            mode = mode or self.creative_task_mode.currentData()
            scoped = self.editing_request_payload(mode)
            background = ({'retouch': 'preserve', 'partial_composite': 'enhance_source', 'full_composite': 'replace'}[scoped['work_type']]
                          if scoped else self.creative_background_mode.currentData())
            constraints = {'preserve_background': background == 'preserve',
                           'no_effects': self.creative_no_effects.isChecked() or bool(scoped and scoped['work_type'] == 'retouch')}
            if background == 'enhance_source': constraints['background_mode'] = background
            if self.same_photo_finish_enabled and mode == 'existing_reference' and background == 'enhance_source':
                constraints['reference_role'] = 'same_photo_finish'
            elif (background == 'enhance_source' and not self.regional_enabled and
                    mode == 'existing_reference' and self.source_conditioned_reference):
                constraints['reference_role'] = 'source_conditioned_design'
            if (mode in {'create_reference', 'existing_reference'}
                    and constraints.get('reference_role') == 'source_conditioned_design'
                    and self.preset.currentData() == 'source_refinement' and self.refinement_profile_choice):
                constraints['refinement_profile'] = copy.deepcopy(self.refinement_profile_choice)
            payload = {'schema_version': 'cosplay_task_intent.v1', 'mode': mode,
                'brief': self.creative_task_brief.toPlainText().strip(),
                'constraints': constraints,
                'services': {'text_vision': (True if mode == 'create_reference' and scoped else
                                             self.creative_text_vision.isChecked() if mode != 'local_portrait' and background != 'enhance_source' else False),
                             'visual_review': (mode != 'local_portrait' and self.creative_text_vision.isChecked()
                                               and (scoped is not None or background != 'enhance_source')
                                               and self.creative_visual_review.isChecked())}}
            if mode == 'existing_reference' and background == 'enhance_source' and self.regional_enabled and not self.same_photo_finish_enabled:
                payload['regional_refinement'] = {'maximum_submissions': 1, 'coordinate_space': 'source_canvas',
                    'regions': [{'id': 'bg-1', 'core_box': list(self.regional_core_box)}] if self.regional_core_box else []}
            return payload

        def set_background_mode(self, value):
            index = self.creative_background_mode.findData(value)
            if index >= 0: self.creative_background_mode.setCurrentIndex(index)

        def background_mode_changed(self, *_args):
            if not self.same_photo_finish_available(): self.same_photo_finish_enabled = False
            self.creative_preserve_background.blockSignals(True)
            self.creative_preserve_background.setChecked(self.creative_background_mode.currentData() == 'preserve')
            self.creative_preserve_background.blockSignals(False)
            self.creative_task_policy_changed()

        def set_regional_enabled(self, enabled):
            if self.same_photo_finish_enabled or self.editing_request_payload() is not None: return
            self.regional_enabled = bool(enabled)
            self.invalidate_creative_task_plan()

        def same_photo_finish_available(self):
            return (self.creative_task_mode.currentData() == 'existing_reference'
                    and self.creative_background_mode.currentData() == 'enhance_source')

        def set_same_photo_finish(self, enabled):
            self.same_photo_finish_enabled = bool(enabled) and self.same_photo_finish_available()
            if self.same_photo_finish_enabled:
                self.source_conditioned_reference = False
                self.regional_enabled = False; self.regional_core_box = None
                self.simple_workspace.select_region.setChecked(False)
            self.invalidate_creative_task_plan()

        def source_native_size(self):
            size = self.project_dimensions.get('source')
            return list(map(int, size)) if isinstance(size, (list, tuple)) and len(size) == 2 and min(size) > 0 else None

        def set_regional_box(self, box):
            size = self.source_native_size()
            if not size or not isinstance(box, list) or len(box) != 4: return
            left, top, right, bottom = map(int, box)
            if not (0 <= left < right <= size[0] and 0 <= top < bottom <= size[1]) or right-left > 768 or bottom-top > 768:
                self.operation_state.setText('请选择原片范围内、不超过 768 × 768 像素的背景区域。'); return
            self.regional_core_box = [left, top, right, bottom]
            self.invalidate_creative_task_plan()

        def set_normalized_regional_box(self, box):
            size = self.source_native_size()
            if not size: return
            left, top = max(0, round(box[0]*size[0])), max(0, round(box[1]*size[1]))
            right = min(size[0], left+768, max(left+1, round(box[2]*size[0])))
            bottom = min(size[1], top+768, max(top+1, round(box[3]*size[1])))
            self.set_regional_box([left, top, right, bottom])

        def update_creative_service_actions(self):
            mode = self.creative_task_mode.currentData()
            scoped = self.editing_request_payload(mode)
            observed = scoped is not None and mode == 'create_reference'
            local_direction = self.creative_background_mode.currentData() == 'enhance_source'
            if observed:
                self.creative_text_vision.blockSignals(True); self.creative_text_vision.setChecked(True); self.creative_text_vision.blockSignals(False)
            elif local_direction:
                for control in (self.creative_text_vision, self.creative_visual_review):
                    control.blockSignals(True); control.setChecked(False); control.blockSignals(False)
            self.creative_text_vision.setEnabled(mode != 'local_portrait' and not local_direction and not observed)
            self.creative_text_vision.setToolTip('本次制作在执行阶段观察原片并设计；检查计划不发送图片。' if observed else
                '已有参考的原景增强使用已绑定设计。' if local_direction else '可选择额外外部角色研究与视觉规划。')
            allowed = self.creative_task_payload()['services']['text_vision']
            if not allowed:
                self.creative_visual_review.blockSignals(True); self.creative_visual_review.setChecked(False); self.creative_visual_review.blockSignals(False)
            self.creative_visual_review.setEnabled(allowed)
            reason = ('原片观察与设计由本次已检查的制作任务执行，避免额外预先规划。' if observed else
                      '原景增强使用本地检查设计；当前路线不外发规划。' if local_direction else
                      '会向已配置的文本/视觉服务发送原片与创作要求。' if allowed else '当前任务未允许外部文本/视觉服务；选择允许服务的路线并启用该服务后才能识别或研究。')
            self.character_recognize.setToolTip(reason)
            self.character_recognize.setEnabled(allowed and not observed)
            self.sync_editing_scope_controls()

        def allow_creative_text_vision(self):
            if self.creative_task_mode.currentData() == 'create_reference' and self.editing_request_payload() is not None:
                self.operation_state.setText('本次原片观察与设计在已检查的制作中执行；未提交额外规划。')
                return False
            if self.creative_task_payload()['services']['text_vision']: return True
            self.operation_state.setText('当前任务未允许文本/视觉服务；识别与研究未提交。')
            self.update_creative_service_actions(); return False

        def creative_task_mode_changed(self, *_args):
            self._startup_service_suppressed=False
            mode = self.creative_task_mode.currentData(); policy = self.creative_task_policies[mode]
            if mode != getattr(self, '_regional_task_mode', mode):
                self.regional_enabled = False
                self.same_photo_finish_enabled = False
                self.source_conditioned_reference = False
            self._regional_task_mode = mode
            for control, key in ((self.creative_preserve_background, 'preserve_background'),
                                 (self.creative_no_effects, 'no_effects'), (self.creative_text_vision, 'text_vision'),
                                 (self.creative_visual_review, 'visual_review')):
                control.blockSignals(True); control.setChecked(policy[key]); control.blockSignals(False)
            self.creative_background_mode.blockSignals(True)
            background = policy.get('background_mode', 'preserve' if policy['preserve_background'] else 'replace')
            scoped = self.editing_request_payload(mode)
            if scoped:
                background = {'retouch': 'preserve', 'partial_composite': 'enhance_source', 'full_composite': 'replace'}[scoped['work_type']]
            self.creative_background_mode.setCurrentIndex(self.creative_background_mode.findData(background))
            self.creative_background_mode.blockSignals(False)
            if not self.same_photo_finish_available(): self.same_photo_finish_enabled = False
            self.creative_text_vision.setEnabled(mode != 'local_portrait')
            self.creative_task_reference.setVisible(mode == 'existing_reference')
            self.clear_generation_design_view()
            self.invalidate_creative_task_plan()
            self.update_creative_service_actions()
            self.schedule_selected_service()

        def creative_task_policy_changed(self, *_args):
            mode = self.creative_task_mode.currentData()
            self.creative_task_policies[mode] = {'preserve_background': self.creative_preserve_background.isChecked(),
                'background_mode': self.creative_background_mode.currentData(),
                'no_effects': self.creative_no_effects.isChecked(),
                'text_vision': self.creative_text_vision.isChecked() if mode != 'local_portrait' else False,
                'visual_review': self.creative_task_payload()['services']['visual_review']}
            self.clear_generation_design_view()
            self.invalidate_creative_task_plan()
            self.update_creative_service_actions()

        def apply_creative_task_options(self, value):
            if not isinstance(value, dict) or value.get('schema_version') != 'cosplay_task_intent.v1': return
            mode = value.get('mode')
            if mode not in CREATIVE_TASK_MODES: return
            local = mode == 'local_portrait'
            constraints = value.get('constraints') or {}; services = value.get('services') or {}
            selected_profile = constraints.get('refinement_profile')
            self.refinement_profile_choice = ({key: copy.deepcopy(selected_profile[key]) for key in ('id', 'version')}
                if isinstance(selected_profile, dict) and all(key in selected_profile for key in ('id', 'version')) else None)
            self.creative_task_policies[mode] = {'preserve_background': bool(constraints.get('preserve_background', local)),
                'background_mode': constraints.get('background_mode') or ('preserve' if constraints.get('preserve_background', local) else 'replace'),
                'no_effects': bool(constraints.get('no_effects', local)), 'text_vision': bool(services.get('text_vision', False)) and not local,
                'visual_review': bool(services.get('visual_review', False)) and bool(services.get('text_vision', False)) and not local}
            self.regional_enabled = False; self.regional_core_box = None
            self.creative_task_mode.setCurrentIndex(self.creative_task_mode.findData(mode))
            self.creative_task_mode_changed(); self.creative_task_brief.setPlainText(str(value.get('brief') or ''))
            self.same_photo_finish_enabled = self.same_photo_finish_available() and constraints.get('reference_role') == 'same_photo_finish'
            self.source_conditioned_reference = constraints.get('reference_role') == 'source_conditioned_design'
            self.invalidate_creative_task_plan()

        def invalidate_creative_task_plan(self, *_args):
            had_plan = self.creative_task_plan is not None or self.creative_task_plan_request is not None
            if had_plan and self.simple_creation and self.simple_creation.get('plan_label'):
                self.simple_creation = None
            self.creative_task_revision += 1; self.creative_task_plan = None; self.creative_task_plan_request = None
            if had_plan: self.creative_plan_summary.setPlainText('输入、要求或配置已改变；请重新检查当前制作计划。')
            self.update_creative_task_actions()

        def update_creative_task_actions(self):
            busy = self.render_pending or self.creative_task_start_pending
            self.creative_plan_button.setEnabled(not busy and self.creative_task_plan_request is None)
            self.creative_start_button.setEnabled(not busy and bool(self.creative_task_plan)
                and self.creative_task_plan['plan'].get('ready') is True)
            self.refresh_cos_workspace()

        def creative_task_request_options(self, mode, *, source=None, batch=False):
            # Shared artistic options contain no source-coordinate editing assets.
            settings = self.generation_request_options() if mode == 'create_reference' else self._template_payload()
            keys = ('preset', 'provider', 'model', 'vision_model', 'variant_count', 'variant_styles', 'model_registry',
                    'character_name', 'character_notes', 'algorithm_profile', 'material_finish_profile')
            request = {key: copy.deepcopy(settings[key]) for key in keys if key in settings}
            request.update(prompt=self.generation_input_prompt(), input_prompt=self.generation_input_prompt(),
                           creative_task=self.creative_task_payload(mode))
            scoped = self.editing_request_payload(mode)
            if scoped is not None:
                request['editing_request'] = scoped
                request.pop('variant_styles', None)
            auto=self.editing_autofinish_payload(mode)
            if auto is not None:request['editing_autofinish']=auto
            if batch:
                request['creative_task'].pop('regional_refinement', None)
            if request['creative_task']['constraints'].get('background_mode') == 'enhance_source':
                request['variant_count'] = 1
                request.pop('variant_styles', None)
                request.pop('defer_reference_selection', None)
            if mode == 'create_reference':
                count = int(request.get('variant_count') or 1)
                if request['creative_task']['constraints'].get('background_mode') == 'enhance_source':
                    count = 1
                    request.pop('variant_styles', None)
                request['variant_count'] = count
                request['defer_reference_selection'] = count > 1
            return request

        def check_creative_task_plan(self):
            mode = self.creative_task_mode.currentData()
            if mode == 'existing_reference' and not self.creative_task_reference.text().strip():
                self.use_existing_design_reference(); return
            source = self.source_edit.text().strip()
            parent = (self.source_run_id if self.source_run_id and self.source_run_source and source
                and Path(source).resolve() == Path(self.source_run_source).resolve()
                and not self._session_dirty and self.source_status in {'needs_visual_review','accepted','delivered'} else None)
            self.request_creative_task_plan(mode, parent_run_id=parent)

        def request_creative_task_plan(self, mode, *, source=None, reference=None, batch=False,
                                       parent_run_id=None, auto_start=False, prepare_existing_beauty=False):
            if self.render_pending or self.creative_task_start_pending: return
            if self.embedded_backend and self.embedded_server is None:
                self.operation_state.setText('请先启动当前工程的后台。'); return
            self.creative_task_mode.setCurrentIndex(self.creative_task_mode.findData(mode))
            if self.creative_task_plan_request is not None: return
            request = self.creative_task_request_options(mode, source=source, batch=batch)
            if (request['creative_task'].get('regional_refinement') or {}).get('maximum_submissions'):
                if not self.regional_core_box:
                    self.operation_state.setText('已开启一次局部精修；请先在原片框选背景区域或设置位置尺寸。'); return
            if batch:
                if mode == 'existing_reference': request['items'] = copy.deepcopy(self.local_batch_pairs)
                else: request['sources'] = list(self.batch_sources)
            else:
                request['source'] = source or self.source_edit.text().strip()
                if not request['source']:
                    QMessageBox.information(self, '缺少原图', '请选择一张 JPG/RAW 原图。'); return
                if mode == 'existing_reference': request['reference'] = reference or self.creative_task_reference.text().strip()
                if parent_run_id: request['parent_run_id'] = parent_run_id
            self.submit_creative_task_plan(request, mode=mode, batch=batch, auto_start=auto_start,
                prepare_existing_beauty=prepare_existing_beauty)

        def submit_creative_task_plan(self, request, *, mode, batch=False, auto_start=False,
                                      prepare_existing_beauty=False, improvement_parent=None, shaping_parent=None):
            if self._production_mutation_busy() or self.creative_task_plan_request is not None: return
            if self.embedded_backend and self.embedded_server is None:
                self.operation_state.setText('请先启动当前工程的后台。'); return
            label = 'creative-task-plan:' + uuid.uuid4().hex
            callback = ('local-batch-started' if mode == 'existing_reference' else 'batch-started') if batch else {
                'local_portrait': 'local-only', 'existing_reference': 'existing-design', 'create_reference': 'one-click'}[mode]
            self.creative_task_plan = None
            self.creative_task_plan_request = {'label': label, 'revision': self.creative_task_revision,
                'request': request, 'batch': batch, 'callback': callback,
                'auto_start': bool(auto_start), 'prepare_existing_beauty': bool(prepare_existing_beauty)}
            if improvement_parent is not None:
                self.creative_task_plan_request['improvement_parent'] = copy.deepcopy(improvement_parent)
            if shaping_parent is not None:
                self.creative_task_plan_request.update(shaping_parent=copy.deepcopy(shaping_parent),
                    shaping_revision=self.shaping_revision, callback='shaping-start:'+uuid.uuid4().hex)
            self.creative_plan_summary.setPlainText('正在读取输入与配置，检查制作计划；不上传、不推理、不开始制作。')
            self.creative_plan_toggle.setChecked(True); self.update_creative_task_actions()
            self.task('POST', '/api/tasks/batch/plan' if batch else '/api/tasks/plan', request, label)

        def receive_creative_task_plan(self, label, value):
            context = self.creative_task_plan_request
            if not context or context['label'] != label or context['revision'] != self.creative_task_revision: return
            if context.get('shaping_parent'):
                self.creative_task_plan_request = None
                state = self.shaping_parent_state()
                if (not state['enabled'] or state.get('parent') != context['shaping_parent']
                        or context['shaping_revision'] != self.shaping_revision):
                    self.creative_task_plan = None
                    self.creative_plan_summary.setPlainText('当前保存版本或形体选择已变化；计划作废，未开始制作。')
                    self.update_creative_task_actions(); return
            if context.get('improvement_parent'):
                # Ignore this request's pending flag when rechecking its parent.
                self.creative_task_plan_request = None
                state = self.editing_improvement_state()
                if not state['enabled'] or state.get('parent') != context['improvement_parent']:
                    self.creative_task_plan = None
                    self.creative_plan_summary.setPlainText('原保存版本、图片或独立审片已变化；改进计划作废，未开始制作。')
                    self.update_creative_task_actions(); return
            self.creative_task_plan_request = None
            plan = value.get('batch_plan' if context['batch'] else 'plan') or {}
            request = value.get('request')
            if value.get('ok') is not True or not plan.get('plan_id') or not isinstance(request, dict):
                self.simple_creation = None
                self.creative_task_plan = None; self.creative_plan_summary.setPlainText('计划未返回有效身份或规范请求；未开始制作。')
                self.update_creative_task_actions(); return
            self.creative_task_plan = {**context, 'plan': copy.deepcopy(plan), 'request': copy.deepcopy(request)}
            if context['batch']:
                budget = plan.get('image_budget_total') or {}
                lines = ['共享要求批量计划 · ' + str(len(plan.get('items', []))) + ' 张',
                    '各图独立准备几何、遮罩、对应与修复资格；逐图审片和批准。',
                    '批次新增图像提交上限：' + str(sum(budget.values())),
                    '批次可开始。' if plan.get('ready') is True else '批次存在缺口，尚未开始。']
                for row in plan.get('items', []):
                    lines += ['', '第 ' + str(row.get('index', 0) + 1) + ' 张', creative_task_plan_text(row.get('plan') or {})]
                self.creative_plan_summary.setPlainText('\n'.join(lines))
            else: self.creative_plan_summary.setPlainText(creative_task_plan_text(plan))
            self.operation_state.setText('计划检查完成；查看范围后可开始制作。' if plan.get('ready') is True else '制作计划有缺口；未创建处理任务。')
            self.update_creative_task_actions()
            if (self.simple_creation and self.simple_creation['stage'] == 'base'
                    and self.simple_creation.get('plan_label') == context['label']
                    and not context['batch'] and context['request'].get('source')
                    and str(Path(context['request']['source']).resolve()) == self.simple_creation['source']):
                if plan.get('ready') is True:
                    if context.get('prepare_existing_beauty'):
                        self.creative_task_plan = None
                        self.simple_creation['stage'] = 'beauty'
                        self.prepare_portrait_beauty()
                        if self.portrait_beauty_request is None: self.simple_creation = None
                    else:
                        self.start_planned_creative_task()
                        if not self.creative_task_start_pending: self.simple_creation = None
                else:
                    self.simple_creation = None
                    self.operation_state.setText('请先完成配置：' + '；'.join(plan.get('blocking_reasons') or []))
                    self.refresh_cos_workspace()
            elif context.get('auto_start') and plan.get('ready') is True:
                self.start_planned_creative_task()

        def start_planned_creative_task(self):
            context = self.creative_task_plan
            if not context or context['plan'].get('ready') is not True or self.render_pending or self.creative_task_start_pending: return
            if context.get('shaping_parent'):
                state = self.shaping_parent_state()
                if (not state['enabled'] or state.get('parent') != context['shaping_parent']
                        or context['shaping_revision'] != self.shaping_revision or self.shaping_execution):
                    self.invalidate_creative_task_plan()
                    self.operation_state.setText('当前保存版本或形体选择已变化；请重新检查。'); return
            if context.get('improvement_parent'):
                state = self.editing_improvement_state()
                if not state['enabled'] or state.get('parent') != context['improvement_parent']:
                    self.invalidate_creative_task_plan()
                    self.creative_plan_summary.setPlainText('原保存版本、图片或独立审片已变化；改进计划作废，未开始制作。')
                    return
            if context['revision'] != self.creative_task_revision:
                self.invalidate_creative_task_plan(); return
            if self.embedded_backend and self.embedded_server is None:
                self.operation_state.setText('请先启动当前工程的后台。'); return
            if self._session_loading: return
            self._checkpoint_session()
            if self._session_checkpoint_failed: return
            body = copy.deepcopy(context['request']); body['plan_id'] = context['plan']['plan_id']
            if not context['batch']:
                body['async'] = True
                self.active_checked_task = {'callback': context['callback'], 'job_id': None,
                    'request': copy.deepcopy(context['request']),
                    'progress_stages': copy.deepcopy(context['plan'].get('stages') or []),
                    'image_budget': copy.deepcopy(context['plan'].get('image_budget') or {}),
                    'creative_direction': copy.deepcopy(context['plan'].get('creative_direction') or context['request'].get('creative_direction'))}
                if self.simple_creation:
                    self.simple_creation['progress_stages'] = copy.deepcopy(context['plan'].get('stages') or [])
            if (self.simple_creation and self.simple_creation.get('stage') == 'base'
                    and context['plan'].get('mode') in {'create_reference', 'existing_reference'}
                    and not body.get('defer_reference_selection')):
                # This checked backend task already owns every production
                # stage. Its saved result must not launch Beauty/TBS again.
                self.simple_creation['stage'] = 'production'
            self.creative_task_plan = None; self.creative_task_start_pending = True
            if context.get('shaping_parent'):
                self.shaping_execution = {'token':context['callback'].split(':', 1)[1],
                    'parent':copy.deepcopy(context['shaping_parent']), 'revision':self.shaping_revision,
                    'request':copy.deepcopy(context['request'].get('shaping_request')), 'cancelled':False}
            self.creative_task_execution_label = context['callback']; self.update_creative_task_actions()
            self.task('POST', '/api/tasks/batch' if context['batch'] else '/api/tasks/start', body, context['callback'])
            self.creative_plan_summary.appendPlainText('\n开始请求已提交；后台将再次核验计划身份与预算。')

        def _template_payload(self):
            try:
                appearance = json.loads(self.appearance.toPlainText())
            except Exception:
                appearance = {}
            try:
                manual = json.loads(self.manual_layers.toPlainText())
            except Exception:
                manual = {}
            model_options = self.current_model_options() or {'algorithm_profile': 'dinov3_vitl16', 'material_finish_profile': 'owner_adaptive_multiscale_skin_v2'}
            return {
                **model_options,
                **({'editing_request': self.editing_request_payload()} if self.editing_request_payload() is not None else {}),
                **({'editing_autofinish':self.editing_autofinish_payload()} if self.editing_autofinish_payload() is not None else {}),
                "provider": self.provider.currentText(),
                "model": self.model.text().strip(),
                "prompt": self.prompt.toPlainText().strip(),
                "preset": self.preset.currentData() if hasattr(self, "preset") else "",
                "character_name": self.character_name.text().strip(), "character_notes": self.character_notes.text().strip(),
                "variant_count": self.variant_count.value(),
                "creative_task": self.creative_task_payload(),
                "appearance_revision": self._appearance_revision_payload(appearance),
                "manual_layers": manual.get("layers", manual) if isinstance(manual, dict) else {},
            }

        def save_workbench_template(self):
            name = self.template_name.text().strip()
            if not name:
                QMessageBox.information(self, "需要模板名称", "先填写模板名称。"); return
            self.task("POST", "/api/templates", {"name": name, "template": self._template_payload()}, "template-saved")

        def apply_cleanup(self):
            answer = QMessageBox.question(self, "清理过时失败记录", "直接删除未被当前任务、版本、缓存或交付引用的失败、阻塞和取消记录；保留原片、待审版本与已批准交付。继续？")
            if answer == QMessageBox.StandardButton.Yes:
                self.task("POST", "/api/cleanup", {"apply": True}, "cleanup-apply")

        def open_recent_project(self):
            row = self.recent_projects.currentData()
            if isinstance(row, dict) and row.get("run_id") and not self.render_pending:
                self.task("GET", f"/api/source-runs/{row['run_id']}", None, "history-run-selected")

        def load_workbench_template(self):
            row = self.template_selector.currentData()
            if not isinstance(row, dict):
                return
            self._checkpoint_session(retain=True)
            if self._session_checkpoint_failed: return
            params = row.get("parameters") if isinstance(row.get("parameters"), dict) else {}
            self.apply_creative_task_options(params.get('creative_task'))
            self.apply_editing_request_options(params.get('editing_request'))
            self.apply_editing_autofinish_options(params.get('editing_autofinish'))
            self.character_name.setText(str(params.get("character_name") or "")); self.character_notes.setText(str(params.get("character_notes") or "")); self.variant_count.setValue(int(params.get("variant_count") or 1))
            if params.get("provider"):
                self.provider.setCurrentText(str(params["provider"]))
            if params.get("model") is not None: self.model.setText(str(params.get("model") or ""))
            self.analysis_models.setCurrentIndex(0)
            preset = str(params.get("preset") or "")
            self.preset.blockSignals(True); self.preset.setCurrentIndex(max(0, self.preset.findData(preset))); self.preset.blockSignals(False); self.update_generation_style_description()
            from generation_prompt import compose_generation_prompt
            self.prompt.setPlainText(str(params.get("prompt") or compose_generation_prompt(style_prompt=self.preset_rows.get(preset, {}).get("prompt"))))
            revision = params.get("appearance_revision") or {}
            parameters = revision.get("params", {})
            if revision.get("regions"): parameters = {**parameters, **revision["regions"]}
            self.appearance.setPlainText(json.dumps({"params": parameters}, ensure_ascii=False, indent=2))
            scope = parameters.get("scope", "global")
            slider_params = parameters.get(scope, {})
            if scope == "local": slider_params = (parameters.get("local") or [{}])[-1]
            for key, slider in self.appearance_sliders.items():
                value = slider_params.get(key, parameters.get(key, 0))
                slider.blockSignals(True); slider.setValue(round(float(value or 0) * 100)); slider.blockSignals(False)
                self.appearance_value_labels[key].setText(str(slider.value()))
            if isinstance(params.get("manual_layers"), dict):
                self.manual_layers.setPlainText(json.dumps({"schema_version": "manual_layers.v1", "layers": params["manual_layers"]}, ensure_ascii=False, indent=2))
                self.sync_manual_controls()
                self.load_clone()
                self.load_tone_curves()
                self.load_hsl()
                self.load_local_brush()
            self.template_name.setText(str(row.get("name") or ""))
            self.log_text(f"[template] loaded {row.get('name', '')}")
            self.operation_state.setText(f"模板：{row.get('name', '')} / {row.get('schema_version', '')} / 参数已载入，点击调整保存为子版本")
        def export_portable_project(self):
            if not self.source_run_id or self.render_pending:return
            if self._unsaved_session_blocks_export(): return
            destination,_=QFileDialog.getSaveFileName(self,'导出当前已保存工程',str(self.root_path/(self.source_run_id+'.cosplayproj')),'Cosplay 工程 (*.cosplayproj)')
            if not destination:return
            if not destination.lower().endswith('.cosplayproj'):destination+='.cosplayproj'
            # An existing archive is preserved. The exporter creates a new file.
            archive=Path(destination)
            if archive.exists():archive=archive.with_name(archive.stem+'-'+uuid.uuid4().hex[:8]+archive.suffix)
            self.portable_project_state.setText('正在导出已保存版本；包括原片、设计与高清资产…')
            self.task('POST',f'/api/source-runs/{self.source_run_id}/export-project',{'archive':str(archive),'include_raw':True,'include_design':True},'project-exported')

        def import_portable_project(self):
            if self.render_pending:return
            self._checkpoint_session()
            if self._session_checkpoint_failed: return
            archive,_=QFileDialog.getOpenFileName(self,'导入 Cosplay 工程',str(self.root_path),'Cosplay 工程 (*.cosplayproj)')
            if not archive:return
            import zipfile
            try:
                with zipfile.ZipFile(archive) as package:identifier=str(json.loads(package.read('project.json'))['run_id'])
            except (OSError,ValueError,KeyError,zipfile.BadZipFile) as error:
                self.portable_project_state.setText('工程包无法读取：'+str(error));return
            body={'archive':archive}
            if (self.root_path/'outputs/runs'/identifier).exists():body['run_id']=identifier+'-import-'+uuid.uuid4().hex[:8]
            self.portable_project_state.setText('正在导入工程；重名版本会作为新版本保留…')
            self.task('POST','/api/projects/import',body,'project-imported')

        def choose_source(self):
            path, _ = QFileDialog.getOpenFileName(self, "选择原图", str(self.root_path), "Images (*.jpg *.jpeg *.png *.tif *.tiff *.arw *.cr2 *.nef *.dng)")
            if path:
                self.set_simple_source(path)

        def choose_batch_sources(self):
            paths, _ = QFileDialog.getOpenFileNames(self, "选择多张原图", str(self.root_path), "Images (*.jpg *.jpeg *.png *.tif *.tiff *.arw *.cr2 *.nef *.dng)")
            if paths:
                self.batch_sources = paths
                self.source_edit.setText(paths[0])
                self.batch_state.setText(f"批量状态：已选择 {len(paths)} 张原图")
                self.local_batch_pairs = []
                self.invalidate_creative_task_plan()
                self.project_view_selector.setCurrentIndex(0); self.request_source_preview(paths[0])

        def choose_local_batch_references(self):
            if not self.batch_sources: self.choose_batch_sources()
            if not self.batch_sources: return
            paths, _ = QFileDialog.getOpenFileNames(self, '按原图顺序选择对应已有参考（可选一张共用参考）', str(self.root_path), 'Images (*.png *.jpg *.jpeg *.tif *.tiff)')
            if not paths: return
            if len(paths) not in {1, len(self.batch_sources)}:
                self.batch_state.setText('参考数量需为一张共用参考，或与原图数量一致。'); return
            self.local_batch_pairs = [{'source': source, 'reference': paths[0] if len(paths) == 1 else paths[index]} for index, source in enumerate(self.batch_sources)]
            self.invalidate_creative_task_plan()
            self.local_batch_id = None; self.show_local_batch({'items': [{**row, 'index': index} for index, row in enumerate(self.local_batch_pairs)]})
            self.batch_state.setText('已配对 ' + str(len(self.local_batch_pairs)) + ' 张；核对后检查共享要求和逐图后期计划，新增图像预算为零。')

        def run_local_reference_batch(self):
            if not self.local_batch_pairs: self.choose_local_batch_references()
            if not self.local_batch_pairs: return
            self.request_creative_task_plan('existing_reference', batch=True)

        @staticmethod
        def is_creative_task_batch(batch):
            return batch.get('kind') == 'creative_task_batch'

        def show_local_batch(self, batch):
            self.local_batch_tree.clear(); failed = False
            mode = (batch.get('shared_task') or {}).get('mode')
            for row in batch.get('items', []):
                request = row.get('request') or {}; source = str(row.get('source') or request.get('source') or '')
                reference = str(row.get('reference') or request.get('reference') or '')
                job = row.get('job') or {}; status = job.get('status') or row.get('status') or ('待提交' if not row.get('job_id') else 'queued')
                reference_label = Path(reference).name if reference else ('本次创作' if mode == 'create_reference' else '无导入参考')
                item = QTreeWidgetItem(self.local_batch_tree, [Path(source).name, reference_label, status])
                item.setToolTip(0, source); item.setToolTip(1, reference); item.setData(0, Qt.UserRole, row['index']); item.setData(1, Qt.UserRole, status)
                failed |= status in {'failed', 'blocked', 'cancelled', 'missing'}
            self.local_batch_retry_button.setEnabled(bool(self.local_batch_id) and failed and not self.creative_task_start_pending)

        def retry_local_batch_item(self):
            if self.creative_task_start_pending or self.render_pending: return
            item = self.local_batch_tree.currentItem()
            if not item or not self.local_batch_id or item.data(1, Qt.UserRole) not in {'failed', 'blocked', 'cancelled', 'missing'}: return
            self.creative_task_start_pending = True; self.creative_task_execution_label = 'local-batch-started'
            self.creative_task_retry_batch_id = self.local_batch_id
            self.local_batch_retry_button.setEnabled(False); self.update_creative_task_actions()
            self.task('POST', f'/api/tasks/batch/{self.local_batch_id}/retry-item', {'index': item.data(0, Qt.UserRole)}, 'local-batch-started')

        def run_batch(self):
            if not self.batch_sources:
                self.choose_batch_sources()
            if not self.batch_sources: return
            self.request_creative_task_plan('create_reference', batch=True)

        def poll_batch(self, batch_id):
            if batch_id: self.task("GET", f"/api/batch/{batch_id}", None, "batch-status")
        def run_source_only(self):
            source = self.source_edit.text().strip()
            if not source: QMessageBox.information(self, "缺少原图", "请选择一张 JPG/RAW 原图"); return
            self.request_creative_task_plan('create_reference', source=source)
        def run_local_source(self):
            source = self.source_edit.text().strip()
            if not source: QMessageBox.information(self, "缺少原图", "请选择一张 JPG/RAW 原图"); return
            self.request_creative_task_plan('local_portrait', source=source)

        def use_existing_design_reference(self):
            if self.render_pending: return
            source = self.source_edit.text().strip()
            if not source:
                QMessageBox.information(self, "缺少原图", "请先导入一张 RAW 或原图，再选择已有设计参考。"); return
            reference, _ = QFileDialog.getOpenFileName(self, "选择已有设计参考", str(self.root_path), "Design images (*.png *.jpg *.jpeg *.tif *.tiff)")
            if not reference: return
            reference = str(Path(reference).expanduser().resolve())
            self.creative_task_reference.setText(reference)
            self.existing_design_state.setText(f"已选参考：{Path(reference).name}。正在检查成对后期计划；默认零生图，局部精修须明确授权。")
            self.request_creative_task_plan('existing_reference', source=source, reference=reference)
        def complete_current_post(self):
            if not self.source_run_id or self.render_pending:
                return
            self._checkpoint_session()
            if self._session_checkpoint_failed:
                self.simple_creation = None
                self.operation_state.setText(self.session_draft_state.text()); self.refresh_cos_workspace(); return
            if self._session_dirty:
                QMessageBox.information(self, '请先保存微调', '保存当前编辑后再完成后期，处理会从这个已保存版本开始。')
                return
            model_options = self.current_model_options()
            if model_options is None:
                self.model_setup_summary.setText('请先检查或配置当前后台的分析模型。'); self.open_model_setup(); return
            self.task('POST', f'/api/source-runs/{self.source_run_id}/complete-post',
                model_options, 'complete-post')
        def _session_snapshot(self):
            from session_draft import editable_payload
            from source_relight import normalize_source_render_plan
            from source_color import normalize_source_color_plan
            appearance = json.loads(self.appearance.toPlainText())
            revision = self._appearance_revision_payload(appearance)
            plan = self.design_reconstruction_plan or {}
            return editable_payload({"params": revision.get("params", {}), "regions": revision.get("regions", {}),
                "scope_controls": self._appearance_scope_options(),
                "slider_values": {key: widget.value() for key, widget in self.appearance_sliders.items()},
                "manual_layers": json.loads(self.manual_layers.toPlainText()).get("layers", {}),
                "source_render_plan": normalize_source_render_plan(self.source_light_plan) if self.source_light_plan is not None else None,
                "source_color_plan": normalize_source_color_plan(self.source_color_plan) if self.source_color_plan is not None else None,
                "receiver_intent": revision["receiver_intent"],
                "receiver_shadow_plan": revision["receiver_shadow_plan"],
                "portrait_beauty": revision.get('portrait_beauty'),
                "workflow_graph": revision.get('workflow_graph'),
                "design_overrides": {"enabled": bool(plan.get("enabled", True)),
                    "prepared_plan": json.loads(json.dumps(plan)) if self.design_reconstruction_prepared else None,
                    "layers": [{"id": layer["id"], "enabled": bool(layer.get("enabled", True)), "strength": float(layer.get("strength", 1)),
                        **({key: float(layer.get(key, 1)) for key in ('colour_strength', 'lightness_strength')} if layer.get('kind') == 'reference_appearance_stack' else {})}
                               for layer in plan.get("layers", [])]},
                "native_matte_strokes": self.native_matte_strokes})

        def _set_session_dirty(self, dirty, message=None):
            self._session_dirty = bool(dirty)
            self.session_draft_state.setText(message or ("未保存精修已自动保留；保存高清后才能更新导出。" if dirty else ""))
            self.update_accept_enabled()
            if hasattr(self, "export_project_button"):
                self.export_project_button.setEnabled(bool(self.source_run_id) and not self.render_pending and not dirty)
            if dirty:
                self.export_state.setText("当前参数与笔迹尚未写入高清母版；保存高清版本后才能导出当前精修。")

        def _checkpoint_session(self, retain=False):
            self._session_checkpoint_failed = False
            if self._session_loading: return None
            if self.source_generated_appearance: return None
            if not self._session_binding:
                if self._session_baseline is not None and self.source_run_id:
                    from session_draft import digest
                    if digest(self._session_snapshot()) != self._session_baseline:
                        self._session_checkpoint_failed = True
                        self._set_session_dirty(True, '当前精修无法保留；请检查工程与绑定资产，保存完成前不能继续后期或导出。')
                return None
            from session_draft import SessionDraftStore, digest
            try:
                snapshot = self._session_snapshot(); sha = digest(snapshot)
                store = SessionDraftStore(self.root_path)
                if sha == self._session_baseline:
                    if self._session_written: store.clear_if_equal(self._session_binding, self._session_written)
                    self._session_written = None
                    if self._session_dirty: self._set_session_dirty(False)
                    return snapshot
                if sha != self._session_written:
                    store.checkpoint(self._session_binding, snapshot)
                    self._session_written = sha
                if retain: store.retain(self._session_binding)
                self._set_session_dirty(True)
                return snapshot
            except (OSError, TypeError, ValueError, KeyError) as error:
                self._session_checkpoint_failed = True
                self._set_session_dirty(True, "未保存精修未能自动保留：" + str(error))
                return None

        def _unsaved_session_blocks_export(self):
            self._checkpoint_session()
            if self._session_dirty:
                self.operation_state.setText("请先保存高清版本，再导出当前精修")
                return True
            return False

        def _restore_session_payload(self, payload):
            self.workflow_graph = copy.deepcopy(payload.get('workflow_graph'))
            appearance = json.loads(self.appearance.toPlainText())
            appearance.update(params=payload["params"], regions=payload.get("regions", {}))
            self.load_portrait_beauty(payload.get('portrait_beauty'))
            self.load_receiver_intent(payload.get("receiver_intent"))
            appearance["receiver_intent"] = json.loads(json.dumps(self.receiver_intent))
            self.load_receiver_shadow_plan(payload.get('receiver_shadow_plan'))
            appearance['receiver_shadow_plan'] = json.loads(json.dumps(self.receiver_shadow_plan))
            self.appearance.setPlainText(json.dumps(appearance, ensure_ascii=False, indent=2))
            scope = payload.get("scope_controls") or {}
            if self.appearance_extra_controls:
                controls = self.appearance_extra_controls[0]
                for key, value in (("region", scope.get("scope", "global")), ("local_scope", scope.get("local_scope", "global"))):
                    widget = controls[key]; widget.blockSignals(True); widget.setCurrentIndex(max(0, widget.findData(value))); widget.blockSignals(False)
                controls["local_rect"].setText(",".join(str(value) for value in scope.get("local_rect", [0, 0, 1, 1])))
                controls["local_feather"].blockSignals(True); controls["local_feather"].setValue(round(float(scope.get("local_feather", 0)) * 100)); controls["local_feather"].blockSignals(False)
            for key, value in payload.get("slider_values", {}).items():
                if key in self.appearance_sliders:
                    widget = self.appearance_sliders[key]; widget.blockSignals(True); widget.setValue(int(value)); widget.blockSignals(False)
                    self.appearance_value_labels[key].setText(str(widget.value()))
            self.load_source_lights(payload.get("source_render_plan"))
            self.load_source_colors(payload.get("source_color_plan"))
            overrides = payload.get('design_overrides') or {}
            prepared = overrides.get('prepared_plan')
            if isinstance(prepared, dict):
                if (prepared.get('source_sha256') == self.source_run_source_sha256
                        and prepared.get('candidate_sha256') == self.source_run_candidate_sha256
                        and not self._paired_subject_state(prepared)['reconstruction_required']):
                    self.design_reconstruction_plan = json.loads(json.dumps(prepared))
                    self.design_reconstruction_prepared = True
            plan = self.design_reconstruction_plan
            if plan:
                plan["enabled"] = bool(overrides.get("enabled", True))
                by_id = {layer["id"]: layer for layer in overrides.get("layers", [])}
                for layer in plan.get("layers", []):
                    if layer["id"] in by_id: layer.update({key: by_id[layer["id"]][key] for key in ("enabled", "strength", "colour_strength", "lightness_strength") if key in by_id[layer["id"]]})
                self.load_design_reconstruction(plan)
            self.manual_layers.setPlainText(json.dumps({"schema_version": "manual_layers.v1", "revision": self.source_revision,
                                                       "layers": payload.get("manual_layers", {})}, ensure_ascii=False, indent=2))
            self.sync_manual_controls(); self.load_clone(); self.load_tone_curves(); self.load_hsl(); self.load_local_brush()
            self.native_matte_strokes = json.loads(json.dumps(payload.get("native_matte_strokes", [])))
            self.native_matte_history = [[], json.loads(json.dumps(self.native_matte_strokes))] if self.native_matte_strokes else [[]]
            self.native_matte_history_index = len(self.native_matte_history) - 1
            self.native_matte_cache_key = None; self.native_matte_preview_version = -1; self.native_matte_result = None
            if self.native_matte_strokes:
                self.native_matte_state.setText(str(len(self.native_matte_strokes)) + " 笔 · 已恢复未保存边缘精修；可预览或保存高清版本。")
            self.update_native_matte_overlay()

        def load_source_run(self, run, stage_report=None, saved_submission=False, selection_explicit=False,
                            record_selection=True, before_load=None):
            from session_draft import SessionDraftStore, binding_for_run, digest
            intent=run.get('task_intent') or (run.get('options') or {}).get('task_intent') or {}
            from scoped_workflow import SOURCE_NATIVE_INITIALIZATION
            if (intent.get('mode')=='create_reference'
                    and (run.get('task_plan') or {}).get('editing_initialization')!=SOURCE_NATIVE_INITIALIZATION):
                self.operation_state.setText('不支持此旧创建格式；请从原图新建当前任务。')
                return False
            self._checkpoint_session()
            if self._session_checkpoint_failed: return False
            store = SessionDraftStore(self.root_path)
            if self._session_binding and run.get("parent_run_id") == self.source_run_id:
                submission = store.submission_for_run(self.source_run_id)
                committed = ((run.get("options") or {}).get("appearance_revision") or {}).get("native_session_submission")
                if submission and committed == submission and digest(self._session_snapshot()) != submission["payload_sha256"]:
                    # The child contains the submitted snapshot. Later edits belong
                    # to the current parent session and must stay visibly editable.
                    self.render_pending = False
                    self._set_session_dirty(True, "高清子版本 " + str(run.get("run_id")) + " 已保存；当前新增修改已保留，尚未保存高清。")
                    self._session_submission = None; self._session_commit_child_id = None
                    return False
            # Shaping's non-waiting explicit selection runs only after both
            # session-preservation gates, before consuming or replacing state.
            if before_load is not None: before_load()
            for kind in tuple(self.shaping_inspections): self._discard_shaping_inspection(kind)
            self.shaping_parent_fact = None; self.shaping_adoption = None
            if store.consume_for_child(run) or saved_submission:
                self._session_submission = None; self._session_commit_child_id = None
            self._session_loading = True; self._session_binding = None; self._session_written = None
            self._session_dirty = False; self.session_draft_state.setText("")
            self.source_light_requests.clear(); self.native_detail_token = ""
            try:
                self._load_source_run_state(run, stage_report)
                saved_shape = (run.get('options') or {}).get('shaping_state') or {}
                self.shaping_parts = copy.deepcopy((saved_shape.get('request') or {}).get('parts') or {}) if saved_shape.get('enabled') else {}
                self.shaping_candidate = None; self.shaping_revision += 1
                for controls in self.shaping_panels:
                    controls['before'].blockSignals(True); controls['before'].setChecked(False); controls['before'].blockSignals(False)
                self._session_baseline = digest(self._session_snapshot())
                try:
                    if self.source_generated_appearance:
                        raise ValueError('生成成片已保存；原片图层会话不适用于此版本。')
                    self._session_binding = binding_for_run(self.root_path, run)
                except (OSError, KeyError, TypeError, ValueError) as error:
                    self._session_binding = None
                    self.session_draft_state.setText("候选选择后再准备可编辑图层。" if run.get('reference_role') == 'candidate_pending_selection'
                        else "当前版本不支持会话精修保留：" + str(error))
                else:
                    record, reason = store.restore(self._session_binding)
                    if record:
                        self._restore_session_payload(record["payload"])
                        self._session_written = record["payload_sha256"]
                        self._session_submission = record.get("submission")
                        self._set_session_dirty(True, "已恢复未保存精修，保存高清后才能更新导出。")
                    elif reason:
                        self.session_draft_state.setText("未恢复会话草稿，原草稿已保留：" + reason)
            finally:
                self.native_matte_timer.stop(); self.source_light_timer.stop(); self._session_loading = False
                self.update_accept_enabled()
                self._advance_simple_creation('base')
            if record_selection and run.get('auto_draft'):
                try:
                    from workbench_store import select_saved_result
                    select_saved_result(self.root_path, str(run.get('run_id') or ''), explicit=selection_explicit)
                except (OSError, ValueError, TimeoutError) as error:
                    self.log_text('[saved result selection] ' + str(error))
            return True

        def show_preparation_version(self, run):
            if self.source_generated_appearance:
                self.preparation_currentness = {'state': 'saved_generated_appearance', 'scope': 'saved_pixels'}
                text = '已保存冻结外观。原生来源与生成来源分别记录；原尺寸母版不自动证明相机细节，完整作品仍需审片。'
                for name in ('preparation_version_workspace', 'preparation_version_review'):
                    label = getattr(self, name, None)
                    if label is not None: label.setText(text); label.setVisible(True)
                return
            from preparation_currentness import preparation_currentness
            report = preparation_currentness(self.root_path, run)
            self.preparation_currentness = report
            state = report.get('state')
            text = (('此保存版本已使用当前算法。' if report.get('scope') == 'saved_local_production'
                     else '此保存版本的仿色融合已使用当前算法。') if state == 'current' else
                    '此保存版本使用旧算法；可以原样查看，重新制作会保存使用当前算法的新版本。' if state == 'stale' else
                    '无法核验此保存版本的处理来源；重新制作会建立当前算法的新版本。' if state == 'unavailable' else '')
            for name in ('preparation_version_workspace', 'preparation_version_review'):
                label = getattr(self, name, None)
                if label is not None:
                    label.setText(text); label.setVisible(bool(text))

        def _load_source_run_state(self, run, stage_report=None):
            self.source_generated_appearance = bool((run.get('options') or {}).get('photo_appearance')
                or (run.get('render') or {}).get('photo_appearance'))
            self.analysis_models.setCurrentIndex(0)
            self.complete_post_available = bool(run.get('auto_draft') and run.get('reference'))
            self.complete_post_button.setEnabled(self.complete_post_available)
            self.receiver_shadow_result = None; self.receiver_shadow_preview_requests = {}
            from pipeline_stage_report import format_stage_report, read_stage_report
            product_report = stage_report if isinstance(stage_report, dict) else read_stage_report(self.root_path, run)
            self.pipeline_summary.setPlainText(format_stage_report(product_report))
            self.native_live_detail_context=None;self.native_live_detail_pending_center=None
            same_source = bool(self.source_run_id) and self.source_run_source == str((run.get("source") or {}).get("path") or "")
            self.source_light_timer.stop(); self.source_light_version += 1
            self.source_color_auto_request = None
            self.design_reconstruction_request = None
            self.design_reconstruction_prepared = False
            self.native_reconstruction_job_id = ''; self.native_reconstruction_result_seen = ''
            self.portrait_beauty_request = None; self.portrait_beauty_job_id = ''; self.portrait_beauty_result_seen = ''
            for name in ("source", "final"): self.previews[name]._color_stroke = None
            self.source_light_preview_path = None
            self.source_run_id = str(run.get("run_id") or "")
            self.source_run_source_sha256 = (run.get('source') or {}).get('sha256')
            self.source_run_candidate_sha256 = (run.get('reference') or {}).get('sha256')
            self.source_run_has_reference = bool(run.get('reference') or (run.get('artifacts') or {}).get('reference_candidate'))
            self.source_run_only_plan = copy.deepcopy((run.get('options') or {}).get('source_only_plan'))
            self.source_run_source = str((run.get("source") or {}).get("path") or "")
            if self.source_run_source: self.source_edit.setText(self.source_run_source)
            self.source_revision = int(run.get("revision") or 1)
            self.source_status = str(run.get("status") or "unknown")
            # Opening an in-flight checkpoint from the task centre is a read.
            # Keep the owning task busy until its terminal observation arrives.
            self.render_pending = bool(self._active_production_job())
            self.export_project_button.setEnabled(True)
            # Historical layers remain in saved manifests for provenance, but
            # the workbench only exposes native source-bound views and actions.
            for _widget in (self.alpha_tool, self.alpha_overlay, self.alpha_radius,
                            self.alpha_opacity, self.alpha_undo, self.alpha_redo,
                            self.alpha_reset, self.alpha_save):
                _widget.setEnabled(False)
            self.alpha.setReadOnly(True)
            self.alpha_canvas.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            self.alpha_canvas.setToolTip("Source matte 仅供查看；人物几何与 alpha 不在 Qt 中手动改写。")
            self.review_title.setText(f"单图：{self.source_run_id} / revision {self.source_revision}")
            self.show_preparation_version(run)
            self.source_sample = str(run.get("sample") or (run.get("source") or {}).get("sample") or Path(str((run.get("source") or {}).get("path") or "source")).stem)
            artifacts = run.get("artifacts") if isinstance(run.get("artifacts"), dict) else {}
            paths = {"source": Path(str((run.get("source") or {}).get("path"))) if (run.get("source") or {}).get("path") else None,
                     "reference": Path(str(artifacts.get("reference_candidate") or (run.get("reference") or {}).get("path"))) if (artifacts.get("reference_candidate") or (run.get("reference") or {}).get("path")) else None,
                     "final": Path(str(artifacts.get("auto_draft") or "")) if artifacts.get("auto_draft") else None,
                     "alpha": Path(str(artifacts.get("alpha") or "")) if artifacts.get("alpha") else None,
                     "risk": Path(str(artifacts.get("structure_mismatch") or "")) if artifacts.get("structure_mismatch") else None,
                     "silhouette": Path(str(artifacts.get("silhouette") or "")) if artifacts.get("silhouette") else None}
            self.source_run_reference = str(paths['reference'] or "")
            self.existing_design_state.setText(f"当前设计参考：{paths['reference'].name}；可复用仿色融合准备。" if paths.get("reference") else "可使用已有参考，默认零生图；局部精修须在新计划中明确授权。")
            self.risk_crop_paths = {}
            risk_dir = Path(str(artifacts.get("risk_crops") or "")) if artifacts.get("risk_crops") else None
            if risk_dir and risk_dir.is_dir():
                self.risk_crop_paths = {path.stem: path for path in sorted(risk_dir.glob("*.jpg"))}
            self.risk_crop.blockSignals(True); self.risk_crop.clear(); self.risk_crop.addItem("结构导航图", "")
            for name in self.risk_crop_paths: self.risk_crop.addItem(name, name)
            self.risk_crop.blockSignals(False)
            proxy_paths = {}
            try:
                from cache_store import ContentCache
                from raw_support import is_raw, RGB_DECODE_VERSION
                cache = ContentCache(self.root_path)
                for name in ("source", "reference"):
                    if paths.get(name):
                        hit = cache.find_latest_proxy(paths[name], params={"max_edge": 2048, "decode_version": RGB_DECODE_VERSION} if is_raw(paths[name]) else None)
                        if hit: proxy_paths[name] = hit
            except Exception as exc:
                self.log_text(f"[cache] proxy preview unavailable: {exc}")
            for name, view in self.previews.items():
                if name == "final" and same_source and paths[name] and view.path:
                    view.replace_preview_path(paths[name])
                else: view.set_path(paths[name], proxy_paths.get(name))
            self.source_light_draft_path = paths.get("final")
            if paths.get('final'):
                native_size=QImageReader(str(paths['final'])).size()
                if native_size.isValid():self.clone_native_size=[native_size.width(),native_size.height()]
            self.native_detail_button.setEnabled(bool(artifacts.get('render_scene_master')))
            self.previews['final'].native_detail_enabled = self.native_detail_button.isEnabled()
            if paths.get("final"):
                self.layer_tabs.setCurrentWidget(self.previews['final'])
            self.source_light_compare.blockSignals(True); self.source_light_compare.setChecked(False); self.source_light_compare.blockSignals(False)
            self.project_images.update({name: paths.get(name) for name in ("source", "reference", "final")}); self.project_proxies = dict(proxy_paths)
            source_size = (run.get("source") or {}).get("oriented_size") or []
            self.project_dimensions = {"source": source_size} if len(source_size) == 2 else {}
            self.update_project_canvas()
            if paths.get("source") and not proxy_paths.get("source"): self.request_source_preview(paths["source"])
            # The structure mismatch heatmap is an inspection overlay only;
            # brush writes still target the alpha image.
            self.alpha_canvas.set_path(paths["alpha"], paths["risk"])
            options = run.get("options") or {}
            self.apply_creative_task_options(run.get('task_intent') or options.get('task_intent') or options.get('creative_task'))
            self.apply_editing_request_options(options.get('editing_request') or (run.get('task_plan') or {}).get('editing_request'))
            self.apply_editing_autofinish_options(options.get('editing_autofinish') or (run.get('task_plan') or {}).get('editing_autofinish'))
            self.creative_task_reference.setText(self.source_run_reference)
            self.character_name.setText(str(options.get("character_name") or "")); self.character_notes.setText(str(options.get("character_notes") or "")); self.variant_count.setValue(int(options.get("variant_count") or len(run.get("variants") or []) or 1))
            self.show_character_profile(run.get("character_profile") or options.get("character_profile") or {})
            self.populate_variants(run)
            selected_preset = options.get("preset") or ""
            if isinstance(selected_preset, dict): selected_preset = selected_preset.get("id", "")
            self.preset.blockSignals(True); self.preset.setCurrentIndex(max(self.preset.findData(selected_preset), 0)); self.preset.blockSignals(False)
            self.update_generation_style_description()
            saved_design = run.get("generation_design") or options.get("generation_design") or {}
            saved_prompt = saved_design.get("input_prompt") or (saved_design.get("inputs") or {}).get("base_prompt") or options.get("input_prompt") or options.get("prompt") or options.get("effective_prompt") or (run.get("reference") or {}).get("prompt")
            from generation_prompt import prompt_for_new_task
            saved_prompt_version = saved_design.get('prompt_template_version') or options.get('prompt_template_version')
            self.prompt.setPlainText(prompt_for_new_task(saved_prompt, saved_version=saved_prompt_version,
                style_prompt=self.preset_rows.get(selected_preset, {}).get('prompt')))
            # A saved image owns its provenance, not the model selection for
            # the next task. Keep the current user/default generation choice.
            self.alpha.setPlainText(json.dumps({"schema_version": "alpha_revision_draft.v1", "run_id": self.source_run_id, "sample": self.source_sample, "revision": run.get("revision", 1), "base_alpha": str(paths["alpha"]) if paths["alpha"] else None, "base_alpha_sha256": hashlib.sha256(paths["alpha"].read_bytes()).hexdigest() if paths["alpha"] and paths["alpha"].is_file() else None, "edits": []}, ensure_ascii=False, indent=2))
            appearance = options.get("appearance_revision") or {}
            self.workflow_graph = copy.deepcopy(appearance.get('workflow_graph'))
            self.simple_comparing = False
            self.load_portrait_beauty(appearance.get('portrait_beauty'))
            parameters = appearance.get("params", {})
            slider_parameters = parameters.get("global", parameters) if isinstance(parameters, dict) else {}
            self.appearance.setPlainText(json.dumps({**appearance, "schema_version": "appearance_revision_draft.v1", "revision": run.get("revision", 1), "params": parameters}, ensure_ascii=False, indent=2))
            from source_relight import source_render_plan_for_run
            has_light_plan = "source_render_plan" in appearance or "source_render_plan" in (saved_design.get("design_brief") or {})
            light_plan = source_render_plan_for_run(run) if has_light_plan else None
            self.source_light_design = appearance.get("source_render_plan_design")
            self.source_light_origin = appearance.get("source_render_plan_origin") or ("model" if has_light_plan and "source_render_plan" not in appearance else "manual")
            self.load_source_lights(light_plan)
            from source_color import source_color_plan_for_run
            has_color_plan = "source_color_plan" in appearance or "source_color_plan" in (saved_design.get("design_brief") or {})
            self.source_color_design = appearance.get("source_color_plan_design")
            self.source_color_origin = appearance.get("source_color_plan_origin") or ("model" if has_color_plan and "source_color_plan" not in appearance else "manual")
            self.source_color_provenance = appearance.get("source_color_plan_provenance")
            self.load_source_colors(source_color_plan_for_run(run) if has_color_plan else None); self.update_source_light_actions()
            self.load_design_reconstruction(appearance.get("design_reconstruction_plan")); self.update_source_light_actions()
            self.local_editor_tabs.setEnabled(True)
            self.receiver_source_canvas_size = (run.get("source") or {}).get("oriented_size") or (self.design_reconstruction_plan or {}).get("canvas", {}).get("source_oriented_size")
            self.load_receiver_intent(appearance.get("receiver_intent"))
            self.load_receiver_shadow_plan(appearance.get('receiver_shadow_plan'))
            self.local_editor_tabs.setCurrentIndex(2 if self.design_reconstruction_plan is not None or paths.get("reference") else 0)
            if self._paired_subject_state()['reconstruction_required']: self.source_light_state.setText('已保存画面可查看；请点击准备仿色融合，更新人物参考观察后再预览或保存。')
            elif self.design_reconstruction_plan is not None: self.source_light_state.setText("设计图层已载入；预览并保存高清成片，整体效果仍需审片。")
            elif paths.get("reference"): self.source_light_state.setText("已选仿色参考；点击准备仿色融合制作可编辑成片。")
            for key, slider in self.appearance_sliders.items():
                slider.blockSignals(True); slider.setValue(round(float(slider_parameters.get(key, 0) or 0) * 100)); slider.blockSignals(False)
                self.appearance_value_labels[key].setText(str(slider.value()))
            review_data = run.get("visual_review") if isinstance(run.get("visual_review"), dict) else {}
            review_path = artifacts.get("visual_review") or review_data.get("path") or run.get("review_path")
            if review_path:
                candidate = Path(str(review_path))
                if not candidate.is_absolute(): candidate = (self.root_path / candidate).resolve()
                try:
                    if candidate.is_file(): review_data = json.loads(candidate.read_text(encoding="utf-8"))
                except Exception as exc:
                    self.log_text(f"[visual-review] 无法读取 {candidate}: {exc}")
            checks = (review_data.get("checks") or {}) if isinstance(review_data, dict) else {}
            for key, combo in getattr(self, "visual_checks", {}).items():
                value = str(checks.get(key, "not_checked"))
                combo.blockSignals(True); combo.setCurrentText(value if value in {"not_checked", "pass", "rework", "hold"} else "not_checked"); combo.blockSignals(False)
            self.update_accept_enabled()
            manual = (run.get("options") or {}).get("manual_layers") if isinstance(run.get("options"), dict) else None
            if isinstance(manual, dict):
                self.manual_layers.setPlainText(json.dumps({"schema_version":"manual_layers.v1","revision":run.get("revision", 1),"layers":manual}, ensure_ascii=False, indent=2))
            else:
                self.manual_layers.setPlainText(json.dumps(self.default_manual_layers(), ensure_ascii=False, indent=2))
            self.sync_manual_controls()
            self.load_clone()
            self.load_tone_curves()
            self.load_hsl()
            self.load_local_brush()
            self.load_native_matte()
            self.load_receiver_shadow_result((self.design_reconstruction_plan or {}).get('execution', {}).get('authored_receiver_shadow'),
                snapshot=self.receiver_edit_snapshot(), matte_version=self.native_matte_version, origin='本版本')
            self.show_product_result(product_report, run)
            self.review_state.setText(f"单图审片：{self.source_status} / run={self.source_run_id} / revision={self.source_revision} / 质量判断：视觉定性")
            self.delivery.setPlainText(json.dumps({"run_id": self.source_run_id, "status": run.get("status"), "artifacts": artifacts, "manual": None, "gt": None}, ensure_ascii=False, indent=2))
            self.export_state.setText(f"当前保存版本：{self.source_run_id} / {self.source_status}；导出回执显示母版的实际尺寸、位深和 ICC 信息。")
            self.export_artifact.setCurrentIndex(1 if self.source_status in {"accepted", "delivered"} else 0)
            if saved_design: self.show_generation_design(saved_design, run.get("character_profile") or options.get("character_profile") or {})
            self.tabs.setCurrentIndex(2 if self.professional_mode else self.simple_index)
            self.load_delivery_history()
        def select_risk_crop(self, index):
            key = self.risk_crop.itemData(index) if hasattr(self, "risk_crop") else ""
            if key and key in self.risk_crop_paths:
                self.previews["risk"].set_path(self.risk_crop_paths[key])
        def accept_source_run(self):
            run_id = getattr(self, "source_run_id", "")
            reviewer = self.source_reviewer.text().strip()
            if self.render_pending: return
            self._checkpoint_session()
            if self._session_dirty:
                self.operation_state.setText("请先保存高清版本，再批准当前精修"); return
            if not run_id: QMessageBox.information(self, "没有单图草稿", "请先运行单图自动生成"); return
            if not reviewer: QMessageBox.information(self, "需要批准人", "填写视觉批准人后再交付"); return
            checks = {key: combo.currentText() for key, combo in self.visual_checks.items()}
            if any(value != "pass" for value in checks.values()):
                QMessageBox.information(self, "视觉检查未完成", "八项视觉检查必须全部选择 pass。"); return
            self.task("POST", f"/api/source-runs/{run_id}/accept", {"reviewer": reviewer, "checks": checks}, "source-accept")

        def update_accept_enabled(self):
            button = getattr(self, "source_accept", None)
            if button is None: return
            checks = getattr(self, "visual_checks", {})
            ready = bool(self.source_run_id) and self.source_status == "needs_visual_review" and not self.render_pending and not self._session_dirty and bool(checks) and all(combo.currentText() == "pass" for combo in checks.values())
            button.setEnabled(ready)
            self.export_button.setEnabled(bool(self.source_run_id) and not self.render_pending and not self._session_dirty)
            if hasattr(self, "export_project_button"):
                self.export_project_button.setEnabled(bool(self.source_run_id) and not self.render_pending and not self._session_dirty)
            self.local_source_button.setEnabled(not self.render_pending); self.run_source_button.setEnabled(not self.render_pending)
            self.existing_design_button.setEnabled(not self.render_pending)
            if hasattr(self, 'complete_post_button'):
                self.complete_post_button.setEnabled(bool(getattr(self, 'complete_post_available', False))
                    and not self.render_pending and not self._session_dirty)
            if hasattr(self, 'reprepare_current_button'):
                self.reprepare_current_button.setEnabled(bool(getattr(self, 'complete_post_available', False))
                    and not self.render_pending and not self._session_dirty)
            if hasattr(self, "variant_use"):
                item = self.variant_list.currentItem(); row = item.data(Qt.UserRole) if item else None
                self.variant_use.setEnabled(bool(self.source_run_id and isinstance(row, dict) and row.get("variant_id")) and not self.render_pending)
            if hasattr(self, "source_light_save"): self.update_source_light_actions()
            self.refresh_product_result_actions()
            self.update_creative_task_actions()

        def rollback_source_delivery(self):
            sample = self.source_sample.strip()
            target = self.rollback_target.text().strip()
            if not sample or not target:
                QMessageBox.information(self, "需要回滚目标", "先加载单图运行，并填写已批准历史 run id。"); return
            self.task("POST", f"/api/source-deliveries/{sample}/rollback", {"run_id": target, "reason": "workbench rollback"}, "source-rollback")

        def load_delivery_history(self):
            sample = self.source_sample.strip()
            if sample:
                self.task("GET", f"/api/source-deliveries/{sample}", None, "source-delivery-history")
                if self.source_run_id:
                    self.task("GET", f"/api/source-runs/{self.source_run_id}/history", None, "source-run-history")

        def populate_source_history(self, value):
            """Render the backend's immutable run tree and retain run ids on items."""
            tree = value.get("tree") if isinstance(value, dict) else []
            if not isinstance(tree, list):
                tree = []
            widget = self.source_history_tree
            widget.blockSignals(True); widget.clear()

            def add_node(node, parent=None):
                if not isinstance(node, dict):
                    return None
                run_id = str(node.get("run_id") or "")
                revision = node.get("revision", 1)
                marker = "  [ACTIVE]" if node.get("active_delivery") else ""
                label = f"r{revision}  {run_id}{marker}"
                status = str(node.get("status") or "unknown")
                delivery = node.get("delivery") if isinstance(node.get("delivery"), dict) else None
                delivery_text = "active" if node.get("active_delivery") else ("approved" if delivery else "—")
                item_parent = parent if parent is not None else widget
                item = QTreeWidgetItem(item_parent, [label, status, delivery_text, str(node.get("updated_at") or node.get("created_at") or "")])
                item.setData(0, Qt.UserRole, run_id)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable); item.setCheckState(0, Qt.Unchecked)
                for child in node.get("children", []) if isinstance(node.get("children"), list) else []:
                    add_node(child, item)
                return item

            for node in tree:
                add_node(node)
            widget.expandAll()
            # Keep the selected run highlighted after a refresh without
            # issuing another HTTP request for the same manifest.
            selected = str(value.get("selected_run_id") or self.source_run_id) if isinstance(value, dict) else self.source_run_id
            if selected:
                matches = widget.findItems(f"*{selected}*", Qt.MatchWildcard | Qt.MatchRecursive, 0)
                if matches:
                    widget.setCurrentItem(matches[0])
            widget.blockSignals(False)

        def select_source_history(self, item, _previous=None):
            if item is None:
                return
            run_id = str(item.data(0, Qt.UserRole) or "").strip()
            if run_id and run_id != self.source_run_id:
                self.task("GET", f"/api/source-runs/{run_id}", None, "history-run-selected")
        def retry_source_run(self):
            run_id = getattr(self, "source_run_id", "")
            if not run_id: QMessageBox.information(self, "没有单图运行", "请先运行单图自动生成"); return
            self.request_creative_task_plan('create_reference', source=self.source_run_source, parent_run_id=run_id)
        def copy_source_run(self):
            if not self.source_run_id: return
            self.task("POST", f"/api/source-runs/{self.source_run_id}/copy", {}, "source-copy")
        def adjust_source_run(self):
            self.save_source_lights()
        def load(self, value):
            report = value.get('report') if isinstance(value.get('report'), dict) else {}
            self.log_text('[status] 当前原片工作区：' + str(report.get('workflow_mode') or report.get('kind') or 'single_source'))
        def alpha_edits_changed(self, edits):
            if not self.source_sample and not self.sample(): return
            try: data = json.loads(self.alpha.toPlainText())
            except Exception: data = {"schema_version":"alpha_revision_draft.v1","revision":1}
            if not data.get("base_alpha_sha256") and self.alpha_canvas.source and self.alpha_canvas.source.is_file():
                try: data["base_alpha_sha256"] = hashlib.sha256(self.alpha_canvas.source.read_bytes()).hexdigest()
                except OSError: pass
            regions = [e.get("region") for e in edits if isinstance(e, dict) and e.get("region")]
            data.update({"sample":self.source_sample or self.sample(),"edits":edits,
                         "stroke_count": len({str(e.get("stroke_id")) for e in edits if isinstance(e, dict) and e.get("stroke_id") is not None}),
                         "edit_regions": regions, "regions": regions})
            self.alpha.blockSignals(True); self.alpha.setPlainText(json.dumps(data, ensure_ascii=False, indent=2)); self.alpha.blockSignals(False)
        def update_appearance_parameter(self, key, value, output_label):
            output_label.setText(str(value))
            try: data = json.loads(self.appearance.toPlainText())
            except Exception: data = {"schema_version":"appearance_revision_draft.v1","revision":1,"params":{}}
            parameters = data.setdefault("params", {})
            parameters[key] = round(float(value) / 100.0, 3)
            self.appearance.blockSignals(True); self.appearance.setPlainText(json.dumps(data, ensure_ascii=False, indent=2)); self.appearance.blockSignals(False)

        def _appearance_revision_payload(self, appearance):
            """Build the current params and regions editing contract."""
            raw = appearance.get("params", {}) if isinstance(appearance, dict) else {}
            params = dict(raw) if isinstance(raw, dict) else {}
            keys = ("exposure", "contrast", "highlights", "shadows", "saturation", "temperature", "tint", "sharpness")
            base = {}
            for key in keys:
                if key in params and not isinstance(params.get(key), (dict, list)):
                    value = params.get(key, 0.0)
                elif key in getattr(self, "appearance_sliders", {}):
                    value = self.appearance_sliders[key].value() / 100.0
                else:
                    value = 0.0
                try: base[key] = float(value or 0.0)
                except (TypeError, ValueError): base[key] = 0.0
            nested = {name: (dict(params.get(name, {})) if isinstance(params.get(name), dict) else {}) for name in ("global", "subject", "background")}
            scope = self._appearance_scope_options()
            target = scope.get("scope", "global")
            if target in {"global", "subject", "background"}:
                nested[target].update(base)
            else:
                local = {"rect": scope.get("local_rect", [0, 0, 1, 1]), "scope": scope.get("local_scope", "global"), "feather": scope.get("local_feather", 0.0), **base, "strength": 1.0}
                existing = params.get("local")
                params["local"] = [local] if not isinstance(existing, list) else ([*existing[:-1], local] if existing else [local])
            params.update(nested)
            params["scope"] = target
            params["local_rect"] = scope.get("local_rect", [0, 0, 1, 1])
            params["local_scope"] = scope.get("local_scope", "global")
            params["local_feather"] = scope.get("local_feather", 0.0)
            revision = {**appearance, "params": params, "regions": {key: params[key] for key in ("global", "subject", "background", "local") if key in params}}
            from scene_receiver_intent import normalize_receiver_intent
            revision["receiver_intent"] = normalize_receiver_intent(self.receiver_intent,
                source_canvas_size=self.receiver_source_canvas_size)
            from authored_receiver_shadow import normalize_receiver_shadow_plan
            revision['receiver_shadow_plan'] = normalize_receiver_shadow_plan(self.receiver_shadow_plan)
            return self.add_local_plans_to_revision(revision)
        def generate(self):
            source = self.source_run_source if self.source_run_id and self.source_run_source else self.source_edit.text().strip()
            if not source: QMessageBox.information(self, "缺少原图", "请在项目页选择原图，或打开已有单图运行。"); return
            self.request_creative_task_plan('create_reference', source=source,
                parent_run_id=self.source_run_id if self.source_run_id and self.source_run_source else None)
        def save_revision(self, kind):
            if kind != 'appearance':
                raise ValueError('Use the current native matte editor for foreground edges')
            if self.source_run_id:
                self.save_source_lights()
            else:
                QMessageBox.information(self, '需要原片版本', '请先打开或创建原片版本，再保存编辑。')
        def start_backend(self):
            if self.embedded_backend:
                if self.embedded_server is not None:
                    self.log_text("[backend] embedded server already running"); return
                try:
                    import backend_server
                    requested = self.url
                    self.embedded_server, self.url = backend_server.build_embedded_server(self.root_path, requested)
                    self.backend_generation += 1
                    self.url_edit.setText(self.url)
                    threading.Thread(target=self.embedded_server.serve_forever, daemon=True).start()
                    self.log_text('[backend] ' + self.url + ' / ' + str(self.root_path))
                    if requested != self.url: self.log_text('[backend] 已为当前工程分配独立后台地址。')
                    QTimer.singleShot(300, self.refresh)
                except Exception as exc:
                    self.error(str(exc))
                return
            if self.process and self.process.poll() is None: self.log_text("[backend] already running"); return
            self._checkpoint_session()
            if self._session_checkpoint_failed: return
            self.root_path = Path(self.root_edit.text()).expanduser().resolve()
            self._session_binding = None; self._session_written = None
            try:
                from runtime_resources import code_path
                parsed = urlparse(self.url_edit.text().strip())
                if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost'}:
                    raise ValueError('启动本地后台需要 http://127.0.0.1:<端口> 地址。')
                port = parsed.port if parsed.port is not None else 8787
                if port == 0: raise ValueError('独立进程后台须指定端口；自动分配端口请使用内嵌后台。')
                self.process = subprocess.Popen([sys.executable, str(code_path('backend_server.py')), "--root", str(self.root_path), '--host', '127.0.0.1', '--port', str(port)], cwd=str(self.root_path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
                self.log_text("[backend] started"); QTimer.singleShot(700, self.refresh); threading.Thread(target=self.read_backend, daemon=True).start()
            except Exception as exc: self.error(str(exc))
        def read_backend(self):
            if self.process and self.process.stdout:
                for line in self.process.stdout: self.backendLog.emit("[backend] " + line)
        def closeEvent(self, event):
            self._checkpoint_session()
            if self._session_checkpoint_failed:
                self.session_draft_state.setText("会话保留失败，窗口保持打开；可重试保存或关闭。")
                event.ignore(); return
            self.cancel_startup_service(closing=True)
            self.shaping_inspection_closed = True; self.shaping_adoption = None
            for kind in tuple(self.shaping_inspections): self._discard_shaping_inspection(kind)
            self.session_timer.stop(); self.source_light_timer.stop(); self.native_matte_timer.stop(); self.job_poll_timer.stop()
            self.backend_generation += 1
            if self.process and self.process.poll() is None: self.process.terminate()
            if self.embedded_server is not None:
                self.embedded_server.shutdown(); self.embedded_server.server_close(); self.embedded_server = None
            event.accept()

def parse_args(argv: Optional[Iterable[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, default=REPO_ROOT); parser.add_argument("--backend-url", default="http://127.0.0.1:8787"); parser.add_argument("--auto-start-backend", action="store_true", help="start backend immediately after opening the workbench"); parser.add_argument("--embedded-backend", action="store_true", help="serve the backend in the workbench process"); parser.add_argument("--prewarm", action="store_true", help="initialize the selected owned service without image inference; model warming is not implied"); parser.add_argument("--smoke", action="store_true"); return parser.parse_args(argv)

def main(argv: Optional[Iterable[str]] = None):
    args = parse_args(argv)
    if args.smoke:
            native_runtime = native_runtime_smoke(args.root.expanduser().resolve())
            capabilities = capability_declaration()
            smoke = {
                "native_runtime": native_runtime,
                "qt_available": QT_AVAILABLE,
                "qt_import_error": globals().get("QT_IMPORT_ERROR"),
                "tabs": ["项目", "候选", "审片", "交付", "任务"],
                "capabilities": capabilities,
                "visual_check_keys": list(VISUAL_CHECK_KEYS),
                "visual_check_choices": ["not_checked", "pass", "rework", "hold"],
                "delivery_history_supported": True,
                "run_history_tree_supported": True,
                "revision_copy_supported": True,
                "durable_job_list_supported": True,
                "delivery_rollback_supported": True,
                "risk_crop_navigation_supported": True,
                "preview_pan_supported": True,
                "native_alpha_revision_supported": False,
                "manual_layer_controls_supported": True,
                "job_event_stream_supported": True,
                "preset_selection_supported": True,
                "modern_workspace_layout_supported": True,
                "default_workspace": "simple_cos",
                "professional_workflow_canvas_supported": True,
                "workflow_arbitrary_connections_supported": False,
                "shared_workspace_draft_supported": True,
                "portrait_beauty_supported": True,
                "source_preview_worker_supported": True,
                "collapsed_advanced_configuration_supported": True,
                "saved_observed_design_view_supported": True,
                "saved_design_readonly": True,
                "character_recognition_supported": True,
                "character_optional_supported": True,
                "character_recommended_styles_supported": True,
                "variant_count_range": [1, 8],
                "variant_thumbnail_browser_supported": True,
                "persistent_provider_settings_supported": True,
                "workbench_templates_supported": True,
                "portable_project_supported": True,
                "portable_project_format": ".cosplayproj",
                "portable_project_export_saved_revision_only": True,
                "recent_projects_supported": True,
                "local_feather_supported": True,
                "project_cleanup_supported": capabilities["project_cleanup_supported"],
                "project_cleanup_scope": capabilities["project_cleanup_scope"],
                "batch_queue_supported": True,
                "creative_task_modes": list(CREATIVE_TASK_MODES),
                "creative_task_plan_before_start": True,
                "creative_task_batch_shared_brief": True,
                "export_formats": capabilities["native"]["export_formats"],
                "export_actual_bit_depth": capabilities["native"]["export_actual_bit_depth"],
                "render_master_supported": True,
                "smoke_scope": "runtime_import_and_capability_declaration",
                "endpoints": ["/api/source-runs/<id>/complete-post",
                    "/api/status", "/api/recovery", "/api/cleanup", "/api/providers", "/api/provider-settings",
                    "/api/provider-settings/discover", "/api/templates", "/api/recent", "/api/batch",
                    "/api/batch/<id>",
                    "/api/source-runs/export-batch", "/api/providers/health", "/api/providers/dry-run", "/api/presets",
                    "/api/jobs", "/api/jobs/<id>", "/api/jobs/<id>/events", "/api/jobs/<id>/events/stream",
                    "/api/jobs/<id>/cancel", "/api/source-deliveries/<sample>",
                    "/api/source-deliveries/<sample>/rollback", "/api/tasks/plan", "/api/tasks/start",
                    "/api/tasks/batch/plan", "/api/tasks/batch", "/api/tasks/batch/<id>/retry-item", "/api/character/recognize",
                    "/api/source-runs/<id>/select-variant", "/api/source-runs/<id>",
                    "/api/source-runs/<id>/history", "/api/source-runs/<id>/accept", "/api/source-runs/<id>/retry",
                    "/api/source-runs/<id>/adjust", "/api/source-runs/<id>/copy",
                    "/api/source-runs/<id>/reconstruct-design",
                    "/api/source-runs/<id>/export-project", "/api/projects/import", "/api/source-runs/<id>/export",
                    "/api/source-runs/<id>/rollback",
                ],
                "revision_drafts": {"native": ["appearance_revision_draft.json"]},
                "embedded_backend_supported": True,
            }
            print(json.dumps(smoke, ensure_ascii=False, indent=2))
            return 0 if QT_AVAILABLE and native_runtime["ok"] else 2
    if not QT_AVAILABLE:
        print("PySide6 is not installed; install the bundled/runtime PySide6 package before starting the Qt workbench.", file=sys.stderr); return 2
    frozen = bool(getattr(sys, 'frozen', False))
    if frozen:
        os.environ.setdefault('COSPLAY_CONFIG_ROOT', str(Path.home()/'CosplayWorkbench/config'))
    app = QApplication(sys.argv); window = Workbench(args.root.expanduser().resolve(), args.backend_url, args.embedded_backend or frozen)
    startup = args.auto_start_backend or frozen
    if startup:
        # Bind the embedded backend before showing the window. The first paint
        # therefore lands on a live local project instead of a cold connection.
        window.set_startup_readiness('initializing')
        window.start_backend()
    window.show()
    if args.prewarm or startup:
        QTimer.singleShot(0, window.prewarm_startup)
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())





