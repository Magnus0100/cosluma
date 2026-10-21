"""Saved-parent read ownership: real Win32 files/maps, no photo or model jobs."""
from __future__ import annotations

from argparse import Namespace
from contextvars import copy_context
import ctypes
from ctypes import wintypes as w
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import appearance_read_scope as reads


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class SavedReadScopeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='saved-input-reads-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.parent = self.root/'outputs/runs/saved-parent'
        self.paths = [self.parent/name for name in ('current', 'base', 'source')]
        for index, path in enumerate(self.paths):
            path.mkdir(parents=True)
            (path/'master.npy').write_bytes(bytes([30+index])*reads.MIN_BYTES+b'payload')
            (path/'receipt.json').write_text('{"synthetic":true}\n', encoding='utf8')
        def descriptor(path, value):
            return {'path':str(path), 'asset_id':str(value)*64, 'receipt_sha256':str(value+3)*64}
        self.manifest = {'options':{'photo_appearance':descriptor(self.paths[0], 1),
            'shaping_state':{'base_appearance':descriptor(self.paths[1], 2)}},
            'initial_source_appearance':descriptor(self.paths[2], 3)}
        self.write_manifest()
        self.body = {'parent_run_id':'saved-parent'}
        self.master = self.paths[0]/'master.npy'
        self.calls = []

    def write_manifest(self):
        (self.parent/'manifest.json').write_text(json.dumps(self.manifest), encoding='utf8')

    def callback(self, path=None):
        path = path or self.master
        def calculate():
            self.calls.append(str(path))
            return digest(path)
        return calculate

    def require_windows(self):
        if not reads.support(self.root):
            self.skipTest('Requires actual local fixed NTFS Win32 read sharing')

    def write_open(self, path):
        kernel = reads._api()[0]
        handle = kernel.CreateFileW(str(path), 0xC0000000, 7, None, 3, 0x00200000, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    def test_outside_scope_always_calls_actual_hash(self):
        self.assertEqual(reads.audit(self.master,self.callback()),reads.audit(self.master,self.callback()))
        self.assertEqual(len(self.calls),2)

    def test_bad_body_and_manifest_do_not_replace_owner_validation(self):
        @reads.saved_parent_reads
        def owner(root, body):
            if not isinstance(body,dict):
                raise ValueError('original request error')
            return body
        for body in (None,False,23,[], 'wrong'):
            with self.subTest(body=body), self.assertRaisesRegex(ValueError,'original request error'):
                owner(self.root,body)
        (self.parent/'manifest.json').write_text('{broken',encoding='utf8')
        with reads.parent_operation(self.root,self.body) as scope:
            self.assertFalse(scope.enabled)
            reads.audit(self.master,self.callback());reads.audit(self.master,self.callback())
        self.assertTrue(scope.closed)
        self.assertEqual(len(self.calls),2)

    def test_unsupported_volume_keeps_full_callback(self):
        with patch.object(reads,'support',return_value=False):
            with reads.parent_operation(self.root,self.body) as scope:
                self.assertFalse(scope.enabled)
                reads.audit(self.master,self.callback());reads.audit(self.master,self.callback())
        self.assertEqual(len(self.calls),2)

    def test_safe_component_and_dotdot_never_enroll(self):
        invalid = ('..','../other','x/y','x\\y','C:other','nul','x.', 'x ')
        for value in invalid:
            with self.subTest(value=value),reads.parent_operation(self.root,{'parent_run_id':value}) as scope:
                self.assertFalse(scope.enabled)
        for spelling in (str(self.parent/'current/../current'),str(self.parent/'../elsewhere')):
            self.manifest = {'options':{'photo_appearance':{'path':spelling,'asset_id':'a'*64,'receipt_sha256':'b'*64}}}
            self.write_manifest()
            with reads.parent_operation(self.root,self.body) as scope:
                self.assertFalse(scope.enabled)

    def test_live_scope_hashes_once_closes_and_next_operation_rehashes(self):
        self.require_windows()
        with reads.parent_operation(self.root,self.body) as scope:
            self.assertTrue(scope.enabled)
            before=reads.audit(self.master,self.callback())
            self.assertEqual(before,reads.audit(self.master,self.callback()))
            self.assertEqual(len(self.calls),1)
        self.assertTrue(scope.closed)
        old=self.master.stat();content=bytearray(self.master.read_bytes());content[0]^=255
        self.master.write_bytes(content);os.utime(self.master,ns=(old.st_atime_ns,old.st_mtime_ns))
        self.assertEqual(self.master.stat().st_mtime_ns,old.st_mtime_ns)
        with reads.parent_operation(self.root,self.body):
            after=reads.audit(self.master,self.callback())
        self.assertNotEqual(before,after)
        self.assertEqual(len(self.calls),2)

    def test_only_three_typed_roots_and_large_files_are_eligible(self):
        self.require_windows()
        unknown=self.parent/'unrelated';unknown.mkdir();outside=unknown/'master.npy'
        outside.write_bytes(self.master.read_bytes())
        for path in [*(p/'master.npy' for p in self.paths),self.paths[0]/'receipt.json',outside]:
            with self.subTest(path=path),reads.parent_operation(self.root,self.body):
                before=len(self.calls)
                reads.audit(path,self.callback(path));reads.audit(path,self.callback(path))
                self.assertEqual(len(self.calls)-before,1 if path in [p/'master.npy' for p in self.paths] else 2)

    def test_new_child_and_every_partial_ancestor_stay_uncached(self):
        self.require_windows()
        child=self.root/'outputs/runs/new-child/inputs/asset';child.mkdir(parents=True)
        fresh=child/'master.npy';fresh.write_bytes(self.master.read_bytes())
        with reads.parent_operation(self.root,self.body):
            reads.audit(fresh,self.callback(fresh));reads.audit(fresh,self.callback(fresh))
            fresh.rename(child/'renamed.npy')
        self.assertEqual(len(self.calls),2)
        partial=self.parent/'.evidence.partial-owned/assets/asset';partial.mkdir(parents=True)
        pending=partial/'master.npy';pending.write_bytes(self.master.read_bytes())
        self.manifest={'options':{'photo_appearance':{'path':str(partial),'asset_id':'a'*64,'receipt_sha256':'b'*64}}}
        self.write_manifest()
        with reads.parent_operation(self.root,self.body) as scope:
            self.assertFalse(scope.enabled)
            reads.audit(pending,self.callback(pending));reads.audit(pending,self.callback(pending))
            (partial.parent.parent).rename(self.parent/'evidence-complete')
        self.assertEqual(len(self.calls),4)

    def test_nested_scope_does_not_expand_to_another_parent(self):
        self.require_windows()
        other=self.root/'outputs/runs/another';other.mkdir(parents=True)
        copied=other/'current';copied.mkdir();file=copied/'master.npy';file.write_bytes(self.master.read_bytes())
        (other/'manifest.json').write_text(json.dumps({'options':{'photo_appearance':{
            'path':str(copied),'asset_id':'a'*64,'receipt_sha256':'b'*64}}}),encoding='utf8')
        with reads.parent_operation(self.root,self.body) as outer:
            reads.audit(self.master,self.callback())
            with reads.parent_operation(self.root,{'parent_run_id':'another'}) as nested:
                self.assertIs(outer,nested)
                reads.audit(file,self.callback(file));reads.audit(file,self.callback(file))
                reads.audit(self.master,self.callback())
            self.assertFalse(outer.closed)
        self.assertEqual(len(self.calls),3)

    def test_initial_inventory_excludes_later_members_and_replaced_before_first_audit(self):
        self.require_windows()
        late=self.paths[0]/'later.npy'
        replacement=self.root/'replacement.npy';replacement.write_bytes(self.master.read_bytes())
        with reads.parent_operation(self.root,self.body) as scope:
            late.write_bytes(self.master.read_bytes())
            reads.audit(late,self.callback(late));reads.audit(late,self.callback(late))
            self.assertFalse(scope.entries)
            # No first pin has happened yet. A changed member gets full hashes
            # but cannot acquire reuse eligibility from its replacement ID.
            os.replace(replacement,self.master)
            reads.audit(self.master,self.callback());reads.audit(self.master,self.callback())
            self.assertFalse(scope.entries)
        self.assertEqual(len(self.calls),4)

    def test_partial_exact_or_suffix_roots_and_large_descendants_are_excluded(self):
        self.require_windows()
        for name in ('.partial','thing.partial','prefix.partial-stage'):
            with self.subTest(name=name):
                path=self.parent/name/'asset';path.mkdir(parents=True)
                master=path/'master.npy';master.write_bytes(self.master.read_bytes())
                self.manifest={'options':{'photo_appearance':{'path':str(path),
                    'asset_id':'a'*64,'receipt_sha256':'b'*64}}};self.write_manifest()
                with reads.parent_operation(self.root,self.body) as scope:
                    self.assertFalse(scope.enabled)
                    before=len(self.calls)
                    reads.audit(master,self.callback(master));reads.audit(master,self.callback(master))
                    self.assertEqual(len(self.calls)-before,2)

    def test_reserved_handles_block_writer_delete_replace_and_ancestor_move_or_reject_namespace(self):
        self.require_windows()
        replacement=self.root/'replacement';replacement.write_bytes(self.master.read_bytes())
        with reads.parent_operation(self.root,self.body):
            reads.audit(self.master,self.callback())
            with self.assertRaises(OSError):self.write_open(self.master)
            with self.assertRaises(OSError):self.master.unlink()
            with self.assertRaises(OSError):os.replace(replacement,self.master)
            try:
                self.parent.rename(self.parent.with_name('moved-parent'))
            except OSError:
                reads.audit(self.master,self.callback())
            else:
                with self.assertRaises((OSError,ValueError)):
                    reads.audit(self.master,self.callback())
        self.assertEqual(len(self.calls),1)

    def test_hardlink_alias_writer_is_blocked_until_scope_exit(self):
        self.require_windows()
        alias=self.root/'alias.npy';os.link(self.master,alias)
        with reads.parent_operation(self.root,self.body):
            reads.audit(self.master,self.callback())
            with self.assertRaises(OSError):self.write_open(alias)
        reads._close(self.write_open(alias))

    def test_existing_writable_view_rejects_even_after_file_and_mapping_handles_close(self):
        self.require_windows();kernel=reads._api()[0]
        kernel.CreateFileMappingW.argtypes=[w.HANDLE,w.LPVOID,w.DWORD,w.DWORD,w.DWORD,w.LPCWSTR]
        kernel.CreateFileMappingW.restype=w.HANDLE
        kernel.MapViewOfFile.argtypes=[w.HANDLE,w.DWORD,w.DWORD,w.DWORD,ctypes.c_size_t]
        kernel.MapViewOfFile.restype=w.LPVOID
        kernel.UnmapViewOfFile.argtypes=[w.LPCVOID];kernel.UnmapViewOfFile.restype=w.BOOL
        original=self.write_open(self.master);mapping=None;view=None
        try:
            mapping=kernel.CreateFileMappingW(original,None,4,0,0,None)
            self.assertTrue(mapping)
            view=kernel.MapViewOfFile(mapping,2,0,0,0);self.assertTrue(view)
            reads._close(original);original=None
            with reads.parent_operation(self.root,self.body):
                with self.assertRaises(OSError):reads.audit(self.master,self.callback())
            reads._close(mapping);mapping=None
            with reads.parent_operation(self.root,self.body):
                with self.assertRaises(OSError):reads.audit(self.master,self.callback())
            self.assertFalse(self.calls)
        finally:
            if view:self.assertTrue(kernel.UnmapViewOfFile(view))
            if mapping:reads._close(mapping)
            if original:reads._close(original)
        with reads.parent_operation(self.root,self.body):reads.audit(self.master,self.callback())
        self.assertEqual(len(self.calls),1)

    def test_first_callback_failure_and_cancellation_release_the_handle(self):
        self.require_windows()
        with self.assertRaisesRegex(RuntimeError,'hash failed'):
            with reads.parent_operation(self.root,self.body):
                reads.audit(self.master,lambda:(_ for _ in ()).throw(RuntimeError('hash failed')))
        reads._close(self.write_open(self.master))
        flag=[False]
        def calculate():
            result=digest(self.master);flag[0]=True;return result
        with self.assertRaises(InterruptedError):
            with reads.parent_operation(self.root,self.body):
                reads.audit(self.master,calculate,cancel=lambda:flag[0])
        reads._close(self.write_open(self.master))

    def test_preexisting_writer_is_not_a_fallback_or_callback(self):
        self.require_windows();handle=self.write_open(self.master)
        try:
            with reads.parent_operation(self.root,self.body):
                with self.assertRaises(OSError):reads.audit(self.master,self.callback())
            self.assertFalse(self.calls)
        finally:reads._close(handle)

    def test_handle_budget_falls_back_to_real_hashes(self):
        self.require_windows();second=self.paths[1]/'master.npy'
        with reads.parent_operation(self.root,self.body) as scope:
            with patch.object(reads,'MAX_FILES',1):
                reads.audit(self.master,self.callback());reads.audit(self.master,self.callback())
                reads.audit(second,self.callback(second));reads.audit(second,self.callback(second))
                self.assertEqual(len(scope.entries),1)
        self.assertEqual(len(self.calls),3)

    def test_initial_inventory_has_a_finite_member_bound(self):
        self.require_windows()
        with patch.object(reads,'MAX_FILES',1),reads.parent_operation(self.root,self.body) as scope:
            self.assertFalse(scope.enabled)
            reads.audit(self.master,self.callback());reads.audit(self.master,self.callback())
        self.assertEqual(len(self.calls),2)

    def test_threads_and_copied_context_do_not_reuse_the_owner_hash(self):
        self.require_windows();errors=[]
        def hash_in_thread():
            try:reads.audit(self.master,self.callback())
            except BaseException as error:errors.append(error)
        with reads.parent_operation(self.root,self.body):
            reads.audit(self.master,self.callback())
            ctx=copy_context()
            for target in (hash_in_thread,lambda:ctx.run(hash_in_thread)):
                thread=threading.Thread(target=target);thread.start();thread.join(5)
                self.assertFalse(thread.is_alive())
            reads.audit(self.master,self.callback())
        self.assertFalse(errors)
        self.assertEqual(len(self.calls),3)

    def test_pid_change_forces_callback_without_closing_owner_handles(self):
        self.require_windows()
        with reads.parent_operation(self.root,self.body) as scope:
            reads.audit(self.master,self.callback())
            with patch.object(reads.os,'getpid',return_value=scope.pid+1):
                reads.audit(self.master,self.callback())
            reads.audit(self.master,self.callback())
        self.assertEqual(len(self.calls),2)

    def test_another_process_performs_its_own_first_audit(self):
        self.require_windows()
        code='''import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import appearance_read_scope as reads
calls=[]
p=Path(sys.argv[3])
def calculate():
 calls.append(1)
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
with reads.parent_operation(Path(sys.argv[2]),{'parent_run_id':'saved-parent'}):
 a=reads.audit(p,calculate);b=reads.audit(p,calculate)
print(json.dumps({'count':len(calls),'same':a==b}))
'''
        with reads.parent_operation(self.root,self.body):
            reads.audit(self.master,self.callback())
            result=subprocess.run([sys.executable,'-B','-c',code,str(Path(reads.__file__).parent),
                str(self.root),str(self.master)],capture_output=True,text=True,timeout=20,close_fds=True,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout),{'count':1,'same':True})
        self.assertEqual(len(self.calls),1)

    def test_decorator_preserves_namespace_signature_and_original_result(self):
        self.require_windows()
        @reads.saved_parent_reads
        def wrapped(root,workflow_args):
            reads.audit(self.master,self.callback());reads.audit(self.master,self.callback())
            return workflow_args
        args=Namespace(parent_run_id='saved-parent')
        self.assertIs(wrapped(self.root,args),args)
        self.assertEqual(wrapped.__name__,'wrapped')
        self.assertEqual(len(self.calls),1)


if __name__=='__main__':unittest.main()
