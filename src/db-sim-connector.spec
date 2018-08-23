# -*- mode: python -*-

# Requires Windows 10 SDK
# Requires matplotlib, numpy, scipy, and pyqt5

block_cipher = None


a = Analysis(['db-sim-connector.py'],
             pathex=['C:\\Program Files (x86)\\Windows Kits\\10\\Redist\\ucrt\\DLLs\\x64', 'C:\\Program Files\\Python36\\Lib\\site-packages\\scipy\\extra-dll', 'C:\\Users\\Samuel Ng\\Desktop\\python'],
             binaries=[],
             datas=[('stylesheets/animator.qss', 'stylesheets')],
             hiddenimports=['scipy._lib.messagestream'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='db-sim-connector',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
