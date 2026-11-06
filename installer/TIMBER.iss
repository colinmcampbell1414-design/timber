; Installer for TIMBER (Inno Setup 6, free: https://jrsoftware.org/isinfo.php).
; Build it with  .\build_installer.ps1  (it builds TIMBER.exe first if needed).
;
; What the installer does:
; - installs just for the current user by default (no administrator password
;   needed, so it also works on school laptops) into %LOCALAPPDATA%\Programs\TIMBER;
;   an administrator can choose "all users" instead (Program Files). No surveys
;   are included;
; - adds TIMBER to the Start menu (and optionally the desktop), with an uninstaller
;   in Settings > Apps.
; Each user's surveys go in Documents\TIMBER\data, which TIMBER creates the first
; time it starts; saved results and the log go in %LOCALAPPDATA%\TIMBER (see
; timber.py user_folders). The uninstaller never deletes surveys.

#define AppName "TIMBER"
#define AppVersion "1.0.0"
#define AppFullName "TIMBER - Tree Instability Mapping By Elevation & Reflectance"

[Setup]
; AppId identifies TIMBER to Windows so new versions replace old ones. Never change it.
AppId={{A8B004D9-7F51-4745-ACE4-53CD1DC51413}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppFullName} {#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installer
OutputBaseFilename=TIMBER-Setup-{#AppVersion}
SetupIconFile=..\assets\timber.ico
UninstallDisplayIcon={app}\TIMBER.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Put a TIMBER shortcut on the desktop"; Flags: unchecked

[Files]
; Everything in the portable build except surveys, saved results, the log and the
; "portable" marker (an installed copy keeps those in the user's own folders).
Source: "..\dist\TIMBER\*"; DestDir: "{app}"; Excludes: "\data\*,\cache\*,\portable.txt,\timber.log"; \
    Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\TIMBER.exe"; Comment: "{#AppFullName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\TIMBER.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\TIMBER.exe"; Description: "Start TIMBER now"; Flags: nowait postinstall skipifsilent
